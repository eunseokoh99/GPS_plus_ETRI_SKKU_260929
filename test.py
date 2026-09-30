from __future__ import print_function, division

import argparse
import importlib
import logging

import numpy as np
import cv2
import os
import random
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from lib.human_loader import StereoHumanDataset
from lib.runtime import freeze_raft_bn, novel_view_to_cuda, pick_render_views
from lib.train_recoder import Logger, file_backup
from lib.GaussianRender import pts2render
from lib.gs_utils.loss_utils import l1_loss, ssim
from lib.gs_utils.image_utils import psnr

from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

import warnings
import trimesh 
warnings.filterwarnings("ignore", category=UserWarning)


class Trainer:
    def __init__(self, cfg_file):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size

        
        from train import build_model, build_dataset  # same dispatch as training
        self.model = build_model(self.cfg)

        self.val_set = build_dataset(self.cfg, self.cfg.seq_name)
        self.val_loader = DataLoader(self.val_set, batch_size=self.bs, shuffle=False, num_workers=8, pin_memory=True)
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
        self.val_iterator = iter(self.val_loader)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.cfg.lr, self.cfg.num_steps + 100,
                                                       pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

        self.logger = Logger(self.scheduler, self.cfg.record)
        self.total_steps = 0

        self.model.cuda()
        if self.cfg.restore_ckpt:
            print('load good ckpt')
            self.load_ckpt(self.cfg.restore_ckpt, load_optimizer=False)

        self.model.eval()
        freeze_raft_bn(self.model)  # We keep BatchNorm frozen in Raft-Stereo

        # Source views to move to GPU / merge when rendering, and the
        # nearest-k render selection -- identical to training.
        self.view_keys = list(getattr(self.val_set, 'view_keys', ['lmain', 'rmain']))
        self.render_nearest_k = int(getattr(self.cfg.dataset, 'render_nearest_k', 0))
        self.view_coincide_tol = float(getattr(self.cfg.dataset, 'view_coincide_tol', 1e-4))
        self.scaler = GradScaler(enabled=self.cfg.raft.mixed_precision)


    def val(self):
        logging.info(f"Doing validation ...")
        torch.cuda.empty_cache()
        psnr_list = []
        for idx in tqdm(range(self.len_val)):
            data = self.fetch_data(phase='val')

            view_id = data['novel_view']['view_id'][0,0].item()
            s_name = data['novel_view']['sample_name']

            with torch.no_grad():
                data, _, _ = self.model(data, is_train=False)
                # fetch_data only moves the source views to GPU.
                data['novel_view'] = novel_view_to_cuda(data['novel_view'])
                data = pts2render(
                    data,
                    bg_color=self.cfg.dataset.bg_color,
                    source_views=pick_render_views(
                        data, data['novel_view'], view_keys=self.view_keys,
                        k=self.render_nearest_k,
                        coincide_tol=self.view_coincide_tol, rng=None,
                    ),
                )

                tmp_novel = data['novel_view']['img_pred'][0].detach()
                tmp_novel *= 255
                tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()

                # Optionally drop the top/bottom `test_save_hcrop` fraction of
                # rows before saving (0.0 = no crop).
                hcrop = float(getattr(self.cfg.dataset, 'test_save_hcrop', 0.0))
                if hcrop > 0.0:
                    h = tmp_novel.shape[0]
                    top = int(round(h * hcrop))
                    if top > 0:
                        tmp_novel = tmp_novel[top:h - top, :, :]

                tmp_img_name = '%s/%s_%02d.png' % (self.cfg.record.show_path, s_name[0], view_id)
                cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))
 

    def fetch_data(self, phase):
        if phase == 'train':
            try:
                data = next(self.train_iterator)
            except:
                self.train_iterator = iter(self.train_loader)
                data = next(self.train_iterator)
        elif phase == 'val':
            try:
                data = next(self.val_iterator)
            except:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)

        for view in self.view_keys:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda()
        return data

    def load_ckpt(self, load_path, load_optimizer=True, strict=True):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=strict)
        logging.info(f"Parameter loading done")
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1
            self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            logging.info(f"Optimizer loading done")

    def save_ckpt(self, save_path, show_log=True):
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)


if __name__ == '__main__':
    # python test.py --config ./config/exp1:src_sparse_view \
    #                --ckpt experiments/<exp>/ckpt/iterXXXXX.pth [--phase val] [--view 3]

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    parser = argparse.ArgumentParser(description='GPS_plus testing')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the config directory (same layout as train.py: '
                             'contains stereo_human_config.py and stage.yaml)')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to the checkpoint (.pth) to evaluate')
    parser.add_argument('--phase', type=str, default='val',
                        help="Dataset split to test on: 'val' / 'train' (uses the "
                             "*_data_root from the config), or a custom '<seq>_process' "
                             "set read from <local_data_root>/test/<phase>")
    parser.add_argument('--view', type=int, nargs='+', default=None,
                        help='Novel view id(s) to render; defaults to '
                             'cfg.dataset.val_novel_id from the config')
    parser.add_argument('--show_path', type=str, default=None,
                        help='Directory for rendered pngs; defaults to '
                             'experiments/<exp_name>/test_show_<phase>')
    args = parser.parse_args()

    # Resolve the config directory the same way train.py does: import its
    # ConfigStereoHuman class and load its stage.yaml.
    config_dir = Path(args.config)
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory does not exist: {config_dir}")
    _config = args.config.rstrip('/').replace('./', '').replace('/', '.')
    config_module = importlib.import_module('%s.stereo_human_config' % _config)
    ConfigStereoHuman = getattr(config_module, 'ConfigStereoHuman')

    cfg = ConfigStereoHuman()
    cfg.load(str(config_dir / 'stage.yaml'))
    cfg = cfg.get_cfg()

    cfg.defrost()
    dt = datetime.today()
    cfg.exp_name = '%s_%s%s' % (cfg.name, str(dt.month).zfill(2), str(dt.day).zfill(2))
    cfg.seq_name = args.phase
    if args.view is not None:
        cfg.dataset.val_novel_id = list(args.view)
    if args.show_path is not None:
        cfg.record.show_path = args.show_path
    else:
        cfg.record.show_path = "experiments/%s/test_show_%s" % (cfg.exp_name, args.phase)
    cfg.restore_ckpt = args.ckpt
    cfg.freeze()

    print('config     :', str(config_dir))
    print('restore_ckpt:', cfg.restore_ckpt)
    print('phase      :', cfg.seq_name)
    print('views      :', cfg.dataset.val_novel_id)
    print('show_path  :', cfg.record.show_path)

    Path(cfg.record.show_path).mkdir(exist_ok=True, parents=True)

    trainer = Trainer(cfg)
    trainer.val()
