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
from lib.runtime import (
    benchmark_forward,
    eval_psnr_pass,
    freeze_raft_bn,
    pick_render_views,
)
from lib.train_recoder import Logger, file_backup

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
        self.novel_view_ids = list(self.cfg.dataset.val_novel_id)
        # Forward-speed benchmark run after the validation pass. Set
        # speed_iters to 0 to skip it.
        self.speed_warmup = int(getattr(self.cfg, 'speed_warmup', 50))
        self.speed_iters = int(getattr(self.cfg, 'speed_iters', 100))
        self.scaler = GradScaler(enabled=self.cfg.raft.mixed_precision)


    def val(self):
        """Render every val sample and report PSNR + forward speed.

        Scoring goes through ``lib/runtime.eval_psnr_pass``, the same function
        train.py's periodic validation uses, so the PSNR printed here matches
        the numbers in README.md exactly.
        """
        logging.info("Doing validation ...")
        torch.cuda.empty_cache()
        self.val_iterator = iter(self.val_loader)

        save_hcrop = float(getattr(self.cfg.dataset, 'test_save_hcrop', 0.0))
        bar = tqdm(total=self.len_val)

        def _save(idx, sample_name, view_id, render_data):
            tmp_novel = render_data['novel_view']['img_pred'][0].detach()
            tmp_novel *= 255
            tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
            # Optionally drop the top/bottom `test_save_hcrop` fraction of rows
            # before saving (0.0 = no crop). Scoring is unaffected -- that uses
            # dataset.eval_img_hcrop.
            if save_hcrop > 0.0:
                h = tmp_novel.shape[0]
                top = int(round(h * save_hcrop))
                if top > 0:
                    tmp_novel = tmp_novel[top:h - top, :, :]
            cv2.imwrite(
                '%s/%s_%02d.png' % (self.cfg.record.show_path, sample_name, view_id),
                tmp_novel[:, :, ::-1].astype(np.uint8),
            )
            bar.update(1 / max(len(self.novel_view_ids), 1))

        out = eval_psnr_pass(
            model=self.model,
            val_set=self.val_set,
            fetch_data=self.fetch_data,
            len_val=self.len_val,
            novel_view_ids=self.novel_view_ids,
            bg_color=self.cfg.dataset.bg_color,
            hcrop=float(getattr(self.cfg.dataset, 'eval_img_hcrop', 0.0)),
            pick_views=lambda d, nv: pick_render_views(
                d, nv, view_keys=self.view_keys, k=self.render_nearest_k,
                coincide_tol=self.view_coincide_tol, rng=None,
            ),
            on_sample=_save,
        )
        bar.close()

        # Speed is measured separately, on one batch with nothing else running --
        # the per-sample time inside the loop above moves with dataloader and
        # GPU load. See lib/runtime.benchmark_forward.
        speed = None
        if self.speed_iters > 0:
            speed = benchmark_forward(
                model=self.model, batches=self._speed_batches(),
                warmup=self.speed_warmup, iters=self.speed_iters,
            )

        hcrop = float(getattr(self.cfg.dataset, 'eval_img_hcrop', 0.0))
        print()
        print('=' * 62)
        print(f"  config        {self.cfg.name}")
        print(f"  checkpoint    {self.cfg.restore_ckpt}")
        print(f"  samples       {self.len_val} x {len(self.novel_view_ids)} novel view "
              f"= {out['n_scored']} scored")
        print(f"  PSNR (full)   {out['full_psnr']}")
        print(f"  PSNR (crop)   {out['crop_psnr']}   (eval_img_hcrop={hcrop})")
        if speed is not None:
            print(f"  forward       {speed['mean_ms']} ms   "
                  f"(median {speed['median_ms']}, min {speed['min_ms']}, "
                  f"std {speed['std_ms']})")
            print(f"  peak memory   {speed['peak_mem_MB']} MB")
            print(f"                {speed['forwards']} forward(s) covering all 4 "
                  f"source cameras of one frame,")
            print(f"                {speed['iters']} iterations after "
                  f"{speed['warmup']} warm-up, inputs already on the GPU;")
            print(f"                model forward only -- data IO and render excluded.")
            print(f"                Needs an otherwise idle GPU to be comparable.")
        print(f"  renders       {self.cfg.record.show_path}")
        print('=' * 62)
        return out

    def _speed_batches(self):
        """Batches whose forwards together cover all 4 source cameras of a frame.

        4-view branches need one (all four cameras go in together); 2-view
        branches need the three camera pairs s1/s2/s3 of the same frame, since
        the model takes one pair at a time. Staged on the GPU before timing so
        data IO stays out of the measurement.
        """
        from torch.utils.data.dataloader import default_collate

        base = self.val_set.sample_list[0]
        if len(self.view_keys) > 2:
            names = [base]
        else:
            # '<seq>_s<N>_<frame>' -> the s1/s2/s3 siblings of the same frame.
            seq, _, frame = base.split('_')
            names = [f'{seq}_s{g}_{frame}' for g in (1, 2, 3)]
            names = [n for n in names if n in self.val_set.sample_list]

        batches = []
        for n in names:
            b = default_collate([self.val_set[self.val_set.sample_list.index(n)]])
            for v in self.view_keys:
                for k in b[v]:
                    b[v][k] = b[v][k].cuda()
            batches.append(b)
        torch.cuda.synchronize()
        logging.info("Speed benchmark input: %s", names)
        return batches

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
    parser.add_argument('--opts', nargs='*', default=[],
                        help='Override config entries as space-separated '
                             'key value pairs, e.g. '
                             '--opts dataset.val_data_root /data/preprocessed/val. '
                             'Only keys declared in that branch\'s '
                             'stereo_human_config.py are accepted.')
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
    if args.opts:
        cfg.cfg.defrost()
        cfg.cfg.merge_from_list(args.opts)
        cfg.cfg.freeze()
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
