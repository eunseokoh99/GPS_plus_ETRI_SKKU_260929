from __future__ import print_function, division

import argparse
import logging

import numpy as np
import cv2
import os
import random
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from lib.human_loader import StereoHumanDataset, MultiViewStereoHumanDataset
from lib.network import RtStereoHumanModel
from lib.runtime import freeze_raft_bn, novel_view_to_cuda, pick_render_views
from models.dav3_model import DAV3Model_MK
from models.dav3_mv_model import DAV3Model_MK_Upsampler_MV
import importlib
from lib.train_recoder import Logger, file_backup
from lib.GaussianRender import pts2render
from lib.gs_utils.loss_utils import l1_loss, ssim
from lib.gs_utils.image_utils import psnr
from pytorch3d.loss import chamfer_distance

import trimesh 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import warnings
from contextlib import contextmanager
from copy import deepcopy
warnings.filterwarnings("ignore", category=UserWarning)


# cfg.model_type -> model class. Kept as a plain dict rather than a registry so
# that `import train` stays cheap and eval_psnr_wandb.py / test.py can reuse it.
#
# NOTE (checkpoint contract): the object assigned to Trainer.self.model defines
# the state_dict key prefixes. GPSGS uses the bare net, so its keys are
# un-prefixed (img_encoder.* / loftr_coarse.* / raft_stereo.* /
# gs_parm_regresser.*); the DAv3 entries are thin wrappers whose .model holds
# the real net, so their keys carry a "model." prefix. Both match the
# checkpoints already on disk -- do not "simplify" either side.
MODELS = {
    'GPSGS': RtStereoHumanModel,
    'DAV3Model_MK': DAV3Model_MK,
    'DAV3Model_MK_Upsampler_MV': DAV3Model_MK_Upsampler_MV,
}


def build_model(cfg):
    model_type = getattr(cfg, 'model_type', None) or 'GPSGS'
    if model_type not in MODELS:
        raise KeyError(
            f"Unknown model_type {model_type!r}. Known: {sorted(MODELS)}"
        )
    logging.info("Building model: %s", model_type)
    return MODELS[model_type](cfg, with_gs_render=True)


def build_dataset(cfg, phase):
    """4+ source views -> the multi-view loader; 2 -> the plain stereo one."""
    if int(cfg.dataset.get('num_source_views', 2)) > 2:
        return MultiViewStereoHumanDataset(cfg.dataset, phase=phase)
    return StereoHumanDataset(cfg.dataset, phase=phase)


class _ModelEMA:
    """Exponential moving average of a model's trainable parameters.

    Only floating-point parameters with ``requires_grad=True`` are tracked, so
    an EMA checkpoint reuses the live ``state_dict`` with the shadow values
    substituted in (see ``state_dict_with_shadow``).
    """

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow = {
            name: p.detach().clone().float()
            for name, p in model.named_parameters()
            if p.requires_grad and p.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for name, p in model.named_parameters():
            s = self.shadow.get(name)
            if s is not None:
                s.mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    def state_dict_with_shadow(self, model):
        """Full model ``state_dict`` with EMA params substituted (for saving)."""
        sd = model.state_dict()
        for name, s in self.shadow.items():
            if name in sd:
                sd[name] = s.to(dtype=sd[name].dtype, device=sd[name].device)
        return sd

    @contextmanager
    def swapped_in(self, model):
        """Temporarily load the EMA params into the live model (for eval),
        restoring the raw weights on exit."""
        params = dict(model.named_parameters())
        backup = {}
        try:
            for name, s in self.shadow.items():
                p = params.get(name)
                if p is not None:
                    backup[name] = p.detach().clone()
                    p.data.copy_(s.to(dtype=p.dtype, device=p.device))
            yield
        finally:
            for name, b in backup.items():
                params[name].data.copy_(b)


class Trainer:
    def __init__(self, cfg_file):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size



        # ORDER-CRITICAL: model construction must stay the first torch-RNG
        # consumer, and nothing may be inserted between it and
        # iter(self.train_loader) below -- the epoch-0 shuffle permutation and
        # the 8 worker seeds are drawn from the global CPU RNG at that point,
        # so any extra draw here changes the data order for the whole run.
        self.model = build_model(self.cfg)
        self.train_set = build_dataset(self.cfg, 'train')
        self.train_loader = DataLoader(self.train_set, batch_size=self.bs, shuffle=True, num_workers=8, pin_memory=True)
        self.train_iterator = iter(self.train_loader)
        self.val_set = build_dataset(self.cfg, 'val')
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
        self.val_iterator = iter(self.val_loader)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.cfg.lr, self.cfg.num_steps + 100,
                                                       pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

        self.logger = Logger(self.scheduler, self.cfg.record)
        self.logger.set_wandb(self.cfg.wandb, self.cfg)
        self.total_steps = 0

        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        elif self.cfg.stage1_ckpt:
            logging.info(f"Using checkpoint from stage1")
            self.load_ckpt(self.cfg.stage1_ckpt, load_optimizer=False, strict=False)
        self.model.train()
        self._freeze_raft_bn()  # We keep BatchNorm frozen in Raft-Stereo
        self.scaler = GradScaler(enabled=self.cfg.raft.mixed_precision)

        # Source-view keys to move to GPU in fetch_data and to merge when
        # rendering. 2-view models keep the original stereo pair.
        self.view_keys = list(getattr(self.train_set, 'view_keys', ['lmain', 'rmain']))

        # Chamfer regularization between the two source-view point clouds.
        # ON for the gps_gs baseline, OFF for the DAv3 branches.
        self.use_chamfer = bool(getattr(self.cfg, 'use_chamfer', True))

        # Render each novel view from only its ``k`` nearest source views
        # instead of merging all N (0 = merge all, the original behaviour).
        # See lib/view_select.py for how "nearest" is resolved when the novel
        # camera coincides with a source camera.
        self.render_nearest_k = int(getattr(self.cfg.dataset, 'render_nearest_k', 0))
        self.view_coincide_tol = float(getattr(self.cfg.dataset, 'view_coincide_tol', 1e-4))
        self._view_rng = np.random.default_rng(
            int(getattr(self.cfg.dataset, 'view_select_seed', 1314))
        )
        if self.render_nearest_k > 0:
            logging.info(
                "[render] each novel view rendered from its %d nearest source "
                "views (coincide_tol=%g)",
                self.render_nearest_k, self.view_coincide_tol,
            )

        # Fraction of image height cropped from BOTH the top and the bottom
        # before scoring validation PSNR (0.0 = no crop). Kept 0.0 by default so
        # other configs are unaffected; set e.g. 0.1 to drop the top/bottom 10%.
        self.eval_img_hcrop = float(getattr(self.cfg.dataset, 'eval_img_hcrop', 0.0))

        # EMA model versions. When enabled we keep a shadow copy of the trainable
        # params at each decay, so on top of the raw ("bare") weights we can save
        # and evaluate smoothed variants. ``ema_specs`` maps a tag (used in ckpt
        # filenames and log keys) to a decay; a ``None`` decay is the raw model.
        self.use_ema = bool(getattr(self.cfg.record, 'use_ema', False))
        self.ema_specs = [('bare', None), ('ema1_e3', 0.999), ('ema1_e4', 0.9999)]
        self.emas = {}
        if self.use_ema:
            for tag, decay in self.ema_specs:
                if decay is not None:
                    self.emas[tag] = _ModelEMA(self.model, decay)
            logging.info("[EMA] enabled with decays: %s",
                         {tag: d for tag, d in self.ema_specs if d is not None})

    def _freeze_raft_bn(self):
        freeze_raft_bn(self.model)

    def _novel_view_to_cuda(self, novel_view):
        return novel_view_to_cuda(novel_view)

    def _pick_render_views(self, data, novel_view, rng=None):
        return pick_render_views(
            data, novel_view,
            view_keys=getattr(self, 'view_keys', None),
            k=self.render_nearest_k,
            coincide_tol=self.view_coincide_tol,
            rng=rng,
        )

    def train(self):
        log_l1 = 0
        log_ssim = 0
        log_chamfer = 0
        log_scale = 0
        if_chamfer = self.use_chamfer
        if_scale = False
        iter_from = -1
        for itr_ in tqdm(range(self.total_steps, self.cfg.num_steps)):
            self.optimizer.zero_grad()
            data = self.fetch_data(phase='train')

            #  Depth / Gaussian regression (RAFT-Stereo or DAv3)
            data, _, metrics = self.model(data, is_train=True)

            #  Gaussian Render. Gaussians are regressed ONCE for all source
            #  views; only the rasterization is repeated, once per novel view.
            #  'novel_views' holds one novel view per stereo segment when
            #  dataset.novel_per_segment is on, otherwise this is the single
            #  novel view the loader returned.
            novel_views = data.get('novel_views') or [data['novel_view']]
            # The model stamps scale_regular onto novel_view during forward;
            # carry it across the per-novel-view swaps below.
            scale_regular = data['novel_view'].get('scale_regular')

            Ll1, Lssim = 0.0, 0.0
            for novel_view in novel_views:
                # fetch_data only moves the source views; a novel view straight
                # off the loader is still on CPU, and the nearest-view selection
                # compares it against the source extrinsics. Pure device move.
                novel_view = self._novel_view_to_cuda(novel_view)
                if scale_regular is not None:
                    novel_view['scale_regular'] = scale_regular
                data['novel_view'] = novel_view
                data = pts2render(
                    data,
                    bg_color=self.cfg.dataset.bg_color,
                    source_views=self._pick_render_views(
                        data, data['novel_view'], rng=self._view_rng,
                    ),
                )
                render_novel = data['novel_view']['img_pred']
                gt_novel = data['novel_view']['img'].cuda()
                Ll1 = Ll1 + l1_loss(render_novel, gt_novel)
                Lssim = Lssim + (1.0 - ssim(render_novel, gt_novel))
            Ll1 = Ll1 / len(novel_views)
            Lssim = Lssim / len(novel_views)

            chamfer_loss = 0
            if if_chamfer and itr_ > iter_from:
                # Chamfer compares the two source views' point clouds, so it is
                # only defined for the 2-view branches. Look the views up by key
                # rather than assuming they are called lmain/rmain.
                l_key, r_key = self.view_keys[0], self.view_keys[1]
                l_xyz = data[l_key]['xyz']
                r_xyz = data[r_key]['xyz']
                for b_i in range(self.bs):
                    l_valid_i = data[l_key]['pts_valid'][b_i, :]  # [S*S]
                    l_xyz_i = l_xyz[b_i, :, :]
                    l_xyz_i = l_xyz_i[l_valid_i].view(1, -1, 3).contiguous()
                    
                    r_valid_i = data[r_key]['pts_valid'][b_i, :]  # [S*S]
                    r_xyz_i = r_xyz[b_i, :, :]
                    r_xyz_i = r_xyz_i[r_valid_i].view(1, -1, 3).contiguous()
                    
                    sample_l = np.random.choice(l_xyz_i.shape[1], 10000, replace = False)
                    sample_r = np.random.choice(r_xyz_i.shape[1], 10000, replace = False)
                    chamfer_loss_i, _ = chamfer_distance(l_xyz_i[:, sample_l], r_xyz_i[:, sample_r])
                    chamfer_loss += chamfer_loss_i
                
                chamfer_loss /= self.bs

            loss = 0.8 * Ll1 + 0.2 * Lssim + 0.5 * chamfer_loss

            log_l1 += 0.8 * Ll1.item()
            log_ssim += 0.2 * Lssim.item()
            log_chamfer += 0.5 * chamfer_loss.item() if if_chamfer and itr_>iter_from else 0 
            log_scale += 0.5 * data['novel_view']['scale_regular'].item() if if_scale else 0

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                self.save_ckpt(save_path=Path('%s/%s_latest.pth' % (self.cfg.record.ckpt_path, self.cfg.name)), show_log=False)
            metrics.update({
                'l1': Ll1.item(),
                'ssim': Lssim.item(),
                'chamfer': 0.5*chamfer_loss.item() if if_chamfer and itr_>iter_from else 0
            })
            self.logger.push(metrics)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scheduler.step()
            self.scaler.update()

            # Update EMA shadows right after the raw weights step.
            if self.use_ema:
                for ema in self.emas.values():
                    ema.update(self.model)

            if self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0:
                self.model.eval()
                self.run_eval()
                self.model.train()
                self._freeze_raft_bn()

            if self.total_steps and self.cfg.record.save_freq > 0 and self.total_steps % self.cfg.record.save_freq == 0:
                self._save_all_versions(self.total_steps)


            self.total_steps += 1
            if self.total_steps % 100 == 99:
                print(
                    'l1 ', log_l1/ 100,
                    'ssim', log_ssim/100,
                    'chamfer', log_chamfer/100,
                    'scale', log_scale/100,
                    )
                log_l1 = 0
                log_ssim = 0
                log_chamfer = 0
                log_scale = 0

        print("FINISHED TRAINING")
        self.logger.close()
        self.save_ckpt(save_path=Path('%s/%s_final.pth' % (self.cfg.record.ckpt_path, self.cfg.name)))

    def _eval_psnr_once(self, tag='bare', save_vis=True):
        """One full validation pass; returns ``(full_psnr, crop_psnr)``, each a
        mean over all (sample, view) pairs.

        Every sample is scored at *every* novel view in ``val_novel_id``: each
        view is rebuilt in the main process, so the loader's per-sample random
        choice is bypassed and all views are always covered. The model runs once
        per sample and is re-rendered per view. The val iterator is reset first
        so each model version sees the same samples. ``full_psnr`` scores the
        whole novel view; ``crop_psnr`` drops the top/bottom
        ``self.eval_img_hcrop`` fraction of rows (0.0 = no crop, in which case
        the two are identical). ``tag`` namespaces the image dump.
        """
        self.val_iterator = iter(self.val_loader)
        novel_view_ids = list(self.cfg.dataset.val_novel_id)
        crop = self.eval_img_hcrop
        psnr_list = []
        crop_psnr_list = []
        show_idx = self.len_val // 2

        for idx in range(self.len_val):
            data = self.fetch_data(phase='val')
            sample_name = data['name'][0]
            with torch.no_grad():
                base_data, _, _ = self.model(data, is_train=False)
                for view_id in novel_view_ids:
                    # Re-render the same gaussians to each novel view in turn.
                    render_data = dict(base_data)
                    # The loader hands back CPU tensors; the nearest-view
                    # selection compares them against the (CUDA) source-view
                    # extrinsics, so move them across first. Pure device move,
                    # no numerical effect.
                    render_data['novel_view'] = self._novel_view_to_cuda(
                        default_collate(
                            [self.val_set.get_novel_view_tensor(sample_name, view_id)]
                        )
                    )
                    # Match the training-time render (same nearest-k source
                    # views) but deterministic (rng=None) so PSNR is stable.
                    render_data = pts2render(
                        render_data,
                        bg_color=self.cfg.dataset.bg_color,
                        source_views=self._pick_render_views(
                            render_data, render_data['novel_view'], rng=None,
                        ),
                    )

                    render_novel = render_data['novel_view']['img_pred']
                    gt_novel = render_data['novel_view']['img'].cuda()

                    # Full-frame PSNR over the whole novel view.
                    psnr_list.append(psnr(render_novel, gt_novel).mean().double().item())

                    # Cropped PSNR: drop the top/bottom `crop` fraction of rows
                    # before scoring (0.0 = no crop -> same as full-frame).
                    render_crop, gt_crop = render_novel, gt_novel
                    if crop > 0.0:
                        h = render_novel.shape[-2]
                        top = int(round(h * crop))
                        if top > 0:
                            render_crop = render_novel[..., top:h - top, :]
                            gt_crop = gt_novel[..., top:h - top, :]
                    crop_psnr_value = psnr(render_crop, gt_crop).mean().double()
                    crop_psnr_list.append(crop_psnr_value.item())

                    if save_vis and idx == show_idx:
                        tmp_novel = render_data['novel_view']['img_pred'][0].detach()
                        tmp_novel *= 255
                        tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                        tmp_img_name = '%s/%s_%s_view%s.png' % (
                            self.cfg.record.show_path, self.total_steps, tag, view_id,
                        )
                        cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))
                        self.logger.log_image(
                            'eval/%s/view%s/image' % (tag, view_id),
                            tmp_novel.astype(np.uint8), self.total_steps,
                        )

        full_psnr = np.round(np.mean(np.array(psnr_list)), 4)
        crop_psnr = np.round(np.mean(np.array(crop_psnr_list)), 4)
        return full_psnr, crop_psnr

    def run_eval(self):
        logging.info(f"Doing validation ...")
        torch.cuda.empty_cache()

        if not self.use_ema:
            val_psnr, val_crop_psnr = self._eval_psnr_once(tag='bare', save_vis=True)
            if val_psnr < 10:
                print('something wrong during training, please change random seed and re-train')
                exit()
            logging.info(
                f"Validation Metrics ({self.total_steps}): psnr {val_psnr} crop_psnr {val_crop_psnr}"
            )
            self.logger.write_dict(
                {'val_psnr': val_psnr, 'eval/psnr': val_psnr, 'eval/crop_psnr': val_crop_psnr},
                write_step=self.total_steps,
            )
            torch.cuda.empty_cache()
            return

        # Evaluate every version: raw ("bare") + each EMA decay. Each logs its
        # full-frame PSNR under eval/{tag}/psnr and its cropped PSNR under
        # eval/{tag}/crop_psnr (eval/bare/*, eval/ema1_e3/*, eval/ema1_e4/*).
        log_payload = {}
        for tag, decay in self.ema_specs:
            if decay is None:  # raw live weights
                val_psnr, val_crop_psnr = self._eval_psnr_once(tag=tag, save_vis=True)
            else:
                with self.emas[tag].swapped_in(self.model):
                    val_psnr, val_crop_psnr = self._eval_psnr_once(tag=tag, save_vis=True)
            logging.info(
                f"Validation Metrics ({self.total_steps}) [{tag}]: "
                f"psnr {val_psnr} crop_psnr {val_crop_psnr}"
            )
            log_payload['eval/%s/psnr' % tag] = val_psnr
            log_payload['eval/%s/crop_psnr' % tag] = val_crop_psnr

        # Sanity check only on the raw model (EMA can be under-warmed early on).
        if log_payload['eval/bare/psnr'] < 10:
            print('something wrong during training, please change random seed and re-train')
            exit()

        # Keep the legacy scalar keys pointed at the raw model.
        log_payload['val_psnr'] = log_payload['eval/bare/psnr']
        log_payload['eval/psnr'] = log_payload['eval/bare/psnr']
        self.logger.write_dict(log_payload, write_step=self.total_steps)
        torch.cuda.empty_cache()

    def _save_all_versions(self, step):
        """Save the raw checkpoint plus one checkpoint per EMA version."""
        ckpt_dir = self.cfg.record.ckpt_path
        self.save_ckpt(save_path=Path('%s/iter%d.pth' % (ckpt_dir, step)))
        if not self.use_ema:
            return
        for tag, decay in self.ema_specs:
            if decay is None:
                continue
            ema = self.emas[tag]
            self.save_ckpt(
                save_path=Path('%s/iter%d_%s.pth' % (ckpt_dir, step, tag)),
                network_state=ema.state_dict_with_shadow(self.model),
            )

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

        # All N source views for the multi-view models; the stereo pair otherwise.
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

    def save_ckpt(self, save_path, show_log=True, network_state=None):
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict() if network_state is None else network_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    parser = argparse.ArgumentParser(description='GPS_plus training')
    parser.add_argument('--config', type=str, default='config/all_scene',
                        help='Path to the config directory (contains '
                             'stereo_human_config.py and stage.yaml)')
    parser.add_argument('--opts', nargs='*', default=[],
                        help='Override config entries as space-separated '
                             'key value pairs, e.g. '
                             '--opts num_steps 400 wandb.project force_none. '
                             'Only keys declared in that branch\'s '
                             'stereo_human_config.py are accepted.')
    args = parser.parse_args()

    # Resolve the config directory: import its ConfigStereoHuman class and load
    # its stage.yaml, so each experiment lives in its own config folder
    # (e.g. config/all_scene, config/only_s1).
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
    cfg.record.ckpt_path = "experiments/%s/ckpt" % cfg.exp_name
    cfg.record.show_path = "experiments/%s/show" % cfg.exp_name
    cfg.record.logs_path = "experiments/%s/logs" % cfg.exp_name
    cfg.record.file_path = "experiments/%s/file" % cfg.exp_name
    cfg.freeze()


    for path in [cfg.record.ckpt_path, cfg.record.show_path, cfg.record.logs_path, cfg.record.file_path]:
        Path(path).mkdir(exist_ok=True, parents=True)
    
    file_backup(cfg.record.file_path, cfg, config_dir=config_dir)

    torch.manual_seed(1314)
    np.random.seed(1314)

    trainer = Trainer(cfg)
    trainer.train()
