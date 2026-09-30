"""Multi-view (N source-view) DAv3 model, backbone-upsampler depth path.

Generalizes ``RtStereoHumanDAV3Model`` (``DAV3Model_MK``) from a single stereo
pair (2 views) to ``N`` source views. The DA3 backbone already does multi-view
attention over its leading ``(bs, N, ...)`` dim, so it is run ONCE jointly over
all N views. Everything else (image encoder, backbone-upsampler head, Gaussian
regressor) either runs once over the N views stacked on the batch axis or once
per view -- see ``forward``.

Each view produces its own Gaussians; ``data['source_view_keys']`` is stamped so
``pts2render`` merges all N views. The optional backbone-token LayerNorm
(``dav3.add_layernorm_upsampler``) is honored via the inherited
``_predict_inverse_depth_from_backbone``.

Assumes ``batch_size == 1``. LoFTR is not used (dropped if built).
"""

from __future__ import annotations

import logging

import torch
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from models.dav3_model import RtStereoHumanDAV3Model
from models.depth_anything_v3_model import _cfg_get

logger = logging.getLogger(__name__)

# 4 distinct source cameras on this rig; see MultiViewStereoHumanDataset.VIEW_KEYS
_DEFAULT_VIEW_KEYS = ["view0", "view1", "view2", "view3"]


class RtStereoHumanDAV3ModelUpsamplerMV(RtStereoHumanDAV3Model):
    """N-source-view backbone-upsampler DAv3 model."""

    def __init__(self, cfg, with_gs_render=False, legacy_dataset=False, inspect_mode=False):
        super().__init__(
            cfg,
            with_gs_render=with_gs_render,
            legacy_dataset=legacy_dataset,
            inspect_mode=inspect_mode,
        )

        n = int(_cfg_get(cfg, "dataset.num_source_views", 4))
        if not 2 <= n <= len(_DEFAULT_VIEW_KEYS):
            raise ValueError(
                f"dataset.num_source_views={n} out of supported range "
                f"[2, {len(_DEFAULT_VIEW_KEYS)}]."
            )
        self.view_keys = list(_DEFAULT_VIEW_KEYS[:n])
        self.num_da3_views = n

        # LoFTR is pairwise (2 views only); rely on the DA3 backbone's own
        # multi-view attention. Drop the module if it was built.
        if hasattr(self, "loftr_coarse"):
            del self.loftr_coarse

        # Gradient checkpointing (opt-in): recompute per-view full-res
        # activations in backward instead of storing all N, so 4 views fit.
        self.grad_checkpoint = bool(_cfg_get(cfg, "dav3.grad_checkpoint", False))
        self.grad_checkpoint_backbone = bool(
            _cfg_get(cfg, "dav3.grad_checkpoint_backbone", self.grad_checkpoint)
        )
        if self.grad_checkpoint and self.grad_checkpoint_backbone:
            self.da3_loader._da3_model.backbone.grad_checkpoint = True

        logger.info(
            "[DAv3 Upsampler MV] num_source_views=%d view_keys=%s add_layernorm_upsampler=%s "
            "grad_checkpoint=%s (backbone=%s)",
            n, self.view_keys, getattr(self, "add_layernorm_upsampler", False),
            self.grad_checkpoint, self.grad_checkpoint_backbone,
        )

    def _stages_per_view(self, data, view_keys, backbone_tokens, bs, ph, pw,
                         render, ckpt):
        """One view at a time. Supports gradient checkpointing."""
        outs = []
        for i, v in enumerate(view_keys):
            img_v = data[v]["img"]
            # Per-view token slices. Passed as EXPLICIT args to checkpoint (not
            # captured via closure) so gradients flow into the DA3 backbone when
            # the adapter (LoRA / LayerNorm) is trainable.
            tokens_v = [t[i * bs:(i + 1) * bs] for t in backbone_tokens]

            def _fn(img_in, *tok, _ph=ph, _pw=pw, _render=render):
                with autocast(
                    device_type=img_in.device.type,
                    enabled=self.use_mixed_precision,
                ):
                    feat = self.img_encoder(img_in)
                pred_inv, _conf = self._predict_inverse_depth_from_backbone(
                    backbone_tokens=list(tok), ph=_ph, pw=_pw,
                    img_feats=feat, rgb=img_in,
                )
                if not _render:
                    return pred_inv
                rot, scale, opacity, resdepth = self.gs_parm_regresser(
                    img_in, pred_inv, feat,
                )
                return pred_inv, rot, scale, opacity, resdepth

            if ckpt:
                outs.append(checkpoint(_fn, img_v, *tokens_v, use_reentrant=False))
            else:
                outs.append(_fn(img_v, *tokens_v))
        return outs

    def _stages_stacked(self, image, backbone_tokens, bs, n, ph, pw, render):
        """All N views at once on the batch axis, then split back per view.

        This is what the 2-view model (``RtStereoHumanDAV3Model``) already does.
        """
        with autocast(device_type=image.device.type, enabled=self.use_mixed_precision):
            feat = self.img_encoder(image)
        pred_inv_all, _conf = self._predict_inverse_depth_from_backbone(
            backbone_tokens=list(backbone_tokens), ph=ph, pw=pw,
            img_feats=feat, rgb=image,
        )
        if not render:
            return [pred_inv_all[i * bs:(i + 1) * bs] for i in range(n)]

        rot, scale, opacity, resdepth = self.gs_parm_regresser(
            image, pred_inv_all, feat,
        )
        stacked = (pred_inv_all, rot, scale, opacity, resdepth)
        return [tuple(t[i * bs:(i + 1) * bs] for t in stacked) for i in range(n)]

    def forward(self, data, is_train=True):
        del is_train
        view_keys = self.view_keys
        n = len(view_keys)
        bs = data[view_keys[0]]["img"].shape[0]
        image = torch.cat([data[v]["img"] for v in view_keys], dim=0)
        ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        render = self.with_gs_render

        # DA3 backbone (usually frozen) -> per-layer tokens, joint over N views.
        backbone_tokens, ph, pw, da3_input_hw = self._run_da3_backbone(
            image, bs, data=data,
        )
        da3_input_hw_t = torch.tensor(
            da3_input_hw, device=image.device, dtype=torch.int32,
        )

        # The image encoder, upsampler head and GSRegresser are all batch
        # agnostic, so they can either run once over the N views stacked on the
        # batch axis, or once per view. Which is faster is not obvious and was
        # measured on an RTX A6000, 1024^2, bs=1 (see README):
        #
        #                      per-view loop      stacked
        #   inference 4-view       326 ms         248 ms   <- stacked wins
        #   training  4-view       622 ms         707 ms   <- loop wins, and
        #                        27.9 GB        29.5 GB       uses 1.6 GB less
        #
        # So: stack for inference, loop for training. Keeping training on the
        # loop also leaves the trajectory of the released checkpoints intact,
        # and it is the only path that supports dav3.grad_checkpoint (which
        # recomputes per-view activations so 4 views fit).
        per_view = (
            self._stages_per_view(data, view_keys, backbone_tokens, bs, ph, pw,
                                  render, ckpt)
            if (ckpt or self.training) else
            self._stages_stacked(image, backbone_tokens, bs, n, ph, pw, render)
        )

        scale_mean_sum = 0.0
        for i, v in enumerate(view_keys):
            out = per_view[i]

            if render:
                pred_inv, rot, scale, opacity, resdepth = out
            else:
                pred_inv = out

            data[v]["depth"] = pred_inv
            data[v]["depth_pre_gs"] = pred_inv
            data[v]["depth_da3"] = pred_inv.detach()
            data[v]["depth_offset"] = torch.zeros_like(pred_inv).detach()
            data[v]["depth_conf"] = torch.ones_like(pred_inv)
            data[v]["da3_input_hw"] = da3_input_hw_t

            if render:
                if self.apply_gs_resdepth:
                    new_depth = pred_inv + resdepth
                else:
                    new_depth = pred_inv
                data[v]["depth"] = new_depth.clamp(
                    self.min_inverse_depth, self.max_inverse_depth,
                )
                data[v]["xyz"] = self.depth2pc_fn(
                    data[v]["depth"], data[v]["extr"], data[v]["intr"],
                ).view(bs, -1, 3)
                valid = data[v]["mask"][:, :1, :, :] > 0.5
                data[v]["pts_valid"] = valid.view(bs, -1)
                data[v]["rot_maps"] = rot
                data[v]["scale_maps"] = scale
                data[v]["opacity_maps"] = opacity
                scale_mean_sum = scale_mean_sum + scale.mean()

        data["source_view_keys"] = list(view_keys)
        if render:
            data["novel_view"]["scale_regular"] = scale_mean_sum / float(n)
        return data, None, {}


class DAV3Model_MK_Upsampler_MV(torch.nn.Module):
    """Registered wrapper around ``RtStereoHumanDAV3ModelUpsamplerMV``."""

    def __init__(self, cfg, with_gs_render=False, legacy_dataset=False, inspect_mode=False):
        super().__init__()
        self.model = RtStereoHumanDAV3ModelUpsamplerMV(
            cfg,
            with_gs_render=with_gs_render,
            legacy_dataset=legacy_dataset,
            inspect_mode=inspect_mode,
        )

    def forward(self, data, is_train=True):
        return self.model(data, is_train=is_train)
