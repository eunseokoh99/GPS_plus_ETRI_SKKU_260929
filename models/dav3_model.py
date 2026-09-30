from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast

from lib.attention_module import LocalFeatureTransformer
from core.extractor import UnetExtractor
from models.depth_anything_v3_model import DA3SmallModel, _cfg_get
from models.lora import (
    apply_backbone_lora,
    enable_bitfit,
    enable_layernorm_tuning,
)
from lib.gs_parm_network import GSRegresser
from lib.utils import depth2pc


logger = logging.getLogger(__name__)


class _NoOpRaftStereo(nn.Module):
    def freeze_bn(self):
        return self


class _BackboneDepthUpsampler(nn.Module):
    """Upsamples DA3 ViT backbone tokens directly to full-resolution depth.

    Replaces the DA3 DPT head: takes the multi-scale backbone tokens (4 stages
    of ``(B, N, C)`` patch tokens at the ViT patch grid) and progressively
    upsamples to image resolution by fusing the stereo encoder's image features
    at H/8, H/4, H/2 via PixelShuffle. The final layer outputs ``log(depth)``;
    the model exponentiates to produce metric depth, which downstream code
    converts to inverse depth.
    """

    def __init__(
        self,
        backbone_dim: int,
        n_layers: int,
        image_feat_dims: tuple[int, int, int],
        hidden_dim: int = 64,
        rgb_channels: int = 3,
        init_log_depth: float = 0.0,
        log_depth_range: float = 3.0,
    ):
        super().__init__()
        c_h2, c_h4, c_h8 = (int(d) for d in image_feat_dims)
        h = int(hidden_dim)
        # Soft-bound parameters for the head: log_depth = init + tanh(.) * range
        # keeps depth in [exp(init-range), exp(init+range)] so the downstream
        # inverse-depth clamp never pins the gradient to zero.
        self.register_buffer(
            "init_log_depth", torch.tensor(float(init_log_depth)), persistent=False,
        )
        self.register_buffer(
            "log_depth_range", torch.tensor(float(log_depth_range)), persistent=False,
        )

        self.layer_projects = nn.ModuleList(
            [nn.Conv2d(int(backbone_dim), h, kernel_size=1) for _ in range(int(n_layers))]
        )

        self.fuse_backbone = nn.Sequential(
            nn.Conv2d(h * int(n_layers), h * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(h * 2, h * 2, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Each upsample stage follows the standard PixelShuffle pattern:
        # concat with the image-encoder feature at the current scale, expand
        # channels to ``out_ch * r²`` in a single 3x3 conv, then PixelShuffle
        # rearranges those channels into spatial detail. No prior reduce step.
        self.up_h8_to_h4 = nn.Sequential(
            nn.Conv2d(h * 2 + c_h8, h * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
        )
        self.up_h4_to_h2 = nn.Sequential(
            nn.Conv2d(h + c_h4, h * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
        )
        self.up_h2_to_h1 = nn.Sequential(
            nn.Conv2d(h + c_h2, h * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Conv2d(h + rgb_channels, h, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(h, 1, kernel_size=1),
        )
        # Stable starting point: zero-weight + bias = log(typical_depth) on the
        # final 1x1 conv makes initial depth uniform = exp(init_log_depth) for
        # all pixels, instead of the wide exp-of-Gaussian distribution that
        # default kaiming init would produce.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, float(init_log_depth))

    def forward(
        self,
        backbone_tokens: list[torch.Tensor],
        ph: int,
        pw: int,
        img_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            backbone_tokens: list of N tensors, each ``(B, num_patches, C)`` at
                the ViT patch grid.
            ph, pw: backbone patch grid resolution (``num_patches == ph * pw``).
            img_feats: ``(feat_h2, feat_h4, feat_h8)`` from the stereo encoder.
            rgb: full-resolution image batch ``(B, 3, H, W)``.

        Returns:
            depth tensor of shape ``(B, 1, H, W)`` in metric space (positive).
        """
        feat_h2, feat_h4, feat_h8 = img_feats
        target_dtype = rgb.dtype
        feat_h2 = feat_h2.to(target_dtype)
        feat_h4 = feat_h4.to(target_dtype)
        feat_h8 = feat_h8.to(target_dtype)
        rgb = rgb.to(target_dtype)

        full_h, full_w = rgb.shape[-2:]

        projected: list[torch.Tensor] = []
        for tokens, proj in zip(backbone_tokens, self.layer_projects):
            B = tokens.shape[0]
            C = tokens.shape[-1]
            x = tokens.permute(0, 2, 1).reshape(B, C, ph, pw).to(target_dtype)
            x = proj(x)
            x = F.interpolate(
                x, size=feat_h8.shape[-2:], mode="bilinear", align_corners=False,
            )
            projected.append(x)
        x = torch.cat(projected, dim=1)
        x = self.fuse_backbone(x)

        x = torch.cat([x, feat_h8], dim=1)
        x = self.up_h8_to_h4(x)

        if x.shape[-2:] != feat_h4.shape[-2:]:
            x = F.interpolate(
                x, size=feat_h4.shape[-2:], mode="bilinear", align_corners=False,
            )
        x = torch.cat([x, feat_h4], dim=1)
        x = self.up_h4_to_h2(x)

        if x.shape[-2:] != feat_h2.shape[-2:]:
            x = F.interpolate(
                x, size=feat_h2.shape[-2:], mode="bilinear", align_corners=False,
            )
        x = torch.cat([x, feat_h2], dim=1)
        x = self.up_h2_to_h1(x)

        if x.shape[-2:] != (full_h, full_w):
            x = F.interpolate(x, size=(full_h, full_w), mode="bilinear", align_corners=False)
        x = torch.cat([x, rgb], dim=1)
        log_depth_raw = self.head(x)
        log_depth = self.init_log_depth + torch.tanh(
            log_depth_raw - self.init_log_depth
        ) * self.log_depth_range
        depth = torch.exp(log_depth)
        return depth


class RtStereoHumanDAV3Model(nn.Module):
    PATCH_SIZE = 14
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, cfg, with_gs_render=False, legacy_dataset=False, inspect_mode=False):
        super().__init__()
        self.cfg = cfg
        self.with_gs_render = with_gs_render
        self.legacy_dataset = legacy_dataset
        self.inspect_mode = inspect_mode

        self.use_mixed_precision = bool(_cfg_get(cfg, "raft.mixed_precision", False))
        self.depth2pc_fn = depth2pc
        # Number of source views fed jointly to the DA3 backbone (2 = one stereo
        # pair). Multi-view subclasses raise this; keep 2 for the default path.
        self.num_da3_views = 2

        self.img_encoder = UnetExtractor(in_channel=3, encoder_dim=self.cfg.raft.encoder_dims)
        # Optional cross-view coarse transformer. Disable (raft.use_loftr_coarse
        # = False) to feed the raw encoder coarse features straight through.
        self.use_loftr_coarse = bool(_cfg_get(cfg, "raft.use_loftr_coarse", True))
        if self.use_loftr_coarse:
            self.loftr_coarse = LocalFeatureTransformer(
                cross_attention_mode=str(_cfg_get(cfg, "raft.cross_attention_mode", "row")),
            )
        if self.with_gs_render:
            self.gs_parm_regresser = GSRegresser(self.cfg, rgb_dim=3, depth_dim=1)

        self.raft_stereo = _NoOpRaftStereo()

        self.da3_process_res = int(
            _cfg_get(
                cfg,
                "dav3.process_res",
                _cfg_get(cfg, "external_models.depth_anything_v3.process_res", 504),
            )
        )
        self.da3_process_res_method = str(
            _cfg_get(
                cfg,
                "dav3.process_res_method",
                _cfg_get(
                    cfg,
                    "external_models.depth_anything_v3.process_res_method",
                    "upper_bound_resize",
                ),
        )
        )
        self.freeze_da3 = bool(_cfg_get(cfg, "dav3.freeze_backbone", True))
        self.use_camera_pose = bool(_cfg_get(cfg, "dav3.use_camera_pose", False))
        self.min_inverse_depth = float(_cfg_get(cfg, "dav3.min_inverse_depth", 1e-4))
        self.max_inverse_depth = float(_cfg_get(cfg, "dav3.max_inverse_depth", 20.0))
        self.apply_gs_resdepth = bool(_cfg_get(cfg, "dav3.apply_gs_resdepth", True))

        adapter_hidden_dim = int(_cfg_get(cfg, "dav3.adapter_hidden_dim", 64))
        # Optionally LayerNorm the first half of each backbone token's channels
        # before feeding the backbone upsampler head (see
        # ``_predict_inverse_depth_from_backbone``). Trainable, one LN per layer.
        self.add_layernorm_upsampler = bool(
            _cfg_get(cfg, "dav3.add_layernorm_upsampler", False)
        )

        self.da3_loader = DA3SmallModel(
            cfg=cfg,
            with_gs_render=False,
            legacy_dataset=legacy_dataset,
            inspect_mode=inspect_mode,
            load_pretrained=_cfg_get(cfg, "dav3.load_pretrained", True),
            process_res=self.da3_process_res,
            process_res_method=self.da3_process_res_method,
            strict=True,
        )
        self.da3_loader._ensure_model_initialized()
        self.da3_loader.eval()
        self._configure_da3_trainability()
        self._silence_da3_runtime_logs()

        encoder_dims = tuple(int(d) for d in self.cfg.raft.encoder_dims)
        if len(encoder_dims) != 3:
            raise ValueError(
                f"raft.encoder_dims must have length 3 (got {encoder_dims})."
            )
        backbone_dim, n_layers = self._infer_backbone_token_shape()
        inv_depth_init = float(_cfg_get(cfg, "dataset.inverse_depth_init", 0.2))
        # NOTE: the attribute is called depth_offset_head for historical reasons
        # -- an earlier head predicted an offset on top of DA3's own depth. This
        # one predicts depth directly. The name is load-bearing: every released
        # checkpoint keys its tensors under it, so do not rename it.
        self.depth_offset_head = _BackboneDepthUpsampler(
            backbone_dim=backbone_dim,
            n_layers=n_layers,
            image_feat_dims=encoder_dims,
            hidden_dim=adapter_hidden_dim,
            init_log_depth=math.log(1.0 / max(inv_depth_init, 1e-6)),
            log_depth_range=float(
                _cfg_get(cfg, "dav3.upsampler_log_depth_range", 3.0)
            ),
        )
        if self.add_layernorm_upsampler:
            # One LayerNorm per backbone layer; each per-token block is
            # ``2 * embed_dim`` (cat_token), normalize the first half.
            self.backbone_token_lns = nn.ModuleList(
                [nn.LayerNorm(backbone_dim // 2) for _ in range(n_layers)]
            )

    def _infer_backbone_token_shape(self) -> tuple[int, int]:
        """Read ``(per_token_channels, num_out_layers)`` from the DA3 backbone.

        For DA3 with ``cat_token=True`` (default for all variants), each output
        layer's per-token channel count is ``2 * embed_dim`` (e.g., 3072 for
        DA3-GIANT). The number of layers equals ``len(out_layers)`` (4 for all
        variants).
        """
        da3_model = self.da3_loader._da3_model
        if da3_model is None:
            raise RuntimeError("DA3 model must be initialized before reading backbone dims.")
        backbone = getattr(da3_model, "backbone", None)
        if backbone is None:
            raise AttributeError("DA3 model does not expose a `backbone` attribute.")

        out_layers = getattr(backbone, "out_layers", None)
        if not out_layers:
            raise AttributeError("DA3 backbone does not expose `out_layers`.")
        n_layers = len(out_layers)

        pretrained = getattr(backbone, "pretrained", None)
        embed_dim = getattr(pretrained, "embed_dim", None) if pretrained is not None else None
        if embed_dim is None:
            raise AttributeError("DA3 backbone does not expose `pretrained.embed_dim`.")
        cat_token = bool(getattr(backbone, "cat_token", True))
        per_token_dim = int(embed_dim) * (2 if cat_token else 1)
        return per_token_dim, int(n_layers)

    def _configure_da3_trainability(self) -> None:
        da3_model = self.da3_loader._da3_model
        if da3_model is None:
            raise RuntimeError("DA3 model must be initialized before configuring trainability.")

        tuning_mode = str(
            _cfg_get(self.cfg, "dav3.tuning_mode", "none")
        ).lower().strip()
        # Legacy fallback: older configs used ``dav3.lora.enabled`` directly.
        if tuning_mode in {"", "none"} and bool(
            _cfg_get(self.cfg, "dav3.lora.enabled", False)
        ):
            tuning_mode = "lora"

        # ``tuning_mode`` forces the backbone frozen so only the chosen
        # adapter / subset of parameters becomes trainable. ``freeze_backbone``
        # remains the default switch when no adapter is selected.
        force_freeze = tuning_mode in {"lora", "bitfit", "layernorm"}
        requires_grad = not (self.freeze_da3 or force_freeze)
        for parameter in da3_model.parameters():
            parameter.requires_grad_(requires_grad)

        if tuning_mode == "lora":
            # LoRA wrappers keep the base weight frozen and add trainable
            # ``lora_A``/``lora_B`` parameters (picked up by the optimizer via
            # ``model.parameters()`` since their ``requires_grad`` defaults
            # to True).
            apply_backbone_lora(
                da3_model=da3_model,
                rank=int(_cfg_get(self.cfg, "dav3.lora.rank", 8)),
                alpha=float(_cfg_get(self.cfg, "dav3.lora.alpha", 16.0)),
                dropout=float(_cfg_get(self.cfg, "dav3.lora.dropout", 0.0)),
                target_modules=tuple(
                    str(t) for t in _cfg_get(
                        self.cfg, "dav3.lora.target_modules", ["qkv", "proj"]
                    )
                ),
            )
        elif tuning_mode == "bitfit":
            enable_bitfit(da3_model)
        elif tuning_mode == "layernorm":
            enable_layernorm_tuning(da3_model)
        elif tuning_mode not in {"", "none"}:
            raise ValueError(
                f"Unknown dav3.tuning_mode {tuning_mode!r}. "
                "Expected one of: 'none', 'lora', 'bitfit', 'layernorm'."
            )

    @staticmethod
    def _silence_da3_runtime_logs() -> None:
        try:
            from depth_anything_3.utils.logger import LOG_LEVELS, logger as da3_logger
            da3_logger.level = LOG_LEVELS["WARN"]
        except ImportError:
            pass

    def _denormalize_rgb(self, image: torch.Tensor) -> torch.Tensor:
        return ((image.float() + 1.0) * 0.5).clamp(0.0, 1.0)

    def _resolve_boundary_hw(self, height: int, width: int) -> tuple[int, int]:
        method = self.da3_process_res_method
        target = self.da3_process_res

        if method.startswith("upper_bound"):
            scale = target / float(max(height, width))
        elif method.startswith("lower_bound"):
            scale = target / float(min(height, width))
        else:
            raise ValueError(f"Unsupported DAV3 resize method: {method}")

        resized_h = max(1, int(round(height * scale)))
        resized_w = max(1, int(round(width * scale)))
        return resized_h, resized_w

    def _resolve_patch_aligned_hw(self, height: int, width: int) -> tuple[int, int]:
        method = self.da3_process_res_method
        if method.endswith("resize"):
            height = self._nearest_multiple(height, self.PATCH_SIZE)
            width = self._nearest_multiple(width, self.PATCH_SIZE)
        elif method.endswith("crop"):
            height = max(self.PATCH_SIZE, (height // self.PATCH_SIZE) * self.PATCH_SIZE)
            width = max(self.PATCH_SIZE, (width // self.PATCH_SIZE) * self.PATCH_SIZE)
        else:
            raise ValueError(f"Unsupported DAV3 resize method: {method}")

        return height, width

    @staticmethod
    def _nearest_multiple(value: int, divisor: int) -> int:
        lower = max(divisor, (value // divisor) * divisor)
        upper = lower if lower == value else lower + divisor
        if abs(upper - value) <= abs(value - lower):
            return upper
        return lower

    def _prepare_da3_input(self, image: torch.Tensor) -> torch.Tensor:
        _, _, height, width = image.shape
        boundary_h, boundary_w = self._resolve_boundary_hw(height, width)

        resized = F.interpolate(
            image,
            size=(boundary_h, boundary_w),
            mode="bilinear",
            align_corners=False,
        )

        final_h, final_w = self._resolve_patch_aligned_hw(boundary_h, boundary_w)
        if self.da3_process_res_method.endswith("resize"):
            if (final_h, final_w) != (boundary_h, boundary_w):
                resized = F.interpolate(
                    resized,
                    size=(final_h, final_w),
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            top = max(0, (boundary_h - final_h) // 2)
            left = max(0, (boundary_w - final_w) // 2)
            resized = resized[..., top : top + final_h, left : left + final_w]

        mean = torch.tensor(self.IMAGENET_MEAN, device=resized.device, dtype=resized.dtype)[
            None, :, None, None
        ]
        std = torch.tensor(self.IMAGENET_STD, device=resized.device, dtype=resized.dtype)[
            None, :, None, None
        ]
        return (resized - mean) / std

    def _build_da3_cam_inputs(
        self,
        data,
        image_hw: tuple[int, int],
        da3_input_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Stack lmain/rmain extr+intr into (B, 2, 4, 4)/(B, 2, 3, 3) and
        rescale intrinsics from the original image resolution to the DA3
        input resolution. ``extr`` is assumed to be w2c (GPS+ convention,
        matching DA3 cam_enc)."""
        if not self.use_camera_pose or data is None:
            return None, None

        l_ext = data["lmain"]["extr"]
        r_ext = data["rmain"]["extr"]
        l_int = data["lmain"]["intr"]
        r_int = data["rmain"]["intr"]

        def _to_4x4(ext: torch.Tensor) -> torch.Tensor:
            if ext.shape[-2:] == (4, 4):
                return ext
            if ext.shape[-2:] != (3, 4):
                raise ValueError(
                    f"extr must be (..., 3, 4) or (..., 4, 4), got {tuple(ext.shape)}"
                )
            pad = torch.zeros(*ext.shape[:-2], 1, 4, dtype=ext.dtype, device=ext.device)
            pad[..., 0, 3] = 1.0
            return torch.cat([ext, pad], dim=-2)

        extr = torch.stack([_to_4x4(l_ext), _to_4x4(r_ext)], dim=1).float()
        intr = torch.stack([l_int, r_int], dim=1).float().clone()

        h_in, w_in = image_hw
        h_da3, w_da3 = da3_input_hw
        sx = float(w_da3) / float(w_in)
        sy = float(h_da3) / float(h_in)
        intr[..., 0, 0] *= sx
        intr[..., 0, 2] *= sx
        intr[..., 1, 1] *= sy
        intr[..., 1, 2] *= sy
        return extr, intr

    def _compute_da3_cam_token(
        self,
        data,
        image_hw: tuple[int, int],
        da3_input_hw: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor | None:
        """Build DA3 cam_token from data poses, matching the path used in
        ``DepthAnything3Net.forward``. Returns None if disabled or cam_enc
        is unavailable."""
        extr, intr = self._build_da3_cam_inputs(data, image_hw, da3_input_hw)
        if extr is None:
            return None
        cam_enc = getattr(self.da3_loader._da3_model, "cam_enc", None)
        if cam_enc is None:
            logger.warning(
                "dav3.use_camera_pose=True but DA3 model has no cam_enc; "
                "skipping pose injection."
            )
            return None
        with torch.autocast(device_type=device.type, enabled=False):
            return cam_enc(extr.to(device), intr.to(device), da3_input_hw)

    def _run_da3_backbone(
        self,
        image: torch.Tensor,
        bs: int,
        data=None,
    ) -> tuple[list[torch.Tensor], int, int, tuple[int, int]]:
        """Run only the DA3 ViT backbone (skipping the DPT head).

        When ``dav3.use_camera_pose`` is enabled and ``data`` is provided,
        GT extrinsics/intrinsics from ``data['lmain']``/``data['rmain']`` are
        passed through DA3's ``cam_enc`` to produce a ``cam_token`` that
        replaces the CLS-slot token inside the DINOv2 backbone (same path
        used by ``DepthAnything3Net.forward``).

        Returns:
            backbone_tokens: list of per-layer patch tokens, each shape
                ``(B*2, num_patches, C)`` where ``num_patches == ph * pw``.
            ph, pw: backbone patch grid dimensions.
            da3_input_hw: ``(H, W)`` of the DA3 input image (504×504 for the
                standard config).
        """
        rgb_01 = self._denormalize_rgb(image)
        da3_input = self._prepare_da3_input(rgb_01)
        da3_input_hw = da3_input.shape[-2:]
        ph = int(da3_input_hw[0]) // self.PATCH_SIZE
        pw = int(da3_input_hw[1]) // self.PATCH_SIZE
        da3_input = da3_input.view(bs, self.num_da3_views, 3, da3_input_hw[0], da3_input_hw[1])

        backbone = self.da3_loader._da3_model.backbone
        cam_enc = getattr(self.da3_loader._da3_model, "cam_enc", None)
        backbone_needs_grad = any(p.requires_grad for p in backbone.parameters())
        cam_enc_needs_grad = (
            any(p.requires_grad for p in cam_enc.parameters())
            if cam_enc is not None
            else False
        )
        grad_context = (
            torch.enable_grad()
            if (self.training and (backbone_needs_grad or cam_enc_needs_grad))
            else torch.inference_mode()
        )
        with grad_context:
            cam_token = self._compute_da3_cam_token(
                data,
                image_hw=image.shape[-2:],
                da3_input_hw=da3_input_hw,
                device=da3_input.device,
            )
            if da3_input.device.type == "cuda":
                autocast_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    feats, _aux = backbone(
                        da3_input,
                        cam_token=cam_token,
                        export_feat_layers=[],
                        ref_view_strategy="saddle_balanced",
                    )
            else:
                feats, _aux = backbone(
                    da3_input,
                    cam_token=cam_token,
                    export_feat_layers=[],
                    ref_view_strategy="saddle_balanced",
                )

        backbone_tokens: list[torch.Tensor] = []
        for pair in feats:
            patch_tokens = pair[0]
            B, S, N, C = patch_tokens.shape
            t = patch_tokens.reshape(B * S, N, C).float()
            if t.is_inference():
                t = t.clone()
            backbone_tokens.append(t)

        return backbone_tokens, ph, pw, da3_input_hw

    def _predict_inverse_depth_from_backbone(
        self,
        backbone_tokens: list[torch.Tensor],
        ph: int,
        pw: int,
        img_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        rgb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the backbone-direct upsampler and return inverse depth at the
        full image resolution. The DPT head is bypassed entirely.

        Returns ``(pred_inverse_depth, depth_conf)`` where ``depth_conf`` is a
        tensor of ones (no separate DA3 confidence is available in this path).
        """
        if self.add_layernorm_upsampler:
            tokens_for_head: list[torch.Tensor] = []
            for tokens, ln in zip(backbone_tokens, self.backbone_token_lns):
                channels = tokens.shape[-1]
                tokens1, tokens2 = tokens.split(channels // 2, dim=-1)
                tokens1 = ln(tokens1)
                tokens_for_head.append(torch.cat([tokens1, tokens2], dim=-1))
        else:
            tokens_for_head = backbone_tokens

        depth_metric = self.depth_offset_head(tokens_for_head, ph, pw, img_feats, rgb)
        depth_metric = depth_metric.clamp_min(1e-4)
        pred_inverse_depth = (1.0 / depth_metric).clamp(
            self.min_inverse_depth, self.max_inverse_depth,
        )
        depth_conf = torch.ones_like(pred_inverse_depth)
        return pred_inverse_depth, depth_conf

    def _fill_debug_depth(
        self,
        data,
        l_da3: torch.Tensor,
        r_da3: torch.Tensor,
        l_offset: torch.Tensor,
        r_offset: torch.Tensor,
    ) -> None:
        if not self.inspect_mode:
            return

        data["lmain"]["depth_da3"] = l_da3.detach()
        data["rmain"]["depth_da3"] = r_da3.detach()
        data["lmain"]["depth_offset"] = l_offset.detach()
        data["rmain"]["depth_offset"] = r_offset.detach()

    def _depth2gsparms(self, lr_img, lr_img_feat, data, bs):
        l_depth = data["lmain"]["depth"]
        r_depth = data["rmain"]["depth"]
        # Preserve the DAv3 output depth (DA3 backbone+head + 1x1 conv offset
        # correction) before ``gs_parm_regresser`` adds its residual, so the
        # distillation trainer can supervise this quantity directly rather
        # than the post-residual depth used for rendering.
        data["lmain"]["depth_pre_gs"] = l_depth
        data["rmain"]["depth_pre_gs"] = r_depth
        lr_depth = torch.cat([l_depth, r_depth], dim=0)

        rot_maps, scale_maps, opacity_maps, depth_maps = self.gs_parm_regresser(
            lr_img,
            lr_depth,
            lr_img_feat,
        )
        l_resdepth, r_resdepth = torch.split(depth_maps, [bs, bs])

        if self.apply_gs_resdepth:
            l_new = data["lmain"]["depth"] + l_resdepth
            r_new = data["rmain"]["depth"] + r_resdepth
        else:
            l_new = data["lmain"]["depth"]
            r_new = data["rmain"]["depth"]
        data["lmain"]["depth"] = l_new.clamp(
            self.min_inverse_depth,
            self.max_inverse_depth,
        )
        data["rmain"]["depth"] = r_new.clamp(
            self.min_inverse_depth,
            self.max_inverse_depth,
        )

        data["lmain"]["xyz"] = self.depth2pc_fn(
            data["lmain"]["depth"],
            data["lmain"]["extr"],
            data["lmain"]["intr"],
        ).view(bs, -1, 3)
        data["rmain"]["xyz"] = self.depth2pc_fn(
            data["rmain"]["depth"],
            data["rmain"]["extr"],
            data["rmain"]["intr"],
        ).view(bs, -1, 3)

        l_valid = data["lmain"]["mask"][:, :1, :, :] > 0.5
        r_valid = data["rmain"]["mask"][:, :1, :, :] > 0.5
        data["lmain"]["pts_valid"] = l_valid.view(bs, -1)
        data["rmain"]["pts_valid"] = r_valid.view(bs, -1)

        data["novel_view"]["scale_regular"] = torch.mean(scale_maps)
        data["lmain"]["rot_maps"], data["rmain"]["rot_maps"] = torch.split(rot_maps, [bs, bs])
        data["lmain"]["scale_maps"], data["rmain"]["scale_maps"] = torch.split(
            scale_maps,
            [bs, bs],
        )
        data["lmain"]["opacity_maps"], data["rmain"]["opacity_maps"] = torch.split(
            opacity_maps,
            [bs, bs],
        )

        return data

    def forward(self, data, is_train=True):
        del is_train
        bs = data["lmain"]["img"].shape[0]
        image = torch.cat([data["lmain"]["img"], data["rmain"]["img"]], dim=0)

        with autocast(device_type=image.device.type, enabled=self.use_mixed_precision):
            img_feat = self.img_encoder(image)
        # img_feat[2] already holds both views stacked on the batch dim; when the
        # coarse transformer is disabled we leave it untouched (split+cat is a
        # no-op), so only the cross-view attention step is skipped.
        if self.use_loftr_coarse:
            feat_c0, feat_c1 = img_feat[2].split(bs)
            feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, None, None)
            img_feat = img_feat[0], img_feat[1], torch.cat((feat_c0, feat_c1), dim=0)

        backbone_tokens, ph, pw, da3_input_hw = self._run_da3_backbone(
            image, bs, data=data,
        )
        pred_inverse, depth_conf_full = self._predict_inverse_depth_from_backbone(
            backbone_tokens=backbone_tokens,
            ph=ph,
            pw=pw,
            img_feats=img_feat,
            rgb=image,
        )
        l_depth, r_depth = torch.split(pred_inverse, [bs, bs], dim=0)
        l_da3_conf, r_da3_conf = torch.split(depth_conf_full, [bs, bs], dim=0)
        l_da3_inverse, r_da3_inverse = l_depth.detach(), r_depth.detach()

        data["lmain"]["depth"] = l_depth
        data["rmain"]["depth"] = r_depth
        data["lmain"]["depth_da3"] = l_da3_inverse.detach()
        data["rmain"]["depth_da3"] = r_da3_inverse.detach()
        data["lmain"]["depth_conf"] = l_da3_conf
        data["rmain"]["depth_conf"] = r_da3_conf
        data["lmain"]["da3_input_hw"] = torch.tensor(
            da3_input_hw,
            device=image.device,
            dtype=torch.int32,
        )
        data["rmain"]["da3_input_hw"] = torch.tensor(
            da3_input_hw,
            device=image.device,
            dtype=torch.int32,
        )

        self._fill_debug_depth(
            data,
            l_da3=l_da3_inverse,
            r_da3=r_da3_inverse,
            l_offset=l_depth - l_da3_inverse,
            r_offset=r_depth - r_da3_inverse,
        )

        flow_loss = None
        metrics = {}

        if not self.with_gs_render:
            return data, flow_loss, metrics

        data = self._depth2gsparms(image, img_feat, data, bs)
        return data, flow_loss, metrics


class DAV3Model(nn.Module):
    def __init__(self, cfg, with_gs_render=False, legacy_dataset=False, inspect_mode=False):
        super().__init__()
        self.model = RtStereoHumanDAV3Model(
            cfg,
            with_gs_render=with_gs_render,
            legacy_dataset=legacy_dataset,
            inspect_mode=inspect_mode,
        )

    def forward(self, data, is_train=True):
        return self.model(data, is_train=is_train)


class DAV3Model_MK(DAV3Model):
    """Alias of ``DAV3Model``. Kept because ``cfg.model_type`` in the released
    dav3_2view config -- and hence in the runs that produced the published
    checkpoints -- spells the model this way."""
    pass
