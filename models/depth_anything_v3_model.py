from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from lib.paths import (
    EXTERNAL_REPOS_DIR,
    ensure_external_repo_on_path,
)


logger = logging.getLogger(__name__)


class _DepthOnlyDA3Head(nn.Module):
    def __init__(self, dual_head: nn.Module):
        super().__init__()
        ensure_external_repo_on_path("Depth-Anything-3", "src")
        from depth_anything_3.model.utils.head_utils import (
            create_uv_grid,
            custom_interpolate,
            position_grid_to_embed,
        )

        self.patch_size = dual_head.patch_size
        self.activation = dual_head.activation
        self.conf_activation = dual_head.conf_activation
        self.pos_embed = dual_head.pos_embed
        self.down_ratio = dual_head.down_ratio
        self.intermediate_layer_idx = tuple(dual_head.intermediate_layer_idx)

        self._create_uv_grid = create_uv_grid
        self._custom_interpolate = custom_interpolate
        self._position_grid_to_embed = position_grid_to_embed

        self.norm = dual_head.norm
        self.projects = dual_head.projects
        self.resize_layers = dual_head.resize_layers

        self.scratch = nn.Module()
        for name in (
            "layer1_rn",
            "layer2_rn",
            "layer3_rn",
            "layer4_rn",
            "refinenet1",
            "refinenet2",
            "refinenet3",
            "refinenet4",
            "output_conv1",
            "output_conv2",
        ):
            setattr(self.scratch, name, getattr(dual_head.scratch, name))

    def forward(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        chunk_size: int = 8,
    ) -> dict[str, torch.Tensor]:
        B, S, N, C = feats[0][0].shape
        feats = [feat[0].reshape(B * S, N, C) for feat in feats]

        if chunk_size is None or chunk_size >= S:
            out_dict = self._forward_impl(feats, H, W, patch_start_idx)
            return {k: v.reshape(B, S, *v.shape[1:]) for k, v in out_dict.items()}

        out_dicts = []
        for s0 in range(0, B * S, chunk_size):
            s1 = min(s0 + chunk_size, B * S)
            out_dicts.append(
                self._forward_impl(
                    [feat[s0:s1] for feat in feats],
                    H,
                    W,
                    patch_start_idx,
                )
            )
        merged = {
            key: torch.cat([out_dict[key] for out_dict in out_dicts], dim=0)
            for key in out_dicts[0].keys()
        }
        return {k: v.view(B, S, *v.shape[1:]) for k, v in merged.items()}

    def _forward_impl(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
    ) -> dict[str, torch.Tensor]:
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size

        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw)
            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)
            resized_feats.append(x)

        fused_main = self._fuse_main(resized_feats)
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)
        fused_main = self._custom_interpolate(
            fused_main,
            (h_out, w_out),
            mode="bilinear",
            align_corners=True,
        )
        if self.pos_embed:
            fused_main = self._add_pos_embed(fused_main, W, H)

        main_logits = self.scratch.output_conv2(fused_main)
        fmap = main_logits.permute(0, 2, 3, 1)
        main_pred = self._apply_activation_single(fmap[..., :-1], self.activation)
        main_conf = self._apply_activation_single(fmap[..., -1], self.conf_activation)
        return {
            "depth": main_pred.squeeze(-1),
            "depth_conf": main_conf,
        }

    def _fuse_main(self, feats: list[torch.Tensor]) -> torch.Tensor:
        l1, l2, l3, l4 = feats
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        out = self.scratch.refinenet1(out, l1_rn)
        out = self.scratch.output_conv1(out)
        return out

    def _add_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        pw, ph = x.shape[-1], x.shape[-2]
        pe = self._create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = self._position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe

    def _apply_activation_single(
        self,
        x: torch.Tensor,
        activation: str = "linear",
    ) -> torch.Tensor:
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return torch.exp(x)
        if act == "expm1":
            return torch.expm1(x)
        if act == "expp1":
            return torch.exp(x) + 1
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return torch.nn.functional.softplus(x)
        if act == "tanh":
            return torch.tanh(x)
        return x


def _cfg_get(cfg: Any, path: str, default: Any) -> Any:
    if cfg is None:
        return default

    current = cfg
    for key in path.split("."):
        if current is None:
            return default
        if hasattr(current, "get"):
            current = current.get(key, None)
        elif isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return default if current is None else current


def _tensor_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    tensor = image.detach().cpu().float()
    if tensor.dim() != 3:
        raise ValueError(f"Expected a CHW tensor, got shape {tuple(tensor.shape)}")

    if tensor.shape[0] in {1, 3, 4}:
        tensor = tensor.permute(1, 2, 0)
    elif tensor.shape[-1] not in {1, 3, 4}:
        raise ValueError(
            "Tensor input must be CHW or HWC with 1, 3, or 4 channels. "
            f"Received shape {tuple(image.shape)}"
        )

    if tensor.min().item() < 0.0:
        tensor = (tensor + 1.0) / 2.0
    if tensor.max().item() <= 1.0:
        tensor = tensor * 255.0

    return tensor.clamp(0, 255).byte().numpy()


def _ndarray_to_image_list(images: np.ndarray) -> list[np.ndarray]:
    if images.ndim == 3:
        if images.shape[0] in {1, 3, 4}:
            return [np.transpose(images, (1, 2, 0))]
        return [images]
    if images.ndim == 4:
        if images.shape[1] in {1, 3, 4}:
            return [np.transpose(frame, (1, 2, 0)) for frame in images]
        return [frame for frame in images]
    raise ValueError(
        "NumPy input must be HWC, CHW, NHWC, or NCHW. "
        f"Received shape {tuple(images.shape)}"
    )


class DA3SmallModel(nn.Module):
    repo_name = "Depth-Anything-3"
    config_key = "da3-small"
    pretrained_repo_id = "depth-anything/DA3-SMALL"
    weights_filename = "model.safetensors"
    _KNOWN_AUX_MISSING_KEY_RE = re.compile(
        r"^head\.scratch\.output_conv2_aux\.\d+\.2\.(weight|bias)$"
    )

    def __init__(
        self,
        cfg: Any = None,
        with_gs_render: bool = False,
        legacy_dataset: bool = False,
        inspect_mode: bool = False,
        *,
        device: str | torch.device | None = None,
        load_pretrained: bool | None = None,
        weights_path: str | Path | None = None,
        hf_repo_id: str | None = None,
        process_res: int | None = None,
        process_res_method: str | None = None,
        num_workers: int | None = None,
        cache_dir: str | Path | None = None,
        strict: bool = False,
        disable_aux_head: bool | None = None,
        config_key: str | None = None,
    ):
        super().__init__()
        del with_gs_render, legacy_dataset, inspect_mode

        self.cfg = cfg
        self.device_name = str(
            device
            or _cfg_get(cfg, "external_models.depth_anything_v3.device", None)
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.load_pretrained_on_init = bool(
            _cfg_get(cfg, "external_models.depth_anything_v3.load_pretrained", True)
            if load_pretrained is None
            else load_pretrained
        )
        self.weights_path = (
            Path(weights_path)
            if weights_path is not None
            else _cfg_get(cfg, "external_models.depth_anything_v3.weights_path", None)
        )
        self.hf_repo_id = str(
            hf_repo_id
            or _cfg_get(cfg, "external_models.depth_anything_v3.hf_repo_id", None)
            or self.pretrained_repo_id
        )
        # Allow per-config override of the DA3 architecture variant
        # (e.g. 'da3-small', 'da3-large', 'da3-giant'). A direct ``config_key``
        # kwarg wins over the cfg-derived value, so callers (e.g. the distill
        # trainer building a giant-variant teacher) don't need a yacs cfg.
        cfg_variant = _cfg_get(cfg, "external_models.depth_anything_v3.config_key", None)
        if config_key:
            self.config_key = str(config_key)
        elif cfg_variant:
            self.config_key = str(cfg_variant)
        self.process_res = int(
            process_res
            or _cfg_get(cfg, "external_models.depth_anything_v3.process_res", None)
            or 504
        )
        self.process_res_method = str(
            process_res_method
            or _cfg_get(cfg, "external_models.depth_anything_v3.process_res_method", None)
            or "upper_bound_resize"
        )
        self.num_workers = int(
            num_workers
            or _cfg_get(cfg, "external_models.depth_anything_v3.num_workers", None)
            or 1
        )
        self.cache_dir = Path(cache_dir) if cache_dir is not None else EXTERNAL_REPOS_DIR / "_hf_cache"
        self.strict = strict
        self.disable_aux_head = bool(
            _cfg_get(cfg, "external_models.depth_anything_v3.disable_aux_head", False)
            if disable_aux_head is None
            else disable_aux_head
        )

        self._da3_model: nn.Module | None = None
        self._input_processor: Any = None

        if self.load_pretrained_on_init:
            self._ensure_model_initialized()

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_name)

    def _import_da3_components(self) -> tuple[Any, Any, Any]:
        ensure_external_repo_on_path(self.repo_name, "src")

        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY as da3_registry
        # try:
        # except ImportError as exc:
        #     raise ImportError(
        #         "Depth Anything 3 dependencies are missing. "
        #         "Install the project with the `external-models` extra or install "
        #         "`huggingface_hub`, `omegaconf`, `einops`, `imageio`, `safetensors`, and `addict`."
        #     ) from exc

        return create_object, load_config, da3_registry

    def _import_input_processor_cls(self) -> Any:
        ensure_external_repo_on_path(self.repo_name, "src")

        try:
            from depth_anything_3.utils.io.input_processor import InputProcessor
        except ImportError as exc:
            raise ImportError(
                "Depth Anything 3 input preprocessing dependencies are missing. "
                "Install the project with the `external-models` extra or install "
                "`imageio` and the rest of the DA3 extras."
            ) from exc

        return InputProcessor

    def _download_weights(self) -> Path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to download DA3-SMALL weights."
            ) from exc

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=self.hf_repo_id,
            filename=self.weights_filename,
            cache_dir=str(self.cache_dir),
        )
        return Path(downloaded)

    def _load_weights(self, model: nn.Module, weights_path: Path) -> None:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("safetensors is required to load DA3-SMALL weights.") from exc

        state_dict = load_file(str(weights_path))
        state_dict = self._align_state_dict_keys(model, state_dict)
        state_dict = self._fill_known_missing_aux_keys(model, state_dict)
        incompatible = model.load_state_dict(state_dict, strict=self.strict)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            logger.warning(
                "DA3-SMALL state dict mismatch. missing=%s unexpected=%s",
                incompatible.missing_keys,
                incompatible.unexpected_keys,
            )

    def _fill_known_missing_aux_keys(
        self,
        model: nn.Module,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        target_state = model.state_dict()
        patched_state_dict = dict(state_dict)

        for key, value in target_state.items():
            if key in patched_state_dict:
                continue
            if self._KNOWN_AUX_MISSING_KEY_RE.match(key):
                patched_state_dict[key] = value.detach().clone()

        return patched_state_dict

    def _align_state_dict_keys(
        self,
        model: nn.Module,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        target_keys = set(model.state_dict().keys())
        candidate_prefixes = (
            "",
            "model.",
            "_orig_mod.model.",
            "_orig_mod.",
            "module.model.",
            "module.",
        )

        best_prefix = ""
        best_overlap = -1
        best_state_dict = state_dict

        for prefix in candidate_prefixes:
            if prefix:
                candidate = {
                    key[len(prefix):]: value
                    for key, value in state_dict.items()
                    if key.startswith(prefix)
                }
                if not candidate:
                    continue
            else:
                candidate = dict(state_dict)

            overlap = len(target_keys.intersection(candidate.keys()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_prefix = prefix
                best_state_dict = candidate

        if best_prefix:
            logger.info(
                "Adjusted DA3 state_dict keys by stripping prefix %r (%d matched keys)",
                best_prefix,
                best_overlap,
            )
        else:
            logger.info("Using DA3 state_dict keys as-is (%d matched keys)", best_overlap)

        return best_state_dict

    def _ensure_model_initialized(self) -> None:
        if self._da3_model is not None:
            return

        create_object, load_config, da3_registry = self._import_da3_components()

        if self.config_key not in da3_registry:
            raise KeyError(
                f"Depth Anything 3 config '{self.config_key}' was not found in "
                f"external repo registry. Available keys: {list(da3_registry.keys())}"
            )

        config = load_config(da3_registry[self.config_key])
        model = create_object(config)
        model.eval().to(self.device)

        weights_path = Path(self.weights_path) if self.weights_path is not None else None
        if self.load_pretrained_on_init:
            if weights_path is None:
                weights_path = self._download_weights()
            self._load_weights(model, weights_path)

        if self.disable_aux_head:
            model = self._strip_aux_head(model)

        self._da3_model = model

    def _strip_aux_head(self, model: nn.Module) -> nn.Module:
        if not hasattr(model, "head"):
            raise AttributeError("DA3 model does not expose a `head` module.")

        model.head = _DepthOnlyDA3Head(model.head)
        return model

    def _ensure_initialized(self) -> None:
        self._ensure_model_initialized()
        if self._input_processor is None:
            self._input_processor = self._import_input_processor_cls()()

    def _normalize_inputs(
        self,
        images: torch.Tensor | np.ndarray | Sequence[Any] | str | Path,
    ) -> list[Any]:
        if isinstance(images, (str, Path)):
            return [str(images)]

        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                return [_tensor_to_uint8_hwc(images)]
            if images.dim() == 4:
                return [_tensor_to_uint8_hwc(frame) for frame in images]
            raise ValueError(
                "Tensor input must be CHW or NCHW. "
                f"Received shape {tuple(images.shape)}"
            )

        if isinstance(images, np.ndarray):
            return _ndarray_to_image_list(images)

        if isinstance(images, Sequence):
            normalized: list[Any] = []
            for item in images:
                normalized.append(str(item) if isinstance(item, Path) else item)
            return normalized

        raise TypeError(f"Unsupported input type for DA3-SMALL: {type(images)}")

    def _prepare_model_inputs(
        self,
        images: list[Any],
        extrinsics: np.ndarray | None,
        intrinsics: np.ndarray | None,
        process_res: int,
        process_res_method: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        assert self._input_processor is not None

        imgs_cpu, ex_t, in_t = self._input_processor(
            image=images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            process_res=process_res,
            process_res_method=process_res_method,
            num_workers=self.num_workers,
            sequential=self.num_workers <= 1,
            print_progress=False,
        )

        imgs = imgs_cpu.to(self.device, non_blocking=True)[None].float()
        ex_t = ex_t.to(self.device, non_blocking=True)[None].float() if ex_t is not None else None
        in_t = in_t.to(self.device, non_blocking=True)[None].float() if in_t is not None else None
        return imgs, ex_t, in_t

    def _run_model(
        self,
        imgs: torch.Tensor,
        ex_t: torch.Tensor | None,
        in_t: torch.Tensor | None,
        export_feat_layers: Iterable[int] | None,
        infer_gs: bool,
        use_ray_pose: bool,
        ref_view_strategy: str,
        *,
        enable_grad: bool = False,
    ) -> dict[str, Any]:
        assert self._da3_model is not None
        if self.disable_aux_head and use_ray_pose:
            raise ValueError("use_ray_pose=True is not supported when aux head is disabled.")

        feat_layers = list(export_feat_layers) if export_feat_layers is not None else []
        grad_context = torch.enable_grad() if enable_grad else torch.inference_mode()
        # If the model's weights are already in a low-precision dtype (e.g. the
        # distill teacher cast to bf16), skip autocast: autocast would force
        # certain ops (``layer_norm``, ``softmax``) back to fp32, producing
        # fp32-vs-bf16 dtype mismatches against the bf16 weights. With the
        # whole model already in bf16, every op runs natively in that dtype.
        model_dtype = next(self._da3_model.parameters()).dtype
        model_is_low_precision = model_dtype in (torch.bfloat16, torch.float16)
        with grad_context:
            if imgs.device.type == "cuda" and not model_is_low_precision:
                autocast_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    output = self._da3_model(
                        imgs,
                        ex_t,
                        in_t,
                        feat_layers,
                        infer_gs,
                        use_ray_pose,
                        ref_view_strategy,
                    )
            else:
                output = self._da3_model(
                    imgs,
                    ex_t,
                    in_t,
                    feat_layers,
                    infer_gs,
                    use_ray_pose,
                    ref_view_strategy,
                )
        return dict(output)

    def _postprocess_output(self, output: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "depth": output["depth"].squeeze(0).squeeze(-1).detach().cpu().numpy(),
            "is_metric": int(output.get("is_metric", 0)),
        }

        for key, source_key in (
            ("depth_conf", "depth_conf"),
            ("extrinsics", "extrinsics"),
            ("intrinsics", "intrinsics"),
            ("sky", "sky"),
        ):
            value = output.get(source_key)
            result[key] = value.squeeze(0).detach().cpu().numpy() if value is not None else None

        aux = output.get("aux")
        if aux is None:
            result["aux"] = {}
        else:
            aux_out: dict[str, Any] = {}
            for key, value in aux.items():
                aux_out[key] = value.squeeze(0).detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
            result["aux"] = aux_out

        if output.get("gaussians") is not None:
            result["gaussians"] = output["gaussians"]
        if output.get("scale_factor") is not None:
            scale_factor = output["scale_factor"]
            result["scale_factor"] = (
                float(scale_factor.detach().cpu().item())
                if isinstance(scale_factor, torch.Tensor)
                else float(scale_factor)
            )

        return result

    def forward(
        self,
        images: torch.Tensor | np.ndarray | Sequence[Any] | str | Path,
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        *,
        process_res: int | None = None,
        process_res_method: str | None = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        export_feat_layers: Iterable[int] | None = None,
        return_raw: bool = False,
    ) -> dict[str, Any]:
        self._ensure_initialized()

        normalized_images = self._normalize_inputs(images)
        imgs, ex_t, in_t = self._prepare_model_inputs(
            normalized_images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            process_res=process_res or self.process_res,
            process_res_method=process_res_method or self.process_res_method,
        )
        output = self._run_model(
            imgs,
            ex_t,
            in_t,
            export_feat_layers=export_feat_layers,
            infer_gs=infer_gs,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            enable_grad=False,
        )
        return output if return_raw else self._postprocess_output(output)

    def infer_depth(
        self,
        images: torch.Tensor | np.ndarray | Sequence[Any] | str | Path,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.forward(images, **kwargs)["depth"]
