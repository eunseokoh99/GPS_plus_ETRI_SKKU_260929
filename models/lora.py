"""LoRA (Low-Rank Adaptation) utilities for fine-tuning frozen backbones.

This module provides a minimal hand-rolled LoRA implementation:

  * ``LoRALinear`` wraps an existing ``nn.Linear`` with a frozen base
    weight/bias and a trainable low-rank update (``lora_A``/``lora_B``).
  * ``apply_lora_to_modules`` walks a model and replaces target
    ``nn.Linear`` submodules (matched by attribute name) with
    ``LoRALinear``.
  * ``apply_backbone_lora`` is the DA3-specific entry point used by
    ``dav3_model.py``: it descends into the DinoV2 ``pretrained.blocks``
    and wraps the attention projections.

The implementation has no external dependencies (no ``peft``); ``lora_B``
is zero-initialised so a freshly-wrapped layer matches the original
forward exactly until training updates the adapters.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn


logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """nn.Linear with a frozen base and a trainable low-rank update.

    Forward: ``y = (x @ W.T + b) + (alpha / rank) * lora_dropout(x) @ A.T @ B.T``

    The base ``weight``/``bias`` are stored as parameters with
    ``requires_grad=False`` so they remain part of ``state_dict``.
    ``lora_A``/``lora_B`` are the only trainable params here; ``lora_B``
    is zero-initialised so the wrapped layer is initially equivalent to
    the original ``nn.Linear``.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects an nn.Linear, got {type(base).__name__}")

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)

        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(torch.zeros(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = (
            nn.Dropout(p=float(dropout)) if dropout and float(dropout) > 0 else nn.Identity()
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"bias={self.bias is not None}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = F.linear(self.lora_dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return base_out + self.scaling * lora_out


def apply_lora_to_modules(
    parents: Iterable[nn.Module],
    target_attrs: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> int:
    """For each parent module, replace each ``nn.Linear`` child whose
    attribute name is in ``target_attrs`` with a ``LoRALinear``.

    Returns the number of layers wrapped.
    """
    n_wrapped = 0
    for parent in parents:
        if parent is None:
            continue
        for attr in target_attrs:
            child = getattr(parent, attr, None)
            if isinstance(child, nn.Linear):
                setattr(
                    parent,
                    attr,
                    LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
                )
                n_wrapped += 1
    return n_wrapped


def apply_backbone_lora(
    da3_model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_modules: Sequence[str] = ("qkv", "proj"),
) -> int:
    """Inject LoRA adapters into a DA3 DinoV2 backbone.

    Walks ``da3_model.backbone.pretrained.blocks`` and wraps the attention
    projections (``attn.qkv``, ``attn.proj`` by default). Base weights are
    kept frozen; LoRA params become the only trainable parameters in the
    affected modules.

    Args:
        da3_model: A ``DepthAnything3Net`` (or compatible) module exposing
            ``backbone.pretrained.blocks``.
        rank: LoRA rank.
        alpha: LoRA scaling alpha.
        dropout: Dropout on the LoRA input path.
        target_modules: Linear attribute names under each ``block.attn``
            to wrap (default: ``("qkv", "proj")``).

    Returns:
        Number of nn.Linear modules wrapped.
    """
    backbone = getattr(da3_model, "backbone", None)
    if backbone is None:
        raise RuntimeError("DA3 model has no `backbone` attribute; cannot attach LoRA.")
    pretrained = getattr(backbone, "pretrained", None)
    if pretrained is None or not hasattr(pretrained, "blocks"):
        raise RuntimeError(
            "DA3 backbone has no `pretrained.blocks`; cannot attach LoRA."
        )

    attn_parents = [getattr(block, "attn", None) for block in pretrained.blocks]
    n_wrapped = apply_lora_to_modules(
        parents=attn_parents,
        target_attrs=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    logger.info(
        "[DAV3 LoRA] wrapped %d Linear modules (rank=%d, alpha=%.4g, targets=%s, dropout=%.4g)",
        n_wrapped,
        int(rank),
        float(alpha),
        list(target_modules),
        float(dropout),
    )
    return n_wrapped


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Return the trainable LoRA parameters (``lora_A``/``lora_B``) under
    ``module``. Convenience helper for parameter-group construction in
    optimizers."""
    params: list[nn.Parameter] = []
    for sub in module.modules():
        if isinstance(sub, LoRALinear):
            params.append(sub.lora_A)
            params.append(sub.lora_B)
    return params


def enable_bitfit(module: nn.Module) -> int:
    """Mark every ``.bias`` parameter under ``module`` as trainable.

    BitFit-style adaptation: leave weights frozen, train only biases.
    Returns the number of bias parameters re-enabled.
    """
    n = 0
    for name, param in module.named_parameters():
        if name.endswith("bias"):
            param.requires_grad_(True)
            n += 1
    logger.info("[DAV3 BitFit] re-enabled %d bias parameters", n)
    return n


def enable_layernorm_tuning(module: nn.Module) -> int:
    """Mark every ``LayerNorm`` parameter under ``module`` as trainable.

    LayerNorm-only adaptation: leave weights frozen, train only the affine
    parameters of LayerNorm submodules. Returns the number of LayerNorm
    parameters re-enabled.
    """
    n = 0
    for sub in module.modules():
        if isinstance(sub, nn.LayerNorm):
            for param in sub.parameters(recurse=False):
                param.requires_grad_(True)
                n += 1
    logger.info("[DAV3 LayerNorm] re-enabled %d LayerNorm parameters", n)
    return n
