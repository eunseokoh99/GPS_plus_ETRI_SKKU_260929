"""Small helpers shared by train.py and test.py.

They live here rather than on the Trainer so the two entry points cannot drift
apart -- render-view selection in particular must behave identically at train
and test time or the reported PSNR stops matching the trained model.
"""

from __future__ import annotations

import torch

from lib.view_select import select_nearest_views


# Keys of a novel-view dict that the rasterizer reads and therefore need to be
# on the GPU. Everything else (sample_name, width/height ints) is left alone.
NOVEL_CUDA_KEYS = (
    'extr', 'FovX', 'FovY', 'world_view_transform',
    'full_proj_transform', 'camera_center',
)


def freeze_raft_bn(model):
    """Keep RAFT-Stereo BatchNorm frozen whenever the model is in train mode.

    The DAv3 models carry a parameter-free ``_NoOpRaftStereo`` stub one level
    deeper (they are thin wrappers around the real net), so look the attribute
    up instead of assuming it sits directly on ``model``.
    """
    raft = getattr(model, 'raft_stereo', None)
    if raft is None:
        raft = getattr(getattr(model, 'model', None), 'raft_stereo', None)
    if raft is not None:
        raft.freeze_bn()


def novel_view_to_cuda(novel_view):
    """Return a copy of ``novel_view`` with the rasterizer's tensors on GPU."""
    out = dict(novel_view)
    for key in NOVEL_CUDA_KEYS:
        val = out.get(key)
        if torch.is_tensor(val):
            out[key] = val.cuda(non_blocking=True)
    return out


def pick_render_views(data, novel_view, view_keys, k, coincide_tol=1e-4, rng=None):
    """Per-batch-item source-view lists to merge when rendering this novel view.

    Returns ``None`` when ``k`` is 0, which leaves ``pts2render`` merging every
    source view exactly as the 2-view models expect.

    ``rng`` is deliberately asymmetric between call sites: training passes the
    trainer's seeded generator (a novel camera coinciding with a source camera
    picks a random rig neighbour), evaluation passes ``None`` (deterministic
    first neighbour, so PSNR is reproducible). Do not unify them -- doing so
    changes both the training trajectory and the reported PSNR.
    """
    if k <= 0:
        return None
    keys = list(data.get('source_view_keys') or view_keys or ['lmain', 'rmain'])
    bs = data[keys[0]]['img'].shape[0]
    return [
        select_nearest_views(
            novel_view['extr'][i],
            [data[v]['extr'][i] for v in keys],
            keys,
            k=k,
            coincide_tol=coincide_tol,
            rng=rng,
        )
        for i in range(bs)
    ]
