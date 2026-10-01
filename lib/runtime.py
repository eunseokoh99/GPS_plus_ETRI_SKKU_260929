"""Small helpers shared by train.py and test.py.

They live here rather than on the Trainer so the two entry points cannot drift
apart -- render-view selection in particular must behave identically at train
and test time or the reported PSNR stops matching the trained model.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from lib.GaussianRender import pts2render
from lib.gs_utils.image_utils import psnr
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


def eval_psnr_pass(
    *,
    model,
    val_set,
    fetch_data,
    len_val,
    novel_view_ids,
    bg_color,
    hcrop,
    pick_views,
    on_sample=None,
):
    """One full validation pass. Returns a dict of PSNRs and forward timings.

    train.py and test.py both call this so their reported PSNR cannot drift.
    Every sample is scored at *every* id in ``novel_view_ids``: each view is
    rebuilt here rather than taken from the loader, so the loader's per-sample
    random choice is bypassed and all views are always covered. The model runs
    once per sample; only the rasterization repeats per view.

    ``hcrop`` is the fraction of rows dropped from BOTH top and bottom before
    computing ``crop_psnr`` (0.0 -> identical to ``full_psnr``).

    ``pick_views(data, novel_view)`` returns the source views to merge; pass a
    closure over the caller's nearest-k settings with ``rng=None`` so the render
    is deterministic.

    ``on_sample(idx, sample_name, view_id, render_data)`` is called after each
    render, for callers that want to dump images.

    This measures no time: forward speed is benchmarked separately, by
    ``benchmark_forward``, because a per-sample average taken here moves with
    dataloader and GPU load.

    The caller is responsible for resetting its val iterator first, and for
    putting the model in eval mode.
    """
    novel_view_ids = list(novel_view_ids)
    psnr_list, crop_psnr_list = [], []

    for idx in range(len_val):
        data = fetch_data(phase='val')
        sample_name = data['name'][0]
        with torch.no_grad():
            base_data, _, _ = model(data, is_train=False)

            for view_id in novel_view_ids:
                # Re-render the same gaussians to each novel view in turn.
                render_data = dict(base_data)
                # The loader hands back CPU tensors; the nearest-view selection
                # compares them against the (CUDA) source-view extrinsics, so
                # move them across first. Pure device move, no numerical effect.
                render_data['novel_view'] = novel_view_to_cuda(
                    default_collate(
                        [val_set.get_novel_view_tensor(sample_name, view_id)]
                    )
                )
                render_data = pts2render(
                    render_data,
                    bg_color=bg_color,
                    source_views=pick_views(render_data, render_data['novel_view']),
                )

                render_novel = render_data['novel_view']['img_pred']
                gt_novel = render_data['novel_view']['img'].cuda()

                # Full-frame PSNR over the whole novel view.
                psnr_list.append(psnr(render_novel, gt_novel).mean().double().item())

                # Cropped PSNR: drop the top/bottom `hcrop` fraction of rows
                # before scoring (0.0 = no crop -> same as full-frame).
                render_crop, gt_crop = render_novel, gt_novel
                if hcrop > 0.0:
                    h = render_novel.shape[-2]
                    top = int(round(h * hcrop))
                    if top > 0:
                        render_crop = render_novel[..., top:h - top, :]
                        gt_crop = gt_novel[..., top:h - top, :]
                crop_psnr_list.append(
                    psnr(render_crop, gt_crop).mean().double().item()
                )

                if on_sample is not None:
                    on_sample(idx, sample_name, view_id, render_data)

    return {
        'full_psnr': np.round(np.mean(np.array(psnr_list)), 4),
        'crop_psnr': np.round(np.mean(np.array(crop_psnr_list)), 4),
        'n_scored': len(psnr_list),
    }


def benchmark_forward(*, model, batches, warmup=50, iters=100):
    """Controlled timing of the model forward over ``batches``, pre-staged on GPU.

    One timed iteration is a forward of EVERY batch in ``batches``, so the unit
    is "all four source cameras of one frame covered":

      * 4-view branches take one batch -- the four cameras go in together.
      * 2-view branches take three, the s1/s2/s3 camera pairs of one frame, run
        back to back, because the model only accepts one pair at a time.

    This is deliberately NOT an average over the validation loop. There the
    dataloader workers and the PNG writes compete for CPU and the GPU may be
    shared, so the per-sample figure moves with system load -- on a GPU already
    running a training job it came out 2x higher. Here the batches are already
    on the GPU, nothing else runs between iterations, and the first ``warmup``
    passes are discarded (CUDA kernel selection, clock ramp-up). These are the
    conditions the speeds in README.md were measured under, so an otherwise idle
    GPU is needed for the number to be comparable.

    Data IO is outside the timer and ``pts2render`` is never called, so the
    number covers only the path from input images to Gaussian parameters.
    """
    def one_pass():
        for b in batches:
            model(b, is_train=False)

    with torch.no_grad():
        for _ in range(warmup):
            one_pass()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            one_pass()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {
        'mean_ms': round(statistics.mean(ts), 1),
        'median_ms': round(statistics.median(ts), 1),
        'min_ms': round(ts[0], 1),
        'std_ms': round(statistics.pstdev(ts), 2),
        'peak_mem_MB': round(torch.cuda.max_memory_allocated() / 2 ** 20),
        'forwards': len(batches),
        'warmup': warmup,
        'iters': iters,
    }
