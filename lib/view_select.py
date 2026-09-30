"""Pick which source views' Gaussians are merged when rendering a novel view.

The THuman_MV rig is a linear array of 4 physical cameras. ``mv_source_chain``
loads them in spatial order, so ``view_keys`` = ``[view0, view1, view2, view3]``
runs left-to-right along the rig, and consecutive stereo groups overlap:

    s1 = {view0, view1}   s2 = {view1, view2}   s3 = {view2, view3}

Each sample folder ``*_s{g}_*`` stores novel cameras that lie inside segment
``g`` only: cam2/cam3 are interpolated between the two endpoints, while cam4 and
cam5 ARE those endpoints (distance 0 from a source camera).

``pts2render`` merges every view in ``data['source_view_keys']``. Rendering a
novel view from all 4 cameras means most Gaussians project from far-away
viewpoints; selecting the 2 nearest keeps the render to the segment that
actually brackets the target.
"""

from __future__ import annotations

import torch


def camera_centers(extr):
    """World-space camera centers from w2c extrinsics.

    ``extr`` is ``(..., 3, 4)`` or ``(..., 4, 4)``. Returns ``(..., 3)``.
    """
    e = torch.as_tensor(extr).to(torch.float64)
    rot = e[..., :3, :3]
    trans = e[..., :3, 3]
    return -torch.einsum("...ij,...j->...i", rot.transpose(-1, -2), trans)


def rig_order(centers):
    """Indices that sort cameras along the rig's dominant axis.

    Consecutive entries of the returned order are spatial neighbours, which is
    what makes "left"/"right" well defined. Uses the first principal axis rather
    than a hard-coded world axis so the ordering survives a re-oriented rig.
    """
    centered = centers - centers.mean(dim=0, keepdim=True)
    axis = torch.linalg.svd(centered, full_matrices=False)[2][0]
    return torch.argsort(centered @ axis)


def select_nearest_views(
    novel_extr,
    source_extrs,
    view_keys,
    k=2,
    coincide_tol=1e-4,
    rng=None,
):
    """Return the ``k`` ``view_keys`` whose cameras sit closest to the novel view.

    When the novel camera coincides with a source camera (cam4 / cam5), the
    "2 nearest" are degenerate -- the coincident view plus whichever side the
    distance tie-break happens to favour. In that case we keep the coincident
    view and pick one of its rig neighbours, i.e. we randomly choose which of
    the two segments meeting at that camera to render from.

    ``rng`` is a ``numpy.random.Generator`` used for that choice; pass ``None``
    for the deterministic (first neighbour) pick wanted at eval time.
    """
    n = len(view_keys)
    k = max(1, min(int(k), n))

    novel_c = camera_centers(novel_extr).reshape(3)
    src_c = torch.stack([camera_centers(e).reshape(3) for e in source_extrs])
    dist = torch.linalg.norm(src_c - novel_c[None], dim=1)

    nearest = int(torch.argmin(dist))
    if k >= 2 and float(dist[nearest]) <= coincide_tol:
        order = rig_order(src_c).tolist()
        pos = order.index(nearest)
        neighbours = ([order[pos - 1]] if pos > 0 else []) + (
            [order[pos + 1]] if pos + 1 < n else []
        )
        if neighbours:
            pick = neighbours[0] if rng is None else neighbours[int(rng.integers(len(neighbours)))]
            chosen = [nearest, pick]
            # k > 2 falls back to distance order for the remaining slots.
            for idx in torch.argsort(dist).tolist():
                if len(chosen) >= k:
                    break
                if idx not in chosen:
                    chosen.append(idx)
            return [view_keys[i] for i in chosen]

    return [view_keys[i] for i in torch.argsort(dist)[:k].tolist()]
