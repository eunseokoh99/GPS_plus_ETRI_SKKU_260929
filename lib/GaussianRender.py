
import torch
from gaussian_renderer import render


def pts2render(data, bg_color, source_views=None):
    '''
    :param data: rgb input color [-1, 1], will be scaled to [0, 1]
    :param bg_color:  [0, 0, 0]
    :param source_views: views whose Gaussians are merged into the render,
        overriding data['source_view_keys']. Either a flat list of view keys
        (same for every batch item) or one list per batch item, e.g. when each
        item renders from its own nearest pair (dataset.render_nearest_k).
        None -> data['source_view_keys'] -> the plain stereo pair.
    :return: rbg render result in [0, 1]
    '''
    if source_views is None:
        source_views = data.get('source_view_keys', ['lmain', 'rmain'])
    # Batch size comes from a view we are actually going to read, since the
    # multi-view loader does not produce a view called 'lmain'.
    first = source_views[0] if isinstance(source_views[0], str) else source_views[0][0]
    bs = data[first]['img'].shape[0]
    # Normalize to one view list per batch item.
    if isinstance(source_views[0], str):
        source_views = [source_views] * bs

    render_novel_list = []
    for i in range(bs):
        xyz_i_valid = []
        rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []
        for view in source_views[i]:
            valid_i = data[view]['pts_valid'][i, :]
            xyz_i = data[view]['xyz'][i, :, :]  # [S*S, 3]
            rgb_i = data[view]['img'][i, :, :, :].permute(1, 2, 0).view(-1, 3)  # [S*S, 3]
            # rgb_i = data[view]['color_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)  # [S*S, 3]
            rot_i = data[view]['rot_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 4)  # [S*S, 4]
            scale_i = data[view]['scale_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)  # [S*S, 3]
            opacity_i = data[view]['opacity_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 1)  # [S*S, 1]

            xyz_i_valid.append(xyz_i[valid_i].view(-1, 3)) #[valid_i]
            rgb_i_valid.append(rgb_i[valid_i].view(-1, 3))
            rot_i_valid.append(rot_i[valid_i].view(-1, 4))
            scale_i_valid.append(scale_i[valid_i].view(-1, 3))
            opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))

        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
        pts_rgb_i = torch.concat(rgb_i_valid, dim=0)
        pts_rgb_i = pts_rgb_i * 0.5 + 0.5
        rot_i = torch.concat(rot_i_valid, dim=0)
        scale_i = torch.concat(scale_i_valid, dim=0)
        opacity_i = torch.concat(opacity_i_valid, dim=0)

        render_novel_i = render(data, i, pts_xyz_i, pts_rgb_i, rot_i, scale_i, opacity_i, bg_color=bg_color)
        render_novel_list.append(render_novel_i.unsqueeze(0))

    data['novel_view']['img_pred'] = torch.concat(render_novel_list, dim=0)
    return data
