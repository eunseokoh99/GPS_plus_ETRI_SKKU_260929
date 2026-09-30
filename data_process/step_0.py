import os
import json
import argparse
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

cv2.setNumThreads(1)  # avoid oversubscription with OpenCV internal threads


def get_stereo_data(main_view_data, ref_view_data):
    # No rectification: keep original undistorted images and calibration.
    img0, intr0, extr0 = main_view_data
    img1, intr1, extr1 = ref_view_data

    r0, t0 = extr0[:3, :3], extr0[:3, 3:]
    r1, t1 = extr1[:3, :3], extr1[:3, 3:]
    inv_r0 = r0.T
    inv_t0 = -r0.T @ t0
    E0 = np.eye(4)
    E0[:3, :3], E0[:3, 3:] = inv_r0, inv_t0
    E1 = np.eye(4)
    E1[:3, :3], E1[:3, 3:] = r1, t1
    E = E1 @ E0
    T = E[:3, 3]
    # Approximation of the rectified-stereo Tf_x = fx * baseline.
    Tf_x = np.array(-intr1[0, 0] * T[0])

    camera = {
        'intr0': intr0.astype(float).copy(),
        'intr1': intr1.astype(float).copy(),
        'extr0': extr0.astype(float).copy(),
        'extr1': extr1.astype(float).copy(),
        'Tf_x':  Tf_x,
    }
    mask0 = np.ones_like(img0) * 255
    mask1 = np.ones_like(img1) * 255
    return {
        'img0':   img0,
        'mask0':  mask0,
        'img1':   img1,
        'mask1':  mask1,
        'camera': camera,
    }


def get_rectified_stereo_data(main_view_data, ref_view_data, img_sz):
    # define view 0 as main and 1 as reference
    img0, intr0, extr0 = main_view_data
    img1, intr1, extr1 = ref_view_data

    W, H = img_sz[0], img_sz[1]
    r0, t0 = extr0[:3, :3], extr0[:3, 3:]
    r1, t1 = extr1[:3, :3], extr1[:3, 3:]
    inv_r0 = r0.T
    inv_t0 = -r0.T @ t0
    E0 = np.eye(4)
    E0[:3, :3], E0[:3, 3:] = inv_r0, inv_t0
    E1 = np.eye(4)
    E1[:3, :3], E1[:3, 3:] = r1, t1
    E = E1 @ E0
    R, T = E[:3, :3], E[:3, 3]
    dist0, dist1 = np.zeros(4), np.zeros(4)
    # https://blog.csdn.net/qq_25458977/article/details/114829674
    R0, R1, P0, P1, _, _, _ = cv2.stereoRectify(intr0, dist0, intr1, dist1, (W, H), R, T, flags=0)

    new_extr0 = R0 @ extr0
    new_intr0 = P0[:3, :3]
    new_extr1 = R1 @ extr1
    new_intr1 = P1[:3, :3]
    Tf_x = np.array(P1[0, 3])

    camera = {
        'intr0': new_intr0,
        'intr1': new_intr1,
        'extr0': new_extr0,
        'extr1': new_extr1,
        'Tf_x':  Tf_x,
    }

    mask0 = np.ones_like(img0) * 255
    mask1 = np.ones_like(img1) * 255
    map0x, map0y = cv2.initUndistortRectifyMap(intr0, dist0, R0, P0, (W, H), cv2.CV_32FC1)
    map1x, map1y = cv2.initUndistortRectifyMap(intr1, dist1, R1, P1, (W, H), cv2.CV_32FC1)

    return {
        'img0':   cv2.remap(img0,  map0x, map0y, cv2.INTER_LINEAR),
        'mask0':  cv2.remap(mask0, map0x, map0y, cv2.INTER_LINEAR),
        'img1':   cv2.remap(img1,  map1x, map1y, cv2.INTER_LINEAR),
        'mask1':  cv2.remap(mask1, map1x, map1y, cv2.INTER_LINEAR),
        'camera': camera,
    }


def load_data(cam, t, ori_dir, calib_full):
    intr   = np.array(calib_full[cam]['K']).astype(float).reshape(3, 3)
    dist   = np.array(calib_full[cam]['distCoeff']).astype(float).reshape(5)
    img_sz = calib_full[cam]['imgSize']
    img    = cv2.imread(os.path.join(ori_dir, f'{t}_{cam}.jpg'))
    img    = cv2.undistort(img, intr, dist, None)
    R_     = np.array(calib_full[cam]['R']).astype(float).reshape(3, 3)
    T_     = np.array(calib_full[cam]['T']).astype(float).reshape(3, 1)
    extr   = np.concatenate([R_, T_], axis=1)
    return (img, intr, extr), img_sz


def save_np_to_json(parm, save_name, img_sz):
    w, h = img_sz
    move_t = (h - w) / 2
    parm['intr0'][1, -1] -= move_t
    parm['intr1'][1, -1] -= move_t
    rescale = 1024 / min(w, h)
    parm['intr0'][:2] *= rescale
    parm['intr1'][:2] *= rescale
    parm['Tf_x']      *= rescale
    for key in parm:
        parm[key] = parm[key].tolist()
    with open(save_name, 'w') as f:
        json.dump(parm, f, indent=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stage 0: build the stereo source-view pair (views 0 and 1) '
                    'for every stereo segment of a THumanMV sequence.')
    parser.add_argument('-i', '--input',    type=str, required=True,
                        help='input sequence, e.g. s1a1')
    parser.add_argument('-t', '--trainval', type=str, required=True,
                        choices=['train', 'val', 'test'], help='split')
    parser.add_argument('--data-root', type=str, required=True,
                        help='raw THumanMV root (contains <seq>/ with '
                             '<frame>_<camserial>.jpg and calibration_full.json)')
    parser.add_argument('--processed-root', type=str, required=True,
                        help='output dataset root, e.g. .../preprocessed_wo_rect')
    rect = parser.add_mutually_exclusive_group(required=True)
    rect.add_argument('--rect', dest='rect', action='store_true',
                      help='stereo-rectify the pair (dataset for the gps_gs branch)')
    rect.add_argument('--no-rect', dest='rect', action='store_false',
                      help='keep the undistorted images as-is (dataset for the dav3_* branches)')
    parser.add_argument('-j', '--num_workers', type=int, default=min(32, os.cpu_count() or 8),
                        help='number of parallel threads')
    arg = parser.parse_args()

    data_root = os.path.join(arg.data_root, '')
    processed_data_root = os.path.join(arg.processed_root, '')
    print(f"[step_0] {'RECTIFIED' if arg.rect else 'NON-RECTIFIED'}  "
          f"{data_root}{arg.input} -> {processed_data_root}{arg.trainval}")

    data_n     = arg.input
    ori_dir    = data_root + data_n
    processed_data_root += arg.trainval
    calib_path = ori_dir + '/calibration_full.json'

    file_list = sorted(os.listdir(ori_dir))
    used_time_id_list = []
    for file in file_list:
        if file[-3:] != 'jpg':
            continue
        time_id = file.split('/')[-1].split('_')[0]
        if time_id not in used_time_id_list:
            used_time_id_list.append(time_id)

    if arg.trainval == 'train':
        used_time_id_list = sorted(used_time_id_list)[:300]
    elif arg.trainval == 'val':
        used_time_id_list = sorted(used_time_id_list)[50:80]
    elif arg.trainval == 'test':
        used_time_id_list = sorted(used_time_id_list)
        processed_data_root = processed_data_root + '/' + data_n + '_process'
        Path(processed_data_root).mkdir(exist_ok=True, parents=True)
        shutil.copyfile(calib_path, processed_data_root + '/calibration_full.json')
    else:
        exit()

    img_dir = os.path.join(processed_data_root, 'img')
    msk_dir = os.path.join(processed_data_root, 'mask')
    par_dir = os.path.join(processed_data_root, 'parameter')
    for d in (img_dir, msk_dir, par_dir):
        Path(d).mkdir(exist_ok=True, parents=True)

    cam_id_list_s = [
        ['22139908', '22139909'],
        ['22139909', '22139914'],
        ['22139914', '22139906'],
    ]

    with open(calib_path, 'r') as f:
        calib_full = json.load(f)

    for cam_id_list in cam_id_list_s:
        if cam_id_list[0] == '22139908':
            scene_n = 's1'
        elif cam_id_list[0] == '22139909':
            scene_n = 's2'
        elif cam_id_list[0] == '22139914':
            scene_n = 's3'
        else:
            print('wrong')
            exit()

        def _process_one_t(t, _cam_id_list=cam_id_list, _scene_n=scene_n):
            tag       = '%s_%s_%04d' % (data_n, _scene_n, int(t))
            t_dir     = os.path.join(img_dir, tag)
            t_msk_dir = os.path.join(msk_dir, tag)
            t_par_dir = os.path.join(par_dir, tag)
            for d in (t_dir, t_msk_dir, t_par_dir):
                os.makedirs(d, exist_ok=True)

            mview, img_sz = load_data(_cam_id_list[0], t, ori_dir, calib_full)
            rview, _      = load_data(_cam_id_list[1], t, ori_dir, calib_full)

            if arg.rect:
                stereo_data = get_rectified_stereo_data(mview, rview, img_sz)
            else:
                stereo_data = get_stereo_data(mview, rview)
            img0 = stereo_data['img0']
            img1 = stereo_data['img1']
            msk0 = stereo_data['mask0']
            msk1 = stereo_data['mask1']

            w, h = img_sz[0], img_sz[1]
            move_t = (h - w) // 2

            ######## scene specific ###########
            img0 = img0[(move_t):(w + move_t), :, :]
            img1 = img1[(move_t):(w + move_t), :, :]
            msk0 = msk0[(move_t):(w + move_t), :, :]
            msk1 = msk1[(move_t):(w + move_t), :, :]
            ######## scene specific ###########

            img0 = cv2.resize(img0, (1024, 1024))
            img1 = cv2.resize(img1, (1024, 1024))
            msk0 = cv2.resize(msk0, (1024, 1024))
            msk1 = cv2.resize(msk1, (1024, 1024))

            cv2.imwrite(os.path.join(t_dir,     '0.png'), img0.astype(np.uint8))
            cv2.imwrite(os.path.join(t_dir,     '1.png'), img1.astype(np.uint8))
            cv2.imwrite(os.path.join(t_msk_dir, '0.png'), msk0.astype(np.uint8))
            cv2.imwrite(os.path.join(t_msk_dir, '1.png'), msk1.astype(np.uint8))
            save_np_to_json(stereo_data['camera'], os.path.join(t_par_dir, '0_1.json'), img_sz)

        with ThreadPoolExecutor(max_workers=arg.num_workers) as ex:
            futures = [ex.submit(_process_one_t, t) for t in used_time_id_list]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=scene_n):
                pass
