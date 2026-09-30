import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

cv2.setNumThreads(1)  # avoid oversubscription with OpenCV internal threads


# python step_1.py -i s2a3 -t val [-j 16]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stage 1: add the novel views (2..5) for every stereo '
                    'segment of a THumanMV sequence. Views are never rectified; '
                    'run this with the SAME --processed-root as step_0.py.')
    parser.add_argument('-i', '--input',    type=str, required=True,
                        help='input sequence, e.g. s1a1')
    parser.add_argument('-t', '--trainval', type=str, required=True,
                        choices=['train', 'val', 'test'], help='split')
    parser.add_argument('--data-root', type=str, required=True,
                        help='raw THumanMV root')
    parser.add_argument('--processed-root', type=str, required=True,
                        help='output dataset root -- MUST match the step_0.py run')
    parser.add_argument('-j', '--num_workers', type=int, default=min(32, os.cpu_count() or 8),
                        help='number of parallel threads')
    arg = parser.parse_args()

    data_root = os.path.join(arg.data_root, '')
    processed_data_root = os.path.join(arg.processed_root, '')
    print(f"[step_1] {data_root}{arg.input} -> {processed_data_root}{arg.trainval}")

    data_n              = arg.input
    ori_dir             = data_root + data_n
    processed_data_root = processed_data_root + arg.trainval

    Path(processed_data_root).mkdir(exist_ok=True, parents=True)

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
    else:
        exit()

    img_dir = os.path.join(processed_data_root, 'img')
    par_dir = os.path.join(processed_data_root, 'parameter')
    for d in (img_dir, par_dir):
        Path(d).mkdir(exist_ok=True, parents=True)

    cam_move = [0, 0, 0, 0]  # TODO

    cam_id_list_s = [
        ['22139907', '22070932', '22139908', '22139909'],
        ['22053927', '22053908', '22139909', '22139914'],
        ['22053925', '22053923', '22139914', '22139906'],
    ]

    calib_path = ori_dir + '/calibration_full.json'
    with open(calib_path, 'r') as f:
        calib_full = json.load(f)

    for cam_id_list in cam_id_list_s:
        if cam_id_list[0] == '22139907':
            scene_n = 's1'
        elif cam_id_list[0] == '22053927':
            scene_n = 's2'
        elif cam_id_list[0] == '22053925':
            scene_n = 's3'
        else:
            exit()

        for cam_i, cam in enumerate(cam_id_list):
            K      = np.array(calib_full[cam]['K']).astype(float).reshape(3, 3)
            dist   = np.array(calib_full[cam]['distCoeff']).astype(float).reshape(5)
            img_sz = calib_full[cam]['imgSize']
            in_mat = K
            w, h   = img_sz[0], img_sz[1]
            # 1500, 2048
            move_t = (h - w) // 2 + cam_move[cam_i]

            tmp = np.array(calib_full[cam]['K']).astype(float).reshape(3, 3).copy()
            tmp[1, -1] -= move_t

            ######## scene specific ###########
            tmp[:2] /= (min(w, h) / 1024.0)
            ######## scene specific ###########

            R_ = np.array(calib_full[cam]['R']).astype(float).reshape(3, 3)
            T_ = np.array(calib_full[cam]['T']).astype(float).reshape(3, 1)
            extr = np.concatenate([R_, T_], axis=1)
            print('intr cam ', cam)
            print(tmp)
            print('-----------------------')

            def _process_one_t(t, _scene_n=scene_n, _cam=cam, _cam_i=cam_i,
                               _move_t=move_t, _in_mat=in_mat, _dist=dist,
                               _w=w, _tmp=tmp, _extr=extr):
                tag       = '%s_%s_%04d' % (data_n, _scene_n, int(t))
                t_dir     = os.path.join(img_dir, tag)
                t_par_dir = os.path.join(par_dir, tag)
                for d in (t_dir, t_par_dir):
                    os.makedirs(d, exist_ok=True)

                np.save(os.path.join(t_par_dir, '%d_extrinsic.npy' % int(_cam_i + 2)), _extr)
                np.save(os.path.join(t_par_dir, '%d_intrinsic.npy' % int(_cam_i + 2)), _tmp)

                file_name = os.path.join(ori_dir, '%s_%s.jpg' % (t, _cam))
                img = cv2.imread(file_name)
                dst = cv2.undistort(img, _in_mat, _dist, None)

                ######## scene specific ###########
                img_tmp = dst[(_move_t):(_w + _move_t), :, :]
                ######## scene specific ###########

                img_out = cv2.resize(img_tmp, (1024, 1024))
                out_path = os.path.join(t_dir, '%d.png' % int(_cam_i + 2))
                cv2.imwrite(out_path, img_out.astype(np.uint8))

            with ThreadPoolExecutor(max_workers=arg.num_workers) as ex:
                futures = [ex.submit(_process_one_t, t) for t in used_time_id_list]
                for _ in tqdm(as_completed(futures), total=len(futures),
                              desc=f'{scene_n}/{cam}'):
                    pass
