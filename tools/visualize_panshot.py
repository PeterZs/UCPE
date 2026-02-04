import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from src.dataset import PanShotDataset
from .visualize_re10k import CameraPoseVisualizer
import imageio


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/UCPE', help='path to the dataset root folder')
    parser.add_argument('--out_path', required=True, help='path to save the visualization results')
    parser.add_argument('--hw_ratio', default=2/3, type=float, help='the height over width of the film plane')
    parser.add_argument('--sample_stride', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=81)
    parser.add_argument('--base_xval', type=float, default=0.25)
    parser.add_argument('--zval', type=float, default=0.5)
    parser.add_argument('--use_exact_fx', action='store_true')
    parser.add_argument('--zero_first_yaw', action='store_true')
    parser.add_argument('--x_min', type=float, default=-2)
    parser.add_argument('--x_max', type=float, default=2)
    parser.add_argument('--y_min', type=float, default=-2)
    parser.add_argument('--y_max', type=float, default=2)
    parser.add_argument('--z_min', type=float, default=-2)
    parser.add_argument('--z_max', type=float, default=2)
    parser.add_argument('--animate_camera', action='store_true')
    parser.add_argument('--fps', type=int, default=16)
    parser.add_argument('--keyframe_interval', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.out_path, exist_ok=True)

    dataset = PanShotDataset(args, 'test', load_keys=['pose'])

    for data in tqdm(dataset, desc='Visualizing camera poses'):
        pose = data['pose']  # (N, 3, 4)
        video_id = data['video_id']
        c2ws = pose[::args.sample_stride]
        transform_matrix = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).reshape(4, 4)
        last_row = np.zeros((1, 4))
        last_row[0, -1] = 1.0
        c2ws = np.concatenate([c2ws, last_row.repeat(c2ws.shape[0], axis=0).reshape(-1, 1, 4)], axis=1)  # (N, 4, 4)
        c2ws = transform_matrix[None, ...] @ c2ws  # (N, 4, 4)

        visualizer = CameraPoseVisualizer([args.x_min, args.x_max], [args.y_min, args.y_max], [args.z_min, args.z_max])
        if args.animate_camera:
            out_file_path = os.path.join(args.out_path, f'{video_id}.mp4')
            visualizer.anim_pose(
                c2ws,
                out_file_path,
                args.hw_ratio,
                args.base_xval,
                args.zval,
                args.fps,
                args.keyframe_interval,
            )
        else:
            out_file_path = os.path.join(args.out_path, f'{video_id}.png')
            visualizer.vis_pose(
                c2ws,
                out_file_path,
                args.hw_ratio,
                args.base_xval,
                args.zval
            )
