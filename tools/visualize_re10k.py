import random
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from tqdm.auto import tqdm
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg


class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(7, 7), dpi=300)
        self.ax = self.fig.add_subplot(projection='3d')
        self.plotly_data = None  # plotly data traces
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.init_ax()
        print('initialize camera pose visualizer')

    def init_ax(self):
        self.ax.cla()
        self.ax.set_aspect("auto")
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_zlim(self.zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

    def extrinsic2pyramid(self, extrinsic, color_map='red', hw_ratio=9/16, base_xval=1, zval=3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [base_xval, -base_xval * hw_ratio, zval, 1],
                               [base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, -base_xval * hw_ratio, zval, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]

        color = color_map if isinstance(color_map, str) else plt.cm.rainbow(color_map)

        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax, orientation='vertical', label='Frame Number')

    def show(self, out_file_path):
        plt.title('Extrinsic Parameters')
        os.makedirs('debug/visualize_re10k', exist_ok=True)
        plt.savefig(out_file_path, format='png', dpi=600)
        plt.show()

    def draw(self):
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf, dtype=np.uint8)
        return img[:, :, :3]

    def vis_pose(self, c2ws, out_file_path, hw_ratio, base_xval, zval):
        num_frames = len(c2ws)
        self.colorbar(num_frames)
        self.init_ax()

        for frame_idx, c2w in enumerate(c2ws):
            self.extrinsic2pyramid(
                c2w,
                frame_idx / num_frames,
                hw_ratio=hw_ratio,
                base_xval=base_xval,
                zval=zval
            )

        self.show(out_file_path)

    def anim_pose(self, c2ws, out_file_path, hw_ratio, base_xval, zval, fps, keyframe_interval):
        num_frames = len(c2ws)
        self.colorbar(num_frames)
        writer = imageio.get_writer(out_file_path, fps=fps)

        keyframes = []
        for frame_idx, c2w in enumerate(c2ws):
            self.init_ax()
            if keyframes:
                for kf in keyframes:
                    self.extrinsic2pyramid(
                        c2ws[kf],
                        kf / num_frames,
                        hw_ratio=hw_ratio,
                        base_xval=base_xval,
                        zval=zval
                    )
            if frame_idx % keyframe_interval == 0:
                keyframes.append(frame_idx)
            self.extrinsic2pyramid(
                c2w,
                frame_idx / num_frames,
                hw_ratio=hw_ratio,
                base_xval=base_xval,
                zval=zval
            )
            img = self.draw()
            writer.append_data(img)

        writer.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_file_path', required=True, help='path to the trajectory txt file')
    parser.add_argument('--filter_file', required=True, help='path to the filter txt file')
    parser.add_argument('--out_path', required=True, help='path to save the visualization results')
    parser.add_argument('--num_videos', type=int, default=150, help='number of videos to visualize')
    parser.add_argument('--hw_ratio', default=2/3, type=float, help='the height over width of the film plane')
    parser.add_argument('--sample_stride', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=81)
    parser.add_argument('--all_frames', action='store_true')
    parser.add_argument('--base_xval', type=float, default=0.25)
    parser.add_argument('--zval', type=float, default=0.5)
    parser.add_argument('--use_exact_fx', action='store_true')
    parser.add_argument('--relative_c2w', action='store_true')
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


def get_c2w(w2cs, transform_matrix, relative_c2w):
    if relative_c2w:
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ np.linalg.inv(w2c) for w2c in w2cs[1:]]
    else:
        ret_poses = [np.linalg.inv(w2c) for w2c in w2cs]
    ret_poses = [transform_matrix @ x for x in ret_poses]
    return np.array(ret_poses, dtype=np.float32)


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.out_path, exist_ok=True)
    with open(args.filter_file, 'r') as f:
        video_ids = f.read().splitlines()
    video_ids = video_ids[:args.num_videos]
    for video_id in tqdm(video_ids, desc='Visualizing camera poses'):
        pose_file_path = os.path.join(args.pose_file_path, f'{video_id}.txt')
        if not os.path.exists(pose_file_path):
            print(f'Pose file {pose_file_path} does not exist, skip.')
            continue
        print(f'Visualizing {pose_file_path}...')

        with open(pose_file_path, 'r') as f:
            poses = f.readlines()
        w2cs = [np.asarray([float(p) for p in pose.strip().split(' ')[7:]]).reshape(3, 4) for pose in poses[1:]]
        fxs = [float(pose.strip().split(' ')[1]) for pose in poses[1:]]
        if args.all_frames:
            args.num_frames = len(fxs)
            args.sample_stride = 1
        cropped_length = args.num_frames * args.sample_stride
        total_frames = len(w2cs)
        start_frame_ind = 0
        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)
        frame_ind = np.linspace(start_frame_ind, end_frame_ind - 1, args.num_frames, dtype=int)
        w2cs = [w2cs[x] for x in frame_ind]
        transform_matrix = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).reshape(4, 4)
        last_row = np.zeros((1, 4))
        last_row[0, -1] = 1.0
        w2cs = [np.concatenate((w2c, last_row), axis=0) for w2c in w2cs]
        c2ws = get_c2w(w2cs, transform_matrix, args.relative_c2w)

        visualizer = CameraPoseVisualizer([args.x_min, args.x_max], [args.y_min, args.y_max], [args.z_min, args.z_max])
        zval = fxs[0] if args.use_exact_fx else args.zval
        if args.animate_camera:
            out_file_path = os.path.join(args.out_path, f'{video_id}.mp4')
            visualizer.anim_pose(
                c2ws,
                out_file_path,
                args.hw_ratio,
                args.base_xval,
                zval,
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
                zval
            )
