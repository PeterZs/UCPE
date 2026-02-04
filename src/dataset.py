import torch
from torch.utils.data import Dataset
from einops import rearrange, repeat
from pathlib import Path
import jsonlines
import json
import numpy as np
import math
from tqdm.auto import tqdm
from src import camera_control as ucpe


class PanShotDataset(Dataset):
    def __init__(self, args, split, load_keys=["video"], video_ids=None, skip_cached=False, result_root=None):
        self.args = args
        self.data_root = Path(args.data_root)
        self.load_keys = load_keys
        self.split = split  # "train" or "test"
        
        near_depth_file = self.data_root / "PanFlow" / f"near_plane_depth-{split}.jsonl"
        near_depths = {}
        with jsonlines.open(near_depth_file) as reader:
            for obj in reader:
                video_id = obj["video_id"]
                clip_id = obj["clip_id"]
                near_depths[f"{video_id}-{clip_id}"] = obj["near_depth"]
        print(f"Loaded {len(near_depths)} near depth entries.")

        meta_path = self.data_root / "PanShot" / f"meta-{split}"
        meta_files = list(meta_path.glob("*.json"))
        metas = {}
        for meta_file in meta_files:
            with open(meta_file, "r") as f:
                matches = json.load(f)
            for match in matches:
                for video in match["videos"]:
                    meta = {
                        "pose_id": video["pose"],
                        "x_fov": float(video["x_fov"]),
                        "xi": video["xi"],
                        "near_depth": near_depths[meta_file.stem],
                    }
                    # estimate y_fov by 16:9 aspect ratio
                    fx = ucpe.compute_fx_from_fov_xi(meta["x_fov"], meta["xi"], 16)
                    y_fov = ucpe.compute_fov_from_fx_xi(fx, meta["xi"], 9)
                    meta["y_fov"] = float(y_fov)

                    metas[video["video"]] = meta
                    
        print(f"Loaded {len(metas)} video metas.")

        self.metas = []
        caption_file = self.data_root / "PanShot" / f"captioned-{split}.jsonl"
        with jsonlines.open(caption_file) as reader:
            for obj in reader:
                if obj["video"] not in metas:
                    continue
                meta = metas[obj["video"]]
                meta["caption"] = obj["caption"]
                meta["video_id"] = obj["video"]
                self.metas.append(meta)
        print(f"Loaded {len(self.metas)} captioned videos.")

        if "model_id" in args:
            self.cache_prefix = f"cache-{args.model_id.split('/')[-1]}"
            self.cache_folder = self.data_root / "PanShot" / f"{self.cache_prefix}-{split}"
            cache_names = set(c.stem for c in self.cache_folder.glob("*.pth"))
            print(f"Found {len(cache_names)} cached videos.")
            if skip_cached:
                self.metas = [m for m in self.metas if m["video_id"] not in cache_names]
                print(f"Skipped cached, {len(self.metas)} videos remaining.")
            elif "cache" in self.load_keys:
                self.metas = [m for m in self.metas if m["video_id"] in cache_names]
                print(f"Only use cached, {len(self.metas)} videos remaining.")

        if video_ids is not None:
            self.metas = [meta for meta in self.metas if meta["video_id"] in video_ids]
            print(f"Filtered by video_ids, {len(self.metas)} videos remaining.")

        self.result_root = None
        if result_root is not None:
            self.result_root = Path(result_root)
            video_ids = set(v.stem for v in self.result_root.glob("*.mp4"))
            self.metas = [m for m in self.metas if m["video_id"] in video_ids]
            print(f"Filtered by result_root, {len(self.metas)} videos remaining.")

    def __len__(self):
        return max(len(self.metas), 1)

    def __getitem__(self, idx):
        data = self.metas[idx].copy()

        video_id = data["video_id"]
        video_path = self.data_root / "PanShot" / f"videos-{self.split}" / f"{video_id}.mp4"
        data["video_path"] = str(video_path)
        data["result_path"] = data["video_path"] if self.result_root is None else str(self.result_root / f"{video_id}.mp4")

        if "image_path" in self.load_keys:
            image_path = self.data_root / "PanShot" / f"images-{self.split}" / f"{video_id}.png"
            data["image_path"] = str(image_path)
            if not image_path.exists():
                from decord import VideoReader, cpu
                from PIL import Image

                vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
                first_frame = vr[0].asnumpy()
                first_frame = Image.fromarray(first_frame)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                first_frame.save(image_path)

        for key, path in [(k, data[f"{k}_path"]) for k in ["video", "result"] if k in self.load_keys]:
            try:
                from decord import VideoReader, cpu
                vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
                video = vr.get_batch(range(len(vr))).asnumpy()
            except Exception as e:
                alt_idx = (idx + 32) % len(self)
                print(f"Error reading video {path}: {e}")
                print(f"Use video {alt_idx} instead.")
                return PanShotDataset.__getitem__(self, alt_idx)
            video = video.astype(np.float32)
            video = video / 255.0 * 2 - 1  # to [-1, 1]
            video = rearrange(video, "T H W C -> C T H W")
            data[key] = video
            data["fps"] = float(vr.get_avg_fps())

        if "input_image" in self.load_keys:
            data["input_image"] = video[:, 0]
        
        if "cache" in self.load_keys:
            cache_path = self.cache_folder / f"{data['video_id']}.pth"
            data |= torch.load(cache_path, map_location="cpu")

        if "pose" in self.load_keys:
            pose_file = self.data_root / "PanShot" / f"pose-{self.split}" / data["pose_id"]
            pose_file = pose_file.with_suffix(".npy")
            pose = np.load(pose_file)
            pose[..., 3] /= data["near_depth"]

            if getattr(self.args, "zero_first_yaw", True):
                # Rotate all cameras around world-y
                # to move forward-z of the first camera to y-z plane
                forward = pose[0, :, 2]  # (3,) the z-axis of the first camera in world
                forward_xy = np.array([forward[0], 0, forward[2]])  # project to x-z plane
                forward_xy /= np.linalg.norm(forward_xy) + 1e-8

                # compute rotation angle theta = atan2(x, z)
                theta = np.arctan2(forward_xy[0], forward_xy[2])

                # rotation matrix around world-y by -theta
                c, s = np.cos(-theta), np.sin(-theta)
                R_y = np.array([[c, 0, s],
                                [0, 1, 0],
                                [-s, 0, c]], dtype=pose.dtype)

                # apply rotation to all camera extrinsics
                pose[..., :3] = (R_y[None] @ pose[..., :3])
                pose[..., 3] = (R_y[None] @ pose[..., 3:4]).squeeze(-1)
            else:
                last_row = repeat(np.array([0,0,0,1], dtype=pose.dtype), "n -> t 1 n", t=pose.shape[0])
                c2w = np.concatenate([pose, last_row], axis=-2)  # (T, 4, 4)
                w2c0= np.linalg.inv(c2w[0])  # (4, 4)
                c2w = w2c0[None] @ c2w  # (T, 4, 4)
                pose = c2w[:, :3]  # (T, 3, 4)

            data["pose"] = pose
                
        return data


class Re10kDataset(Dataset):
    def __init__(self, args, split, load_keys=["pose"], video_ids=None, result_root=None):
        self.args = args
        self.data_root = Path(args.data_root)
        self.load_keys = load_keys
        self.split = split  # "train" or "test"
        self.normalize_traj = getattr(args, "normalize_traj", None)

        self.pose_path = self.data_root / "pose_files" / split

        caption_file = self.data_root / "captions" / f"{split}.json"
        with open(caption_file, "r") as f:
            captions = json.load(f)
        print(f"Loaded {len(captions)} captions.")

        self.metas = []
        for video_name, caption in captions.items():
            video_id = Path(video_name).stem
            pose_file = self.pose_path / f"{video_id}.txt"
            if not pose_file.exists():
                continue
            self.metas.append({
                "video_id": video_id,
                "caption": caption[0],
            })
        print(f"Total {len(self.metas)} videos with poses.")

        self.result_root = None
        filter_file = Path(args.data_root) / "filter_files" / f"filter_{split}_{args.num_frames}.txt"
        if not filter_file.exists():
            self.filter_frames(filter_file)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        with open(filter_file, "r") as f:
            filtered_ids = set(ln.strip() for ln in f.readlines() if ln.strip())
        self.metas = [m for m in self.metas if m["video_id"] in filtered_ids]
        print(f"After filtering, {len(self.metas)} videos remaining.")

        if video_ids is not None:
            self.metas = [meta for meta in self.metas if meta["video_id"] in video_ids]
            print(f"Filtered by video_ids, {len(self.metas)} videos remaining.")

        if result_root is not None:
            self.result_root = Path(result_root)
            video_ids = set(v.stem for v in self.result_root.glob("*.mp4"))
            self.metas = [m for m in self.metas if m["video_id"] in video_ids]
            print(f"Filtered by result_root, {len(self.metas)} videos remaining.")

    def __len__(self):
        return len(self.metas)

    def filter_frames(self, filter_file):
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        Path(filter_file).parent.mkdir(parents=True, exist_ok=True)
        with open(filter_file, "w") as f:
            for data in tqdm(self, desc=f"Filtering {self.split} set"):
                video_id = data["video_id"]
                if len(data["pose"]) >= self.args.num_frames:
                    f.write(f"{video_id}\n")
        print(f"Filter file saved to {filter_file}.")

    def __getitem__(self, idx):
        data = self.metas[idx].copy()
        pose_file = self.pose_path / f"{data['video_id']}.txt"

        if self.result_root is not None:
            video_id = data["video_id"]
            data["result_path"] = str(self.result_root / f"{video_id}.mp4")

        if "result" in self.load_keys:
            path = data["result_path"]
            try:
                from decord import VideoReader, cpu
                vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
                video = vr.get_batch(range(len(vr))).asnumpy()
            except Exception as e:
                alt_idx = (idx + 32) % len(self)
                print(f"Error reading video {path}: {e}")
                print(f"Use video {alt_idx} instead.")
                return PanShotDataset.__getitem__(self, alt_idx)
            video = video.astype(np.float32)
            video = video / 255.0 * 2 - 1  # to [-1, 1]
            video = rearrange(video, "T H W C -> C T H W")
            data["result"] = video
            data["fps"] = float(vr.get_avg_fps())

        if "pose" in self.load_keys:
            num_frames = self.args.num_frames
        else:
            num_frames = 1
        with open(pose_file, "r") as f:
            # lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.startswith("http")]
            lines = []
            for line in f:
                line = line.strip()
                if line and not line.startswith("http"):
                    lines.append(line)
                    if len(lines) >= num_frames:
                        break

        if "pose" in self.load_keys:
            poses = []
            for line in lines:
                parts = line.split()
                pose = np.array(list(map(float, parts[7:]))).reshape(3, 4)
                poses.append(pose)
            poses = np.stack(poses, axis=0)  # [T, 3, 4]
            last_row = repeat(np.array([0,0,0,1], dtype=poses.dtype), "n -> t 1 n", t=poses.shape[0])
            w2c = np.concatenate([poses, last_row], axis=-2)  # (T, 4, 4)
            c2w = np.linalg.inv(w2c)  # (T, 4, 4)
            w2c0= np.linalg.inv(c2w[0])  # (4, 4)
            c2w = w2c0[None] @ c2w  # (T, 4, 4)
            poses = c2w[:, :3]  # (T, 3, 4)
            if self.normalize_traj is not None:
                poses[..., 3] *= self.normalize_traj
            data["pose"] = poses

        fx = float(lines[0].split()[1])
        fy = float(lines[0].split()[2])
        x_fov = float(2 * math.atan(0.5 / fx) * 180 / math.pi)
        y_fov = float(2 * math.atan(0.5 / fy) * 180 / math.pi)
        overwrite_xfov = getattr(self.args, "overwrite_xfov", None)
        data["x_fov"] = x_fov if overwrite_xfov is None else overwrite_xfov
        data["y_fov"] = y_fov
        data["xi"] = 0.0  # pinhole camera

        return data


class DemoDataset(Dataset):
    def __init__(self, args, split=None, load_keys=["pose"], video_ids=None, result_root=None):
        self.args = args
        self.panshot_data_root = Path(args.panshot_data_root)
        self.load_keys = load_keys
        with open(args.input_file, "r") as f:
            self.metas = json.load(f)
        self.normalize_traj = getattr(args, "re10k_normalize_traj", None)

        near_depth_file = self.panshot_data_root / "PanFlow" / "near_plane_depth-test.jsonl"
        near_depths = {}
        with jsonlines.open(near_depth_file) as reader:
            for obj in reader:
                video_id = obj["video_id"]
                clip_id = obj["clip_id"]
                near_depths[f"{video_id}-{clip_id}"] = obj["near_depth"]
        print(f"Loaded {len(near_depths)} near depth entries.")
        for meta in self.metas:
            pose_file = Path(meta["pose_path"])
            if pose_file.suffix == ".npy":
                clip_name = pose_file.stem.rsplit("-", 2)[0]
                meta["near_depth"] = near_depths[clip_name]

        for idx, data in enumerate(self.metas):
            pose_file = Path(data["pose_path"])
            prefix = f"{idx}-{pose_file.stem}-fov{int(data['x_fov'])}-xi{data['xi']:.2f}-"
            data["video_id"] = prefix + data["caption"][:50].replace(" ", "_")

        if video_ids is not None:
            self.metas = [meta for meta in self.metas if meta["video_id"] in video_ids]
            print(f"Filtered by video_ids, {len(self.metas)} videos remaining.")

        self.result_root = None
        if result_root is not None:
            self.result_root = Path(result_root)
            new_metas = []
            metas = {m["video_id"]: m for m in self.metas}
            results = list(self.result_root.glob("*.mp4"))
            results.sort()
            for v in results:
                video_id = v.stem.rsplit("-", 1)[0]
                if video_id in metas:
                    meta = metas[video_id].copy()
                    meta["result_path"] = str(v)
                    new_metas.append(meta)
            self.metas = new_metas
            print(f"Filtered by result_root, {len(self.metas)} videos remaining.")

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        data = self.metas[idx].copy()

        if "result" in self.load_keys:
            path = data["result_path"]
            try:
                from decord import VideoReader, cpu
                vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
                video = vr.get_batch(range(len(vr))).asnumpy()
            except Exception as e:
                alt_idx = (idx + 32) % len(self)
                print(f"Error reading video {path}: {e}")
                print(f"Use video {alt_idx} instead.")
                return PanShotDataset.__getitem__(self, alt_idx)
            video = video.astype(np.float32)
            video = video / 255.0 * 2 - 1  # to [-1, 1]
            video = rearrange(video, "T H W C -> C T H W")
            data["result"] = video
            data["fps"] = float(vr.get_avg_fps())

        if "pose" in self.load_keys:
            pose_file = Path(data["pose_path"])
            if pose_file.suffix == ".txt":
                with open(pose_file, "r") as f:
                    lines = []
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("http"):
                            lines.append(line)
                            if len(lines) >= self.args.num_frames:
                                break
                poses = []
                for line in lines:
                    parts = line.split()
                    pose = np.array(list(map(float, parts[7:]))).reshape(3, 4)
                    poses.append(pose)
                poses = np.stack(poses, axis=0)  # [T, 3, 4]
                last_row = repeat(np.array([0,0,0,1], dtype=poses.dtype), "n -> t 1 n", t=poses.shape[0])
                w2c = np.concatenate([poses, last_row], axis=-2)  # (T, 4, 4)
                c2w = np.linalg.inv(w2c)  # (T, 4, 4)
                w2c0= np.linalg.inv(c2w[0])  # (4, 4)
                c2w = w2c0[None] @ c2w  # (T, 4, 4)
                poses = c2w[:, :3]  # (T, 3, 4)
                if self.normalize_traj is not None:
                    poses[..., 3] *= self.normalize_traj
                data["pose"] = poses
            elif pose_file.suffix == ".npy":
                pose = np.load(pose_file)
                pose[..., 3] /= data["near_depth"]
                    
                if getattr(self.args, "zero_first_yaw", True):
                    # Rotate all cameras around world-y
                    # to move forward-z of the first camera to y-z plane
                    forward = pose[0, :, 2]  # (3,) the z-axis of the first camera in world
                    forward_xy = np.array([forward[0], 0, forward[2]])  # project to x-z plane
                    forward_xy /= np.linalg.norm(forward_xy) + 1e-8

                    # compute rotation angle theta = atan2(x, z)
                    theta = np.arctan2(forward_xy[0], forward_xy[2])

                    # rotation matrix around world-y by -theta
                    c, s = np.cos(-theta), np.sin(-theta)
                    R_y = np.array([[c, 0, s],
                                    [0, 1, 0],
                                    [-s, 0, c]], dtype=pose.dtype)

                    # apply rotation to all camera extrinsics
                    pose[..., :3] = (R_y[None] @ pose[..., :3])
                    pose[..., 3] = (R_y[None] @ pose[..., 3:4]).squeeze(-1)
                else:
                    last_row = repeat(np.array([0,0,0,1], dtype=pose.dtype), "n -> t 1 n", t=pose.shape[0])
                    c2w = np.concatenate([pose, last_row], axis=-2)  # (T, 4, 4)
                    w2c0= np.linalg.inv(c2w[0])  # (4, 4)
                    c2w = w2c0[None] @ c2w  # (T, 4, 4)
                    pose = c2w[:, :3]  # (T, 3, 4)

                data["pose"] = pose
            else:
                raise NotImplementedError(f"Unsupported pose file: {pose_file}")

        return data
