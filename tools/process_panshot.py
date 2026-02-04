from pathlib import Path
import jsonlines
from tqdm.auto import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import ffmpeg
import json
import csv
import numpy as np
import cv2
from einops import rearrange, repeat
from visualize_pose import vis_to_html
import decord
from decord import VideoReader, cpu
from copy import deepcopy
from equilib import equi2pers
from thirdparty.PanFlow.utils.erp_utils import equilib_rotation
from scipy.spatial.transform import Rotation as R, Slerp
import shutil
import torch
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import yt_dlp
from threading import Lock
import gc


split = "train"  # "train" or "test"
panflow_root = Path("data/360-1M")
pf_pose_root = panflow_root / "slam_pose"
ucpe_root = Path("data/UCPE")
camerabench_root = ucpe_root / "CameraBench"
match_pf_root = ucpe_root / "PanFlow" / f"match_to_camerabench-{split}"
pf_clip_root = ucpe_root / "PanFlow" / f"match_clips-{split}"
pf_video_root = ucpe_root / "PanFlow" / "videos"
ps_video_root = ucpe_root / "PanShot" / f"videos-{split}"
ps_video_root.mkdir(parents=True, exist_ok=True)
ps_pose_root = ucpe_root / "PanShot" / f"pose-{split}"
ps_pose_root.mkdir(parents=True, exist_ok=True)
ps_meta_root = ucpe_root / "PanShot" / f"meta-{split}"
ps_meta_root.mkdir(parents=True, exist_ok=True)
debug_root = Path("debug/process_panshot")

random_seed = 42
num_workers = 8
target_fps = 16
target_height = 480
target_width = 832

x_fovs = [
    [90, 110],    # 典型 pinhole / 准广角
    [110, 140],   # 广角（轻微畸变）
    [140, 180],   # 常见鱼眼（GoPro / 全景相机）
    [160, 200],   # 极鱼眼 / 安防全景
]

xis = [
    [0.0, 0.0],   # pinhole
    [0.5, 0.95],   # 广角，畸变很小
    [1.05, 2.0],   # 常见鱼眼
    [1.5, 2.3],   # 极鱼眼
]
rot_augs = {
    "no_rot_aug": {
        "num": 1,
    },
    "yaw_aug": {
        "num": 1,
        "fixed_yaw": 180,
    },
    "yaw_pitch_aug": {
        "num": 1,
        "fixed_yaw": 180,
        "fixed_pitch": 80,
    },
    "linear_aug": {
        "num": 1,
        "fixed_yaw": 90,
        "fixed_pitch": 40,
        "fixed_roll": 30,
        "linear_yaw": 90,
        "linear_pitch": 40,
        "linear_roll": 30,
    },
}

visualize_ref = False
debug_aug = False
debug_aug_lower = False
debug = False
pose_only = False
overwrite = False
offline = False

clip_metas = list(pf_clip_root.glob("*.json"))
clip_metas.sort()
print(f"Found {len(clip_metas)} PanFlow clip files.")
video_metas = defaultdict(dict)
seed = random_seed
for clip_meta in clip_metas:
    with open(clip_meta, "r") as f:
        meta = json.load(f)
    video_metas[meta["video_id"]][meta["clip_id"]] = meta
    meta["matches"] = []
    meta["seed"] = seed
    seed += 1
print(f"Found {len(video_metas)} unique PanFlow videos.")

match_metas = list(match_pf_root.glob("*.json"))
match_metas.sort()
cb_poses = {}
num_matches = 0
for match_meta in tqdm(match_metas, desc="Loading match metas"):
    with open(match_meta, "r") as f:
        cb_matches = json.load(f)
    cb_video = match_meta.stem

    if cb_video not in cb_poses:
        pose_file = camerabench_root / "pose" / f"{cb_video}.npy"
        pose= np.load(pose_file)  # (T, 4, 4)
        cb_poses[cb_video] = pose

    for cb_match in cb_matches:
        matches = video_metas[cb_match['video_id']][cb_match['clip_id']]["matches"]
        matches.append({
            "match_id": len(matches),
            "cb_video": cb_video,
            "frames": cb_match["frames"],
            "R": cb_match["R"],
        })
        num_matches += 1
print(f"Found {num_matches} matches to {len(cb_poses)} CameraBench videos.")

num_rot_augs = sum(aug["num"] for aug in rot_augs.values()) + 1
estimated_clips = num_rot_augs * num_matches
print(f"Estimated total {estimated_clips} PanShot clips to be generated.")


def apply_rotation_align(rotation, R):
    """
    将旋转对齐矩阵 R 应用于相机轨迹的旋转矩阵序列。
    左乘 R: R'_cw = R * R_cw

    Args:
        rotation: (T, 3, 3)   # 待对齐的旋转序列
        R: (3, 3)             # 对齐用的旋转矩阵

    Returns:
        rotation_aligned: (T, 3, 3)
    """
    # einsum: i j, t j k -> t i k
    return np.einsum("ij,tjk->tik", R, rotation)


def make_mixed_rotation(rotation: np.ndarray,
                        yaw_end: float = 0.0,
                        pitch_end: float = 0.0,
                        roll_end: float = 0.0,
                        sample_frames: np.ndarray = None) -> np.ndarray:
    """
    混合增强: yaw 在世界坐标系, pitch/roll 在相机坐标系
    rotation: (T,3,3)   采样后的原始 R_cw
    yaw_end/pitch_end/roll_end: 增强角度(度)
    sample_frames: (T,) 原始帧索引; 如果提供, 将按首尾 idx 做连续插值后再采样
    """

    # === 确定插值时间轴 ===
    if sample_frames is not None:
        # 原始轨迹长度
        start_idx = int(sample_frames[0])
        end_idx = int(sample_frames[-1])
        full_T = end_idx - start_idx + 1
        full_times = np.linspace(0, 1, full_T)
    else:
        full_T = rotation.shape[0]
        full_times = np.linspace(0, 1, full_T)

    # ===== 1) 世界坐标系 yaw 插值 =====
    R_yaw_start = R.identity()
    R_yaw_end = R.from_euler('y', yaw_end, degrees=True)
    slerp_yaw = Slerp([0, 1], R.from_matrix([R_yaw_start.as_matrix(), R_yaw_end.as_matrix()]))
    R_yaw_full = slerp_yaw(full_times).as_matrix()  # (full_T,3,3)

    # ===== 2) 相机坐标系 pitch/roll 插值 =====
    R_pr_start = R.identity()
    R_pr_end = R.from_euler('xz', [pitch_end, roll_end], degrees=True)
    slerp_pr = Slerp([0, 1], R.from_matrix([R_pr_start.as_matrix(), R_pr_end.as_matrix()]))
    R_pr_full = slerp_pr(full_times).as_matrix()    # (full_T,3,3)

    # ===== 3) 对齐采样帧 =====
    if sample_frames is not None:
        # 通过偏移索引选出采样帧对应的旋转
        idxs = (sample_frames - sample_frames[0]).astype(int)
        R_yaw = R_yaw_full[idxs]
        R_pr = R_pr_full[idxs]
    else:
        R_yaw = R_yaw_full
        R_pr = R_pr_full

    # ===== 4) 合成 =====
    # yaw 世界系 → 左乘, pitch/roll 相机系 → 右乘
    R_out = np.einsum('tij,tjk,tkl->til', R_yaw, rotation, R_pr)
    return R_out


def download_video(video_id, output_path):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "outtmpl": str(output_path),
        "format": "bestvideo[height<=2500][height>1500]",
        "quiet": False,
        "no_warnings": True,
        "simulate": False,
        "cookiefile": "~/.config/cookies.txt",
        "print": [
            "before_dl:Format: %(format_id)s | Res: %(resolution)s | FPS: %(fps)s",
            "after_move:Size: %(filesize:.2fMB)s"
        ],
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "retries": 3,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


download_lock = Lock()

def process_one_video(video_id, video_meta):
    video_path = Path(pf_video_root) / f"{video_id}.mp4"
    if not video_path.exists() and not pose_only:
        if offline:
            tqdm.write(f"Offline mode, skipping download of {video_id}.")
            return False
        try:
            with download_lock:
                download_video(video_id, video_path)
        except Exception as e:
            tqdm.write(f"Failed to download {video_id}: {e}")
            return False

    if video_path.exists():
        for clip_meta in video_meta.values():
            if not process_one_clip(clip_meta):
                return False
    return True


def process_one_clip(clip_meta):
    video_id = clip_meta["video_id"]
    clip_id = clip_meta["clip_id"]
    pf_key = f"{video_id}-{clip_id}"
    meta_file = ps_meta_root / f"{pf_key}.json"
    if not overwrite and meta_file.exists():
        tqdm.write(f"Meta file {meta_file} exists, skipping.")
        return

    seed = clip_meta["seed"]
    np.random.seed(seed)
    random.seed(seed)

    video_file = pf_video_root / f"{video_id}.mp4"
    try:
        vr = VideoReader(str(video_file), ctx=cpu(0), num_threads=1)
    except Exception as e:
        tqdm.write(f"Failed to read video {video_file}: {e}")
        return False
    total_frames = len(vr)
    del vr

    matches = deepcopy(clip_meta["matches"])
    for match in matches:
        match_id = match["match_id"]
        cb_video = match["cb_video"]
        cb_pose = cb_poses[cb_video]
        clip_start = clip_meta["frames"][0]
        start_frame, end_frame = match["frames"]
        start_frame += clip_start
        end_frame += clip_start

        if visualize_ref:
            vis_file = debug_root / "cb_videos" / f"{pf_key}-{match_id}.mp4"
            vis_file.parent.mkdir(parents=True, exist_ok=True)
            if not vis_file.exists():
                src_file = camerabench_root / "videos" / f"{cb_video}.mp4"
                shutil.copy2(src_file, vis_file)

        if end_frame >= total_frames:
            tqdm.write(f"End frame {end_frame} > total frames {total_frames} in {video_file}, skipping.")
            continue

        num_frames_sampled = len(cb_pose)
        sample_frames = np.linspace(start_frame, end_frame, num_frames_sampled)
        sample_frames = np.round(sample_frames).astype(int)

        if not pose_only:
            vr = VideoReader(str(video_file), ctx=cpu(0), num_threads=1)
            decord.bridge.set_bridge("torch")
            erp_frames = vr.get_batch(sample_frames)  # (T, H, W, 3)
            del vr
            gc.collect()
            erp_frames = rearrange(erp_frames, "t h w c -> t c h w")

            if visualize_ref:
                out_path = debug_root / "pf_videos" / f"{pf_key}-{match_id}.mp4"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # ---- 1. 转换到 float 并降采样 ----
                erp_down = torch.nn.functional.interpolate(
                    erp_frames.float(),  # [1, T, C, H, W]
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=False
                ).clamp(0, 255).to(torch.uint8)

                # ---- 2. 改为 [T, H, W, C] ----
                frames_np = rearrange(erp_down, "t c h w -> t h w c").contiguous().numpy()

                # ---- 3. 写入视频 ----
                process = (
                    ffmpeg
                    .input(
                        'pipe:', 
                        format='rawvideo',
                        pix_fmt='rgb24',
                        s=f'{target_width}x{target_height}',
                        framerate=target_fps
                    )
                    .output(str(out_path), vcodec='libx264', pix_fmt='yuv420p', crf=18, preset='slow')
                    .overwrite_output()
                    .run_async(pipe_stdin=True, quiet=True)
                )
                process.stdin.write(frames_np.tobytes())
                process.stdin.close()
                process.wait()
                print(f"✅ saved to {out_path}")

        clip_name = f"Clip-{clip_id:03d}"
        pose_file = pf_pose_root / video_id / f"{clip_name}.npy"
        pf_pose = np.load(pose_file)  # (N, 3, 4)
        c2w = pf_pose[sample_frames - clip_start]  # (T, 3, 4)
        last_row = repeat(np.array([0,0,0,1], dtype=c2w.dtype), "n -> t 1 n", t=c2w.shape[0])
        c2w = np.concatenate([c2w, last_row], axis=-2)  # (T, 4, 4)
        w2c0 = np.linalg.inv(c2w[0])  # (4, 4)
        c2w = w2c0[None] @ c2w  # (T, 4, 4)
        pf_pose = c2w[:, :3, :]  # (T, 3, 4)

        rotation = cb_pose[:, :3, :3]  # (T, 3, 3)
        R_align = np.array(match["R"]).T
        rotation = apply_rotation_align(rotation, R_align)

        videos = []
        i_rot = 0
        for aug_key, aug_setting in rot_augs.items():
            for i_aug in range(aug_setting["num"]):
                aug_meta = deepcopy(aug_setting)
                del aug_meta["num"]

                rotation_aug = rotation.copy()
                if "fixed_yaw" in aug_setting:
                    yaw_aug = int(np.random.randint(-aug_setting["fixed_yaw"], aug_setting["fixed_yaw"]))
                    yaw_rad = float(np.deg2rad(yaw_aug))
                    aug_meta["yaw"] = yaw_aug
                    R_yaw_world = R.from_euler("y", yaw_rad, degrees=False).as_matrix()
                    rotation_aug = np.einsum("ij,tjk->tik", R_yaw_world, rotation_aug)
                if "fixed_pitch" in aug_setting:
                    pitch_aug = int(np.random.randint(-aug_setting["fixed_pitch"], aug_setting["fixed_pitch"]))
                    pitch_rad = float(np.deg2rad(pitch_aug))
                    aug_meta["pitch"] = pitch_aug
                    R_pitch_cam = R.from_euler("x", pitch_rad, degrees=False).as_matrix()
                    rotation_aug = np.einsum("tij,jk->tik", rotation_aug, R_pitch_cam)
                if "fixed_roll" in aug_setting:
                    roll_aug = int(np.random.randint(-aug_setting["fixed_roll"], aug_setting["fixed_roll"]))
                    roll_rad = float(np.deg2rad(roll_aug))
                    aug_meta["roll"] = roll_aug
                    R_roll_cam = R.from_euler("z", roll_rad, degrees=False).as_matrix()
                    rotation_aug = np.einsum("tij,jk->tik", rotation_aug, R_roll_cam)
                if any(k in aug_setting for k in ("linear_yaw", "linear_pitch", "linear_roll")):
                    yaw_range = aug_setting.get("linear_yaw", 0)
                    pitch_rang = aug_setting.get("linear_pitch", 0)
                    roll_rang = aug_setting.get("linear_roll", 0)
                    yaw_end = int(np.random.randint(-yaw_range, yaw_range)) if yaw_range > 0 else 0
                    pitch_end = int(np.random.randint(-pitch_rang, pitch_rang)) if pitch_rang > 0 else 0
                    roll_end = int(np.random.randint(-roll_rang, roll_rang)) if roll_rang > 0 else 0
                    rotation_aug = make_mixed_rotation(rotation_aug, yaw_end, pitch_end, roll_end, sample_frames)
                    aug_meta.update({"linear_yaw": yaw_end, "linear_pitch": pitch_end, "linear_roll": roll_end})
                # rotation_aug = repeat(np.eye(3), "i j -> t i j", t=rotation_aug.shape[0])  # Debug

                # generate camera pose
                ps_pose_file = Path(f"{pf_key}-{match_id}-{aug_key}_{i_aug}.npy")
                ps_pose = pf_pose.copy()
                ps_pose[..., :3] = pf_pose[..., :3] @ rotation_aug
                np.save(ps_pose_root / ps_pose_file, ps_pose)

                idx_lens = i_rot % len(x_fovs) if debug_aug else random.randint(0, len(x_fovs) - 1)
                x_fov_range = x_fovs[idx_lens]
                xi_range = xis[idx_lens]
                if debug_aug:
                    if debug_aug_lower:
                        x_fov = x_fov_range[0]
                        xi = xi_range[0]
                    else:
                        x_fov = x_fov_range[1]
                        xi = xi_range[1]
                else:
                    x_fov = int(np.round(np.random.uniform(*x_fov_range)))
                    xi = float(np.round(np.random.uniform(*xi_range), 2))

                if not pose_only:
                    rotation_aug = equilib_rotation(rotation_aug)
                    pers_frames = equi2pers(erp_frames, rotation_aug, target_height, target_width, x_fov, xi=xi)
                    pers_frames = rearrange(pers_frames, "t c h w -> t h w c")
                    pers_frames = pers_frames.numpy()

                    out_file = ps_video_root / f"{pf_key}-{match_id}-{aug_key}_{i_aug}-fov{x_fov}-xi{xi:.2f}.mp4"
                    process = (
                        ffmpeg
                        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{target_width}x{target_height}', framerate=target_fps)
                        .output(str(out_file), pix_fmt='yuv420p', vcodec='libx264', r=target_fps, crf=16, preset='slow')
                        .overwrite_output()
                        .run_async(pipe_stdin=True, quiet=True)
                    )
                    process.stdin.write(pers_frames.tobytes())
                    del pers_frames
                    process.stdin.close()
                    process.wait()

                    videos.append({
                        "video": str(out_file.stem),
                        "pose": str(ps_pose_file.stem),
                        "x_fov": x_fov,
                        "xi": xi,
                        "rot_aug": aug_meta,
                    })

                gc.collect()
                torch.cuda.empty_cache()

                i_rot += 1

        match["videos"] = videos

    if not pose_only:
        with open(meta_file, "w") as f:
            json.dump(matches, f, indent=4)

    return True

success = []
if debug:
    # === 单线程调试模式 ===
    for video_id, video_meta in tqdm(
        video_metas.items(),
        desc="Processing clips (debug single-thread)"
    ):
        success.append(process_one_video(video_id, video_meta))
else:
    # === 正常并行模式 ===
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(process_one_video, video_id, video_meta)
            for video_id, video_meta in video_metas.items()
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing clips"):
            try:
                success.append(fut.result())
            except Exception as e:
                print("Error:", e)

print(f"All done. {sum(success)}/{len(success)} videos processed successfully.")