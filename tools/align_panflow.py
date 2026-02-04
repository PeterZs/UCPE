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
from decord import VideoReader, cpu
from copy import deepcopy


panflow_root = Path("data/360-1M")
panshot_root = Path("data/UCPE")
debug_root = Path("debug/align_panflow")
debug_root.mkdir(parents=True, exist_ok=True)

filter_root = panshot_root / "PanFlow" / "filtered"
filter_clips_thres = 0.5  # reject if >50% clips are filtered

score_jsonl = panshot_root / "PanFlow" / "scores.jsonl"
qalign_keys = ["image_aesthetic", "image_quality", "video_quality"]
max_clips_per_video = 10
max_clips_per_poi = 5

split = "train"  # "train" or "test"
camerabench_root = panshot_root / "CameraBench"
output_root = panshot_root / "PanFlow" / f"align_to_camerabench-{split}"
output_root.mkdir(parents=True, exist_ok=True)
summary_root = output_root.parent / f"{output_root.name}-summary"
summary_root.mkdir(parents=True, exist_ok=True)
static_words = ["static", "fixed"]
rotation_score_thres = 5.0  # degrees, reject if rotation_score < threshold
rotating_clips_thres = 0.5  # reject if >50% clips have large rotation

top_k = 30 if split == "train" else 10  # how many matches to keep per 360-1M clip
target_fps = 16
match_step = 5  # sweep step in frames

visualize_gravity = False
visualize_pose = False
visualize_motion = False
visualize_rotation = False
visualize_fps = 1.

cb_geocalib_file = camerabench_root / "geocalib.jsonl"
with jsonlines.open(cb_geocalib_file, "r") as reader:
    cb_geocalib = {obj["video"]: obj for obj in reader}

# 读取 CameraBench 的位姿数据
cb_meta_file = camerabench_root / f"processed_{split}.jsonl"
with jsonlines.open(cb_meta_file, "r") as reader:
    cb_meta_all = list(reader)
cb_R_gravity = []
cb_poses = []
cb_meta = []
for obj in tqdm(cb_meta_all, desc="Loading CameraBench poses"):
    obj["video"] = Path(obj["path"]).stem
    video_id = obj["video"]
    camera_caption = obj["caption"].lower()
    if any(word in camera_caption for word in static_words):
        tqdm.write(f"Skipping static camera {video_id} ({camera_caption})")
        continue

    pose_file = camerabench_root / "vipe" / "pose" / f"{video_id}.npz"
    if not pose_file.exists():
        tqdm.write(f"Pose file not found: {pose_file}, skipping.")
        continue

    cb_meta.append(obj)
    pose = np.load(pose_file)["data"]  # (T, 4, 4)
    cb_poses.append(pose)
    cb_R_gravity.append(cb_geocalib[video_id]["R"])

print(f"Loaded {len(cb_poses)} / {len(cb_meta_all)} CameraBench poses.")
cb_poses = np.array(cb_poses)  # (N, T, 4, 4)
cb_R_gravity = np.array(cb_R_gravity)        # (N, 3, 3)

cb_w2c0 = np.linalg.inv(cb_poses[:, 0])        # (N, 4, 4) batched inv
cb_poses_origin = cb_w2c0[:, None] @ cb_poses       # (N, T, 4, 4)  第一帧在原点

# rotate cb_poses based on cb_R to world align with gravity direction
cb_T_gravity = repeat(np.eye(4), 'h w -> n h w', n=cb_R_gravity.shape[0])  # (N,4,4)
cb_T_gravity[:, :3, :3] = cb_R_gravity
cb_poses_gravity = cb_T_gravity[:, None, :, :] @ cb_poses_origin

# save cb_poses_gravity
cb_pose_root = camerabench_root / "pose"
cb_pose_root.mkdir(parents=True, exist_ok=True)
for i, obj in enumerate(cb_meta):
    cb_pose_dir = cb_pose_root / f"{obj['video']}.npy"
    np.save(cb_pose_dir, cb_poses_gravity[i])

cb_pos = cb_poses_gravity[:, :, :3, 3]               # (N, T, 3)
target_frames = cb_pos.shape[1]

if visualize_gravity:
    for i, obj in enumerate(cb_meta[:3]):
        video_id = obj["video"]
        cb_pose_dir = debug_root / "gravity" / video_id
        cb_pose_dir.mkdir(parents=True, exist_ok=True)
        origin_pose_file = cb_pose_dir / "origin.npy"
        np.save(origin_pose_file, cb_poses_gravity[i])
        gravity_pose_file = cb_pose_dir / "gravity_aligned.npy"
        np.save(gravity_pose_file, cb_poses_origin[i])
        combined_file = cb_pose_dir / "comparison.npy"
        combined_pose = np.concatenate([cb_poses_gravity[i], cb_poses_origin[i]], axis=0)
        np.save(combined_file, combined_pose)
        # vis_to_html(cb_pose_dir, [origin_pose_file, gravity_pose_file])
        vis_to_html(cb_pose_dir, [combined_file])

# 读取 PanFlow 的质量分数
with jsonlines.open(score_jsonl, "r") as reader:
    qalign_scores = {
        (obj["video_id"], obj["clip_id"]): {k: obj[k] for k in qalign_keys}
        for obj in reader
    }


def get_traj_align(A, B, allow_scale=True, eps=1e-12):
    """
    计算将轨迹 B 对齐到轨迹 A 的相似变换 (R, s)，
    但 R 被约束为仅绕世界 y 轴的旋转（偏航）。
    假设所有轨迹的第一帧已在原点。
    A: (N1, T, 3)
    B: (N2, T, 3)
    Returns:
        R: (N1, N2, 3, 3)  使得 A ≈ s * R * B
        s: (N1, N2)
    """
    N1, T, _ = A.shape
    N2 = B.shape[0]

    # 扩维到配对形状 (N1, N2, T, 3)
    A_rel = A[:, None, :, :]          # (N1, N2, T, 3)
    B_rel = B[None, :, :, :]          # (N1, N2, T, 3)

    # 仅取 x,z 分量：[..., 0] 为 x，[..., 2] 为 z
    Ax = A_rel[..., 0]                # (N1, N2, T)
    Az = A_rel[..., 2]
    Bx = B_rel[..., 0]
    Bz = B_rel[..., 2]

    # 计算 H_xz 的四个元素（按时间平均）
    h11 = np.einsum("nmt,nmt->nm", Ax, Bx) / T
    h12 = np.einsum("nmt,nmt->nm", Ax, Bz) / T
    h21 = np.einsum("nmt,nmt->nm", Az, Bx) / T
    h22 = np.einsum("nmt,nmt->nm", Az, Bz) / T

    # 最优偏航角 theta（只绕 y 轴）
    theta = np.arctan2(h12 - h21, h11 + h22)    # (N1, N2)

    c = np.cos(theta)
    s_th = np.sin(theta)

    # 组装 3x3 的绕 y 轴旋转矩阵
    R = np.zeros((N1, N2, 3, 3), dtype=A.dtype)
    R[..., 0, 0] = c
    R[..., 0, 2] = s_th
    R[..., 1, 1] = 1.0
    R[..., 2, 0] = -s_th
    R[..., 2, 2] = c

    # 尺度：只用 xz 平面的能量（与 yaw-only 一致）
    if allow_scale:
        var_B_xz = (np.sum(Bx**2 + Bz**2, axis=2) / T) + eps  # (N1, N2) 通过 broadcast
        # tr(R2D*H_xz) = c*(h11+h22) + s*(h21 - h12)
        numer = c * (h11 + h22) + s_th * (h21 - h12)
        s = numer / var_B_xz
    else:
        s = np.ones((N1, N2), dtype=A.dtype)

    return R, s


def apply_traj_align(B, R, s):
    """
    应用 (R, s) 到轨迹 B，使其对齐到 A。
    B: (N2, T, 3)
    R: (N1, N2, 3, 3)
    s: (N1, N2)
    return: (N1, N2, T, 3)
    """
    # (N1,N2,3,3) @ (N2,T,3) -> (N1,N2,T,3)
    rotated = np.einsum("nmij,mtj->nmti", R, B)
    return s[..., None, None] * rotated


def apply_pose_align(c2w, R, s):
    """
    应用 (R, s) 到 c2w 外参，左乘 R，平移乘 s：
    R'_cw = R R_cw,  t'_cw = s R t_cw
    c2w: (N2,T,4,4) 或 (T,4,4)
    R: (N1,N2,3,3), s: (N1,N2)
    return: (N1,N2,T,4,4)
    """
    if c2w.ndim == 3:  # (T,4,4) -> (1,T,4,4)
        c2w = c2w[None, ...]
    N2, T, _, _ = c2w.shape

    Rc2w = c2w[..., :3, :3]     # (N2,T,3,3)
    tc2w = c2w[..., :3, 3]      # (N2,T,3)

    Rc2w_new = np.einsum("nmij,mtjk->nmtik", R, Rc2w)          # (N1,N2,T,3,3)
    tc2w_new = s[..., None, None] * np.einsum("nmij,mtj->nmti", R, tc2w)  # (N1,N2,T,3)

    c2w_aligned = np.zeros((R.shape[0], N2, T, 4, 4), dtype=c2w.dtype)
    c2w_aligned[..., :3, :3] = Rc2w_new
    c2w_aligned[..., :3, 3]  = tc2w_new
    c2w_aligned[..., 3, 3]   = 1.0
    return c2w_aligned


def compute_rmse(A, B_aligned):
    """
    计算对齐后轨迹之间的 RMSE (Root Mean Square Error)。

    Args:
        A: (N1, T, 3)           # 目标轨迹
        B_aligned: (N1, N2, T, 3)  # 已对齐到 A 的轨迹

    Returns:
        rmse: (N1, N2)          # 每对 (A_i, B_j) 的误差
    """
    if A.ndim == 2:  # 单条轨迹
        A = A[None, ...]

    # 误差 (N1,N2,T)
    diff = A[:, None, :, :] - B_aligned
    sqerr = np.sum(diff**2, axis=-1)  # (N1,N2,T)
    mse = np.mean(sqerr, axis=-1)     # (N1,N2)
    rmse = np.sqrt(mse)

    return rmse


def traj_length(traj):
    """
    traj: (..., T, 3)
    return: (...,), 每条轨迹路径长度
    """
    diffs = traj[..., 1:] - traj[..., :-1]       # (..., T-1, 3)
    lengths = np.linalg.norm(diffs, axis=-1).sum(axis=-1)  # (...)
    return lengths


def normalize_traj(traj, eps=1e-8):
    """
    traj: (..., T, 3)
    return: (..., T, 3), 每条轨迹路径长度归一化
    """
    return traj / (traj_length(traj)[..., None, None] + eps)


def visualize_clip(video_file, frames, out_clip_file):
    vr = VideoReader(str(video_file), ctx=cpu(0), num_threads=1)
    start_frame, end_frame = frames
    sample_frames = np.arange(start_frame, end_frame, fps / visualize_fps)
    sample_frames = np.round(sample_frames).astype(int)
    clip_data = vr.get_batch(sample_frames).asnumpy()

    out_clip_file.parent.mkdir(parents=True, exist_ok=True)
    process = (
        ffmpeg
        .input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{clip_data.shape[2]}x{clip_data.shape[1]}", framerate=visualize_fps)
        .output(str(out_clip_file), pix_fmt="yuv420p", vcodec="libx264", r=visualize_fps, crf=23, preset="medium")
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )
    process.stdin.write(clip_data.tobytes())
    process.stdin.close()
    process.wait()


def max_rot_from_anchor(rot_seq, degrees=True, robust_percentile=None):
    """
    计算一个相机姿态序列中相对于首帧的最大旋转角。

    Args:
        rot_seq: (T,3,3) 单个clip的旋转矩阵序列
        degrees: True=返回角度, False=弧度
        robust_percentile: 如果指定, 用分位数而不是max

    Returns:
        float: 最大(或分位数)旋转角
    """
    T = rot_seq.shape[0]
    R0 = rot_seq[0]
    # 相对旋转: R0^T @ Rt
    R_rel = R0.T @ rot_seq              # (3,3)@(T,3,3) -> (T,3,3)
    # np.matmul 自动广播: (3,3)@(T,3,3)不可直接，需要einsum
    R_rel = np.einsum('ij,tjk->tik', R0.T, rot_seq)

    trace = np.trace(R_rel, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)        # 弧度, shape (T,)
    theta = theta[1:]                   # 去掉首帧

    if robust_percentile is None:
        val = np.max(theta)
    else:
        val = np.percentile(theta, robust_percentile)
    if degrees:
        val = np.degrees(val)
    return val


# 读取 360-1M 的位姿数据
pf_meta_root = panflow_root / "meta"
pf_meta_files = list((pf_meta_root).glob("*.json"))
pf_meta_files.sort()
print(f"Found {len(pf_meta_files)} PanFlow meta files.")
# pf_meta_files = pf_meta_files[:100]

filter_summary = defaultdict(lambda: 0)
camera_summary = defaultdict(lambda: 0)
for meta_file in tqdm(pf_meta_files, desc="Matching 360-1M poses"):
    # tqdm.write(f"Processing {meta_file}")

    with open(meta_file, "r") as f:
        pf_meta = json.load(f)
    if "slam_clips" not in pf_meta:
        tqdm.write(f"No slam_clips in {meta_file}, skipping.")
        continue

    filter_file = filter_root / meta_file.name
    if not filter_file.exists():
        tqdm.write(f"Filter file not found: {filter_file}, skipping.")
        continue
    with open(filter_file, "r") as f:
        filter_meta = json.load(f)
    if not filter_meta:
        tqdm.write(f"Empty filter file: {filter_file}, skipping.")
        continue
    reject_clips = [any(clip["filter"].values()) for clip in filter_meta]
    reject_ratio = np.mean(reject_clips)
    if reject_ratio > filter_clips_thres:
        tqdm.write(f"Reject video {meta_file} due to too many ({reject_ratio:.2%}) filtered clips.")
        filter_summary["filter"] += len(pf_meta["slam_clips"]["clips"])
        continue

    video_id = meta_file.stem
    video_file = panflow_root / "videos" / video_id
    video_file = video_file.with_suffix(".mp4")
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        tqdm.write(f"Failed to open video: {video_file}, skipping.")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)

    if "motion_score" not in pf_meta:
        tqdm.write(f"No motion_score in {meta_file}, skipping.")
        continue

    if "watermark_score" not in pf_meta:
        tqdm.write(f"No watermark_score in {meta_file}, skipping.")
        continue

    clips = deepcopy(pf_meta["slam_clips"]["clips"])
    missing_qalign = 0
    for clip, slam_pose, motion_score, watermark_score in zip(
        clips,
        pf_meta["slam_pose"]["clips"],
        pf_meta["motion_score"]["clips"],
        pf_meta["watermark_score"]["clips"],
    ):
        clip["scores"] = {}
        clip["scores"]["motion_score"] = motion_score["score"]
        clip["scores"]["watermark_score"] = watermark_score["score"]
        clip["info"] = slam_pose["info"]
        if (video_id, clip["clip_id"]) in qalign_scores:
            scores = qalign_scores[(video_id, clip["clip_id"])]
            clip["scores"] |= scores
            clip["scores"]["avg_qalign"] = float(np.mean(list(scores.values())))
        else:
            clip["scores"]["avg_qalign"] = -1
            missing_qalign += 1
    if missing_qalign > 0:
        tqdm.write(f"Warning: {missing_qalign} / {len(clips)} clips missing q-align scores in {video_id}.")
    clips.sort(key=lambda x: x["scores"]["avg_qalign"], reverse=True)
    
    ps_meta = {
        "fps": fps,
        "clips": [],
    }
    poi_counter = defaultdict(lambda: 0)
    rotation_counter = 0
    for i_clip, clip in enumerate(tqdm(clips, desc="Processing clips", leave=False)):
        if len(ps_meta["clips"]) >= max_clips_per_video:
            tqdm.write(f"Reached max clips per video ({max_clips_per_video}), stopping.")
            filter_summary["max_clips_per_video"] += len(clips) - i_clip
            break

        clip_name = clip["clip_name"]
        clip_id = clip["clip_id"]

        clip_filter = filter_meta[clip_id - 1]
        assert clip_filter["clip_id"] == clip_id
        if any(clip_filter["filter"].values()):
            reasons = [k for k, v in clip_filter["filter"].items() if v]
            # tqdm.write(f"Clip {video_id}/{clip_name} filtered due to {reasons}, skipping.")
            filter_summary["filter"] += 1
            continue

        poi_category = clip_filter["poi_category"]
        if all(poi_counter[c] >= max_clips_per_poi for c in poi_category):
            # tqdm.write(f"Clip {video_id}/{clip_name} skipped due to max clips per POI {poi_category}.")
            filter_summary["max_clips_per_poi"] += 1
            continue

        slam_info = clip["info"]
        motion_score = clip["scores"]["motion_score"]
        frames = clip["frames"]
        clip_dict = {
            "video_id": video_id,
            "clip_id": clip_id,
            "clip_name": clip_name,
            "frames": frames,
            "scores": clip["scores"].copy(),
            "poi_category": poi_category,
            "slam_info": slam_info,
        }

        num_frames = frames[-1] - frames[0] + 1
        num_frames_sampled = int(round(num_frames / fps * target_fps))
        if num_frames_sampled < target_frames:
            # tqdm.write(f"Clip {video_id}/{clip_name} too short ({num_frames_sampled} < {target_frames}), skipping.")
            filter_summary["too_short"] += 1
            continue

        if slam_info == "Small camera motion":
            ps_meta["clips"].append(clip_dict)
            if visualize_motion:
                out_clip_file = debug_root / "static_clips" / f"{motion_score:.4f}-{video_id}-{clip_name}.mp4"
                visualize_clip(video_file, frames, out_clip_file)
            filter_summary["small_camera_motion"] += 1
            continue

        if visualize_motion:
            continue  # 只保留静止片段

        if slam_info != "Success":
            # tqdm.write(f"Clip {video_id}/{clip_name} SLAM not successful ({slam_info}), skipping.")
            filter_summary["slam_fail"] += 1
            continue

        pose_file = panflow_root / "slam_pose" / video_id / clip_name
        pose_file = pose_file.with_suffix(".npy")
        if not pose_file.exists():
            tqdm.write(f"Pose file not found: {pose_file}, skipping.")
            continue
        pf_pose = np.load(pose_file)  # (T, 3, 4)

        rot_seg = pf_pose[:, :3, :3]  # (num_segs,T,3,3)
        rotation_score = max_rot_from_anchor(rot_seg, degrees=True, robust_percentile=95)  # (num_segs,)
        rotation_score = float(rotation_score)
        if visualize_rotation:
            out_clip_file = debug_root / "rotation_score" / f"{rotation_score:.4f}-{video_id}-{clip_name}.mp4"
            visualize_clip(video_file, frames, out_clip_file)
        if rotation_score > rotation_score_thres:
            # tqdm.write(f"Clip {video_id}/{clip_name} rejected due to large rotation ({rotation_score:.2f}° > {rotation_score_thres}°), skipping.")
            rotation_counter += 1
            filter_summary["large_rotation"] += 1
            continue
        clip_dict["scores"]["rotation_score"] = rotation_score

        sample_frames = np.linspace(0, num_frames - 1, num_frames_sampled)
        sample_frames = np.round(sample_frames).astype(int)

        # 1. 提取所有片段索引
        max_start = num_frames_sampled - target_frames
        num_segs = max_start // match_step + 1
        starts = np.arange(0, max_start+1, match_step)
        num_segs = len(starts)
        idx = starts[:, None] + np.arange(target_frames)[None, :]
        idx = sample_frames[idx]

        # 2. 取出 c2w
        c2w = pf_pose[idx]  # (num_segs, T, 3, 4) 或 (num_segs, T, 4, 4)
        if c2w.shape[-2:] == (3, 4):
            # 补成 4x4
            last_row = repeat(np.array([0,0,0,1], dtype=c2w.dtype), "n -> s t 1 n", s=c2w.shape[0], t=c2w.shape[1])
            c2w = np.concatenate([c2w, last_row], axis=-2)  # (num_segs,T,4,4)

        # 3. 每段归一化到第一帧
        w2c0 = np.linalg.inv(c2w[:, 0])  # (num_segs,4,4)
        c2w = w2c0[:, None] @ c2w  # (num_segs,T,4,4)

        # 4. 提取相机中心轨迹 (num_segs, T, 3)
        pos_seg = c2w[:, :, :3, 3]

        # 5. 批量对齐 + RMSE
        # pos_seg -> (num_segs,T,3)，扩展成 (num_segs,1,T,3)，与 cb_pos (N_cb,T,3) 对齐
        R, s = get_traj_align(cb_pos, pos_seg)          # R:(N_cb,num_segs,3,3), s:(N_cb,num_segs)
        pos_aligned = apply_traj_align(pos_seg, R, s)   # (N_cb,num_segs,T,3)
        rmse = compute_rmse(normalize_traj(cb_pos), normalize_traj(pos_aligned))        # (N_cb,num_segs)

        best_idx = np.argmin(rmse, axis=1)     # (N_cb,)
        best_seg = idx[best_idx]         # (N_cb, T)
        best_rmse = rmse[np.arange(len(cb_pos)), best_idx]  # (N_cb,)

        topk_cb = np.argsort(best_rmse)[:top_k]

        if visualize_pose:
            pose_aligned = apply_pose_align(c2w, R, s)  # (N_cb,num_segs,T,4,4)

        clip_dict["matches"] = []
        for i in topk_cb:
            j = best_idx[i]
            cb_name = cb_meta[i]["video"]
            if cb_meta[i].get("camera_labels", False):
                for label in cb_meta[i]["camera_labels"]:
                    camera_summary[label] += 1
            else:
                camera_summary[cb_name] += 1
            clip_dict["matches"].append({
                "video": cb_name,
                "frames": (int(best_seg[i, 0]), int(best_seg[i, -1])),
                "rmse": float(rmse[i, j]),
                "R": R[i, j].tolist(),
                "s": float(s[i, j]),
            })

            if visualize_pose:
                cb_pose_dir = debug_root / video_id / clip_name / cb_name
                cb_pose_dir.mkdir(parents=True, exist_ok=True)
                cb_pose_file = cb_pose_dir / "target.npy"
                np.save(cb_pose_file, cb_poses_gravity[i])
                pf_pose_file = cb_pose_dir / "aligned.npy"
                np.save(pf_pose_file, pose_aligned[i, j])
                combined_file = cb_pose_dir / "comparison.npy"
                combined_pose = np.concatenate([cb_poses_gravity[i], pose_aligned[i, j]], axis=0)
                np.save(combined_file, combined_pose)
                # vis_to_html(cb_pose_dir, [cb_pose_file, pf_pose_file])
                vis_to_html(cb_pose_dir, [combined_file])

        for c in poi_category:
            poi_counter[c] += 1
        ps_meta["clips"].append(clip_dict)

    if not ps_meta["clips"]:
        tqdm.write(f"No valid clips in {meta_file}, skipping.")
        continue

    if rotation_counter / (len(clips) + rotation_counter) > rotating_clips_thres:
        tqdm.write(f"Reject video {meta_file} due to too many ({rotation_counter}/{len(clips)}) high-rotation clips.")
        filter_summary["large_rotation"] += len(ps_meta["clips"])
        continue

    filter_summary["success"] += len(ps_meta["clips"])
    out_file = output_root / f"{video_id}.json"
    with open(out_file, "w") as f:
        json.dump(ps_meta, f, indent=4)

for name, summary in [
    ("filter_summary", filter_summary),
    ("camera_summary", camera_summary),
]:
    summary_file = summary_root / f"{name}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Wrote summary to {summary_file}")

# 可视化统计结果
fig, ax = plt.subplots(figsize=(10, 6))
items = sorted(camera_summary.items(), key=lambda x: x[1], reverse=True)
values = list(camera_summary.values())
ax.hist(values, bins='auto', color="skyblue", edgecolor="black")

ax.set_xlabel("Match Count")
ax.set_ylabel("Number of Videos")
ax.set_title(f"CameraBench Match Distribution")

plt.tight_layout()
summary_file = summary_root / "camera_summary.png"
fig.savefig(summary_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved camera summary histogram to {summary_file}")

# 按数量从大到小排序
items = sorted(filter_summary.items(), key=lambda x: x[1], reverse=True)
labels, counts = zip(*items)

fig, ax = plt.subplots(figsize=(10, 6))
# 建议用水平条形图，长标签也容易展示
ax.barh(labels, counts, color="salmon")
ax.invert_yaxis()  # 让数量最多的排在最上
ax.set_xlabel("Number of filtered clips")
ax.set_title("Filter Summary")

plt.tight_layout()
summary_file = summary_root / "filter_summary.png"
fig.savefig(summary_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved filter summary plot to {summary_file}")
