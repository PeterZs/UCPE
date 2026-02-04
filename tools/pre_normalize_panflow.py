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
from einops import einsum, rearrange, repeat
from visualize_pose import vis_to_html
import decord
from decord import VideoReader, cpu
decord.bridge.set_bridge("torch")
import os
import shutil
import torch
from thirdparty.PanFlow.utils.erp_utils import transformation_to_flow
from thirdparty.PanoFlowAPI.apis.PanoRaft import PanoRAFTAPI


panflow_root = Path("data/360-1M")
panshot_root = Path("data/UCPE")
debug_root = Path("debug/match_panflow")
match_cb_root = panshot_root / "PanFlow" / "align_to_camerabench"
output_jsonl = panshot_root / "PanFlow" / "near_plane_depth.jsonl"
output_jsonl.parent.mkdir(parents=True, exist_ok=True)
summary_root = output_jsonl.parent / f"{output_jsonl.stem}-summary"
summary_root.mkdir(parents=True, exist_ok=True)

flow_height = 512
flow_width = 1024
epipole_thres = 30
upper_edge_mask = 0.35
lower_edge_mask = 0.2
sample_fps = 2
frame_near_quantile = 5
video_near_quantile = 10
batch_size = 24

visualize_disp = True
if visualize_disp:
    disp_root = summary_root / "disp"
    disp_root.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda")
flow_estimater_ckpt = "models/PanoFlow/PanoFlow(RAFT)-wo-CFE.pth"

flow_estimater = PanoRAFTAPI(
    device=device, model_path=flow_estimater_ckpt
)

def epipole_flow(pose):
    pose = pose[1:].inverse() @ pose[:-1]
    return transformation_to_flow(pose, (flow_height, flow_width))


def flow_depth(flow, pose, eps=1e-6):
    epi_flow = epipole_flow(pose)
    dot = einsum(flow, epi_flow, "n c h w, n c h w -> n h w")
    epi_norm = epi_flow.norm(dim=1)
    depth = epi_norm ** 2 / dot.clamp_min(eps)

    flow_norm = flow.norm(dim=1)
    cos = dot / (epi_norm * flow_norm).clamp_min(eps)
    cos = cos.clamp(-1, 1)
    degree = torch.rad2deg(torch.acos(cos))
    invalid = degree > epipole_thres

    depth[invalid] = float("nan")
    depth[dot < eps] = float("nan")
    height = depth.shape[-2]
    depth[:, :int(upper_edge_mask * height)] = float("nan")
    depth[:, int((1 - lower_edge_mask) * height):] = float("nan")
    return depth


def near_plane_depth(depth):
    depth = rearrange(depth, "n h w -> n (h w)")
    near_depth = torch.nanquantile(depth, frame_near_quantile / 100, dim=-1)
    near_depth = torch.nanquantile(near_depth, video_near_quantile / 100)
    return near_depth.item()


match_meta_files = list(match_cb_root.glob("*.json"))
match_meta_files.sort()
match_meta_files = match_meta_files[:10]  # DEBUG
print(f"Found {len(match_meta_files)} match meta files.")

near_depths = []
with jsonlines.open(output_jsonl, "w") as writer:
    for meta_file in tqdm(match_meta_files, desc=f"Processing matched clips"):
        with open(meta_file, "r") as f:
            meta = json.load(f)

        video_id = meta_file.stem
        video_file = panflow_root / "videos" / f"{video_id}.mp4"
        vr = VideoReader(str(video_file), width=flow_width, height=flow_height, ctx=cpu(0), num_threads=1)

        for pf_clip in meta["clips"]:
            if "matches" not in pf_clip:
                tqdm.write(f"No matches in clip {pf_clip['clip_name']}, skipping.")
                continue

            clip_id = pf_clip["clip_id"]
            clip_name = pf_clip["clip_name"]
            frames = pf_clip["frames"]
            num_frames = frames[-1] - frames[0] + 1
            fps = meta["fps"]
            num_frames_sampled = int(round(num_frames / fps * sample_fps))
            if num_frames_sampled < 2:
                tqdm.write(f"Too few frames ({num_frames}) in clip {clip_name}, skipping.")
                continue
            sample_frames = np.linspace(frames[0], frames[-1], num_frames_sampled)
            sample_frames = np.round(sample_frames).astype(int)

            pose_file = panflow_root / "slam_pose" / video_id / clip_name
            pose_file = pose_file.with_suffix(".npy")
            if not pose_file.exists():
                tqdm.write(f"Pose file not found: {pose_file}, skipping.")
                continue
            c2w = np.load(pose_file)  # (T, 3, 4)
            c2w = c2w[sample_frames - frames[0]]  # (N, 3, 4)
            c2w_4x4 = np.eye(4, dtype=np.float32)
            c2w = np.hstack((c2w, repeat(c2w_4x4[-1], "n -> f 1 n", f=c2w.shape[0])))
            pf_pose = torch.from_numpy(c2w)

            video = vr.get_batch(sample_frames)
            flow_in = video.float()
            flow_in = rearrange(flow_in, "n h w c -> n c h w")

            flow_in = flow_in.to(device)
            flows = flow_estimater.chunk_estimate_flow_cfe(flow_in, chunk_size=batch_size)
            flows = rearrange(flows, "n h w c -> n c h w")
            pf_pose = pf_pose.to(device)
            depth = flow_depth(flows, pf_pose)
            near_depth = near_plane_depth(depth)

            if visualize_disp:
                # 保存 disparity
                disp_file = disp_root / f"{video_id}-{clip_id}-disp.png"
                rgb_file  = disp_root / f"{video_id}-{clip_id}-rgb.png"

                # ---------- disparity ----------
                depth_vis = depth[0].detach().cpu().numpy()  # [H,W]
                disp_vis = np.zeros_like(depth_vis, dtype=np.float32)
                valid = np.isfinite(depth_vis) & (depth_vis > 1e-6)
                disp_vis[valid] = 1.0 / depth_vis[valid]

                if valid.any():
                    dmin = np.nanpercentile(disp_vis[valid], 1)
                    dmax = np.nanpercentile(disp_vis[valid], 99)
                    dmin = max(dmin, 0)
                    norm = np.clip((disp_vis - dmin) / (dmax - dmin + 1e-6), 0, 1)
                    disp_color = (plt.cm.magma(norm)[..., :3] * 255).astype(np.uint8)
                else:
                    disp_color = np.zeros((*disp_vis.shape, 3), dtype=np.uint8)

                cv2.imwrite(str(disp_file), cv2.cvtColor(disp_color, cv2.COLOR_RGB2BGR))

                # ---------- 对应的 RGB ----------
                # video: decord 返回 [N,H,W,C], 取第一帧
                if video.shape[0] > 0:
                    rgb = video[0].cpu().numpy()  # [H,W,C], float32 0~255?
                    if rgb.dtype != np.uint8:
                        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                    cv2.imwrite(str(rgb_file), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            near_depths.append(near_depth)
            result = {
                "video_id": video_id,
                "clip_name": clip_name,
                "clip_id": clip_id,
                "near_depth": near_depth,
            }
            writer.write(result)

near_depths = [d for d in near_depths if np.isfinite(d)]
summary_file = summary_root / "near_plane_depth.png"

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(near_depths, bins="auto", color="skyblue", edgecolor="black")
ax.set_xlabel("Near Plane Depth")
ax.set_ylabel("Number of Clips")
ax.set_title("Near Plane Depth Distribution")

plt.tight_layout()
fig.savefig(summary_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved near plane depth histogram to {summary_file}")
