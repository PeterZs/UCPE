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
import os
import shutil
from scipy.spatial.transform import Rotation as R
import seaborn as sns
import pandas as pd


split = "train"  # "train" or "test"
panflow_root = Path("data/360-1M")
panshot_root = Path("data/UCPE")
camerabench_root = panshot_root / "CameraBench"
debug_root = Path("debug/match_panflow")
match_cb_root = panshot_root / "PanFlow" / f"align_to_camerabench-{split}"
match_pf_root = panshot_root / "PanFlow" / f"match_to_camerabench-{split}"
match_pf_root.mkdir(parents=True, exist_ok=True)
pf_clip_root = panshot_root / "PanFlow" / f"match_clips-{split}"
pf_clip_root.mkdir(parents=True, exist_ok=True)
summary_root = match_pf_root.parent / f"{match_pf_root.name}-summary"
summary_root.mkdir(parents=True, exist_ok=True)

max_pf_per_cb = 100 if split == "train" else 1
max_cb_per_pf = 5 if split == "train" else 3
matches_per_poi = 1000 if split == "train" else 1
rotation_score_thres = 5.0
avg_qalign_thres = 0.7
watermark_score_thres = 0.3
motion_score_bins = 10
max_match_per_bin = 2000 if split == "train" else 10
visualize_video = False
visualize_scores = ["motion_score", "watermark_score", "avg_qalign", "rotation_score"]
skip_undownloaded = True
filter_cb_per_pf = True
filter_cb_percentile = 0.8

if skip_undownloaded:
    pf_video_root = panshot_root / "PanFlow" / "videos"
    downloaded_videos = set([p.stem for p in pf_video_root.glob("*.mp4")])
    print(f"Found {len(downloaded_videos)} downloaded PanFlow videos.")

match_meta_files = list(match_cb_root.glob("*.json"))
match_meta_files.sort()
print(f"Found {len(match_meta_files)} match meta files.")

if split == "test":
    pf_clip_train = panshot_root / "PanFlow" / "match_clips-train"
    pf_clip_train = pf_clip_train.glob("*.json")
    pf_clip_train = set([p.stem for p in pf_clip_train])
    print(f"Found {len(pf_clip_train)} training PanFlow clips.")

    print(f"Loading CameraBench poses for filtering...")
    cb_meta_file = camerabench_root / f"processed_{split}.jsonl"
    with jsonlines.open(cb_meta_file, "r") as reader:
        cb_meta_all = list(reader)

    cb_max_rotations = {}
    for obj in tqdm(cb_meta_all, desc="Loading CameraBench poses"):
        video_id = Path(obj["path"]).stem
        pose_file = camerabench_root / "vipe" / "pose" / f"{video_id}.npz"
        if not pose_file.exists():
            tqdm.write(f"Pose file not found: {pose_file}, skipping.")
            continue

        pose = np.load(pose_file)["data"]  # (T, 4, 4)
        R_all = pose[:, :3, :3]  # (T, 3, 3)
        # 计算相邻帧之间的相对旋转
        rel_rot = np.einsum("tij,tjk->tik", np.linalg.inv(R_all[:-1]), R_all[1:])
        # 将相对旋转矩阵转换为旋转角度（度数）
        rel_angles = R.from_matrix(rel_rot).magnitude() * 180.0 / np.pi
        # 取最大旋转角
        cb_max_rotations[video_id] = float(np.max(rel_angles))

    # sort by max rotation
    cb_max_rotations = sorted(cb_max_rotations.items(), key=lambda x: x[1])
    num_cb_to_keep = int(len(cb_max_rotations) * filter_cb_percentile)
    cb_videos_to_keep = set([v[0] for v in cb_max_rotations[:num_cb_to_keep]])
    print(f"Keeping {len(cb_videos_to_keep)} CameraBench videos for testing.")


def plot_match_histogram(summary, summary_name, summary_file):
    fig, ax = plt.subplots(figsize=(10, 6))

    items = sorted(summary.items(), key=lambda x: -x[1])
    labels, counts = zip(*items)
    ax.barh(labels, counts, color="salmon")

    ax.set_xlabel("Match Count")
    ax.set_ylabel("Number of Videos")
    ax.set_title(f"{summary_name} Distribution")

    plt.tight_layout()
    fig.savefig(summary_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved camera summary histogram to {summary_file}")


def plot_match_histogram_category(summary, summary_name, summary_file, max_categories=30):
    """
    seaborn 版本的类别分布绘图：
      - 横轴为类别
      - 纵轴为 density（相对频率）
      - 超过 max_categories 的类别合并为 "others"
      - 输出 PDF（矢量图）
    """

    # ---------------------------
    # Step 1: 排序并保留 top-K 类别
    # ---------------------------
    items = sorted(summary.items(), key=lambda x: -x[1])

    if len(items) > max_categories:
        top_items = items[:max_categories]
        others_total = sum([c for _, c in items[max_categories:]])
        top_items.append(("others", others_total))
    else:
        top_items = items

    labels, counts = zip(*top_items)

    # ---------------------------
    # Step 2: density = count / total
    # ---------------------------
    total = sum(counts)
    density = [c / total for c in counts]

    # ---------------------------
    # Step 3: seaborn 绘图
    # ---------------------------
    df = pd.DataFrame({
        "category": labels,
        "density": density,
        "count": counts,
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df,
        x="category",
        y="density",
        ax=ax,
        palette="viridis"
    )

    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Proportion", fontsize=12)
    # ax.set_title(f"{summary_name} (Proportion)", fontsize=14)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # ---------------------------
    # Step 4: 保存为 PDF（矢量图）
    # ---------------------------
    summary_file = Path(summary_file)
    summary_file = summary_file.with_suffix(".pdf")
    fig.savefig(summary_file, dpi=300, bbox_inches="tight", format="pdf")

    plt.close(fig)
    print(f"[Saved PDF] {summary_file}")


def plot_score_histogram(motion_scores, summary_file):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(motion_scores, bins="auto", color="skyblue", edgecolor="black")
    ax.set_xlabel("Score")
    ax.set_ylabel("Number of Clips")
    ax.set_title("PanFlow Clip Score Distribution")

    plt.tight_layout()
    fig.savefig(summary_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved score histogram to {summary_file}")


# match_meta_files = match_meta_files[:100]
poi_category_count = defaultdict(int)
metas = []
for meta_file in tqdm(match_meta_files, desc=f"Loading match metadata"):
    with open(meta_file, "r") as f:
        meta = json.load(f)
    meta["clips"] = [clip for clip in meta["clips"] if "matches" in clip]
    for pf_clip in meta["clips"]:
        for c in pf_clip["poi_category"]:
            poi_category_count[c] += 1
    metas.append(meta)
        
summary_file = summary_root / "poi_category_count.pdf"
plot_match_histogram_category(poi_category_count, "poi_category_count", summary_file)
num_clips = sum(poi_category_count.values())
print(f"Found {len(poi_category_count)} POI categories, covering {num_clips} clips.")

match_cands = []
pf_clips = {}
for meta in tqdm(metas, desc=f"Preprocessing match metadata"):
    for pf_clip in meta["clips"]:
        pf_key = f"{pf_clip['video_id']}-{pf_clip['clip_id']}"
        if split == "test" and pf_key in pf_clip_train:
            continue

        pf_clips[pf_key] = {k: pf_clip[k] for k in [
            "scores", "video_id", "clip_id", "clip_name", "poi_category", "frames"
        ]}
        pf_clips[pf_key]["fps"] = meta["fps"]
        pf_clips[pf_key]["time_range"] = (
            pf_clip["frames"][0] / meta["fps"],
            pf_clip["frames"][-1] / meta["fps"],
        )

        scores = pf_clip["scores"]
        for cb_clip in pf_clip["matches"]:
            if split == "test" and cb_clip["video"] not in cb_videos_to_keep:
                continue

            if scores["watermark_score"] > watermark_score_thres:
                continue
            if scores["avg_qalign"] < avg_qalign_thres:
                continue
            
            rotation_score = (rotation_score_thres - scores["rotation_score"]) / rotation_score_thres
            total_score = \
                0.3 * (1 - scores["watermark_score"]) \
                + 0.5 * scores["avg_qalign"] \
                + 0.2 * rotation_score

            match_info = {
                "total_score": total_score,
                "video_id": pf_clip["video_id"],
                "clip_id": pf_clip["clip_id"],
                "frames": cb_clip["frames"],
                "rmse": cb_clip["rmse"],
                "R": cb_clip["R"],
                "s": cb_clip["s"],
            }
            match_info["time_range"] = (
                cb_clip["frames"][0] / meta["fps"],
                cb_clip["frames"][-1] / meta["fps"],
            )
            match_cands.append((cb_clip["video"], match_info))

print(f"Found {len(match_cands)} match candidates.")

score_summary = defaultdict(list)
for pf_clip in pf_clips.values():
    for key in visualize_scores:
        score_summary[key].append(pf_clip["scores"][key])
for key, score in score_summary.items():
    summary_file = summary_root / f"{key}_before.png"
    plot_score_histogram(score, summary_file)

motion_scores = score_summary["motion_score"]
motion_bin_edges = np.quantile(motion_scores, np.linspace(0, 1, motion_score_bins + 1))
motion_bin_edges = np.unique(motion_bin_edges)
print(f"Motion score histogram bins: {motion_bin_edges}")

cb_matches = defaultdict(list)
panflow_summary = defaultdict(int)
poi_category_summary = defaultdict(int)
motion_bins = defaultdict(int)
for cb_clip, match_info in tqdm(match_cands, desc=f"Analysing match metadata"):
    pf_key = f"{match_info['video_id']}-{match_info['clip_id']}"
    pf_clip = pf_clips[pf_key]
    motion_score = pf_clip["scores"]["motion_score"]
    bin_idx = np.searchsorted(motion_bin_edges, motion_score, side='right') - 1
    bin_idx = min(bin_idx, len(motion_bin_edges) - 2)

    if skip_undownloaded and match_info["video_id"] not in downloaded_videos:
        continue
    if motion_bins[bin_idx] >= max_match_per_bin:
        continue
    if pf_key in panflow_summary and panflow_summary[pf_key] >= max_cb_per_pf:
        continue
    if cb_clip in cb_matches and len(cb_matches[cb_clip]) >= max_pf_per_cb:
        continue
    if all(poi_category_summary[c] > matches_per_poi for c in pf_clip["poi_category"]):
        continue

    for c in pf_clip["poi_category"]:
        poi_category_summary[c] += 1
    cb_matches[cb_clip].append(match_info)
    panflow_summary[pf_key] += 1
    motion_bins[bin_idx] += 1

camerabench_summary = {k: len(v) for k, v in cb_matches.items()}
for summary_name, summary in [
    ("camerabench_summary", camerabench_summary),
    ("panflow_summary", panflow_summary),
]:
    summary_file = summary_root / f"{summary_name}.png"
    plot_match_histogram(summary, summary_name, summary_file)

plot_match_histogram_category(
    poi_category_summary,
    "poi_category_summary",
    summary_root / "poi_category_summary.pdf"
)

print(f"{len(pf_clips)} PanFlow clips in total.")
print(f"{len(panflow_summary)} PanFlow clips have >= 1 CameraBench match.")
print(f"Selected {len(cb_matches)} CameraBench videos.")
if split == "train" and filter_cb_per_pf:
    pf_keys = {k: v for k, v in panflow_summary.items() if v >= max_cb_per_pf}
    print(f"{len(pf_keys)} PanFlow clips have >= {max_cb_per_pf} CameraBench matches.")
else:
    pf_keys = panflow_summary
pf_clips = {k: v for k, v in pf_clips.items() if k in pf_keys}
panflow_videos = set([c["video_id"] for c in pf_clips.values()])
print(f"Selected {len(panflow_videos)} PanFlow videos.")
print(f"Selected {sum([len(v) for v in cb_matches.values()])} total matches.")

score_summary = defaultdict(list)
for pf_clip in pf_clips.values():
    for key in visualize_scores:
        score_summary[key].append(pf_clip["scores"][key])
for key, score in score_summary.items():
    summary_file = summary_root / f"{key}_after.png"
    plot_score_histogram(score, summary_file)

for pf_key, pf_clip in tqdm(pf_clips.items(), desc="Saving selected PanFlow clips"):
    out_file = pf_clip_root / f"{pf_key}.json"
    with open(out_file, "w") as f:
        json.dump(pf_clip, f, indent=4)

for cb_video, match_list in tqdm(cb_matches.items(), desc="Saving match results"):
    match_list = [m for m in match_list if f"{m['video_id']}-{m['clip_id']}" in pf_keys]
    out_file = match_pf_root / f"{cb_video}.json"
    with open(out_file, "w") as f:
        json.dump(match_list, f, indent=4)
    # tqdm.write(f"Wrote {len(match_list)} matches to {out_file}")
    # tqdm.write(f"Best score {match_list[0]['total_score']:.4f}, worst score {match_list[-1]['total_score']:.4f}")

    if visualize_video:
        cb_video_dir = debug_root / cb_video
        cb_video_dir.mkdir(parents=True, exist_ok=True)
        src_video_file = panshot_root / "CameraBench" / "videos" / f"{cb_video}.mp4"
        tgt_video_file = cb_video_dir / "CameraBench.mp4"
        shutil.copy2(src_video_file, tgt_video_file)

        cap = cv2.VideoCapture(str(src_video_file))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        for match in match_list:
            pf_video_file = panflow_root / "videos" / f"{match['video_id']}.mp4"
            vr = VideoReader(str(pf_video_file), ctx=cpu(0), num_threads=1)
            pf_clip = pf_clips[f"{match['video_id']}/{match['clip_id']}"]
            clip_start = pf_clip["frames"][0]
            start_frame, end_frame = match["frames"]
            start_frame = start_frame + clip_start
            end_frame = end_frame + clip_start
            sample_frames = np.linspace(start_frame, end_frame, num=frame_count)
            sample_frames = np.round(sample_frames).astype(int)
            clip = vr.get_batch(sample_frames).asnumpy()
            out_clip_file = cb_video_dir / f"{match['video_id']}-{start_frame}-{end_frame}.mp4"

            process = (
                ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{clip.shape[2]}x{clip.shape[1]}", r=fps)
                .output(str(out_clip_file), pix_fmt="yuv420p", vcodec="libx264", r=fps, crf=23)
                .overwrite_output()
                .run_async(pipe_stdin=True, quiet=True)
            )
            process.stdin.write(clip.tobytes())
            process.stdin.close()
            process.wait()
