from pathlib import Path
import jsonlines
from tqdm.auto import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import ffmpeg
import json
import cv2


# 数据与路径设置
data_root = Path("data/CameraBench")
output_root = Path("data/UCPE/CameraBench")
target_height = 720
target_width = 1280
target_frames = 81
target_fps = 16
banned_labels = ["zoom-in", "zoom-out"]
banned_keywords = ["zoom"]
split = "train"  # "train" or "test"
dryrun = False

# 读取数据
if split == "test":
    meta_file = data_root / "test.jsonl"
    with jsonlines.open(meta_file, "r") as reader:
        metadata = list(reader)
else:
    meta_file = data_root / "cam_motion" / "captionset.json"
    with open(meta_file, "r") as f:
        captionset = json.load(f)
    metadata = []
    videos = set()
    for obj in captionset:
        video = obj["videos"][0]
        if video in videos:
            continue
        videos.add(video)
        metadata.append({
            "path": video,
            "caption": obj["messages"][1]["content"]
        })

# 处理与过滤视频
labels = defaultdict(int)
frames = []
heights = []
fps_list = []
filtered_meta = []
for obj in tqdm(metadata, desc="Reading videos"):
    video_file = obj["path"]
    if split == "test":
        for label in obj["labels"]:
            labels[label] += 1
    video_path = data_root / video_file

    # 获取视频信息
    # meta = ffmpeg.probe(str(video_path))  # 使用 ffprobe 获取全 metadata
    # vstream = next(s for s in meta["streams"] if s["codec_type"] == "video")
    # w = int(vstream["width"])
    # h = int(vstream["height"])
    # fps_str = vstream.get("avg_frame_rate", "0/1")
    # num, den = map(int, fps_str.split('/'))
    # fps = num / den if den != 0 else None
    # num_frames = int(vstream.get("nb_frames"))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        tqdm.write(f"  - failed to open {video_path}, skipping")
        continue
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 打印视频信息
    num_frames_sampled = int(num_frames / fps * target_fps)
    frames.append(num_frames)
    heights.append(h)
    fps_list.append(fps)
    tqdm.write(f"{video_path} — {num_frames} frames, fps ~ {fps:.2f}, sampled to {num_frames_sampled}, height {h}")

    # 过滤条件
    if num_frames_sampled < target_frames:
        tqdm.write(f"  - filtered by frames: {num_frames_sampled} < {target_frames}")
        continue
    if h < target_height:
        tqdm.write(f"  - filtered by height: {h} < {target_height}")
        continue
    if w < target_width:
        tqdm.write(f"  - filtered by width: {w} < {target_width}")
        continue
    if split == "test":
        if any(label in banned_labels for label in obj["labels"]):
            tqdm.write(f"  - filtered by label: {obj['labels']}")
            continue
    else:
        caption = obj["caption"]
        if any(kw in caption.lower() for kw in banned_keywords):
            tqdm.write(f"  - filtered by keyword in caption: {caption}")
            continue
    filtered_meta.append(obj)

    # 导出视频
    output_video = (output_root / video_file).with_suffix(".mp4")
    tqdm.write(f"  - exporting to {output_video}")
    if dryrun:
        continue
    output_video.parent.mkdir(parents=True, exist_ok=True)

    # 说明：
    # 1) fps 过滤器把时间轴重采样为 target_fps（不会改变播放速度）
    # 2) scale 先把画面按长宽比“铺满”16:9 画幅（a=iw/ih）：
    #    - 如果原视频更宽(gt(a,16/9))：固定高=720，宽按比例（-2 表示自动，且保证可被2整除）
    #    - 否则固定宽=1280，高按比例
    # 3) 再做中心裁剪到 1280x720
    # 4) vframes 只写前 target_frames 帧
    in_stream = ffmpeg.input(str(video_path))
    v = (
        in_stream.video
        .filter('fps', fps=target_fps)  # 重新采样到目标帧率
        .filter('scale',
                f'if(gt(a,{target_width}/{target_height}),-2,{target_width})',
                f'if(gt(a,{target_width}/{target_height}),{target_height},-2)')
        .filter('crop',
                target_width, target_height,
                f'(in_w-{target_width})/2', f'(in_h-{target_height})/2')
    )

    # 如无需音频可去掉 audio；若要保留音频，建议也做 asetpts=PTS-STARTPTS
    out = ffmpeg.output(
        v,
        str(output_video),
        vcodec='libx264',
        pix_fmt='yuv420p',
        r=target_fps,             # 容器帧率元数据
        vframes=target_frames     # 只导出前 N 帧
    )
    ffmpeg.run(out, overwrite_output=True, quiet=True)

print(f"Total videos: {len(frames)}, after filtering: {len(filtered_meta)}")

# 导出 jsonl
if not dryrun:
    jsonl_file = output_root / f"processed_{split}.jsonl"
    with jsonlines.open(jsonl_file, "w") as writer:
        writer.write_all(filtered_meta)

# 创建输出目录
output_dir = Path(f"debug/summarize_camerabench_{split}")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. labels 柱状图
if labels:
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    label_names = list(labels.keys())
    label_counts = [labels[k] for k in label_names]
    ax1.bar(label_names, label_counts, color='skyblue')
    ax1.set_xticklabels(label_names, rotation=45, ha='right')
    ax1.set_ylabel('Count')
    ax1.set_title('Label Frequencies')
    plt.tight_layout()
    fig1.savefig(output_dir / "label_frequencies.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

# 2. frames 直方图
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.hist(frames, bins=30, color='orange', edgecolor='black')
ax2.set_xlabel('Number of Frames')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Frames')
plt.tight_layout()
fig2.savefig(output_dir / "frames_distribution.png", dpi=300, bbox_inches='tight')
plt.close(fig2)

# 3. heights 直方图
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.hist(heights, bins=30, color='green', edgecolor='black')
ax3.set_xlabel('Height (pixels)')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Frame Heights')
plt.tight_layout()
fig3.savefig(output_dir / "heights_distribution.png", dpi=300, bbox_inches='tight')
plt.close(fig3)

# 4. fps 直方图
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.hist(fps_list, bins=30, color='purple', edgecolor='black')
ax4.set_xlabel('FPS (frames per second)')
ax4.set_ylabel('Frequency')
ax4.set_title('FPS Distribution')
plt.tight_layout()
fig4.savefig(output_dir / "fps_distribution.png", dpi=300, bbox_inches='tight')
plt.close(fig4)
