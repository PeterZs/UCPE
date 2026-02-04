from q_align import QAlignVideoScorer, QAlignAestheticScorer, QAlignScorer
from pathlib import Path
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
from decord import VideoReader, cpu
import jsonlines
from PIL import Image
import torch
import numpy as np

# -----------------------------
# basic configuration
# -----------------------------
panflow_root = Path("data/360-1M")
panshot_root = Path("data/UCPE")
output_root = panshot_root / "PanFlow"
output_root.mkdir(parents=True, exist_ok=True)
output_jsonl = output_root / "scores.jsonl"
batch_size = 16
max_workers = min(64, os.cpu_count() or 4)
inflight_limit = max_workers * 2
max_frames = 10
print(f"Using max_workers={max_workers}, inflight_limit={inflight_limit}")

meta_root = panflow_root / "meta"
meta_files = list((meta_root).glob("*.json"))
meta_files.sort()
print(f"Found {len(meta_files)} PanFlow meta files.")

def load_one_meta(meta_file):
    try:
        with open(meta_file, "r") as f:
            meta = json.load(f)
        video_id = meta.get("video_id", Path(meta_file).stem)
        video_path = panflow_root / "videos" / f"{video_id}.mp4"

        if "slam_clips" not in meta or "clips" not in meta["slam_clips"]:
            return []  # skip
        if not video_path.exists():
            return []  # skip

        clips_local = []
        for clip in meta["slam_clips"]["clips"]:
            clips_local.append({
                "clip_id": clip["clip_id"],
                "frames": clip["frames"],
                "video_id": video_id,
                "video_path": str(video_path),
            })
        return clips_local
    except Exception as e:
        tqdm.write(f"Error reading {meta_file}: {e}")
        return []


# ================================================================
# 已处理 clip 检查
# ================================================================
processed = set()
if output_jsonl.exists():
    print(f"Resuming from {output_jsonl}")
    with open(output_jsonl, "r", encoding="utf-8") as f_in:
        for line in f_in:
            try:
                rec = json.loads(line)
                processed.add((rec["video_id"], rec["clip_id"]))
            except Exception:
                continue
    print(f"Found {len(processed)} processed clips to skip.")


# -----------------------------
# 并行加载 meta 文件
# -----------------------------
clips = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = [ex.submit(load_one_meta, mf) for mf in meta_files]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading PanFlow metadata"):
        clips.extend(fut.result())

# 过滤掉已处理的 clips
clips = [c for c in clips if (c["video_id"], c["clip_id"]) not in processed]
print(f"Total {len(clips)} remaining clips from {len(meta_files)} videos.")


# -----------------------------
# scorer 初始化
# -----------------------------
video_scorer = QAlignVideoScorer()
scorers = {
    "image_aesthetic": QAlignAestheticScorer(
        tokenizer=video_scorer.tokenizer,
        model=video_scorer.model,
        image_processor=video_scorer.image_processor
    ),
    "image_quality": QAlignScorer(
        tokenizer=video_scorer.tokenizer,
        model=video_scorer.model,
        image_processor=video_scorer.image_processor
    ),
    "video_quality": video_scorer,
}

# ================================================================
# util functions
# ================================================================
def process_one_clip(clip):
    """
    从 clip 中读取指定起止帧范围内的帧（1fps），返回 {clip, frames(list of PIL)}.
    """
    video_path = clip["video_path"]
    frame_range = clip.get("frames", None)

    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    except Exception as e:
        print(f"[process_one_clip] cannot open {video_path}: {e}")
        return None

    fps = vr.get_avg_fps()
    start, end = frame_range[0], frame_range[-1]
    start = max(0, start)
    end = min(len(vr) - 1, end)

    # 按 1fps 抽帧
    frame_count = min((end - start + 1) / fps, max_frames)
    frame_indices = np.linspace(start, end, num=max(int(frame_count), 1))
    frame_indices = np.round(frame_indices).astype(int)

    try:
        frames_np = vr.get_batch(frame_indices).asnumpy()
    except Exception as e:
        print(f"[process_one_clip] error decoding {video_path}: {e}")
        return None

    frames = [Image.fromarray(frames_np[i]) for i in range(frames_np.shape[0])]
    video = [video_scorer.expand2square(frame, tuple(int(x*255) for x in video_scorer.image_processor.image_mean)) for frame in frames]
    video_tensors = video_scorer.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half()
    return {"clip": clip, "frames": video_tensors}


def infer_and_flush(results, writer):
    """
    批量推理并写出结果。
    results: list of {clip, frames}
    """
    if not results:
        return

    video_batch = [r["frames"] for r in results]
    scores = {}
    with torch.inference_mode():
        video_tensors = [vid.to(video_scorer.model.device) for vid in video_batch]
        video_frames = [len(vid) for vid in video_tensors]
        for key, scorer in scorers.items():
            # image 分支拼接所有帧，video 分支传列表
            images = torch.cat(video_tensors) if "image" in key else video_tensors
            output_logits = scorer.model(
                scorer.input_ids.repeat(len(images), 1),
                images=images
            )["logits"][:, -1, scorer.preferential_ids_]

            values = torch.softmax(output_logits, -1) @ scorer.weight_tensor

            if "image" in key:
                # 按每个 clip 的帧数切开并取均值
                values_split = torch.split(values, video_frames)
                values = torch.stack([v.mean(0) for v in values_split])

            scores[key] = values  # shape: [n_clips]

    # 遍历每个 clip，把三个分数写出
    n_clips = len(results)
    for i in range(n_clips):
        clip_info = results[i]["clip"]
        out_obj = {
            "video_id": clip_info.get("video_id"),
            "clip_id": clip_info.get("clip_id"),
        }
        for key, value in scores.items():
            out_obj[key] = float(value[i].item())
        writer.write(out_obj)


# ================================================================
# 主流程
# ================================================================
output_jsonl.parent.mkdir(parents=True, exist_ok=True)
prepared_buffer = []

try:
    f = open(output_jsonl, "a", buffering=1, encoding="utf-8")
    writer = jsonlines.Writer(f)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        pbar = tqdm(total=len(clips), desc="processing clips")

        pending = set()
        i_submit = 0

        # 先提交一些任务
        while i_submit < len(clips) and len(pending) < inflight_limit:
            fut = ex.submit(process_one_clip, clips[i_submit])
            pending.add(fut)
            i_submit += 1

        while pending:
            for fut in as_completed(list(pending), timeout=None):
                pending.remove(fut)
                result = fut.result()
                if result is not None:
                    prepared_buffer.append(result)

                # 满一批就推理一次
                if len(prepared_buffer) >= batch_size:
                    infer_and_flush(prepared_buffer[:batch_size], writer)
                    prepared_buffer = prepared_buffer[batch_size:]
                    pbar.update(batch_size)

                # 提交新的任务
                while i_submit < len(clips) and len(pending) < inflight_limit:
                    fut_new = ex.submit(process_one_clip, clips[i_submit])
                    pending.add(fut_new)
                    i_submit += 1
                break

        # flush 剩余 buffer
        while prepared_buffer:
            chunk = prepared_buffer[:batch_size]
            infer_and_flush(chunk, writer)
            prepared_buffer = prepared_buffer[len(chunk):]

        pbar.close()
finally:
    try:
        writer.close()
    except Exception:
        pass
    try:
        f.close()
    except Exception:
        pass
