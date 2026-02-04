from vllm import LLM, SamplingParams

from pathlib import Path
import jsonlines
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import re

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


# -----------------------------
# basic configuration
# -----------------------------
split = "train"  # "train" or "test"
data_root = Path("data/UCPE/PanShot")
video_dir = data_root / f"videos-{split}"
output_jsonl = data_root / f"captioned-{split}.jsonl"

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"   # try 7B first; switch to 32B if resources allow
# model_id = "chancharikm/qwen2.5-vl-7b-cam-motion-preview"
nframes = 32                                # hint for frame sampling inside qwen_vl_utils
fps_hint = None                             # None or a small integer like 1/2/4 (optional)
batch_size = 8                              # how many videos per vLLM.generate batch
max_workers = min(8, os.cpu_count() or 4)   # 线程数按机器调整
inflight_limit = batch_size * 2             # 同时在制的样本上限
print(f"Using max_workers={max_workers}, inflight_limit={inflight_limit}")


max_new_tokens = 512
temperature = 0.2
top_p = 0.9
repetition_penalty = 1.05
gpu_memory_utilization = 0.9
tensor_parallel_size = 1
limit_mm_per_prompt = {"video": 1}


def build_llm_input(video_path: Path, prompt_text: str, processor: AutoProcessor):
    # compose messages (system + user text + video item)
    video_item = {"type": "video", "video": str(video_path), "nframes": nframes}
    if fps_hint is not None:
        video_item["fps"] = fps_hint

    messages = [
        {"role": "system", "content": "You are a helpful video captioning assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_text},
            video_item
        ]}
    ]

    # extract frames / prepare tensors for the model (CPU/I/O-heavy)
    image_inputs, video_inputs = process_vision_info(messages)

    # text template → prompt string
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    mm_data = {}
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {"prompt": prompt, "multi_modal_data": mm_data}


def prepare_one(video, processor: AutoProcessor):
    """单样本：基于 labels/seed_caption 构造提示 → 抽帧预处理 → 组装 vLLM 输入"""
    vpath = video_dir / f"{video}.mp4"
    prompt_text = "Please describe this video in detail."
    try:
        llm_in = build_llm_input(vpath, prompt_text, processor)
    except Exception as e:
        print(f"Error processing {video}: {e}")
        return None
    return video, llm_in


# -----------------------------
# load metadata
# -----------------------------
videos = [v.stem for v in video_dir.glob("*.mp4")]
print(f"Found {len(videos)} videos in {video_dir}")
processed = set()
if output_jsonl.exists():
    print(f"Resuming from {output_jsonl}")
    with open(output_jsonl, "r", encoding="utf-8") as f_in:
        for line in f_in:
            try:
                rec = json.loads(line)
                processed.add(rec["video"])
            except Exception:
                continue
    print(f"Found {len(processed)} processed videos to skip.")
videos = [v for v in videos if v not in processed]
print(f"Total videos to process: {len(videos)}")

# -----------------------------
# init vLLM + processor
# -----------------------------
llm = LLM(
    model=model_id,
    tensor_parallel_size=tensor_parallel_size,
    gpu_memory_utilization=gpu_memory_utilization,
    # enforce_eager=True,
    limit_mm_per_prompt=limit_mm_per_prompt,
)
processor = AutoProcessor.from_pretrained(model_id)
sampling_params = SamplingParams(
    max_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
    repetition_penalty=repetition_penalty,
)


# -----------------------------
# pipeline：边准备边推理边写出（动态提交 + 行缓冲）
# -----------------------------
output_jsonl.parent.mkdir(parents=True, exist_ok=True)
prepared_buffer = []  # 缓存已准备好的 (obj, llm_in)


def infer_and_flush(buffer, writer):
    """对 buffer 中的若干样本推理，并写出结果"""
    if not buffer:
        return
    batch_videos = [it[0] for it in buffer]
    batch_inputs = [it[1] for it in buffer]
    gens = llm.generate(batch_inputs, sampling_params)

    for video, g in zip(batch_videos, gens):
        text_out = g.outputs[0].text.strip()

        writer.write({
            "video": video,
            "caption": text_out
        })

try:
    # 用行缓冲打开文件，便于“边写边可见”
    f = open(output_jsonl, "a", buffering=1, encoding="utf-8")
    writer = jsonlines.Writer(f)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        pbar = tqdm(total=len(videos), desc="preparing & inferring")

        # 动态 pending 集合
        pending = set()
        i_submit = 0

        # 先填满 in-flight
        while i_submit < len(videos) and len(pending) < inflight_limit:
            fut = ex.submit(prepare_one, videos[i_submit], processor)
            pending.add(fut)
            i_submit += 1

        # 循环直到所有任务完成
        while pending:
            # 只等待当前 pending 集合中的任务
            for fut in as_completed(list(pending), timeout=None):
                pending.remove(fut)
                result = fut.result()
                pbar.update(1)
                if result is None:
                    continue
                prepared_buffer.append(result)

                # 满一批就立刻推理并清空对应部分
                if len(prepared_buffer) >= batch_size:
                    infer_and_flush(prepared_buffer[:batch_size], writer)
                    prepared_buffer = prepared_buffer[batch_size:]

                # 补交新任务，保持 in-flight 数量
                while i_submit < len(videos) and len(pending) < inflight_limit:
                    fut_new = ex.submit(prepare_one, videos[i_submit], processor)
                    pending.add(fut_new)
                    i_submit += 1

                # 跳出到 while pending，重新评估 pending 集合（已更新）
                break

        # 把“尾巴”按 batch 循环清空，确保不丢最后一个或多个 batch
        while prepared_buffer:
            chunk = prepared_buffer[:batch_size]
            infer_and_flush(chunk, writer)
            prepared_buffer = prepared_buffer[len(chunk):]

        pbar.close()
finally:
    # 关闭 writer / 文件句柄
    try:
        writer.close()
    except Exception:
        pass
    try:
        f.close()
    except Exception:
        pass
    # 优雅关闭 vLLM 引擎
    try:
        llm.shutdown()
    except Exception:
        pass

print(f"done. captions saved to: {output_jsonl}")
