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
data_root = Path("data/UCPE/CameraBench")
videos = sorted((data_root / "videos").glob("*.mp4"))  # 直接遍历视频文件
output_jsonl = data_root / f"filtered.jsonl"

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"   # try 7B first; switch to 32B if resources allow
# model_id = "chancharikm/qwen2.5-vl-7b-cam-motion-preview"
nframes = 32                                # hint for frame sampling inside qwen_vl_utils
fps_hint = None                             # None or a small integer like 1/2/4 (optional)
batch_size = 8                              # how many videos per vLLM.generate batch
max_workers = min(8, os.cpu_count() or 4)   # 线程数按机器调整
inflight_limit = batch_size * 2  # 同时在制的样本上限
print(f"Using max_workers={max_workers}, inflight_limit={inflight_limit}")

max_new_tokens = 512
temperature = 0.2
top_p = 0.9
repetition_penalty = 1.05
gpu_memory_utilization = 0.9
tensor_parallel_size = 1
limit_mm_per_prompt = {"video": 1}

prompt_text = """
You are a video filtering assistant.
Your task is to analyze the given panoramic video and decide whether it meets certain quality and format requirements.
Check the following conditions and output results strictly in JSON format, with boolean values (true/false) for each label:

- non_fullscreen_or_black_borders: true if the video is not full-screen, has black borders, or is vertical instead of wide.
- has_subtitles_or_watermarks: true if subtitles, captions, or watermarks are visible.
- is_cartoon_or_flat_style: true if the video is animated, cartoon-like, 2D flat style, or non-photorealistic.

Important:
– Output strictly in JSON format.
– Do not add extra text or explanation.
– Each key must exist, with true/false values.

Example output:
{
  "non_fullscreen_or_black_borders": false,
  "has_subtitles_or_watermarks": true,
  "is_cartoon_or_flat_style": false
}
"""


# ===================================================================
# 从这里开始：重写为“VLM 布尔标签 + JSONL 写出”的完整流程
# ===================================================================

def build_llm_input(video_path: Path, processor: AutoProcessor):
    """构造单条输入，包含视频与文本 prompt。"""
    video_item = {"type": "video", "video": str(video_path), "nframes": nframes}
    if fps_hint is not None:
        video_item["fps"] = fps_hint

    messages = [
        {"role": "system", "content": "You are a helpful video filtering assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                video_item
            ]
        }
    ]

    # 预处理多模态输入（抽帧/张量装配）
    _, video_inputs = process_vision_info(messages)

    # 模板化为可生成的字符串
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    mm_data = {}
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {"prompt": prompt, "multi_modal_data": mm_data}


def prepare_one(vpath: Path, processor: AutoProcessor):
    """单样本：基于固定过滤 prompt → 组装 vLLM 输入。"""
    llm_in = build_llm_input(vpath, processor)
    obj = {"path": vpath.relative_to(data_root)}
    return obj, llm_in


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
# utils
# -----------------------------
def clean_json_output(text: str) -> str:
    """清理 Qwen 输出中的 ```json / ``` 包裹，并尝试只保留第一段 JSON。"""
    s = text.strip()

    # 去除 Markdown 代码围栏
    s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    s = s.replace("```", "").strip()

    # 尝试截取最外层花括号的 JSON 片段
    # 找到第一个 '{' 与最后一个 '}' 之间的内容
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        s = s[first:last + 1].strip()
    return s


# -----------------------------
# 推理与写出
# -----------------------------
output_jsonl.parent.mkdir(parents=True, exist_ok=True)
prepared_buffer = []  # 缓存已准备好的 (obj, llm_in)


def infer_and_flush(buffer, writer):
    """对 buffer 中的若干样本推理，并写出结果（VLM 布尔标签 JSON）。"""
    if not buffer:
        return
    batch_objs = [it[0] for it in buffer]
    batch_inputs = [it[1] for it in buffer]

    gens = llm.generate(batch_inputs, sampling_params)

    for ob, g in zip(batch_objs, gens):
        # vLLM 生成的第一个候选
        text_out = (g.outputs[0].text if g.outputs and g.outputs[0].text is not None else "").strip()

        try:
            cleaned = clean_json_output(text_out)
            parsed = json.loads(cleaned)

            # 写出：path + 模型布尔标签
            writer.write({
                "path": str(ob["path"]),
                "filter": parsed
            })

        except Exception as e:
            # 输出无法解析成 JSON：打印日志并跳过
            print(f"[skip] JSON parse failed for {ob['path']}: {e}")
            print(f"Raw output: {text_out}")
            continue


# -----------------------------
# pipeline：边准备边推理边写出（动态提交 + 行缓冲）
# -----------------------------
# 基础检查
for v in videos:
    assert v.exists(), f"File not found: {v}"

try:
    # 用行缓冲打开文件，便于“边写边可见”
    f = open(output_jsonl, "w", buffering=1, encoding="utf-8")
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
                obj, llm_in = fut.result()
                prepared_buffer.append((obj, llm_in))
                pbar.update(1)

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

print(f"done. VLM labels saved to: {output_jsonl}")
