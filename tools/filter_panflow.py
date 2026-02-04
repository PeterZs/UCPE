from vllm import LLM, SamplingParams

from pathlib import Path
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import re
from decord import VideoReader, cpu
import numpy as np
import torch
from PIL import Image
ngpus = torch.cuda.device_count()

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


# -----------------------------
# basic configuration
# -----------------------------
panflow_root = Path("data/360-1M")
panshot_root = Path("data/UCPE")
output_root = panshot_root / "PanFlow" / "filtered"
output_root.mkdir(parents=True, exist_ok=True)

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"   # try 7B first; switch to 32B if resources allow
batch_size = 16 * ngpus                             # how many videos per vLLM.generate batch
max_workers = min(4 * ngpus, os.cpu_count() or 4)
inflight_limit = max_workers * 2
print(f"Using max_workers={max_workers}, inflight_limit={inflight_limit}")

max_new_tokens = 512
temperature = 0.2
top_p = 0.9
repetition_penalty = 1.05
gpu_memory_utilization = 0.9
tensor_parallel_size = ngpus
limit_mm_per_prompt = {"image": 1}

meta_root = panflow_root / "meta"
meta_files = list((meta_root).glob("*.json"))
meta_files.sort()
print(f"Found {len(meta_files)} PanFlow meta files.")

existing_out = list(output_root.glob("*.json"))
existing_ids = {f.stem for f in existing_out}
meta_files = [f for f in meta_files if f.stem not in existing_ids]
print(f"{len(meta_files)} files to process after skipping existing.")

prompt_text = """
You are a video understanding assistant specialized in analyzing panoramic ERP-format videos.  
Given one frame of a panoramic video, your tasks are:  

1. **Filtering**: Identify if the video should be filtered out.  
Output boolean flags for the following conditions (true if the issue exists, false otherwise):  

- non_ERP_format: The video is **not in ERP (Equirectangular Projection)** panoramic format. For example, if the video looks like a flat perspective, fisheye, cube-map, or any projection other than ERP, set this to true.  

- has_subtitle_or_watermark: The video contains **text overlays, subtitles, logos, or watermarks**. Look carefully for visible text at the bottom, center, or corners of the video. If such elements are present and not part of the real scene, set this to true.  

- edge_missing: The top or bottom edges of the ERP panorama are **cut off, blacked out, cropped, or covered by logos/watermarks**, so the full 360° vertical coverage is missing or obstructed. If you cannot clearly see the poles (sky/ground) or if the edges are hidden by overlays, set this to true.  

- has_overlay: The frame contains **artificial overlays**, such as embedded UI elements, pop-up graphics, stickers, video-in-video inserts, menus, or other synthetic elements that are not part of the natural scene. If you see signs of AR/VR interface, streaming UI, or added images, set this to true.  

- low_quality: The video is of **poor visual quality**, such as being blurry, noisy, heavily pixelated, very low resolution, or distorted in a way that prevents recognizing the scene. If the content is hard to interpret due to quality issues, set this to true.  

- unnatural_content: The video contains **cartoons, animations, CGI, synthetic 3D renderings, or game engine graphics** rather than real-world panoramic footage. If the content is not realistic, set this to true.  

2. **POI Categorization**: From the provided list of categories, select **one or more most relevant** labels that best describe the scene.  
   Only use the given categories, do not invent new ones.  

---

**poi_category list (choose only from below):**

Restaurant, Coffee-Shop, Bars-and-Pubs, Residential-area, Hotels-Motels, Vaccation-Rentals, Hospitals-Clinics, Pharmacies, Dentists, School-Universities, Library, Supermarkets, Shopping-Malls, Clothing-Stores, Shoe-Stores, Bookstores, Flowerstore, Furniture-Stores, Electorical-Store, Pet-Store, Toy-Shop, Airports, Train-Stations, Bus-Stops, Gas-Station, Car-Rental-Agencies, Theaters, Concert-Halls, Sports-Stadiums, Parks-and-Recreation-Areas, Museums, Art-Galleries, Zoos-Aquariums, Botanical-Gardens, Landmarks, Cultural-Centers, Post-Offices, Police-Stations, Courthouses, CityHalls, Banks-ATMs, Events-Conferences-halls, Beaches, Hiking-Trails, Campgrounds, Lakes, Mountains, Forest-Mountains, Farms, Street-View, Square, Business-Centers, Tech-Companies, Co-working-Spaces, Gyms-and-Fitness-Centers, Sports-Clubs, Swimming-Pools, Tennis-Courts, Auto-Repair-Shops, Car-Washes, Parking-Lots, Churches, Mosques, Temples, Graveyards.

---

**Output strictly in JSON format** as follows:

```json
{
  "filter": {
    "non_ERP_format": false,
    "has_subtitle_or_watermark": false,
    "edge_missing": false,
    "has_overlay": false,
    "low_quality": false,
    "unnatural_content": false
  },
  "poi_category": ["Mountains"]
}
```
"""

# ================================================================
# utils
# ================================================================

def clean_json_output(text: str) -> str:
    s = text.strip()
    s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    s = s.replace("```", "").strip()
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        s = s[first:last + 1].strip()
    return s


def process_one_video(meta_file, processor):
    """子进程：为一个视频的 clips 抽中间帧，生成 LLM 输入"""
    pf_meta = json.load(open(meta_file))
    video_id = pf_meta.get("video_id", Path(meta_file).stem)
    video_path = panflow_root / "videos" / f"{video_id}.mp4"

    if "slam_clips" not in pf_meta or "clips" not in pf_meta["slam_clips"]:
        print(f"[skip] No slam_clips in meta: {meta_file}")
        return None

    if not video_path.exists():
        print(f"[skip] Video file not found: {video_path}")
        return None

    try:
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    except Exception as e:
        print(f"[skip] Failed to open video {video_path}: {e}")
        return None

    clip_infos, llm_inputs = [], []

    for clip in pf_meta["slam_clips"]["clips"]:
        start, end = clip["frames"][0], clip["frames"][-1]
        mid = (start + end) // 2

        frame = vr[mid].asnumpy().astype(np.uint8)
        frame_pil = Image.fromarray(frame)

        image_item = {"type": "image", "image": frame_pil}
        messages = [
            {"role": "system", "content": "You are a helpful video filtering assistant."},
            {"role": "user", "content": [{"type": "text", "text": prompt_text}, image_item]},
        ]

        image_inputs, _ = process_vision_info(messages)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        llm_inputs.append({"prompt": prompt, "multi_modal_data": mm_data})
        clip_infos.append(clip)

    return video_id, clip_infos, llm_inputs


# ================================================================
# init vLLM
# ================================================================
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


# ================================================================
# flush results
# ================================================================
def flush_one_video(video_id, clip_infos, llm_inputs):
    """对单个视频的所有 clips 批量推理并写 JSON。"""
    video_results = []
    total_clips = len(llm_inputs)

    # 二级进度条：按 clip 数量更新
    with tqdm(total=total_clips, desc=f"Video {video_id}", leave=False) as pbar:
        for i in range(0, total_clips, batch_size):
            chunk = llm_inputs[i:i + batch_size]
            gens = llm.generate(chunk, sampling_params)
            for clip, g in zip(clip_infos[i:i + batch_size], gens):
                text_out = g.outputs[0].text.strip() if g.outputs else ""
                try:
                    cleaned = clean_json_output(text_out)
                    parsed = json.loads(cleaned)
                    video_results.append({
                        "clip_id": clip["clip_id"],
                        "clip_name": clip["clip_name"],
                        "filter": parsed.get("filter", {}),
                        "poi_category": parsed.get("poi_category", [])
                    })
                except Exception as e:
                    print(f"[skip] JSON parse failed for {clip['clip_name']}: {e}")
                    print(f"Raw output: {text_out}")
                finally:
                    pbar.update(1)  # 每处理一个 clip 就更新进度

    out_file = output_root / f"{video_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(video_results, f, indent=4)


# ================================================================
# 主流程
# ================================================================

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    pbar = tqdm(total=len(meta_files), desc="processing videos")
    pending = set()
    i_submit = 0

    # 初始填充 inflight
    while i_submit < len(meta_files) and len(pending) < inflight_limit:
        fut = ex.submit(process_one_video, meta_files[i_submit], processor)
        pending.add(fut)
        i_submit += 1

    while pending:
        for fut in as_completed(list(pending), timeout=None):
            pending.remove(fut)
            result = fut.result()
            if result is None:
                continue
            video_id, clip_infos, llm_inputs = result
            flush_one_video(video_id, clip_infos, llm_inputs)

            pbar.update(1)

            # 补交新任务
            while i_submit < len(meta_files) and len(pending) < inflight_limit:
                fut_new = ex.submit(process_one_video, meta_files[i_submit], processor)
                pending.add(fut_new)
                i_submit += 1
            break
    pbar.close()

try:
    llm.shutdown()
except Exception:
    pass

print(f"done. Results saved to {output_root}")
