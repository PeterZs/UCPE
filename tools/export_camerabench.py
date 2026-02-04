import csv
import jsonlines
from pathlib import Path
from tqdm.auto import tqdm
import jsonlines

# -----------------------------
# paths
# -----------------------------
data_root = Path("data/UCPE/CameraBench")
filtered_jsonl = data_root / "filtered.jsonl"
split = "train"  # "train" or "test"
input_jsonl = data_root / f"captioned_{split}.jsonl"
camera_jsonl = data_root / f"processed_{split}.jsonl"
output_csv = data_root / f"metadata_{split}.csv"
output_jsonl = data_root / f"metadata_{split}.jsonl"

with jsonlines.open(filtered_jsonl, "r") as reader:
    filtered_videos = {obj["path"] for obj in reader if any(obj["filter"].values())}

# 保证输出目录存在
output_csv.parent.mkdir(parents=True, exist_ok=True)

total = 0
written = 0
skipped = 0
filtered = 0

with jsonlines.open(input_jsonl, "r") as reader, \
     open(output_csv, "w", newline="", encoding="utf-8") as fout, \
        jsonlines.open(output_jsonl, "w") as jsonl_writer, \
            jsonlines.open(camera_jsonl, "r") as camera_reader:
    camera_metas = {obj["path"]: obj for obj in camera_reader}
    writer = csv.writer(fout)
    # 表头
    writer.writerow(["video", "prompt"])

    for obj in tqdm(reader, desc="Converting JSONL → CSV"):
        total += 1
        path = obj.get("path", None)
        caption = obj.get("caption", None)

        if path is None or caption is None:
            skipped += 1
            continue

        if path in filtered_videos:
            filtered += 1
            continue

        # 规范化 caption 的空白字符，避免 CSV 里出现杂乱换行
        caption_norm = " ".join(str(caption).split())
        jsonl_writer.write({
            "video": Path(path).stem,
            "prompt": caption_norm,
            "camera_caption": camera_metas[path]["caption"],
            "camera_labels": camera_metas[path].get("labels", []),
        })
        writer.writerow([path, caption_norm])
        written += 1

print(f"Done. Total lines: {total}, written: {written}, skipped (missing fields): {skipped}, filtered: {filtered}")
print(f"CSV saved to: {output_csv}")
print(f"JSONL saved to: {output_jsonl}")
