import jsonlines
from pathlib import Path
from tqdm.auto import tqdm
import jsonlines
from geocalib import GeoCalib
from decord import VideoReader, cpu
import torch
from einops import rearrange


data_root = Path("data/UCPE/CameraBench")
videos = sorted((data_root / "videos").glob("*.mp4"))  # 直接遍历视频文件
output_jsonl = data_root / "geocalib.jsonl"

processed = set()
if output_jsonl.exists():
    print(f"Resuming from {output_jsonl}")
    with jsonlines.open(output_jsonl, "r") as reader:
        for obj in reader:
            processed.add(obj["video"])
    print(f"Found {len(processed)} processed videos to skip.")
videos = [v for v in videos if v.stem not in processed]
print(f"Total videos to process: {len(videos)}")

gc = GeoCalib(weights="pinhole").cuda()

with jsonlines.open(output_jsonl, "a") as writer:
    for video in tqdm(videos, desc="Calibrating videos"):
        vr = VideoReader(str(video), ctx=cpu(0), num_threads=1)
        frames = vr.get_batch(range(0, len(vr))).asnumpy()
        frames = torch.from_numpy(frames)
        frames = rearrange(frames, "n h w c -> n c h w")
        frames = frames.float() / 255.0
        frames = frames.cuda()

        result = gc.calibrate(frames, shared_intrinsics=True)
        gravity = result["gravity"][0]
        R = gravity.R.cpu().numpy().tolist()
        roll, pitch = gravity.rp.cpu().numpy().tolist()

        writer.write({
            "video": video.stem,
            "R": R,
            "roll": roll,
            "pitch": pitch,
        })
