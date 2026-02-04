import json
import pandas as pd
from pathlib import Path
from typing import Optional
import tyro
from pydantic import BaseModel
import numpy as np
from PIL import Image
import shutil
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from src.dataset import PanShotDataset, Re10kDataset, DemoDataset
from einops import rearrange
import src.camera_control as ucpe
import torch
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt


class Args(BaseModel):
    methods: list[str]
    data: str = "DemoDataset"
    num_frames: int = 81
    data_root: Path = Path("data/UCPE")
    panshot_data_root: Path = Path("data/UCPE")
    re10k_data_root: Path = Path("data/RealEstate10k")
    input_file: Path = Path("demo/teaser.json")
    num_workers: int = 2
    zero_first_yaw: bool = True
    output_dir: Path = Path("outputs/figures")
    sample_frames: Optional[int] = 4
    padding: int = 25
    quality: int = 95
    video_ids: Optional[list[str]] = None
    fps: int = 16
    animate_latup: bool = False


def collate_fn(samples):
    data = samples[0]
    return data


def prepare_dataloader(args, result_root=None, video_ids=None):
    dataset_class = globals().get(args.data, None)
    dataset = dataset_class(args, "test", load_keys=["pose", "result"], result_root=result_root, video_ids=video_ids)
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
    )
    return dataloader


def main():
    args = tyro.cli(Args)

    dataloaders = {}
    for method, result in zip(args.methods[::2], args.methods[1::2]):
        dataloader = prepare_dataloader(args, result, args.video_ids)
        dataloaders[method] = dataloader

    if args.video_ids is None:
        args.video_ids = None
        for dataloader in dataloaders.values():
            video_ids = set()
            for meta in dataloader.dataset.metas:
                video_ids.add(meta["video_id"])
            args.video_ids = video_ids if args.video_ids is None else video_ids & args.video_ids
        args.video_ids = list(args.video_ids)
        dataloaders = {}
        for method, result in zip(args.methods[::2], args.methods[1::2]):
            dataloader = prepare_dataloader(args, result, args.video_ids)
            dataloaders[method] = dataloader

    print(f"Found {len(args.video_ids)} videos to process.")

    for datas in tqdm(zip(*dataloaders.values()), desc=f"Exporting figures", total=len(dataloader)):
        data = datas[0]
        video_id = data["video_id"]
        result_id = Path(data["result_path"]).stem

        # Save prompts
        prompt_dir = args.output_dir / "prompts"
        prompt_path = prompt_dir / f"{video_id}.txt"
        if not prompt_path.exists():
            prompt_dir.mkdir(parents=True, exist_ok=True)
            with open(prompt_path, "w") as f:
                f.write(data["caption"])

        # Save Lat-up map visualization
        lat_up_dir = args.output_dir / "lat_up_map"
        lat_up_path = lat_up_dir / f"{video_id}.png"
        if not lat_up_path.exists():
            rot = torch.from_numpy(data["pose"][..., :3, :3]).float()  # [T, 3, 3]
            rot = rot[:1].unsqueeze(0)  # [B=1, 1, 3, 3]
            up_map, lat_map = ucpe.compute_up_lat_map(
                R=rot,  # [B, T, 3, 3]
                x_fov=torch.tensor(data["x_fov"]).float().unsqueeze(0),  # [B=1, 1]
                xi=torch.tensor(data["xi"]).float().unsqueeze(0),  # [B=1, 1]
                height=30,
                width=52,
            )

            lat_up_dir.mkdir(parents=True, exist_ok=True)
            ucpe.visualize_up_lat_map(
                up_map[0, 0],
                lat_map[0, 0],
                str(lat_up_path),
            )
            tqdm.write(f"Saved lat-up map to {lat_up_path}")

        lat_up_dir = args.output_dir / "lat_up_video"
        lat_up_path = lat_up_dir / f"{video_id}.mp4"
        if args.animate_latup and not lat_up_path.exists():
            rot = torch.from_numpy(data["pose"][..., :3, :3]).float()  # [T, 3, 3]
            rot = rot.unsqueeze(0)  # [B=1, T, 3, 3]
            up_map, lat_map = ucpe.compute_up_lat_map(
                R=rot,  # [B, T, 3, 3]
                x_fov=torch.tensor(data["x_fov"]).float().unsqueeze(0),  # [B=1, 1]
                xi=torch.tensor(data["xi"]).float().unsqueeze(0),  # [B=1, 1]
                height=30,
                width=52,
            )

            lat_up_dir.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(lat_up_path, fps=args.fps)
            for up, lat in zip(up_map[0], lat_map[0]):
                fig = ucpe.visualize_up_lat_map(up, lat)
                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                buf = canvas.buffer_rgba()
                img = np.asarray(buf, dtype=np.uint8)
                writer.append_data(img[:, :, :3])
                plt.close(fig)
            writer.close()
            tqdm.write(f"Saved lat-up video to {lat_up_path}")

        grid_image_path = args.output_dir / "grid" / f"{result_id}.jpg"
        if not grid_image_path.exists():
            frame_methods = []
            H, W = datas[-1]["result"].shape[2:4]
            for method, data in zip(dataloaders.keys(), datas):
                frames = data["result"]
                frames = (frames + 1.0) / 2.0 * 255.0
                frames = frames.astype(np.uint8)
                frames = rearrange(frames, "C T H W -> T H W C")  # (T, H, W, 3)
                total_frames = len(frames)
                if args.sample_frames < total_frames:
                    frame_indices = np.linspace(0, total_frames - 1, args.sample_frames, dtype=int)
                else:
                    frame_indices = np.arange(total_frames)
                frames = frames[frame_indices]  # (sample_frames, H, W, 3)

                # Save frames as images
                output_frames_dir = args.output_dir / method / result_id
                output_frames_dir.mkdir(parents=True, exist_ok=True)
                scaled = []
                for i, frame in enumerate(frames):
                    frame_path = output_frames_dir / f"{i}.jpg"
                    frame = Image.fromarray(frame)
                    frame.save(frame_path, quality=args.quality)
                    frame = frame.resize((W, H), Image.LANCZOS)
                    scaled.append(frame)

                frame_methods.append(scaled)

            # Save frames as a grid image
            grid_image = Image.new(
                'RGB',
                (
                    W * len(frame_indices) + args.padding * (len(frame_indices) - 1),
                    H * len(frame_methods) + args.padding * (len(frame_methods) - 1),
                ),
                (255, 255, 255)
            )
            for j, frames in enumerate(frame_methods):
                for i, frame in enumerate(frames):    
                    grid_image.paste(frame, (
                        i * (W + args.padding),
                        j * (H + args.padding),
                    ))
            grid_image_path.parent.mkdir(parents=True, exist_ok=True)
            grid_image.save(grid_image_path, quality=args.quality)
            tqdm.write(f"Saved grid image to {grid_image_path}")


if __name__ == "__main__":
    main()
