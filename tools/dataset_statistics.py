import json
import pandas as pd
from pathlib import Path
from typing import Optional
import tyro
from pydantic import BaseModel
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import PanShotDataset, Re10kDataset, DemoDataset
from einops import rearrange


# ======================================================
#   seaborn global theme (官方模板)
# ======================================================
sns.set_theme(
    context="paper",
    style="whitegrid",
    palette="deep",
    font_scale=0.9,
)


# ======================================================
#   Args
# ======================================================
class Args(BaseModel):
    data: str = "PanShotDataset"
    num_frames: int = 81
    data_root: Path = Path("data/UCPE")
    num_workers: int = 4
    zero_first_yaw: bool = True
    output_dir: Path = Path("outputs/suppl")
    num_samples: Optional[int] = None
    split: str = "train"
    color: str = "C0"   # NEW: color


# ======================================================
#   DataLoader
# ======================================================
def collate_fn(samples):
    return samples[0]


def prepare_dataloader(args):
    dataset_class = globals().get(args.data, None)
    dataset = dataset_class(
        args, args.split,
        load_keys=["pose", "xi", "y_fov"]
    )

    if args.num_samples is not None:
        dataset = Subset(dataset, list(range(args.num_samples)))

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
    )


# ======================================================
#   Camera Euler: x-right, y-down, z-forward
# ======================================================
def rotmat_to_euler_cam(R):
    fx, fy, fz = R[:, 2]   # camera forward in world
    yaw = np.arctan2(fx, fz)
    pitch = np.arctan2(-fy, np.sqrt(fx**2 + fz**2))
    roll = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([yaw, pitch, roll])


def rot_angle(R0, Ri):
    R = R0.T @ Ri
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.degrees(np.arccos(cos_theta))


# ======================================================
#   MAIN
# ======================================================
def main():
    args = tyro.cli(Args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataloader = prepare_dataloader(args)

    # seaborn palette → choose refined colors
    COLOR_MAP = {
        "C0": sns.color_palette("deep")[0],
        "C1": sns.color_palette("deep")[1],
        "C2": sns.color_palette("deep")[2],
        "C3": sns.color_palette("deep")[3],
    }
    COLOR = COLOR_MAP.get(args.color, args.color)

    # ======================================================
    # containers
    # ======================================================
    final_rel_yaw = []   # displacement-based azimuth
    init_pitch = []
    init_roll = []
    rotation_magnitude = []
    max_rel_yaw = []     # NEW: per-clip max camera relative yaw (signed)
    xis = []
    fovs = []

    # ======================================================
    # iterate dataset
    # ======================================================
    for data in tqdm(dataloader, desc=f"Processing {args.data}"):

        poses = data["pose"]  # (N, 3, 4)
        xi = float(data["xi"])
        fov = float(data["y_fov"])

        xis.append(xi)
        fovs.append(fov)

        R_all = poses[:, :, :3]
        R0 = R_all[0]

        # Initial orientation
        yaw0, pitch0, roll0 = rotmat_to_euler_cam(R0)
        init_pitch.append(pitch0)
        init_roll.append(roll0)

        # ------- Displacement-based azimuth (position) -------
        p0 = poses[0, :, 3]
        pN = poses[-1, :, 3]
        v = pN - p0
        yaw_pos = np.arctan2(v[0], v[2])
        final_rel_yaw.append(np.degrees(yaw_pos))

        # ------- Rotation magnitude wrt frame 0 (unsigned) -------
        rotation_magnitude.append(max(
            rot_angle(R0, Ri) for Ri in R_all
        ))

        # ------- NEW: max camera relative yaw (signed) -------
        rel_yaws = []
        for Ri in R_all:
            yaw_i, _, _ = rotmat_to_euler_cam(Ri)
            rel_yaws.append(yaw_i)
        rel_yaws = np.array(rel_yaws)

        # pick the frame with largest |relative yaw|, keep sign
        idx_max = np.argmax(np.abs(rel_yaws))
        max_rel_yaw.append(rel_yaws[idx_max])

    # convert to numpy
    final_rel_yaw = np.array(final_rel_yaw)
    init_pitch = np.array(init_pitch)
    init_roll = np.array(init_roll)
    rotation_magnitude = np.array(rotation_magnitude)
    max_rel_yaw = np.array(max_rel_yaw)
    xis = np.array(xis)
    fovs = np.array(fovs)

    # remove top 1% roll outliers
    roll_thr = np.percentile(init_roll, 99)
    init_roll = init_roll[init_roll <= roll_thr]

    # remove bottom 1% roll outliers
    roll_thr = np.percentile(init_roll, 1)
    init_roll = init_roll[init_roll >= roll_thr]

    # ======================================================
    # save helper
    # ======================================================
    def save_pdf(fig, name):
        fig.savefig(args.output_dir / f"{name}.pdf",
                    dpi=300,
                    bbox_inches="tight",
                    format="pdf")
        plt.close(fig)

    # ======================================================
    # 1. Rose plot: displacement azimuth (position)
    # ======================================================
    fig = plt.figure(figsize=(2.0, 2.0))
    ax = plt.subplot(111, polar=True)

    rad = np.radians(final_rel_yaw)

    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_facecolor("white")

    ax.hist(rad, bins=36, alpha=0.55, color=COLOR,
            edgecolor=".3", linewidth=0.4)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_thetagrids(
        angles=np.arange(0, 360, 45),
        labels=[f"{(a if a <= 180 else a - 360)}°" for a in np.arange(0, 360, 45)]
    )

    # ax.set_title("Azimuth", pad=8)
    save_pdf(fig, "rose_direction")

    # ======================================================
    # NEW: 1b. Rose plot: max camera relative yaw (signed)
    # ======================================================
    fig = plt.figure(figsize=(2.0, 2.0))
    ax = plt.subplot(111, polar=True)

    rad_max_yaw = np.radians(max_rel_yaw)

    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_facecolor("white")

    ax.hist(rad_max_yaw, bins=36, alpha=0.55, color=COLOR,
            edgecolor=".3", linewidth=0.4)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_thetagrids(
        angles=np.arange(0, 360, 45),
        labels=[f"{(a if a <= 180 else a - 360)}°" for a in np.arange(0, 360, 45)]
    )

    # ax.set_title("Max Relative Yaw", pad=8)
    save_pdf(fig, "rose_rotation")  # 文件名仍叫 rose_rotation

    # ======================================================
    # 2. Pitch histogram (seaborn)
    # ======================================================
    fig = plt.figure(figsize=(2.2, 2.0))
    sns.histplot(init_pitch, bins=40, kde=False, stat="density",
                 color=COLOR, edgecolor=".3", linewidth=0.4, alpha=0.6)
    plt.xlabel("Pitch (°)")
    plt.ylabel("Density")
    # plt.title("Initial Pitch")
    save_pdf(fig, "pitch_hist")


    # ======================================================
    # 3. Roll histogram
    # ======================================================
    fig = plt.figure(figsize=(2.2, 2.0))
    sns.histplot(init_roll, bins=40, kde=False, stat="density",
                 color=COLOR, edgecolor=".3", linewidth=0.4, alpha=0.6)
    plt.xlabel("Roll (°)")
    plt.ylabel("Density")
    # plt.title("Initial Roll")
    save_pdf(fig, "roll_hist")


    # ======================================================
    # 4. Rotation magnitude
    # ======================================================
    fig = plt.figure(figsize=(2.2, 2.0))
    sns.histplot(rotation_magnitude, bins=40, kde=False, stat="density",
                 color=COLOR, edgecolor=".3", linewidth=0.4, alpha=0.6)
    plt.xlabel("Maximum Rotation (°)")
    plt.ylabel("Density")
    # plt.title("Rotation Magnitude")
    save_pdf(fig, "rotation_magnitude_hist")


    # ======================================================
    # 5. ξ histogram
    # ======================================================
    fig = plt.figure(figsize=(2.2, 2.0))
    sns.histplot(xis, bins=30, kde=False, stat="density",
                 color=COLOR, edgecolor=".3", linewidth=0.4, alpha=0.6)
    plt.xlabel("ξ")
    plt.ylabel("Density")
    # plt.title("ξ Distribution")
    save_pdf(fig, "xi_hist")


    # ======================================================
    # 6. FoV histogram
    # ======================================================
    # remove top 1% FoV outliers
    fov_thr = np.percentile(fovs, 99)
    fovs = fovs[fovs <= fov_thr]
    # remove bottom 1% FoV outliers
    fov_thr = np.percentile(fovs, 1)
    fovs = fovs[fovs >= fov_thr]

    fig = plt.figure(figsize=(2.2, 2.0))
    sns.histplot(fovs, bins=30, kde=False, stat="density",
                 color=COLOR, edgecolor=".3", linewidth=0.4, alpha=0.6)
    plt.xlabel("FoV (°)")
    plt.ylabel("Density")
    # plt.title("FoV Distribution")
    save_pdf(fig, "fov_hist")


# ======================================================
if __name__ == "__main__":
    main()