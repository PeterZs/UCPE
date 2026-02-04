import json
import pandas as pd
from pathlib import Path
from typing import Optional
import tyro
from pydantic import BaseModel


class Args(BaseModel):
    methods: list[str]
    metrics: Optional[list[str]] = None
    pad_cols: int = 0


metric_infos = {
    "video_metrics/rho_pred": {
        "name": "${\\rho}_{\\text{A}}$",
        "group": "Camera Lens Control",
        "higher_is_better": True,
        "scale": 100,
        "decimal_places": 2,
    },
    "video_metrics/rho_gt": {
        "name": "${\\rho}_{\\text{A-gt}}$",
        "group": "Camera Lens Control",
        "higher_is_better": True,
        "scale": 100,
        "decimal_places": 2,
    },
    "video_metrics/vfov_err": {
        "name": "FoV (°)",
        "group": "Camera Lens Control",
        "higher_is_better": False,
        "decimal_places": 2,
    },
    "video_metrics/k1_err": {
        "name": "${k}_{1}$",
        "group": "Camera Lens Control",
        "higher_is_better": False,
        "decimal_places": 3,
    },
    "video_metrics/k2_err": {
        "name": "${k}_{2}$",
        "group": "Camera Lens Control",
        "higher_is_better": False,
        "decimal_places": 3,
    },
    "video_metrics/pitch_err": {
        "name": "Pitch (°)",
        "group": "Absolute Orientation",
        "higher_is_better": False,
        "decimal_places": 2,
    },
    "video_metrics/roll_err": {
        "name": "Roll (°)",
        "group": "Absolute Orientation",
        "higher_is_better": False,
        "decimal_places": 2,
    },
    "video_metrics/gravity_err": {
        "name": "Gravity (°)",
        "group": "Absolute Orientation",
        "higher_is_better": False,
        "decimal_places": 2,
    },
    "video_metrics/latitude_err": {
        "name": "Latitude (°)",
        "group": "Absolute Orientation",
        "higher_is_better": False,
        "decimal_places": 2,
    },
    "video_metrics/up_err": {
        "name": "Up (°)",
        "group": "Absolute Orientation",
        "higher_is_better": False,
        "decimal_places": 2,
    },
    "video_metrics/lpips": {
        "name": "LPIPS",
        "group": "Relative Camera Pose Control",
        "higher_is_better": False,
        "decimal_places": 3,
    },
    "video_metrics/psnr": {
        "name": "PSNR",
        "group": "Relative Camera Pose Control",
        "higher_is_better": True,
        "decimal_places": 2,
    },
    "video_metrics/ssim": {
        "name": "SSIM",
        "group": "Relative Camera Pose Control",
        "higher_is_better": True,
        "decimal_places": 3,
    },
    "pose/rot_err": {
        "name": "RotErr (°)",
        "group": "Relative Camera Pose Control",
        "higher_is_better": False,
        "decimal_places": 2,
    },
    "pose/trans_err": {
        "name": "TransErr",
        "group": "Relative Camera Pose Control",
        "higher_is_better": False,
        "decimal_places": 2,
    },
    "pose/cammc": {
        "name": "CamMC",
        "group": "Relative Camera Pose Control",
        "higher_is_better": False,
        "decimal_places": 2,
    },
    "pose/rot_err_vipe": {
        "name": "RotErr - Vipe (°)",
        "group": "Relative Camera Pose Control",
        "higher_is_better": False,
        "decimal_places": 2,
    },
    "pose/trans_err_vipe": {
        "name": "TransErr - Vipe",
        "group": "Relative Camera Pose Control",
        "higher_is_better": False,
        "decimal_places": 2,
    },
    "pose/cammc_vipe": {
        "name": "CamMC - Vipe",
        "group": "Relative Camera Pose Control",
        "higher_is_better": False,
        "decimal_places": 2,
    },
    "video_metrics/fvd_center": {
        "name": "FVD-center",
        "group": "Video Generation Quality",
        "higher_is_better": False,
        "decimal_places": 2,
    },
    "video_metrics/fvd": {
        "name": "FVD",
        "group": "Video Generation Quality",
        "higher_is_better": False,
        "decimal_places": 2,
    },
    "video_metrics/fid": {
        "name": "FID",
        "group": "Video Generation Quality",
        "higher_is_better": False,
        "decimal_places": 2,
    },
    "video_metrics/cs_text": {
        "name": "CLIP",
        "group": "Video Generation Quality",
        "higher_is_better": True,
        "decimal_places": 2,
    },
    "video_metrics/cs_image": {
        "name": "CLIP-image",
        "group": "Video Generation Quality",
        "higher_is_better": True,
        "decimal_places": 2,
    },
    "video_metrics/is": {
        "name": "IS",
        "group": "Video Generation Quality",
        "higher_is_better": True,
        "decimal_places": 2,
        "std_dev": "video_metrics/is_std"
    },
    "qalign/image_quality": {
        "name": "Image Quality",
        "group": "Video Generation Quality",
        "higher_is_better": True,
        "decimal_places": 4,
    },
    "qalign/image_aesthetic": {
        "name": "Image Aesthetic",
        "group": "Video Generation Quality",
        "higher_is_better": True,
        "decimal_places": 4,
    },
    "qalign/video_quality": {
        "name": "Video Quality",
        "group": "Video Generation Quality",
        "higher_is_better": True,
        "decimal_places": 4,
    },
}

color_cells = [
    "firstcell",
    "secondcell",
    "thirdcell",
]


def main():
    args = tyro.cli(Args)

    rows = []
    metrics = metric_infos.keys() if args.metrics is None else [
        metric for metric in metric_infos.keys() if metric in args.metrics
    ]
    for method, result in zip(args.methods[::2], args.methods[1::2]):
        if result == Path(""):
            rows.append({
                "Method": method,
                **{metric_infos[metric]["name"]: None for metric in metrics}
            })
            continue

        with open(result, "r") as f:
            data = json.load(f)

        row = {"Method": method}
        for metric in metrics:
            metric_name = metric_infos[metric]["name"]
            row[metric_name] = data[metric] * metric_infos[metric].get("scale", 1.)
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.round({metric_infos[metric]["name"]: metric_infos[metric]["decimal_places"] for metric in metrics})
    print(df)

    for metric_info in metric_infos.values():
        col = metric_info["name"]
        if col not in df.columns:
            continue

        ranks = df[col].dropna().rank(
            method="dense",
            ascending=not metric_info["higher_is_better"],
        ).astype(int) - 1

        df[col] = df.apply(
            lambda row: (f"\\{color_cells[ranks[row.name]]} " if pd.notna(row[col]) and ranks[row.name] < len(color_cells) else "") +
                (f"{row[col]:.{metric_info['decimal_places']}f}" if pd.notna(row[col]) else ""),
            axis=1
        )

    for i in range(args.pad_cols):
        df.insert(loc=i, column=i, value="")

    tuples = [("", "")] * args.pad_cols + [("", "Method")]
    for metric in metrics:
        info = metric_infos[metric]
        group = info["group"]
        name_with_arrow = info["name"] + ("$\\uparrow$" if info["higher_is_better"] else "$\\downarrow$")
        tuples.append((group, name_with_arrow))
    df.columns = pd.MultiIndex.from_tuples(tuples, names=["Group", "Metric"])

    latex_table = df.to_latex(
        index=False,
        multicolumn=True,
        multicolumn_format="c",
        multirow=True,
        column_format="r" * args.pad_cols + "l" + "c" * (len(df.columns) - 1 - args.pad_cols),
    )
    print(latex_table)


if __name__ == "__main__":
    main()
