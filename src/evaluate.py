import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch
from torch import Tensor
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from datetime import datetime
from typing import Any, List, Literal, Tuple, Optional, Dict
from src.dataset import PanShotDataset, Re10kDataset
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
import json
from tqdm.auto import tqdm
import subprocess
from scipy.spatial.transform import Rotation as R


class Args(BaseSettings):
    data: str = "PanShotDataset"
    num_frames: int = 81  # for Re10kDataset
    test_steps: List[str] = ["qalign", "video", "vipe", "pose", "overall"]
    conda_envs: Dict = {"qalign": "qalign"}
    data_root: Path = Path("data/UCPE")
    num_workers: int = 2
    test_device: Literal["cuda", "cpu"] = "cuda"
    test_res_path: Optional[Path] = None
    evaluate_gt: bool = False
    test_name: str = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    limit_eval_videos: Optional[int] = None
    save_last: bool = True
    load_last: bool = True
    jitter_filter_percent: float = 0.8

    # qalign
    qalign_fps: float = 4.0

    # video
    test_chunk_size: int = 8

    # pose
    valid_pose_percent: float = 0.5
    frame_stride: Optional[int] = None
    pose_frames: Optional[int] = None

    model_config = SettingsConfigDict(
        env_prefix="EVAL_",
        cli_parse_args=True,
        cli_ignore_unknown_args=False,
    )


def get_path(args):
    if args.evaluate_gt:
        assert args.data == "PanShotDataset", "GT evaluation only supports PanShotDataset."
        paths = {"i2v": (args.data_root / "PanShot" / "videos-test", args.data_root / "evaluate")}
    elif args.test_res_path is not None:
        paths = {args.test_res_path.name: (args.test_res_path, args.test_res_path.parent / f"evaluate_{args.test_res_path.name}")}
    else:
        run_id = os.environ.get('WANDB_RUN_ID', None)
        assert run_id is not None, "WANDB_RUN_ID environment variable must be set."
        paths = {}
        split = "predict" if args.data == "PanShotDataset" else Path(args.data_root).name
        predict_dir = Path("logs") / run_id / split
        for task in ["t2v", "i2v"]:
            task_path = predict_dir / task
            if task_path.exists():
                paths[task] = (task_path, predict_dir / f"evaluate_{task}")
    print(f"Evaluation paths: {paths}")
    return paths


def collate_fn(samples):
    data = samples[0]
    return data


def filter_jitter(args):
    if args.jitter_filter_percent >= 1.0:
        return

    # Fix jittering rotation issue
    # Filter out videos with rapid rotations
    max_rotation_file = args.data_root / "PanShot" / "max_rotation-test.json"
    if not max_rotation_file.exists() and args.jitter_filter_percent < 1.:
        max_rotations = {}
        dataset = PanShotDataset(args, "test", load_keys=["pose"])
        for data in dataset:
            R_all = data["pose"][:, :3, :3]  # (T, 3, 3)
            # 计算相邻帧之间的相对旋转
            rel_rot = np.einsum("tij,tjk->tik", np.linalg.inv(R_all[:-1]), R_all[1:])
            # 将相对旋转矩阵转换为旋转角度（度数）
            rel_angles = R.from_matrix(rel_rot).magnitude() * 180.0 / np.pi
            max_rotations[data['video_id']] = float(np.max(rel_angles))
        max_rotations = dict(sorted(max_rotations.items(), key=lambda x: x[1]))
        with open(max_rotation_file, "w") as f:
            json.dump(max_rotations, f, indent=4)
    else:
        with open(max_rotation_file, "r") as f:
            max_rotations = json.load(f)

    num_videos = int(len(max_rotations) * args.jitter_filter_percent)
    valid_video_ids = set(list(max_rotations.keys())[:num_videos])
    print(f"Filtered {len(max_rotations) - num_videos} videos with high jittering rotations.")

    return valid_video_ids


def prepare_dataloader(args, load_keys, result_root=None, video_ids=None):
    dataset_class = globals().get(args.data, None)

    if dataset_class is PanShotDataset:
        valid_video_ids = filter_jitter(args)
        if valid_video_ids is not None:
            if video_ids is not None:
                video_ids = set(video_ids) & valid_video_ids
            else:
                video_ids = valid_video_ids

    dataset = dataset_class(args, "test", load_keys=load_keys, result_root=result_root, video_ids=video_ids)
    if args.limit_eval_videos and args.limit_eval_videos < len(dataset):
        print(f"Limiting evaluation to {args.limit_eval_videos} videos.")
        sample_ids = np.linspace(0, len(dataset) - 1, args.limit_eval_videos).astype(int).tolist()
        dataset = Subset(dataset, sample_ids)
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
    )
    return dataloader


def link_last(output_path):
    last_path = output_path.parent / "last.json"
    if last_path.exists():
        os.remove(last_path)
    os.symlink(output_path.name, last_path)
    print(f"Saved last evaluation results to {last_path}")


def save_evaluation(args, test_dir, eval_results, subfolder):
    for key, values in eval_results.items():
        if isinstance(values, list):
            results = [v["video_results"] for v in values]
            results = float(np.mean(results))
            eval_results[key] = [results, values]
        else:
            eval_results[key] = [values]

    output_folder = test_dir / subfolder
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{args.test_name}_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Evaluation results saved to {output_path}")

    if args.save_last:
        link_last(output_path)


@torch.inference_mode()
def qalign(args):
    from q_align import QAlignVideoScorer, QAlignAestheticScorer, QAlignScorer
    from PIL import Image
    from einops import rearrange, repeat

    print("Running QAlign evaluation...")
    tasks = get_path(args)

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

    for task, (test_res_path, test_dir) in tasks.items():
        print(f"Evaluating task: {task}")
        dataloader = prepare_dataloader(args, ["result"], test_res_path)

        eval_results = defaultdict(list)
        for data in tqdm(dataloader, desc="Evaluating videos"):
            frames = data["result"]
            frames = rearrange(frames, "C T H W -> T H W C")
            frame_count = len(frames) * args.qalign_fps / data["fps"]
            frame_count = min(int(frame_count), len(frames))
            frame_count = max(frame_count, 1)
            frame_indices = np.linspace(0, len(frames) - 1, num=frame_count)
            frame_indices = np.round(frame_indices).astype(int)
            frames = frames[frame_indices]
            frames = frames / 2. + 0.5
            frames = (frames * 255.0).astype(np.uint8)
            frames = [Image.fromarray(frame) for frame in frames]
            video = [video_scorer.expand2square(frame, tuple(int(x*255) for x in video_scorer.image_processor.image_mean)) for frame in frames]
            video_tensors = video_scorer.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half()

            video_tensors = video_tensors.to(video_scorer.model.device)
            for key, scorer in scorers.items():
                images = video_tensors if "image" in key else [video_tensors]
                output_logits = scorer.model(
                    scorer.input_ids.repeat(len(images), 1),
                    images=images
                )["logits"][:, -1, scorer.preferential_ids_]

                values = torch.softmax(output_logits, -1) @ scorer.weight_tensor
                score = values.mean().cpu().item()
                eval_results[key].append({
                    "video_id": data["video_id"],
                    "video_results": score,
                })

        save_evaluation(args, test_dir, eval_results, "qalign")


@torch.inference_mode()
def video(args):
    import src.camera_control as ucpe
    from unik3d.models import UniK3D
    from unik3d.utils.evaluation_depth import rho
    from torchmetrics import MeanMetric, Metric
    from einops import rearrange, repeat
    from torchmetrics.image import (
        LearnedPerceptualImagePatchSimilarity,
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
    from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
    from torchmetrics.image.inception import InceptionScore
    from torchmetrics.multimodal import CLIPScore
    from thirdparty.fvd.fvd import (
        load_i3d_pretrained,
        get_fvd_logits,
    )
    from geocalib import GeoCalib
    sys.path.append("thirdparty/GeoCalib")
    from siclib.models.utils.metrics import (
        gravity_error,
        latitude_error,
        pitch_error,
        roll_error,
        up_error,
        vfov_error,
    )

    class GeoCalibError(Metric):
        higher_is_better: bool = False

        def __init__(
            self,
            chunk_size: int | None = None,
            skip_frames: int = 4,
        ):
            super().__init__()
            self.gc = GeoCalib(weights="distorted")
            self.errors = torch.nn.ModuleDict({k: MeanMetric() for k in [
                "pitch", "roll", "gravity", "vfov", "k1", "k2", "latitude", "up"
            ]})
            self.chunk_size = chunk_size
            self.skip_frames = skip_frames

        def update(
            self,
            pred: torch.Tensor,  # [B, 3, H, W]
            gt: torch.Tensor,    # [B, 3, H, W]
        ):
            if self.skip_frames > 1:
                pred = pred[::self.skip_frames]
                gt = gt[::self.skip_frames]
            chunk_size = len(pred) if self.chunk_size is None else self.chunk_size
            for pred_chunk, gt_chunk in zip(
                pred.split(chunk_size, dim=0),
                gt.split(chunk_size, dim=0),
            ):
                pred_result = self.gc.calibrate(pred_chunk, camera_model="radial", shared_intrinsics=True)
                gt_result = self.gc.calibrate(gt_chunk, camera_model="radial", shared_intrinsics=True)

                pred_gravity, gt_gravity = pred_result["gravity"], gt_result["gravity"]
                self.errors["pitch"].update(pitch_error(pred_gravity, gt_gravity))
                self.errors["roll"].update(roll_error(pred_gravity, gt_gravity))
                self.errors["gravity"].update(gravity_error(pred_gravity, gt_gravity))

                pred_cam, gt_cam = pred_result["camera"], gt_result["camera"]
                self.errors["vfov"].update(vfov_error(pred_cam, gt_cam))
                self.errors["k1"].update(torch.abs(pred_cam.k1 - gt_cam.k1))
                self.errors["k2"].update(torch.abs(pred_cam.k2 - gt_cam.k2))

                self.errors["latitude"].update(latitude_error(
                    pred_result["latitude_field"],
                    gt_result["latitude_field"],
                ).mean(axis=(1, 2)))
                self.errors["up"].update(up_error(
                    pred_result["up_field"],
                    gt_result["up_field"],
                ).mean(axis=(1, 2)))

        def compute(self):
            return {f"{k}_err": v.compute() for k, v in self.errors.items()}

    class UcmCameraRayAngularErrorRho(Metric):
        higher_is_better = True

        def __init__(
            self,
            model_id: str = "lpiccinelli/unik3d-vitl",
            chunk_size: int = 16,
            resolution_level: int = 0,
        ):
            super().__init__()
            self.model = UniK3D.from_pretrained(model_id)
            self.rho = torch.nn.ModuleDict({k: MeanMetric() for k in [
                "gt", "pred"
            ]})
            self.model.resolution_level = resolution_level
            self.chunk_size = chunk_size

        def update(
            self,
            pred: torch.Tensor,  # [B, 3, H, W]
            gt: torch.Tensor,    # [B, 3, H, W]
            x_fov: float,
            xi: float,
        ):
            _, _, height, width = pred.shape
            d_cam = ucpe.ucm_unproject_grid_fov(
                x_fov=x_fov,
                xi=xi,
                height=height,
                width=width,
                device=pred.device,
            )
            for pred_chunk, gt_chunk in zip(
                pred.split(self.chunk_size, dim=0),
                gt.split(self.chunk_size, dim=0),
            ):
                pred_result = self.model.infer(pred_chunk)
                rays = pred_result["rays"]
                rays = rearrange(rays, "B C H W -> B H W C")
                d_cams = repeat(d_cam, "... -> B ...", B=rays.shape[0])
                rho_errors = rho(d_cams, rays)  # [B]
                self.rho["gt"].update(rho_errors)

                gt_result = self.model.infer(gt_chunk)
                gt_rays = gt_result["rays"]
                gt_rays = rearrange(gt_rays, "B C H W -> B H W C")
                rho_errors = rho(gt_rays, rays)  # [B]
                self.rho["pred"].update(rho_errors)

        def compute(self):
            return {f"rho_{k}": v.compute() for k, v in self.rho.items()}

    class FrechetVideoDistance(Metric):
        higher_is_better: bool = False
        full_state_update: bool = False

        def __init__(
            self,
            crop_center: bool = True,
            batch_size: int = 10,
        ):
            super().__init__()
            self.crop_center = crop_center
            self.batch_size = batch_size
            self.i3d = load_i3d_pretrained()

            num_features = 400
            mx_num_feats = (num_features, num_features)
            self.add_state("real_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
            self.add_state("real_features_cov_sum", torch.zeros(mx_num_feats).double(), dist_reduce_fx="sum")
            self.add_state("real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

            self.add_state("fake_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
            self.add_state("fake_features_cov_sum", torch.zeros(mx_num_feats).double(), dist_reduce_fx="sum")
            self.add_state("fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        def update(self, videos: Tensor, real: bool) -> None:
            features = get_fvd_logits(videos, self.i3d, self.device, bs=self.batch_size, crop_center=self.crop_center)
            self.orig_dtype = features.dtype
            features = features.double()

            if features.dim() == 1:
                features = features.unsqueeze(0)
            if real:
                self.real_features_sum += features.sum(dim=0)
                self.real_features_cov_sum += features.t().mm(features)
                self.real_features_num_samples += videos.shape[0]
            else:
                self.fake_features_sum += features.sum(dim=0)
                self.fake_features_cov_sum += features.t().mm(features)
                self.fake_features_num_samples += videos.shape[0]

        def compute(self) -> Tensor:
            if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
                raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")
            mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
            mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

            cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
            cov_real = cov_real_num / (self.real_features_num_samples - 1)
            cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
            cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
            return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(self.orig_dtype)

    print("Running video evaluation...")
    tasks = get_path(args)

    for task, (test_res_path, test_dir) in tasks.items():
        print(f"Evaluating task: {task}")
        dataloader = prepare_dataloader(args, ["video", "result"], test_res_path)

        image_metrics = {
            "geocalib": GeoCalibError(),
            "rho": UcmCameraRayAngularErrorRho(),
            "lpips": LearnedPerceptualImagePatchSimilarity(
                net_type="vgg",
                normalize=True,
            ),
            "psnr": PeakSignalNoiseRatio(
                data_range=1.,
                dim=(1, 2, 3)
            ),
            "ssim": StructuralSimilarityIndexMeasure(
                data_range=1.,
            ),
            "cs_text": CLIPScore(
                model_name_or_path="zer0int/LongCLIP-L-Diffusers",
            ),
            "cs_image": CLIPScore(),
        }
        image_metrics = {k: v.to(args.test_device) for k, v in image_metrics.items()}
        data_metrics = {
            "fvd_center": FrechetVideoDistance(),
            "fvd": FrechetVideoDistance(
                crop_center=False,
            ),
            "fid": FrechetInceptionDistance(
                normalize=True,
            ),
            "is": InceptionScore(
                normalize=True,
            )
        }
        data_metrics = {k: v.to(args.test_device) for k, v in data_metrics.items()}

        eval_results = defaultdict(list)
        for data in tqdm(dataloader, desc="Evaluating videos"):
            if "video" in data:
                gt_video = torch.from_numpy(data["video"]).to(args.test_device)  # [C, T, H, W]
                gt_video = rearrange(gt_video, "C T H W -> T C H W")  # [T, C, H, W]
                gt_video = gt_video / 2. + 0.5  # to [0, 1]

            video = torch.from_numpy(data["result"]).to(args.test_device)  # [C, T, H, W]
            video = rearrange(video, "C T H W -> T C H W")  # [T, C, H, W]
            video = video / 2. + 0.5  # to [0, 1]

            for metric_name, metric in image_metrics.items():
                if metric_name == "geocalib":
                    if "video" not in data:
                        continue
                    metric.update(
                        pred=video,
                        gt=gt_video,
                    )
                elif metric_name == "rho":
                    if "video" not in data:
                        continue
                    metric.update(
                        pred=video,
                        gt=gt_video,
                        x_fov=data["x_fov"],
                        xi=data["xi"],
                    )
                elif metric_name == "cs_text":
                    for pred in video.split(args.test_chunk_size, dim=0):
                        pred = pred * 255.0
                        metric.update(
                            pred.to(torch.uint8),
                            [data["caption"]] * len(pred),
                        )
                elif metric_name == "cs_image":
                    for pred, gt in zip(
                        video[:-1].split(args.test_chunk_size, dim=0),
                        video[1:].split(args.test_chunk_size, dim=0),
                    ):
                        pred, gt = pred * 255.0, gt * 255.0
                        metric.update(
                            pred.to(torch.uint8),
                            gt.to(torch.uint8),
                        )
                elif metric_name in ["lpips", "psnr", "ssim"]:
                    if "video" not in data:
                        continue
                    for pred, gt in zip(
                        video.split(args.test_chunk_size, dim=0),
                        gt_video.split(args.test_chunk_size, dim=0),
                    ):
                        metric.update(
                            pred.contiguous(),
                            gt
                        )
                else:
                    raise NotImplementedError(f"Image metric {metric_name} not implemented.")

                if metric_name in ("geocalib", "rho"):
                    results = metric.compute()
                    for key, value in results.items():
                        eval_results[key].append({
                            "video_id": data["video_id"],
                            "video_results": value.cpu().item(),
                        })
                else:
                    eval_results[metric_name].append({
                        "video_id": data["video_id"],
                        "video_results": metric.compute().cpu().item(),
                    })
                metric.reset()

            for metric_name, metric in data_metrics.items():
                if metric_name == "is":
                    for pred in video.split(args.test_chunk_size, dim=0):
                        metric.update(pred)
                if "video" not in data:
                    continue
                if metric_name == "fid":
                    for pred, gt in zip(
                        video.split(args.test_chunk_size, dim=0),
                        gt_video.split(args.test_chunk_size, dim=0),
                    ):
                        metric.update(pred, real=False)
                        metric.update(gt, real=True)
                elif metric_name in ("fvd", "fvd_center"):
                    metric.update(video.unsqueeze(0), real=False)
                    metric.update(gt_video.unsqueeze(0), real=True)

        for metric_name, metric in data_metrics.items():
            if not metric.update_called:
                continue
            if metric_name in ("fid", "fvd", "fvd_center"):
                eval_results[metric_name] = metric.compute().item()
            elif metric_name == "is":
                eval_results[metric_name], eval_results[f"{metric_name}_std"] = metric.compute()
                eval_results[metric_name] = eval_results[metric_name].cpu().item()
                eval_results[f"{metric_name}_std"] = eval_results[f"{metric_name}_std"].cpu().item()

        save_evaluation(args, test_dir, eval_results, "video_metrics")


def overall(args):
    tasks = get_path(args)
    eval_res_name = "last.json" if args.load_last else f"{args.test_name}_eval_results.json"
    for task, (_, test_dir) in tasks.items():
        overall_res = {}
        for key in ["qalign", "video_metrics", "pose"]:
            eval_res_path = test_dir / key / eval_res_name
            if not eval_res_path.exists():
                print(f"Evaluation results for {key} not found at {eval_res_path}. Skipping.")
                continue
            with open(eval_res_path, "r") as f:
                eval_res = json.load(f)
            for metric, values in eval_res.items():
                overall_res[f"{key}/{metric}"] = values[0]
        overall_res_path = test_dir / "overall" / f"{args.test_name}.json"
        overall_res_path.parent.mkdir(parents=True, exist_ok=True)
        with open(overall_res_path, "w") as f:
            json.dump(overall_res, f, indent=4)
        print(f"Overall evaluation results saved to {overall_res_path}")
        print(json.dumps(overall_res, indent=4))

        if args.save_last:
            link_last(overall_res_path)


def vipe(args):
    from einops import rearrange, repeat
    import ffmpeg
    import torch.nn.functional as F
    import src.camera_control as ucpe

    def rectify_ucm_to_pinhole(video, x_fov, xi, max_xfov=100.0):
        """
        UCM video → rectified pinhole video (undistortion)
        Args:
            video: torch.Tensor [T, C, H, W], dtype=float32, range [-1,1]
            x_fov: float, horizontal field of view (deg) in UCM model
            xi:    float, UCM mirror parameter
            max_xfov: float, limit effective horizontal FOV (deg)
        Returns:
            rectified: numpy array [T, H, W, 3], uint8, rectified pinhole video
        """

        T, C, H, W = video.shape
        device = video.device

        # Normalize input to [0,1]
        video = (video + 1.0) / 2.0

        # ---------- 1) UCM camera intrinsics ----------
        theta = torch.deg2rad(torch.tensor(x_fov / 2, device=device))
        # Limit maximal horizontal FOV (helps reduce distortion & black edges)
        max_theta = torch.deg2rad(torch.tensor(max_xfov / 2, device=device))
        theta_x = torch.min(theta, max_theta)

        # ---------- 2) Compute vertical FOV from UCM physical rays ----------
        d_cam = ucpe.ucm_unproject_grid_fov(
            x_fov=x_fov,
            xi=xi,
            height=H,
            width=W,
            device=device,
        )  # [H,W,3]

        mid_x = W // 2
        verts = d_cam[:, mid_x, :]  # sample center column rays

        # vertical angle wrt Z forward axis
        theta_y_rc = torch.atan2(
            torch.abs(verts[:, 1]),  # vertical component (Y)
            verts[:, 2].clamp(min=1e-8)  # forward Z
        )
        theta_y_eff = torch.max(theta_y_rc) * 0.98  # avoid edge overflow

        # ---------- 3) Target pinhole intrinsics ----------
        fx_p = fy_p = torch.max(
            (W * 0.5) / torch.tan(theta_x),
            (H * 0.5) / torch.tan(theta_y_eff)
        )
        cx_p = (W - 1) * 0.5
        cy_p = (H - 1) * 0.5

        # ✅ pinhole grid coordinates
        u = torch.linspace(0, W - 1, W, device=device)
        v = torch.linspace(0, H - 1, H, device=device)
        uu, vv = torch.meshgrid(u, v, indexing="xy")  # [W,H]

        X = (uu - cx_p) / fx_p
        Y = (vv - cy_p) / fy_p
        Z = torch.ones_like(X)

        # ---------- 5) Map pinhole rays → UCM pixels ----------
        du, dv = ucpe.project_ucm_points_fov(X, Y, Z, x_fov, xi, H, W)

        # grid normalize to [-1,1]
        grid_x = 2.0 * (du / (W - 1)) - 1.0
        grid_y = 2.0 * (dv / (H - 1)) - 1.0

        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H,W,2]
        grid = grid.unsqueeze(0).expand(T, -1, -1, -1)  # [T,H,W,2]

        # ---------- 6) Warp ----------
        rectified = F.grid_sample(
            video,
            grid,
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)

        rectified = (rectified * 255.0).byte()
        rectified = rectified.permute(0, 2, 3, 1).contiguous()  # [T,H,W,3]
        return rectified.cpu().numpy()

    print("Running Vipe pose generation...")
    tasks = get_path(args)

    for task, (test_res_path, test_dir) in tasks.items():
        print(f"Evaluating task: {task}")
        dataloader = prepare_dataloader(args, ["result"], test_res_path)
        rectify_res_path = test_res_path.parent / f"{test_res_path.name}_rectify"
        rectify_res_path.mkdir(parents=True, exist_ok=True)

        for data in tqdm(dataloader, desc="Exporting videos"):
            video = torch.from_numpy(data["result"]).to(args.test_device)  # [C, T, H, W]
            video = rearrange(video, "C T H W -> T C H W")  # [T, C, H, W]
            _, _, height, width = video.shape
            x_fov = data["x_fov"]
            xi = data["xi"]

            rectify_video = rectify_ucm_to_pinhole(video, x_fov, xi)

            out_file = rectify_res_path / f"{data['video_id']}.mp4"
            process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', framerate=data["fps"])
                .output(str(out_file), pix_fmt='yuv420p', vcodec='libx264', r=data["fps"], crf=16, preset='slow')
                .overwrite_output()
                .run_async(pipe_stdin=True, quiet=True)
            )
            process.stdin.write(rectify_video.tobytes())
            process.stdin.close()
            process.wait()
        
        vipe_path = test_dir / "vipe"
        cmd = [
            "conda", "run", "-n", "vipe",
            "--no-capture-output",
            "python", "/mnt/pfs/users/zhangchen/panshot/UCPE/thirdparty/vipe/run.py",
            "pipeline=default",
            "streams=raw_mp4_stream",
            f"streams.base_path={rectify_res_path}",
            f"pipeline.output.path={vipe_path}",
            "pipeline.output.save_artifacts=true",
            "pipeline.post.depth_align_model=null",
        ]
        print(f"[CMD] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def pose(args):
    from torchmetrics import MeanMetric, Metric
    from einops import rearrange, repeat

    print("Running pose evaluation...")
    tasks = get_path(args)

    if not args.evaluate_gt and args.valid_pose_percent < 1.0:
        pose_eval_path = args.data_root / "evaluate" / "pose" / "last.json"
        if not pose_eval_path.exists():
            print(f"GT pose evaluation results not found at {pose_eval_path}. Cannot limit to valid poses.")
            valid_video_ids = None
        else:
            with open(pose_eval_path, "r") as f:
                gt_eval_res = json.load(f)
            cammc = {v["video_id"]: v["video_results"] for v in gt_eval_res["cammc"][1]}
            sorted_videos = sorted(cammc.items(), key=lambda x: x[1])
            sorted_videos = sorted_videos[:int(len(sorted_videos) * args.valid_pose_percent)]
            valid_video_ids = set(v[0] for v in sorted_videos)
    else:
        valid_video_ids = None


    def normalize_t(rt):
        # normalize translation by max-norm within the same trajectory
        t = rt[:, :3, 3]
        scale = np.max(np.linalg.norm(t, axis=-1)) + 1e-9
        rt[:, :3, 3] /= scale
        return rt

    def relative_pose(rt):
        # C2W → relative to first frame
        rel = np.zeros_like(rt)
        rel[0] = np.eye(4)
        inv0 = np.linalg.inv(rt[0])
        rel[1:] = inv0 @ rt[1:]
        return rel

    def calc_rot_err(r1, r2):
        # r1, r2: (T, 3, 3)
        R = np.matmul(np.transpose(r1, (0,2,1)), r2)
        trace = np.trace(R, axis1=-2, axis2=-1)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))  # (T)
        return np.sum(angle)

    def calc_trans_err(t1, t2):
        return np.sum(np.linalg.norm(t1 - t2, axis=-1))

    def calc_cammc(rt1, rt2):
        # flatten camera motion difference
        diff = (rt2 - rt1).reshape(rt1.shape[0], -1)
        return np.sum(np.linalg.norm(diff, axis=-1))

    for task, (test_res_path, test_dir) in tasks.items():
        print(f"Evaluating task: {task}")
        vipe_path = test_dir / "vipe"
        vipe_pose_path = vipe_path / "pose"
        vipe_video_ids = set(p.stem for p in vipe_pose_path.glob("*.npz"))
        # if valid_video_ids is not None:
        #     vipe_video_ids = vipe_video_ids.intersection(valid_video_ids)
        #     print(f"Evaluating {len(vipe_video_ids)} valid videos with high-quality GT poses.")
        dataloader = prepare_dataloader(args, ["pose"], video_ids=vipe_video_ids)

        eval_results = defaultdict(list)
        for data in tqdm(dataloader, desc="Evaluating poses"):
            gt_c2w = data["pose"]
            last_row = repeat(np.array([0,0,0,1], dtype=gt_c2w.dtype), "n -> t 1 n", t=gt_c2w.shape[0])
            gt_c2w = np.concatenate([gt_c2w, last_row], axis=-2)  # (T, 4, 4)

            pred_c2w = np.load(vipe_pose_path / f"{data['video_id']}.npz")["data"]  # (T, 4, 4)

            if args.frame_stride is not None:
                gt_c2w = gt_c2w[::args.frame_stride]
                pred_c2w = pred_c2w[::args.frame_stride]

            if args.pose_frames is not None:
                gt_c2w = gt_c2w[:args.pose_frames]
                pred_c2w = pred_c2w[:args.pose_frames]

            # Relative + translation normalized
            gt_rel = normalize_t(relative_pose(gt_c2w.copy()))
            pred_rel = normalize_t(relative_pose(pred_c2w.copy()))

            # Metrics
            rot_err = calc_rot_err(gt_rel[:, :3, :3], pred_rel[:, :3, :3])
            trans_err = calc_trans_err(gt_rel[:, :3, 3], pred_rel[:, :3, 3])
            cammc = calc_cammc(gt_rel[:, :3, :4], pred_rel[:, :3, :4])

            results = {
                "rot_err": rot_err,
                "trans_err": trans_err,
                "cammc": cammc,
            }

            vipe_gt_path = args.data_root / "evaluate" / "vipe" / "pose"
            if not args.evaluate_gt and vipe_gt_path.exists() \
                    and valid_video_ids is not None and data["video_id"] in valid_video_ids:
                gt_c2w = np.load(vipe_gt_path / f"{data['video_id']}.npz")["data"]  # (T, 4, 4)
                gt_rel = normalize_t(relative_pose(gt_c2w.copy()))

                # Metrics
                rot_err = calc_rot_err(gt_rel[:, :3, :3], pred_rel[:, :3, :3])
                trans_err = calc_trans_err(gt_rel[:, :3, 3], pred_rel[:, :3, 3])
                cammc = calc_cammc(gt_rel[:, :3, :4], pred_rel[:, :3, :4])
                results.update({
                    "rot_err_vipe": rot_err,
                    "trans_err_vipe": trans_err,
                    "cammc_vipe": cammc,
                })

            for key, value in results.items():
                eval_results[key].append({
                    "video_id": data["video_id"],
                    "video_results": float(value),
                })

        save_evaluation(args, test_dir, eval_results, "pose")


def main():
    args = Args()

    for step in args.test_steps:
        if args.conda_envs and step in args.conda_envs:
            conda_env = args.conda_envs[step]
            print(f"[INFO] Running step '{step}' in conda env: {conda_env}")

            # 当前脚本路径
            script_path = Path(__file__).resolve()
            script_path = script_path.relative_to(Path.cwd())

            # 构造命令：使用 conda run 调用
            cmd = [
                "conda", "run", "-n", conda_env,
                "--no-capture-output",
                "python", str(script_path),
                f"--test_steps=[{step}]",   # 只运行该 step
                "--conda_envs={}",        # 避免递归调用 conda
            ]

            # 把其他命令行参数透传下去
            # 注意：Args 使用了 pydantic-settings + tyro 等 CLI 解析工具，
            # 你可以根据需要加上传入的 CLI 参数，这里简化为当前 sys.argv
            extra_args = sys.argv[1:]
            cmd.extend(extra_args)

            print(f"[CMD] {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        else:
            globals()[step](args)


if __name__ == "__main__":
    main()
