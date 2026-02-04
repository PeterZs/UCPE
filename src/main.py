import lightning as pl
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from jsonargparse import lazy_instance
from torch.utils.data import DataLoader
from pathlib import Path
import torch
from diffsynth.pipelines.wan_video_panshot import WanVideoPipeline, ModelConfig
import wandb
import os
from src.dataset import PanShotDataset, Re10kDataset, DemoDataset
from diffsynth import save_video
from src.camera_control import patch_dit, enable_grad
from typing import Literal
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import numpy as np
from tqdm.auto import tqdm
from typing import Optional


class PanShotDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: Path = Path("data/UCPE"),
        batch_size: int = 1,
        num_workers: int = 4,
        zero_first_yaw: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.test_load_keys = ["video", "pose"]

    def setup(self, stage):
        self.hparams.model_id = self.trainer.model.hparams.model_id

    def train_dataloader(self):
        dataset = PanShotDataset(self.hparams, split="train", load_keys=["cache", "pose"])
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        dataset = PanShotDataset(self.hparams, split="test", load_keys=self.test_load_keys)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        dataset = PanShotDataset(self.hparams, split="test", load_keys=self.test_load_keys)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.hparams.num_workers)

    def predict_dataloader(self):
        dataset = PanShotDataset(self.hparams, split="test", load_keys=self.test_load_keys)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.hparams.num_workers)


class Re10kDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: Path = Path("data/RealEstate10k"),
        batch_size: int = 1,
        num_workers: int = 4,
        overwrite_xfov: float = 100.0,
    ):
        super().__init__()
        self.save_hyperparameters()

    @rank_zero_only
    def normalize_traj(self, normalization_file):
        tgt_data = PanShotDataModule()
        tgt_data.test_load_keys = ["pose"]
        tgt_dataloader = tgt_data.predict_dataloader()
        src_dataloader = self.predict_dataloader()
        traj_length_mean = []
        for dataloader in (tgt_dataloader, src_dataloader):
            traj_length_sum = 0.
            traj_num = 0
            for data in tqdm(dataloader, desc="Calculating trajectory length"):
                pose = data["pose"]  # (B, T, 3, 4)
                traj = pose[..., 3]  # (B, T, 3)
                traj_length = torch.sum(torch.linalg.norm(traj[:, 1:] - traj[:, :-1], dim=-1), dim=-1)  # (B,)
                traj_length_sum += traj_length.sum().item()
                traj_num += traj.shape[0]
            traj_length_mean.append(traj_length_sum / traj_num)
        normalize_traj = traj_length_mean[0] / traj_length_mean[1]
        print(f"Trajectory length normalization factor: {normalize_traj}")
        Path(normalization_file).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(normalization_file, np.array([normalize_traj], dtype=np.float32))

    def setup(self, stage):
        self.hparams.num_frames = self.trainer.model.hparams.num_frames        
        normalization_file = Path(self.hparams.data_root) / "traj_normalization.txt"
        if not normalization_file.exists():
            self.normalize_traj(normalization_file)
        self.trainer.strategy.barrier()
        self.hparams.normalize_traj = float(np.loadtxt(normalization_file))

    def predict_dataloader(self):
        dataset = Re10kDataset(self.hparams, split="test")
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)


class DemoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        panshot_data_root: Path = Path("data/UCPE"),
        re10k_data_root: Path = Path("data/RealEstate10k"),
        input_file: Path = Path("demo/teaser.json"),
        batch_size: int = 1,
        num_workers: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage):
        self.hparams.num_frames = self.trainer.model.hparams.num_frames        
        normalization_file = Path(self.hparams.re10k_data_root) / "traj_normalization.txt"
        self.hparams.re10k_normalize_traj = float(np.loadtxt(normalization_file))

    def predict_dataloader(self):
        dataset = DemoDataset(self.hparams)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)


class PanShotTrainModule(pl.LightningModule):
    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        learning_rate: float = 1e-4,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
        ckpt_path: Path = None,
        fps: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        tiled: bool = False,
        camera_condition: str = "relray_absmap",
        adaptation_method: Literal[
            "before",
            "after",
            "parallel",
        ] = "parallel",
        ti2v_input_image_prob: float = 0.5,
        attn_compress: int = 8,
        num_predict: Optional[int] = None,
    ):
        super().__init__()
        file_patterns = [
            "models_t5_umt5-xxl-enc-bf16.pth",
            "diffusion_pytorch_model*.safetensors",
            "Wan2.1_VAE.pth",
        ]
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=[
                ModelConfig(model_id=model_id, origin_file_pattern=pattern, offload_device="cpu")
                for pattern in file_patterns
            ]
        )

        keywords = patch_dit(
            self.pipe, camera_condition, height, width,
            attn_compress=attn_compress, adaptation_method=adaptation_method
        )
        enable_grad(self.pipe, keywords)

        self.strict_loading = False
        if ckpt_path is not None:
            print(f"Loading weights from {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        self.save_hyperparameters()

    def setup(self, stage=None):
        self.pipe.device = self.device

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        for i in range(self.hparams.num_predict or 1):
            video = self(batch, seed=i)
            is_i2v = "input_image" in batch and self.pipe.dit.fuse_vae_embedding_in_latents
            video_folder = "i2v" if is_i2v else "t2v"
            if isinstance(self.trainer.datamodule, PanShotDataModule):
                split = "predict"
            elif isinstance(self.trainer.datamodule, DemoDataModule):
                split = "demo"
            else:
                split = Path(self.trainer.datamodule.hparams.data_root).name
            self.save_output(
                video,
                batch,
                split=split,
                video_folder=video_folder,
                quality=8,
                suffix=f"-{i}" if self.hparams.num_predict else None
            )
            if is_i2v:
                del batch["input_image"]
                self.predict_step(batch, batch_idx, dataloader_idx)

    def save_output(self, video, batch, split, video_folder, step=None, quality=5, suffix=None):
        video_id = batch["video_id"][0]

        video_prefix = os.path.join(self.logger.save_dir, split, video_folder, video_id)
        if step is not None:
            video_prefix = f"{video_prefix}-{step}"
        if suffix is not None:
            video_prefix = video_prefix + suffix
        video_path = video_prefix + ".mp4"
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        save_video(video, video_path, fps=self.hparams.fps, quality=quality)

        reference_path = os.path.join(self.logger.save_dir, split, "reference", f"{video_id}.mp4")
        os.makedirs(os.path.dirname(reference_path), exist_ok=True)
        if not os.path.exists(reference_path) and "video" in batch:
            reference_video = self.pipe.vae_output_to_video(batch["video"])
            save_video(reference_video, reference_path, fps=self.hparams.fps, quality=quality)

        caption_path = os.path.join(self.logger.save_dir, split, "caption", f"{video_id}.txt")
        os.makedirs(os.path.dirname(caption_path), exist_ok=True)
        if not os.path.exists(caption_path):
            with open(caption_path, "w") as f:
                f.write(batch["caption"][0])

        print(f"Saved video to {video_path}")

        return video_path, reference_path

    def forward(self, batch, seed=None):
        video = self.pipe(
            prompt=batch["caption"][0],
            input_image=batch.get("input_image", None),
            camera_control_panshot={k: batch[k] for k in ["pose", "xi", "x_fov"]},
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            num_inference_steps=self.hparams.num_inference_steps,
            tiled=self.hparams.tiled,
            seed=seed,
            height=self.hparams.height,
            width=self.hparams.width,
            num_frames=self.hparams.num_frames,
        )
        return video

    def on_fit_start(self):
        if self.trainer.is_global_zero and hasattr(self.logger, "experiment"):
            self.logger.experiment.watch(self, log_graph=False, log_freq=1000)

    def training_step(self, batch, batch_idx):
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Data
        _, _, length, height, width = batch["input_latents"].shape
        num_frames = (length - 1) * 4 + 1
        height = height * self.pipe.vae.upsampling_factor
        width = width * self.pipe.vae.upsampling_factor
        inputs_posi = {}
        inputs_nega = {}
        inputs_shared = {
            "camera_control_panshot": {k: batch[k] for k in ["pose", "xi", "x_fov"]},
            "input_latents": batch["input_latents"],
            "context": batch["context"],
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": 1,
            "rand_device": self.device,
            "use_gradient_checkpointing": self.hparams.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.hparams.use_gradient_checkpointing_offload,
            "cfg_merge": False,
        }

        if "first_frame_latents" in batch \
            and self.pipe.dit.fuse_vae_embedding_in_latents \
                and torch.rand(1).item() < self.hparams.ti2v_input_image_prob:
            inputs_shared["first_frame_latents"] = batch["first_frame_latents"]

        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        inputs = {**inputs_shared, **inputs_posi}

        # Compute loss
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)

        # Record log
        self.log("train/loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        video = self(batch, seed=0)
        is_i2v = "input_image" in batch and self.pipe.dit.fuse_vae_embedding_in_latents
        video_folder = "i2v" if is_i2v else "t2v"
        video_path, reference_path = self.save_output(
            video, batch, split="validation", video_folder=video_folder, step=self.global_step)
        log_dict = self.visualize(video_path, reference_path, batch)
        log_dict = {f"val/{k}": v for k, v in log_dict.items()}
        self.logger.experiment.log(log_dict)
        if is_i2v:
            del batch["input_image"]
            self.validation_step(batch, batch_idx)

    def visualize(self, video_path, reference_path, batch):
        log_dict = {}
        log_dict["video"] = wandb.Video(
            video_path,
            caption=batch["caption"][0],
            format="mp4",
        )
        log_dict["reference"] = wandb.Video(
            reference_path,
            caption=batch["video_id"][0],
            format="mp4",
        )
        return log_dict

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.dit.parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.hparams.learning_rate)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        for key in list(checkpoint["state_dict"].keys()):
            if not key.startswith("pipe.dit."):
                del checkpoint["state_dict"][key]


class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        parser.set_defaults({
            # "data": "PanShotDataModule",
            "checkpoint.dirpath": os.path.join(self.trainer_defaults["default_root_dir"], "checkpoints"),
            "checkpoint.save_last": True,
            # "checkpoint.every_n_train_steps": 10000,
            # "checkpoint.every_n_epochs": 1,
        })


def main():
    torch.set_float32_matmul_precision('high')

    wandb_id = os.environ.get("WANDB_RUN_ID", wandb.util.generate_id())
    exp_dir = os.path.join("logs", wandb_id)
    wandb_logger = lazy_instance(
        WandbLogger,
        # entity="pidan1231239",
        project="ucpe",
        id=wandb_id,
        save_dir=exp_dir,
        resume="allow",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    cli = MyCLI(
        model_class=PanShotTrainModule,
        # datamodule_class=PanShotDataModule,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf", "default_env": True},
        seed_everything_default=int(os.environ.get("LOCAL_RANK", 0)),
        trainer_defaults={
            "accelerator": "gpu",
            "devices": "auto",
            "strategy": "deepspeed_stage_1",
            "log_every_n_steps": 10,
            "num_sanity_val_steps": 1,
            "limit_train_batches": 1000,
            "limit_val_batches": 3,
            # "limit_predict_batches": 10,
            "limit_test_batches": 10,
            "benchmark": True,
            "max_epochs": 10,
            # "accumulate_grad_batches": 16,
            "precision": "bf16-true",
            "callbacks": [lr_monitor],
            "logger": wandb_logger,
            "default_root_dir": exp_dir,
        },
        
    )


if __name__ == "__main__":
    main()
