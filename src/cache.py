import lightning as pl
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torch
import lightning as pl
from diffsynth.pipelines.wan_video_panshot import WanVideoPipeline, ModelConfig
from types import SimpleNamespace
from src.dataset import PanShotDataset



class PanShotDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: Path = Path("data/UCPE"),
        batch_size: int = 1,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage):
        self.hparams.model_id = self.trainer.model.hparams.model_id
        load_keys = ["video"]
        self.dataset = PanShotDataset(self.hparams, split="train", load_keys=load_keys, skip_cached=True)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)


class PanShotCacheModule(pl.LightningModule):
    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    ):
        super().__init__()
        model_configs=[
            ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="Wan2.1_VAE.pth"
            ),
        ]
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
        )
        self.pipe.dit = SimpleNamespace(
            require_vae_embedding=True,
            require_clip_embedding=True,
            fuse_vae_embedding_in_latents=False,
        )
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.save_hyperparameters()

    def test_step(self, batch, batch_idx):
        text, video, video_id = batch["caption"][0], batch["video"], batch["video_id"][0]
        self.pipe.device = self.device
        pth_path = self.trainer.datamodule.dataset.cache_folder / f"{video_id}.pth"
        if pth_path.exists():
            return
        pth_path.parent.mkdir(parents=True, exist_ok=True)
        _, _, num_frames, height, width = video.shape
        inputs_posi = {"prompt": text}
        inputs_nega = {}
        inputs_shared = {
            "input_video": video,
            "input_image": batch.get("input_image", None),
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": None,
            "use_gradient_checkpointing": False,
            "use_gradient_checkpointing_offload": False,
            "cfg_merge": False,
        }
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        inputs = {**inputs_shared, **inputs_posi}
        data = {k: inputs[k][0] for k in ["input_latents", "context", "first_frame_latents"] if k in inputs}
        torch.save(data, pth_path)


def main():
    cli = LightningCLI(
        model_class=PanShotCacheModule,
        datamodule_class=PanShotDataModule,
        seed_everything_default=42,
        run=False,
        trainer_defaults={
            "precision": "bf16-true",
            "logger": False,
        },
        save_config_callback=None,
    )
    trainer = cli.trainer
    model = cli.model
    datamodule = cli.datamodule

    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
