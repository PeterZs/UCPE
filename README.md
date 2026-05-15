# 📷 UCPE

<p align="center">
<h1 align="center">Unified Camera Positional Encoding for Controlled Video Generation</h1>
<p align="center">
  <p align="center">
    <a href="https://chengzhag.github.io/">Cheng Zhang</a><sup>1</sup><sup>,2</sup>
    ·
    <a href="https://leeby68.github.io/">Boying Li</a><sup>1</sup>
    ·
    <a href="https://www.linkedin.com/in/meng-wei-66687a105/?originalSubdomain=au">Meng Wei</a><sup>1</sup>
    ·
    <a href="https://yanpei.me/">Yan-Pei Cao</a><sup>3</sup>
    ·
    <a href="https://www.monash.edu/mada/architecture/people/camilo-cruz-gambardella/">Camilo Cruz Gambardella</a><sup>1,2</sup>
    ·
    <a href="https://research.monash.edu/en/persons/dinh-phung/">Dinh Phung</a><sup>1</sup>
    ·
    <a href="https://jianfei-cai.github.io/">Jianfei Cai</a><sup>1</sup><br>
    <sup>1</sup>Monash University <sup>2</sup>Building 4.0 CRC <sup>3</sup>VAST
  </p>
  <h2 align="center"><a href="https://arxiv.org/abs/2512.07237">Paper</a> | <a href="https://chengzhag.github.io/publication/ucpe/">Project Page</a> | <a href="https://youtu.be/rMX7gxH8jBM">Video</a> | <a href="https://huggingface.co/datasets/chengzhag/PanShot">Hugging Face</a></h2>
</p>

[![Watch the video](images/thumbnail.png)](https://youtu.be/rMX7gxH8jBM)
*Our UCPE introduces a geometry-consistent alternative to Plücker rays as one of the core contributions, enabling better generalization in Transformers. We hope to inspire future research on camera-aware architectures.

## 📢 Updates
- \[2026.05.14\] 🔥 **UCPE is used in [SANA-WM](https://nvlabs.github.io/Sana/WM/)**
- \[2026.04.12\] 📦 **Raw 4K Panoramic Videos** released on [HuggingFace](https://huggingface.co/datasets/chengzhag/UCPE) — skips CameraBench and PanFlow curation; also provides ERP videos for PanShot (YouTube now serves 360° videos in EAC format, breaking the original download script).
- \[2026.03.19\] 🔧 Fixed a bug in Plücker encoding (thanks to [@fengq1a0](https://github.com/fengq1a0)'s [issue #5](https://github.com/chengzhag/UCPE/issues/5)).
- \[2026.02.21\] 🎉 **UCPE accepted to CVPR 2026**
- \[2026.02.04\] 📁 **PanShot Dataset And Curation Code** (controllable camera data synthesized from [PanFlow](https://github.com/chengzhag/PanFlow))
- \[2026.02.04\] 🎯 **Full Training, Evaluation, Visualization Code**
- \[2025.12.07\] ⚡ **Quick Demo** code released

## 🚀 TLDR

🔥 **Camera-controlled text-to-video generation**, now with **intrinsics**, **distortion** and **orientation** control!

<p align="center">
  <img src="images/cameras.png" alt="Camera lenses" height="120px">
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="images/orientation.png" alt="Orientation control" height="140px">
</p>

📷 UCPE integrates **Relative Ray Encoding**—which delivers significantly better generalization than Plücker across diverse camera motion, intrinsics and lens distortions—with **Absolute Orientation Encoding** for controllable pitch and roll, enabling a unified camera representation for Transformers and state-of-the-art camera-controlled video generation with just **0.5% extra parameters** (35.5M over the 7.3B parameters of the base model)

<p align="center">
  <img src="images/video-ucpe.gif"
       alt="UCPE"
       style="max-height:480px; width:auto;">
</p>

## 🛠️ Installation

```bash
conda create -n UCPE python=3.11 -y
conda activate UCPE
conda install -c conda-forge "ffmpeg<8" libiconv libgl -y
pip install -r requirements.txt
pip install --no-build-isolation --no-cache-dir flash-attn==2.8.0.post2
pip install -e .

cd thirdparty/equilib
pip install -e .
```

We use wandb to log and visualize the training process. You can create an account then login to wandb by running the following command:

```bash
wandb login
```

<details>
<summary>Below are installations for tools used in evaluation and dataset processing
that can be skipped if you do not need these tools.</summary>

```bash
cd ../GeoCalib
pip install -e .
pip install -e siclib

cd ../UniK3D
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121

cd ../Q-Align
conda create -n qalign python=3.9 -y
conda activate qalign
pip install -e .
pip install jsonlines "numpy<2" protobuf pydantic-settings

cd ../vipe
conda env create -f envs/base.yml
conda activate vipe
pip install -r envs/requirements.txt
pip install --no-build-isolation -e .
```
</details>
<br>

## ⚡ Quick Demo

Download our finetuned weights from [OneDrive](https://monashuni-my.sharepoint.com/:f:/g/personal/cheng_zhang_monash_edu/IgCoTNrYOJRJRKtk5A6I1yiCAR9c64-BOrsId5GYsUxE9y4?e=hD26qU) and put it in `logs/` folder. Then run:

```bash
bash scripts/demo.sh
```

The generated videos will be saved in `logs/6wodf04s/demo`, examples shown below:

* `demo/lens.json`: Our **Relative Ray Encoding** not only generalizes to but also enables controllability over a wide range of camera intrinsics and lens distortions.

<p align="center">
  <img src="images/video-lens.gif"
       alt="Lens control"
       style="max-height:480px; width:auto;">
</p>

* `demo/pose.json`: The geometry-consistent design of **Relative Ray Encoding** further allows strong generalization and controllability over diverse camera motions.

<p align="center">
  <img src="images/video-pose.gif"
       alt="Pose control"
       style="max-height:480px; width:auto;">
</p>

* `demo/teaser.json`: Our **Absolute Orientation Encoding** further eliminate the ambiguity in pitch and roll in previous T2V methods, enabling precise control over initial camera orientation.

<p align="center">
  <img src="images/video-orientation.gif"
       alt="Orientation control"
       style="max-height:480px; width:auto;">
</p>


## 🌏 PanShot Dataset

Please download the PanShot dataset from [Hugging Face](https://huggingface.co/datasets/chengzhag/PanShot) to `data/UCPE/PanShot-7z` by:

```bash
huggingface-cli download chengzhag/PanShot --repo-type dataset --local-dir data/UCPE/PanShot-7z
```

Then extract the dataset by:
```bash
cd data/UCPE/PanShot-7z
bash extract_panshot.sh
cd ../../..
```
The extracted dataset will be saved in `data/UCPE/PanShot`.
Please then copy the other files to form the following folder structure:

```
├── captioned-test.jsonl
├── captioned-train.jsonl
├── max_rotation-test.json
├── meta-test
├── meta-train
├── pose-test
├── pose-train
├── videos-test
└── videos-train
```

<details>
<summary>If you want to go through the dataset curation process, Please follow these three steps.</summary>

> **Shortcut for steps 1 & 2:** You can skip the CameraBench and PanFlow curation steps by downloading our pre-processed data directly:
> ```bash
> huggingface-cli download --repo-type dataset chengzhag/UCPE --local-dir data/UCPE
> cd data/UCPE && bash unpack_hf.sh && cd ../..
> ```
> Note: step 3 (PanShot) still depends on the PanFlow dataset's videos and slam_poses, so you'll also need to download those following the instructions in the [PanFlow](#panflow) section below — only the processing scripts can be skipped.

### CameraBench

Download the dataset from multiple sources:

```bash
cd data
huggingface-cli download --repo-type dataset syCen/CameraBench --local-dir CameraBench
cd CameraBench
huggingface-cli download --repo-type dataset syCen/Videos4CameraBnech --local-dir data/videos
wget https://huggingface.co/datasets/chancharikm/cambench_train_videos/resolve/main/videos.zip
unzip videos.zip -d videos
cd ../..
```

Process the dataset:

```bash
conda activate UCPE
python tools/process_camerabench.py  # set split = "train" and split = "test"

conda activate vipe
cd thirdparty/vipe
python thirdparty/vipe/run.py pipeline=default streams=raw_mp4_stream streams.base_path=data/UCPE/CameraBench/videos/ pipeline.output.path=data/UCPE/CameraBench/vipe/ pipeline.output.save_artifacts=true pipeline.post.depth_align_model=null

conda activate UCPE
python tools/geocalib_camerabench.py
python tools/filter_camerabench.py
```

Processed dataset will be saved in `data/UCPE/CameraBench`.

### PanFlow

Download the pretrained model `PanoFlow(RAFT)-wo-CFE.pth` of Panoflow at [weiyun](https://share.weiyun.com/SIpeQTNE), then put it in `models/PanoFlow` folder.

Our PanShot dataset is built upon [PanFlow](https://github.com/chengzhag/PanFlow) dataset's videos and slam_poses. Please download follow their [instructions](https://github.com/chengzhag/PanFlow/tree/main/curation#download-data) on how to download the full videos and download their meta and slam_poses files following [Full Dataset](https://github.com/chengzhag/PanFlow/tree/main#-full-dataset).

Then process the dataset with:

```bash
conda activate UCPE
python tools/filter_panflow.py

conda activate qalign
python tools/score_panflow.py

conda activate UCPE
python tools/align_panflow.py  # set split = "train" and split = "test"
python tools/match_panflow.py  # set split = "train" and split = "test"
python tools/normalize_panflow.py  # set split = "train" and split = "test"
```


### PanShot

> **Note:** YouTube recently changed its 360° video format from ERP (Equirectangular Projection) to EAC (Equi-Angular Cubemap). As a result, the video download part in `process_panshot.py` no longer works. Use the **Shortcut above** to download our pre-processed ERP videos first — `process_panshot.py` will then automatically skip the download step and proceed with the remaining processing.

Export your YouTube cookies to `~/.config/cookies.txt` in Netscape format for 4k download. Then download and process the dataset:

```bash
conda activate UCPE
python tools/process_panshot.py  # set split = "train" and split = "test"
python tools/caption_panshot.py  # set split = "train" and split = "test"
```

</details>
<br>

## 🏡 RealEstate10k Dataset

We use RealEstate10k Dataset for evaluation, so only poses and captions are needed. Plesae download the RealEstate10k poses from the official [website](https://google.github.io/realestate10k/) ([RealEstate10K.tgz](https://storage.cloud.google.com/realestate10k-public-files/RealEstate10K.tar.gz)) and unpack it to `data/RealEstate10k` folder. Then download the captions from [CameraCtrl](https://github.com/hehao13/CameraCtrl) ([train](https://drive.google.com/file/d/1nytBYjTa0bJ-8AMJWVCtKT2XwkJR3Jra/view) and [test](https://drive.google.com/file/d/1AGEJYbfip0jcp-ymgU9uCjUHzqETivYP/view))

The final folder structure should be like this:
```
├── captions
│   ├── test.json
│   └── train.json
├── pose_files
│   ├── test
│   └── train
└── traj_normalization.txt
```

## 🎯 Training and Evaluation

Prepare the latent cache and train the model with:

```bash
python src/cache.py
bash scripts/train.sh
```

We used 8 A800 GPUs for training, which takes about 1 day. You'll get a WANDB_RUN_ID (e.g., `6wodf04s`) after starting the training. The logs will be synced to your wandb account and the checkpoints will be saved in `logs/<WANDB_RUN_ID>/checkpoints/`. You can use other commented settings in `scripts/train.sh` for ablation studies and baselines.

For evaluation, first download the pretrained model `i3d_pretrained_400.pt` in [common_metrics_on_video_quality](https://github.com/JunyaoHu/common_metrics_on_video_quality/blob/main/fvd/videogpt/i3d_pretrained_400.pt), then put it in `models/FVD` folder. Evaluate results with:

```bash
bash scripts/evaluate.sh
```

Please change the `WANDB_RUN_ID` in `scripts/evaluate.sh` on your own trained model and check other commented settings for ablation studies and baselines.
We note that there are some jitters in the synthesized videos due to inaccurate ViPE pose estimation. Therefore, our evaluation script uses the filtered RealEstate10k test set to avoid those cases.


## 🔧 Tools

<details>
<summary>We also provide tools for visualizing camera trajectories, exporting figures and tables for paper, and visualizing camera statistics.</summary>

Visualize camera trajectories:

```bash
# Export static camera trajectory visualizations
python -m tools.visualize_panshot --out_path=data/UCPE/PanShot/pose_vis-test/ --zero_first_yaw
python -m tools.visualize_re10k --pose_file_path=data/RealEstate10k/pose_files/test/ --filter_file=data/RealEstate10k/filter_files/filter_test_81.txt --relative_c2w --num_videos=150 --out_path=data/RealEstate10k/pose_vis/test/

# Export animated camera trajectory visualizations
python -m tools.visualize_panshot --out_path=data/UCPE/PanShot/pose_anim-test/ --zero_first_yaw --animate_camera
python -m tools.visualize_re10k --pose_file_path=data/RealEstate10k/pose_files/test/ --filter_file=data/RealEstate10k/filter_files/filter_test_81.txt --relative_c2w --num_videos=150 --out_path=data/RealEstate10k/pose_anim/test/ --animate_camera
```

Export figures for paper:

```bash
# Teaser figure
python -m tools.export_figure \
    --methods \
    "UCPE" "logs/6wodf04s/demo/t2v" \
    --input_file \
    "demo/teaser.json" \
    --output_dir \
    "outputs/figures/teaser" \
    --animate_latup

# Try other demo configs
    # "demo/pose.json" \
    # "demo/lens.json" \

# Comparison on PanShot dataset
python -m tools.export_figure \
    --data=PanShotDataset \
    --data_root="data/UCPE" \
    --methods \
    "ReCamMaster" "logs/khnmur4b/predict/t2v" \
    "Wan CameraCtrl" "logs/9hjx47bc/predict/t2v" \
    "UCPE" "logs/6wodf04s/predict/t2v" \
    --output_dir \
    "outputs/figures/panshot" \
    --sample_frames=3 \
    --animate_latup

# Comparison on RealEstate10k dataset
python -m tools.export_figure \
    --data=Re10kDataset \
    --data_root="data/RealEstate10k" \
    --methods \
    "ReCamMaster" "logs/lg1mxf9u/RealEstate10k/t2v" \
    "Wan CameraCtrl" "logs/3yf7psvi/RealEstate10k/t2v" \
    "CameraCtrl" "/mnt/pfs/users/zhangchen/panshot/CameraCtrl/out/re10k" \
    "AC3D" "/mnt/pfs/users/zhangchen/panshot/ac3d/out/5B/test/10000" \
    "UCPE" "logs/coo9rjaq/RealEstate10k/t2v" \
    --output_dir \
    "outputs/figures/re10k" \
    --sample_frames=3
```

Export table for paper:

```bash
# Comparison on PanShot (w/o Absolute Orientation Control)
python -m tools.export_table \
    --pad_cols 1 \
    --methods \
    "ReCamMaster" "logs/lg1mxf9u/predict/evaluate_t2v/overall/last.json" \
    "Wan CameraCtrl" "logs/3yf7psvi/predict/evaluate_t2v/overall/last.json" \
    "UCPE" "logs/coo9rjaq/predict/evaluate_t2v/overall/last.json" \
    --metrics \
    "video_metrics/vfov_err" "video_metrics/k1_err" "video_metrics/k2_err" \
    "video_metrics/pitch_err" "video_metrics/roll_err" \
    "pose/rot_err" "pose/trans_err" "pose/cammc" \
    "video_metrics/fvd" "video_metrics/fid" \
    "video_metrics/cs_text"

# Comparison on PanShot (w/ Absolute Orientation Control)
python -m tools.export_table \
    --pad_cols 1 \
    --methods \
    "ReCamMaster" "logs/khnmur4b/predict/evaluate_t2v/overall/last.json" \
    "Wan CameraCtrl" "logs/9hjx47bc/predict/evaluate_t2v/overall/last.json" \
    "UCPE" "logs/6wodf04s/predict/evaluate_t2v/overall/last.json" \
    --metrics \
    "video_metrics/vfov_err" "video_metrics/k1_err" "video_metrics/k2_err" \
    "video_metrics/pitch_err" "video_metrics/roll_err" \
    "pose/rot_err" "pose/trans_err" "pose/cammc" \
    "video_metrics/fvd" "video_metrics/fid" \
    "video_metrics/cs_text"

# Ablation Study on PanShot
python -m tools.export_table \
    --pad_cols 1 \
    --methods \
    "1/2-dim (\$128 \times 6\$)" "logs/r0hmwcag/predict/evaluate_t2v/overall/last.json" \
    "1/4-dim (\$128 \times 3\$)" "logs/nv4al3mj/predict/evaluate_t2v/overall/last.json" \
    "1/8-dim (\$192 \times 1\$)" "logs/6wodf04s/predict/evaluate_t2v/overall/last.json" \
    "1/12-dim (\$128 \times 1\$)" "logs/lkxh4srz/predict/evaluate_t2v/overall/last.json" \
    "Pre-Attn" "logs/p03o7rqy/predict/evaluate_t2v/overall/last.json" \
    "Post-Attn" "logs/82awngqn/predict/evaluate_t2v/overall/last.json" \
    "PRoPE" "logs/wekc4yx6/predict/evaluate_t2v/overall/last.json" \
    "GTA" "logs/z0cfx65s/predict/evaluate_t2v/overall/last.json" \
    --metrics \
    "video_metrics/vfov_err" "video_metrics/k1_err" "video_metrics/k2_err" \
    "video_metrics/pitch_err" "video_metrics/roll_err" \
    "pose/rot_err" "pose/trans_err" "pose/cammc" \
    "video_metrics/fvd" "video_metrics/fid" \
    "video_metrics/cs_text"

# Comparison on RealEstate10k
python -m tools.export_table \
    --methods \
    "ReCamMaster" "logs/lg1mxf9u/RealEstate10k/evaluate_t2v/overall/last.json" \
    "Wan CameraCtrl" "logs/3yf7psvi/RealEstate10k/evaluate_t2v/overall/last.json" \
    "CameraCtrl" "../CameraCtrl/out/evaluate_re10k/overall/last.json" \
    "AC3D" "../ac3d/out/5B/test/evaluate_10000/overall/last.json" \
    "UCPE" "logs/coo9rjaq/RealEstate10k/evaluate_t2v/overall/last.json" \
    --metrics \
    "pose/rot_err" "pose/trans_err" "pose/cammc" \
    "qalign/image_quality" "qalign/image_aesthetic" "qalign/video_quality"
```

Visualize camera statistics:
```bash
# PanShot
python -m tools.dataset_statistics \
    --data=PanShotDataset \
    --data_root=data/UCPE \
    --output_dir=outputs/suppl/panshot \
    --color=C0

# RE10K
python -m tools.dataset_statistics \
    --data=Re10kDataset \
    --data_root=data/RealEstate10k \
    --output_dir=outputs/suppl/re10k \
    --color=C1
```

</details>
<br>

## 💡 Acknowledgements

Our paper cannot be completed without the amazing open-source projects [Wan2.1](https://github.com/Wan-Video/Wan2.1), [AC3D](https://github.com/snap-research/ac3d), [ReCamMaster](https://github.com/KlingTeam/ReCamMaster), [CameraCtrl](https://github.com/hehao13/CameraCtrl), [prope](https://github.com/liruilong940607/prope), [vllm](https://github.com/vllm-project/vllm), [stella_vslam](https://github.com/stella-cv/stella_vslam)...

Also check out our Pan-Series works [PanFlow](https://github.com/chengzhag/PanFlow), [PanFusion](https://github.com/chengzhag/PanFusion) and [PanSplat](https://github.com/chengzhag/PanSplat) towards 3D scene generation with panoramic images!
