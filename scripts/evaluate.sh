export EVAL_DATA_ROOT="data/UCPE"
export EVAL_NUM_FRAMES=81
# export EVAL_TEST_STEPS='["overall"]'
# export EVAL_LIMIT_EVAL_VIDEOS=10
export HF_HUB_OFFLINE=1


# python src/evaluate.py --evaluate_gt True
# WANDB_RUN_ID=lg1mxf9u python src/evaluate.py  # recammaster_norm
# WANDB_RUN_ID=3yf7psvi python src/evaluate.py  # plucker_norm
# WANDB_RUN_ID=khnmur4b python src/evaluate.py  # recammaster_noyaw
# WANDB_RUN_ID=9hjx47bc python src/evaluate.py  # plucker_noyaw
WANDB_RUN_ID=6wodf04s python src/evaluate.py  # relray_absmap_comp8
# WANDB_RUN_ID=nv4al3mj python src/evaluate.py  # relray_absmap_comp4
# WANDB_RUN_ID=lkxh4srz python src/evaluate.py  # relray_absmap_comp12
# WANDB_RUN_ID=r0hmwcag python src/evaluate.py  # relray_absmap_comp2
# WANDB_RUN_ID=p03o7rqy python src/evaluate.py  # relray_absmap_comp8_before
# WANDB_RUN_ID=82awngqn python src/evaluate.py  # relray_absmap_comp8_after
# WANDB_RUN_ID=coo9rjaq python src/evaluate.py  # relray
# WANDB_RUN_ID=wekc4yx6 python src/evaluate.py  # prope_absmap
# WANDB_RUN_ID=z0cfx65s python src/evaluate.py  # gta_absmap

export EVAL_DATA="Re10kDataset"
export EVAL_DATA_ROOT="data/RealEstate10k"
export EVAL_POSE_FRAMES=16
export EVAL_LIMIT_EVAL_VIDEOS=100
# python src/evaluate.py --frame_stride=2 --test_res_path=/mnt/pfs/users/zhangchen/panshot/ac3d/out/5B/test/10000 # ac3d
# python src/evaluate.py --frame_stride=1 --test_res_path=/mnt/pfs/users/zhangchen/panshot/CameraCtrl/out/re10k # cameractrl
export EVAL_FRAME_STRIDE=4
# WANDB_RUN_ID=lg1mxf9u python src/evaluate.py  # recammaster_norm
# WANDB_RUN_ID=3yf7psvi python src/evaluate.py  # plucker_norm
# WANDB_RUN_ID=khnmur4b python src/evaluate.py  # recammaster_noyaw
# WANDB_RUN_ID=9hjx47bc python src/evaluate.py  # plucker_noyaw
WANDB_RUN_ID=6wodf04s python src/evaluate.py  # relray_absmap_comp8
# WANDB_RUN_ID=nv4al3mj python src/evaluate.py  # relray_absmap_comp4
# WANDB_RUN_ID=lkxh4srz python src/evaluate.py  # relray_absmap_comp12
# WANDB_RUN_ID=r0hmwcag python src/evaluate.py  # relray_absmap_comp2
# WANDB_RUN_ID=p03o7rqy python src/evaluate.py  # relray_absmap_comp8_before
# WANDB_RUN_ID=82awngqn python src/evaluate.py  # relray_absmap_comp8_after
# WANDB_RUN_ID=coo9rjaq python src/evaluate.py  # relray
# WANDB_RUN_ID=wekc4yx6 python src/evaluate.py  # prope_absmap
# WANDB_RUN_ID=z0cfx65s python src/evaluate.py  # gta_absmap
