source scripts/set_Wan2.1-T2V-1.3B.sh
export WANDB_MODE=disabled
# export PL_PREDICT__TRAINER__LIMIT_PREDICT_BATCHES=20

# WANDB_RUN_ID=46ooy8no python src/main.py predict \
#     --model.camera_condition="ucpe"

# WANDB_RUN_ID=7ur7pldr python src/main.py predict \
#     --model.camera_condition="prope"

# WANDB_RUN_ID=txloxo2j python src/main.py predict \
#     --model.camera_condition="plucker"

# WANDB_RUN_ID=gdlseut5 python src/main.py predict \
#     --model.camera_condition="gta"

# WANDB_RUN_ID=b3rv5pk8 python src/main.py predict \
#     --model.camera_condition="recammaster"

# WANDB_RUN_ID=shgomfd7 python src/main.py predict \
#     --model.camera_condition="ucpe" \
#     --model.attn_compress=4
