source scripts/set_Wan2.1-T2V-1.3B.sh

# export WANDB_NAME="plucker_noyaw"
# python src/main.py fit --model.camera_condition="plucker" --model.learning_rate=1e-5
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="plucker"
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="plucker" --data=Re10kDataModule --trainer.limit_predict_batches=13

# export WANDB_NAME="recammaster_noyaw"
# python src/main.py fit --model.camera_condition="recammaster"
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="recammaster"
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="recammaster" --data=Re10kDataModule --trainer.limit_predict_batches=13

# export WANDB_NAME="plucker_norm"
# python src/main.py fit --model.camera_condition="plucker" --model.learning_rate=1e-5 --data.zero_first_yaw=False
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="plucker" --data.zero_first_yaw=False
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="plucker" --data=Re10kDataModule --trainer.limit_predict_batches=13

# export WANDB_NAME="recammaster_norm"
# python src/main.py fit --model.camera_condition="recammaster" --data.zero_first_yaw=False
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="recammaster" --data.zero_first_yaw=False
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="recammaster" --data=Re10kDataModule --trainer.limit_predict_batches=13

export WANDB_NAME="relray_absmap_comp8"
python src/main.py fit --model.camera_condition="relray_absmap" --model.attn_compress=8
WANDB_MODE=offline python src/main.py predict --model.camera_condition="relray_absmap" --model.attn_compress=8
WANDB_MODE=offline python src/main.py predict --model.camera_condition="relray_absmap" --model.attn_compress=8 --data=Re10kDataModule --trainer.limit_predict_batches=13

# export WANDB_NAME="relray_absmap_comp4"
# python src/main.py fit --model.camera_condition="relray_absmap" --model.attn_compress=4
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="relray_absmap" --model.attn_compress=4
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="relray_absmap" --model.attn_compress=4 --data=Re10kDataModule --trainer.limit_predict_batches=13

# export WANDB_NAME="relray_absmap_comp12"
# python src/main.py fit --model.camera_condition="relray_absmap" --model.attn_compress=12
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="relray_absmap" --model.attn_compress=12
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="relray_absmap" --model.attn_compress=12 --data=Re10kDataModule --trainer.limit_predict_batches=13

# export WANDB_NAME="relray_absmap_comp2"
# python src/main.py fit --model.camera_condition="relray_absmap" --model.attn_compress=2
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="relray_absmap" --model.attn_compress=2
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="relray_absmap" --model.attn_compress=2 --data=Re10kDataModule --trainer.limit_predict_batches=13

# export WANDB_NAME="relray_absmap_comp8_before"
# python src/main.py fit --model.camera_condition="relray_absmap" --model.adaptation_method="before" --model.attn_compress=8
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="relray_absmap" --model.adaptation_method="before" --model.attn_compress=8
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="relray_absmap" --model.adaptation_method="before" --model.attn_compress=8 --data=Re10kDataModule --trainer.limit_predict_batches=13

# export WANDB_NAME="relray_absmap_comp8_after"
# python src/main.py fit --model.camera_condition="relray_absmap" --model.adaptation_method="after" --model.attn_compress=8
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="relray_absmap" --model.adaptation_method="after" --model.attn_compress=8
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="relray_absmap" --model.adaptation_method="after" --model.attn_compress=8 --data=Re10kDataModule --trainer.limit_predict_batches=13

# export WANDB_NAME="relray"
# python src/main.py fit --model.camera_condition="relray" --model.attn_compress=8
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="relray" --model.attn_compress=8
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="relray" --model.attn_compress=8 --data=Re10kDataModule --trainer.limit_predict_batches=13

# export WANDB_NAME="prope_absmap"
# python src/main.py fit --model.camera_condition="prope_absmap" --model.attn_compress=8
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="prope_absmap" --model.attn_compress=8
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="prope_absmap" --model.attn_compress=8 --data=Re10kDataModule --trainer.limit_predict_batches=13

# export WANDB_NAME="gta_absmap"
# python src/main.py fit --model.camera_condition="gta_absmap" --model.attn_compress=8
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="gta_absmap" --model.attn_compress=8
# WANDB_MODE=offline python src/main.py predict --model.camera_condition="gta_absmap" --model.attn_compress=8 --data=Re10kDataModule --trainer.limit_predict_batches=13
