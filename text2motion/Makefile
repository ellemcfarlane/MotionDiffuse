# get git root
ROOT_DIR:=$(shell git rev-parse --show-toplevel)/text2motion
PYTHON_BIN:=/work3/s222376/MotionDiffuseNew/bin/python3
run:
	cd ${ROOT_DIR} && ${PYTHON_BIN} -m tools.visualization --opt_path checkpoints/t2m/t2m_motiondiffuse/opt.txt --text "a man sits"

train:
	cd ${ROOT_DIR} && ${PYTHON_BIN} -m tools.train \
	--name kit_baseline_dp_2gpu_8layers_1000 \
	--batch_size 64 \
	--times 50 \
	--num_epochs 50 \
	--dataset_name kit \
	--num_layers 8 \
	--diffusion_steps 1000 \
	--data_parallel \
	--gpu_id 0 1 \
	--use_wandb \
	--wandb_user "elles" \

queue:
	cd ${ROOT_DIR} && bsub < jobscript.sh
