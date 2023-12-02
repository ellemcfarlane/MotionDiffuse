# get git root
ROOT_DIR:=$(shell git rev-parse --show-toplevel)/text2motion
PYTHON_BIN:=/work3/s222376/MotionDiffuseNew/bin/python3


infer:
# checkpoints/t2m/t2m_motiondiffuse/opt.txt
	cd ${ROOT_DIR} && ${PYTHON_BIN} -m tools.visualization \
	--opt_path checkpoints/grab/grab_baseline_dp_2gpu_8layers_1000/opt.txt \
	--text "airplane pass" \
	--npy_path checkpoints/grab/grab_baseline_dp_2gpu_8layers_1000/outputs \

train:
	cd ${ROOT_DIR} && ${PYTHON_BIN} -m tools.train \
	--name grab_baseline_dp_2gpu_8layers_1000 \
	--batch_size 1 \
	--times 50 \
	--num_epochs 50 \
	--dataset_name grab \
	--num_layers 8 \
	--diffusion_steps 1000 \
	--data_parallel \
	--gpu_id 0 \
	--use_wandb \
	--wandb_user "elles" \


play:
	echo "make sure to run on hpc dtu gui with graphics support and that you use 'vglrun' before python3 call!"
	cd $(ROOT_DIR) && vglrun ${PYTHON_BIN} -m datasets.motionx_explorer \
	--model_path ./checkpoints/grab/grab_baseline_dp_2gpu_8layers_1000 \
	--prompt "airplane pass" \
	--display_mesh \

print-data:
	cd $(ROOT_DIR) && ${PYTHON_BIN} -m datasets.motionx_explorer \
	--model_path ./checkpoints/grab/grab_baseline_dp_2gpu_8layers_1000 \
	--prompt "a man sits"

queue:
	cd ${ROOT_DIR} && bsub < jobscript.sh
