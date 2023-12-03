# get git root
ROOT_DIR:=$(shell git rev-parse --show-toplevel)/text2motion
PYTHON_BIN:=/work3/s222376/MotionDiffuseNew/bin/python3
EXP:=motiondiffuse

infer:
# checkpoints/t2m/t2m_motiondiffuse/opt.txt
	cd ${ROOT_DIR} && ${PYTHON_BIN} -m tools.visualization \
	--opt_path checkpoints/grab/grab_baseline_dp_2gpu_8layers_1000/opt.txt \
	--text "airplane pass" \
	--npy_path checkpoints/grab/grab_baseline_dp_2gpu_8layers_1000/outputs \

train:
	echo "experiment name ${EXP}"
	cd ${ROOT_DIR} && ${PYTHON_BIN} -m tools.train \
	--name md_${EXP} \
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
	--experiment_name md_${EXP} \

# put the model your trained in MODEL_DIR (set at top of file) and generate poses with it conditioned on prompt
# smpl-x model then displays poses as meshes
# WARNING: make sure to run 'make infer' first to generate the npy files for make gen
gen:
	echo "make sure to run on hpc dtu gui with graphics support and that you use 'vglrun' before python3 call!"
	echo "WARNING: run 'make infer' first to generate the npy files for make gen"
	cd $(ROOT_DIR) && vglrun ${PYTHON_BIN} -m datasets.motionx_explorer \
	--model_path ${MODEL_DIR} \
	--prompt "airplane pass" \
	--display_mesh \

# smpl-x displays poses from seq_file as meshes
play:
	cd $(ROOT_DIR) && vglrun ${PYTHON_BIN} -m datasets.motionx_explorer \
	--seq_file "s5/airplane_pass_1" \
	--display_mesh \

print-data:
	cd $(ROOT_DIR) && ${PYTHON_BIN} -m datasets.motionx_explorer \

queue:
	cd ${ROOT_DIR} && bsub < jobscript.sh

w_stats:
	cd ${ROOT_DIR} && ${PYTHON_BIN} -m datasets.statistics_writer \

w_custom:
	cd ${ROOT_DIR} && ${PYTHON_BIN} -m datasets.custom_data_writer \

print-data:
	cd $(ROOT_DIR) && ${PYTHON_BIN} -m datasets.motionx_explorer \
	--model_path ./checkpoints/grab/grab_baseline_dp_2gpu_8layers_1000 \
	--prompt "a man sits"

queue:
	cd ${ROOT_DIR} && bsub < jobscript.sh
