# get git root
ROOT_DIR:=$(shell git rev-parse --show-toplevel)/text2motion
PYTHON_BIN:=python3
EXP:=motiondiffuse
SEED = 42
MODEL_DIR:=checkpoints/grab/demo

infer:
# checkpoints/t2m/t2m_motiondiffuse/opt.txt
	cd ${ROOT_DIR} && ${PYTHON_BIN} -m tools.visualization \
	--opt_path ${MODEL_DIR}/opt.txt \
	--text "airplane pass" \
	--npy_path ${MODEL_DIR}/outputs \
	--seed 42 \

eval:
	cd ${ROOT_DIR} && ${PYTHON_BIN} -m tools.evaluation ${MODEL_DIR}/opt.txt \

# TODO (elmc): increase batch_size from 1 when not debugging!!
train: w_stats
	echo "experiment name md_${EXP}_seed${SEED}"
	cd ${ROOT_DIR} && ${PYTHON_BIN} -m tools.train \
	--name md_${EXP}_seed${SEED} \
	--batch_size 128 \
	--times 50 \
	--num_epochs 50 \
	--dataset_name grab \
	--num_layers 8 \
	--diffusion_steps 1000 \
	--data_parallel \
	--gpu_id 0 1 \
	--wandb_user "elles" \
	--experiment_name md_${EXP}_seed${SEED} \
	--log_every 50 \
	--seed ${SEED} \
	--use_wandb \

# put the model your trained in MODEL_DIR (set at top of file) and generate poses with it conditioned on prompt
# smpl-x model then displays poses as meshes
# WARNING: make sure to run 'make infer' first to generate the npy files for make gen
gen:
	echo "make sure to run on hpc dtu gui with graphics support and that you use 'vglrun' before python3 call!"
	echo "WARNING: run 'make infer' first to generate the npy files for make gen"
	cd $(ROOT_DIR) && ${PYTHON_BIN} -m datasets.motionx_explorer \
	--model_path ${MODEL_DIR} \
	--prompt "a_person_walking_happily" \
	--display_mesh \
	--min_t 0 \
	--max_t 60 \
	--save_gif \

aug:
	cd $(ROOT_DIR) && ${PYTHON_BIN} -m Motion-X.mocap-dataset-process.face_motion_augmentation \

# smpl-x displays poses from seq_file as meshes
play:
	cd $(ROOT_DIR) && vglrun ${PYTHON_BIN} -m datasets.motionx_explorer \
	--seq_file "s5/airplane_pass_1" \
	--display_mesh \
	--save_gif \

print-data:
	cd $(ROOT_DIR) && ${PYTHON_BIN} -m datasets.motionx_explorer \

queue:
	cd ${ROOT_DIR} && bsub < jobscript.sh

w_stats:
	cd ${ROOT_DIR} && ${PYTHON_BIN} -m datasets.statistics_writer \

w_custom:
	cd ${ROOT_DIR} && ${PYTHON_BIN} -m datasets.custom_data_writer \

stat:
	@err_file=$$(ls -v gpu_*.err | tail -n 1); \
	out_file=$$(ls -v gpu_*.out | tail -n 1); \
	echo "Latest .err file: $$err_file"; \
	echo "Latest .out file: $$out_file"; \

# checks gpu utilization of latest job
gpu:
	@err_file=$$(ls -v gpu_*.err | tail -n 1); \
	err_number=$$(echo $$err_file | grep -oP 'gpu_\K\d+(?=\.err)'); \
	echo "Latest .err file: $$err_file with number $$err_number"; \
	bnvtop $$err_number; \

space:
	getquota_work3.sh

hog:
	du -h --max-depth=1 --apparent /work3/s222376/

env_setup:
	@echo "module load cuda/10.1 cudnn/v7.6.5.32-prod-cuda-10.1 gcc/5.4.0"