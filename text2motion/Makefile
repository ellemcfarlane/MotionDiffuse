# get git root
ROOT_DIR:=$(shell git rev-parse --show-toplevel)/text2motion
SEED = 42

PYTHON_BIN:=/dtu/blackhole/13/181395/MotionDiffuse/env/bin/python3
EXP:=alexecp
MODEL_DIR:=checkpoints/grab/md_${EXP}_seed${SEED}

infer:
# checkpoints/t2m/t2m_motiondiffuse/opt.txt
	cd ${ROOT_DIR} && ${PYTHON_BIN} -m tools.visualization \
	--opt_path checkpoints/grab/md_GRABalldata_seed42/opt.txt \
	--text "airplane pass" \
	--seed 42 \
	--npy_path checkpoints/grab/md_GRABalldata_seed42/outputs \

eval:
	cd ${ROOT_DIR} && ${PYTHON_BIN} -m tools.evaluation ${MODEL_DIR}/opt.txt \

# TODO (elmc): increase batch_size from 1 when not debugging!!
train:
	echo "experiment name md_${EXP}_seed${SEED}"
	cd ${ROOT_DIR} && ${PYTHON_BIN} -m tools.train \
	--name md_${EXP}_seed${SEED} \
	--batch_size 128 \
	--times 50 \
	--num_epochs 250 \
	--dataset_name grab \
	--num_layers 8 \
	--diffusion_steps 1000 \
	--data_parallel \
	--gpu_id 0 \
	--experiment_name md_${EXP}_seed${SEED} \
	--log_every 50 \
	--seed ${SEED} \
	--use_wandb \
	--wandb_user "alexciru" \
	--experiment_name md_${EXP} \

# put the model your trained in MODEL_DIR (set at top of file) and generate poses with it conditioned on prompt
# WARNING: make sure to run 'make infer' first to generate the npy files for make gen
gen:
# smpl-x model then displays poses as meshes
	echo "make sure to run on hpc dtu gui with graphics support and that you use 'vglrun' before python3 call!"
	echo "WARNING: run 'make infer' first to generate the npy files for make gen"
	cd $(ROOT_DIR) && vglrun ${PYTHON_BIN} -m datasets.motionx_explorer \
	--model_path checkpoints/grab/md_alexExp2fixedseed_seed42 \
	--prompt "airplane pass" \
	--display_mesh \
	--min_t 0 \
	--max_t 60 \

# smpl-x displays poses from seq_file as meshes
play:
	cd $(ROOT_DIR) && vglrun ${PYTHON_BIN} -m datasets.motionx_explorer \
	--seq_file "s5/airplane_pass_1" \
	--display_mesh \

aug:
	cd $(ROOT_DIR) && ${PYTHON_BIN} -m Motion-X.mocap-dataset-process.face_motion_augmentation \


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