# get git root
ROOT_DIR:=$(shell git rev-parse --show-toplevel)/text2motion
PYTHON_BIN:=python3
EXP:=motiondiffuse
SEED = 42
MODEL_DIR:=checkpoints/grab/demo
# MODEL_DIR:=checkpoints/grab/md_fulem_2g_excl_196_seed42
EPOCH:=latest
PROMPT:=a person walking happily
# PROMPT:=happiness airplane pass
GT_FILE=s2/cubesmall_lift # ground-truth
FRAMES=60
MEAN_EMOTION=surprise

mean-mesh:
	cd $(ROOT_DIR) && vglrun ${PYTHON_BIN} -m datasets.mean_mesh \
	--emotion ${MEAN_EMOTION} \
	--file train.txt \

expl-train:
	cd ${ROOT_DIR} && ${PYTHON_BIN} -m datasets.train_explorer \

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

# get makes model generate seq according to text and writes result to npy file
gen-npy:
# checkpoints/t2m/t2m_motiondiffuse/opt.txt
	cd ${ROOT_DIR} && ${PYTHON_BIN} -m tools.inference \
	--opt_path ${MODEL_DIR}/opt.txt \
	--which_epoch ${EPOCH} \
	--text "${PROMPT}" \
	--npy_path ${MODEL_DIR}/outputs \
	--seed 42 \
	--motion_length ${FRAMES} \

# put the model your trained in MODEL_DIR (set at top of file) and generate poses with it conditioned on prompt
# smpl-x model then displays poses as meshes
# WARNING: make sure to run 'make gen' first to generate the npy files for make gen
play-gen-gif:
	echo "make sure to run on hpc dtu gui with graphics support and that you use 'vglrun' before python3 call!"
	echo "WARNING: run 'make gen' first to generate the npy files for make gen"
	cd $(ROOT_DIR) && vglrun ${PYTHON_BIN} -m datasets.motionx_explorer \
	--model_path ${MODEL_DIR} \
	--which_epoch ${EPOCH} \
	--prompt "${PROMPT}" \
	--display_mesh \
	--save_gif \
	--max_t ${FRAMES} \

play-gen-gui:
	echo "make sure to run on hpc dtu gui with graphics support and that you use 'vglrun' before python3 call!"
	echo "WARNING: run 'make infer' first to generate the npy files for make gen"
	cd $(ROOT_DIR) && ${PYTHON_BIN} -m datasets.motionx_explorer \
	--model_path ${MODEL_DIR} \
	--which_epoch ${EPOCH} \
	--prompt "${PROMPT}" \
	--display_mesh \
	--max_t ${FRAMES} \

# smpl-x displays poses from seq_file as meshes
play-gt-gif:
	cd $(ROOT_DIR) && vglrun ${PYTHON_BIN} -m datasets.motionx_explorer \
	--seq_file ${GT_FILE} \
	--display_mesh \
	--save_gif \
	--max_t ${FRAMES} \

play-gt-gui:
	cd $(ROOT_DIR) && vglrun ${PYTHON_BIN} -m datasets.motionx_explorer \
	--seq_file ${GT_FILE} \
	--display_mesh \
	--max_t ${FRAMES} \

gen: gen-npy play-gen-gif

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

env_setup:
	@echo "module load cuda/10.1 cudnn/v7.6.5.32-prod-cuda-10.1 gcc/5.4.0"