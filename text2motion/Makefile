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

# TODO (elmc): increase batch_size from 1 when not debugging!!
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
	--gpu_id 0 1 \
	--wandb_user "elles" \
	--experiment_name md_${EXP} \
	--use_wandb \

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