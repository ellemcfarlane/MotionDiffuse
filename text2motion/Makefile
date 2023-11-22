# get git root
ROOT_DIR:=$(shell git rev-parse --show-toplevel)/text2motion
run:
	cd ${ROOT_DIR} && python3 -m tools.visualization --opt_path checkpoints/t2m/t2m_motiondiffuse/opt.txt --text "a man sits"

train:
	cd ${ROOT_DIR} && python3 -m tools.train \
	--name kit_baseline_dp_2gpu_8layers_1000 \
	--batch_size 128 \
	--times 50 \
	--num_epochs 50 \
	--dataset_name kit \
	--num_layers 8 \
	--diffusion_steps 1000 \
	--data_parallel \
	--gpu_id 0 1