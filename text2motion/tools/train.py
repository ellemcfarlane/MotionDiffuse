import os
# print cwd
import sys
from os.path import join as pjoin

sys.path.append(os.getcwd())
import torch
import torch.distributed as dist
import utils.paramUtil as paramUtil
import wandb
from datasets import Text2MotionDataset
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist
from models import MotionTransformer
from options.train_options import TrainCompOptions
from trainers import DDPMTrainer
from utils.plot_script import *
from utils.utils import *


def build_models(opt, dim_pose):
    encoder = MotionTransformer(
        input_feats=dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder


if __name__ == '__main__':

    parser = TrainCompOptions()
    opt = parser.parse()
    rank, world_size = get_dist_info()

    print(f"setting random seed to {opt.seed}")
    set_random_seed(opt.seed)
    opt.device = torch.device("cuda")
    torch.autograd.set_detect_anomaly(True)
    print(f"device id: {torch.cuda.current_device()}")
    print(f"selected device ids: {opt.gpu_id}")
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.noise_dir = pjoin(opt.save_root, 'noise')

    if rank == 0:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.meta_dir, exist_ok=True)
        os.makedirs(opt.noise_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    if opt.use_wandb:
        wandb_id = wandb.util.generate_id()
        wandb.init(
            project="text2motion",
            name=f"{opt.experiment_name}",
            entity=opt.wandb_user,
            # notes=opt.EXPERIMENT_NOTE,
            config=opt,
            id=wandb_id,
            resume="allow",
            # monitor_gym=True,
            sync_tensorboard=True,
        )
        # opt.wandb = wandb
    if opt.dataset_name == 't2m':
        opt.data_root = './data/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        radius = 4
        fps = 20
        opt.max_motion_length = 196
        dim_pose = 263
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'grab':
        opt.data_root = './data/GRAB'
        opt.motion_dir = pjoin(opt.data_root, 'joints')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.face_text_dir = pjoin(opt.data_root, 'face_texts')
        opt.joints_num = 72 # TODO (elmc): verify this BUT ALSO I'M NOT USING IT FOR NOW!
        # radius = 4 # TODO (elmc): verify this, think it's only for visualization purposes
        # fps = 20 # TODO (elmc): verify this, also for visualization I think
        dim_pose = 212 # drop betas (body shape) and face-shape from Motion data (via to_smplx_params & smplx_dict_to_array method)
        opt.dim_pose = dim_pose
        opt.max_motion_length = 196  # TODO (elmc): verify this; do this dynamically..??
        # TODO (elmc): verify what this does and if we can use the t2m one
        # NOTE: think, again, it's only for visualization
        # kinematic_chain = paramUtil.t2m_kinematic_chain
        # kinematic_chain = paramUtil.grab_kinematic_chain
        print(f"loading data root: {opt.data_root}")
        # print(f"kinematic chain: {kinematic_chain}")
    elif opt.dataset_name == 'kit':
        opt.data_root = './data/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain

    else:
        raise KeyError('Dataset Does Not Exist')

    # TODO (elmc): check dim_word and add back in???
    # dim_word = 300
    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    train_split_file = pjoin(opt.data_root, 'train.txt')
    print(f"cwd is {os.getcwd()}")
    print(f"train_split_file: {train_split_file}")
    encoder = build_models(opt, dim_pose)
    if world_size > 1:
        encoder = MMDistributedDataParallel(
            encoder.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True)
    elif opt.data_parallel:
        encoder = MMDataParallel(
            encoder.cuda(opt.gpu_id[0]), device_ids=opt.gpu_id)
    else:
        encoder = encoder.cuda()

    trainer = DDPMTrainer(opt, encoder)
    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file, opt.times)
    print(f"loaded data, now training")
    trainer.train(train_dataset)
