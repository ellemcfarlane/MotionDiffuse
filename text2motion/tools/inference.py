import argparse
import os
from os.path import join as pjoin

import numpy as np
import torch
import utils.paramUtil as paramUtil
from models import MotionTransformer
from torch.utils.data import DataLoader
from trainers import DDPMTrainer
from utils.get_opt import get_opt
from utils.motion_process import recover_from_ric
from utils.plot_script import *
from utils.utils import *
from utils.word_vectorizer import POS_enumerator


def plot_t2m(data, result_path, npy_path, caption, joints_n):
    joint = recover_from_ric(torch.from_numpy(data).float(), joints_n).numpy()

def get_numpy_file_path(prompt, epoch, n_frames):
    # e.g. "airplane_fly_1_1000_60f.npy"
    prompt_no_spaces = prompt.replace(' ', '_')
    return f"{prompt_no_spaces}_{epoch}_{n_frames}f"

def get_wordvec_model(opt):
    encoder = MotionTransformer(
        input_feats=opt.dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder


if __name__ == '__main__':
    print("inferencemake started")
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, help='Opt path')
    parser.add_argument('--text', type=str, default="", help='Text description for motion generation')
    parser.add_argument('--motion_length', type=int, default=60, help='Number of frames for motion generation')
    parser.add_argument('--result_path', type=str, default="test_sample.gif", help='Path to save generation result')
    parser.add_argument('--npy_path', type=str, default="", help='Path to save 3D keypoints sequence')
    parser.add_argument('--gpu_id', type=int, default=-1, help="which gpu to use")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    # add which_epoch
    parser.add_argument('--which_epoch', type=str, default="latest", help="which epoch to load")
    args = parser.parse_args()
    
    set_random_seed(args.seed)
    print(f"set random seed to {args.seed}")
    device = torch.device('cuda:%d' % args.gpu_id if args.gpu_id != -1 else 'cpu')
    opt = get_opt(args.opt_path, device)
    opt.do_denoise = True
    opt.which_epoch = args.which_epoch

    # TODO (elmc): re-enable this
    # assert opt.dataset_name == "t2m"
    # assert args.motion_length <= 196
    # opt.data_root = './dataset/HumanML3D'
    opt.data_root = './data/GRAB'
    # opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    opt.text_dir = pjoin(opt.data_root, 'texts')
    # TODO (elmc): re-enable this
    # opt.joints_num = 22
    # opt.dim_pose = 263
    opt.dim_pose = 212
    dim_word = 300
    dim_pos_ohot = len(POS_enumerator)
    # TODO (elmc): re-enable this
    # num_classes = 200 // opt.unit_length

    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))

    print("Loading word vectorizer...")
    encoder = get_wordvec_model(opt).to(device)
    print("Loading model...")
    trainer = DDPMTrainer(opt, encoder)
    trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))

    trainer.eval_mode()
    trainer.to(opt.device)

    result_dict = {}
    with torch.no_grad():
        if args.motion_length != -1:
            caption = [args.text]
            file_name = f"{opt.which_epoch}_{args.motion_length}f.npy"
            m_lens = torch.LongTensor([args.motion_length]).to(device)
            pred_motions = trainer.generate(caption, m_lens, opt.dim_pose)
            motion = pred_motions[0].cpu().numpy()
            motion = motion * std + mean
            title = args.text + " #%d" % motion.shape[0]
            print(f"trying to plot {title}")
            # write motion to numpy file
            if not os.path.exists(args.npy_path):
                os.makedirs(args.npy_path) 
            full_npy_path = f"{args.npy_path}/{get_numpy_file_path(caption[0], opt.which_epoch, args.motion_length)}.npy"
            with open(full_npy_path, 'wb') as f:
                print(f"saving output to {full_npy_path}")
                np.save(f, motion)

            # plot_t2m(motion, args.result_path, args.npy_path, title)
