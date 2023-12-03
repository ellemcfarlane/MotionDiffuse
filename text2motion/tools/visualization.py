import argparse
import os
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils.data import DataLoader

import utils.paramUtil as paramUtil
from datasets.evaluator_models import MotionLenEstimatorBiGRU
from models import MotionTransformer
from trainers import DDPMTrainer
from utils.get_opt import get_opt
from utils.motion_process import recover_from_ric
from utils.plot_script import *
from utils.utils import *
from utils.word_vectorizer import POS_enumerator, WordVectorizer


def plot_t2m(data, result_path, npy_path, caption):
    joint = recover_from_ric(torch.from_numpy(data).float(), opt.joints_num).numpy()
    joint = motion_temporal_filter(joint, sigma=1)
    plot_3d_motion(result_path, paramUtil.t2m_kinematic_chain, joint, title=caption, fps=20)
    print(f"saving to {result_path}")
    caption_str = caption.replace(" ", "_")
    result_path += f"_{caption_str}.gif"
    if npy_path != "":
        np.save(npy_path, joint)


def build_models(opt):
    encoder = MotionTransformer(
        input_feats=opt.dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder


if __name__ == '__main__':
    print("visualization started")
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, help='Opt path')
    parser.add_argument('--text', type=str, default="", help='Text description for motion generation')
    parser.add_argument('--motion_length', type=int, default=60, help='Number of frames for motion generation')
    parser.add_argument('--result_path', type=str, default="test_sample.gif", help='Path to save generation result')
    parser.add_argument('--npy_path', type=str, default="", help='Path to save 3D keypoints sequence')
    parser.add_argument('--gpu_id', type=int, default=-1, help="which gpu to use")
    args = parser.parse_args()
    
    device = torch.device('cuda:%d' % args.gpu_id if args.gpu_id != -1 else 'cpu')
    opt = get_opt(args.opt_path, device)
    opt.do_denoise = True

    # TODO (elmc): re-enable this
    # assert opt.dataset_name == "t2m"
    assert args.motion_length <= 196
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

    # TODO (elmc): add back in
    # mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    # std = np.load(pjoin(opt.meta_dir, 'std.npy'))
    # print(f"mean shape: {mean.shape}, std shape: {std.shape}")
    print("Loading word vectorizer...")
    encoder = build_models(opt).to(device)
    print("Loading model...")
    trainer = DDPMTrainer(opt, encoder)
    trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))

    trainer.eval_mode()
    trainer.to(opt.device)

    result_dict = {}
    with torch.no_grad():
        if args.motion_length != -1:
            caption = [args.text]
            m_lens = torch.LongTensor([args.motion_length]).to(device)
            pred_motions = trainer.generate(caption, m_lens, opt.dim_pose)
            motion = pred_motions[0].cpu().numpy()
            # TODO (elmc): add back in
            # motion = motion * std + mean
            title = args.text + " #%d" % motion.shape[0]
            print(f"trying to plot {title}")
            # write motion to numpy file
            text_no_spaces = args.text.replace(" ", "_")
            if not os.path.exists(args.npy_path):
                os.makedirs(args.npy_path)
            full_npy_path = f"{args.npy_path}/{text_no_spaces}.npy"
            with open(full_npy_path, 'wb') as f:
                print(f"saving output to {full_npy_path}")
                np.save(f, motion)

            # plot_t2m(motion, args.result_path, args.npy_path, title)
