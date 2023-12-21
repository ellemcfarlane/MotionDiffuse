# %% [markdown]
# # Motion Generation Demo
# There are three steps to generate motion sequences. 
#   1. System setup
#   2. download pretrained model
#   3. generate
# 
# Steps 1 and 2 are specified in [dtu_install.md](./demo/dtu_install.md). Once done with those steps, carry on hereunder.

# %% [markdown]
# # Motion Generation
# The model generates the sequence and stores it in a `<caption>.npy` in the `args.npy_path` directory.
# 
# To be able to see the sequence, the third-party `SMPLX` model needs to be run in an environment with a graphical interface (so no ssh). Detailed instructions are found below in the [SMPL-X sequence as a GIF](#replay-motion-sequence-as-smpl-x-motion-gif)

# %%
import os
import torch
import numpy as np

import utils.get_opt as opt_utils
import utils.utils as utils 

from os.path import join as pathjoin

from utils.word_vectorizer import POS_enumerator
from tools.arguments import get_case_arguments
from tools.visualization import get_wordvec_model
from trainers import DDPMTrainer

# %%
#### INSERT THE CONDITION HERE. IF UNCONDITIONED, MAKE IT AN EMPTY STRING (i.e. "")
caption = 'binoculars lift'

# %% [markdown]
# ## Parameters for generation process

# %%
print('\nStarting...')
SMPLX_MODEL_DIRPATH = "./models/smplx/"
MODEL_DIRPATH = "./checkpoints/grab/demo"

args = get_case_arguments('generation')

MOTION_FRAME_N = 100

# Custom definition of run arguments
args.opt_path = pathjoin(MODEL_DIRPATH,"opt.txt")
args.seed = 42
args.model_path = MODEL_DIRPATH
args.motion_length = MOTION_FRAME_N
args.min_t = 0
args.max_t = MOTION_FRAME_N
args.npy_path = pathjoin(MODEL_DIRPATH, "outputs") # path to the pretrained model

args.text = caption

utils.set_random_seed(args.seed)
device = utils.get_device(args)
opt = opt_utils.get_opt(args.opt_path, device)

# opt custom definitions
opt.model_name = 'ckpt_e015' # wout .tar
# opt.model_name = 'latest_MotionX_minibatch' # wout .tar

opt.do_denoise = True
assert args.motion_length <= 196
opt.data_root = './data/GRAB' # QUESTION (iony): Needed?
opt.text_dir = pathjoin(opt.data_root, 'texts') # QUESTION (iony): Needed?
opt.dim_pose = 212
opt.max_motion_length = 196
opt.joints_num = 22

# Other configurations
dim_word = 300
dim_pos_ohot = len(POS_enumerator)

mean = np.load(pathjoin(opt.meta_dir, 'mean.npy'))
std = np.load(pathjoin(opt.meta_dir, 'std.npy'))

# %% [markdown]
# ## Motion sequence generation

# %%
# Word vectorizer model
encoder = get_wordvec_model(opt).to(device)

print(f"Loading model {opt.model_name}...")
print(f"caption: {args.text}")

trainer = DDPMTrainer(opt, encoder)
trainer.load(pathjoin(opt.model_dir, opt.model_name + '.tar'))
trainer.eval_mode()
trainer.to(opt.device)

result_dict = {}
with torch.no_grad():
    if args.motion_length != -1:
        caption = [args.text]
        m_lens = torch.LongTensor([args.motion_length]).to(device)
        pred_motions = trainer.generate(caption, m_lens, opt.dim_pose)
        motion = pred_motions[0].cpu().numpy()
        motion = motion * std + mean # TODO: Check if this are the correct values of mean and atd
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

print("Motion generated")


# To retrieve the saved sequence, here u go, uncomment this
# motion = np.load(pathjoin(args.npy_path, 'man_walking.npy'))

# %% [markdown]
# # replay motion sequence as SMPL-X motion gif
# Change directory to the `/text2motion/` and run the following command in a graphical user interface terminal (i.e. if running in DTU's HPC, run in the _thinlinc_ client).
# 
# The values for the different arguments indicated in the command below are displayes in the next code cell
# 
# ***Note:** Remember 2 thing:
# 1. To activate the conda environment mentioned in [dtu_install.md](./demo/dtu_install.md)
# 2. (IF running in the hpc-thinlinc client) to run the command with vglrun in the start (`vglrun python -m ...`)
# 
# ```bash
# python -m datasets.motionx_explorer \
# --model-path [model-path] \
# --prompt [prompt]\
# --min_t [min-t] \
# --max_t [max-t] \
# --display-mesh  --save-gif
# ```
# 
# ## If that does not work...
# you can dc into the `text2motion/` and run the command `make gen`.\
# **But** you'll need to change some parameters of the makefile to fit the promt and motion length

# %%
py_cmd = \
f"python -m datasets.motionx_explorer --model-path {args.model_path} --prompt {args.text} --min_t {args.min_t} --max_t {args.max_t} --display-mesh --save-gif"
print(py_cmd)


