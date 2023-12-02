## INSTALLATION
if you've noticed your python3 bin doens't point to your conda env when using --prefix to point to your scratch dir, then you need to do the following:
* conda config --set always_copy True
* conda config --show | grep always_copy
now continue as normal:
* conda create --prefix <your-scratch-path>/MotionDiffuse/env python=3.7
* conda activate <your-scratch-path>/MotionDiffuse/env
* double check your GCC is 5+ by running `gcc --version`; if not, do module load gcc/5.4.0
* module load cuda/10.1 # you must run these icuda commands before installing torch otherwise it will say version not found!!
* module load cudnn/v7.6.5.32-prod-cuda-10.1
* conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=10.1 -c pytorch
* python3 -m pip install "mmcv-full>=1.3.17,<=1.5.3" -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.1/index.html
* python3 -m pip install -r requirements.txt
* python3 -m pip install --upgrade protobuf==3.20.0

fyi there is an annoying warning in the logs (https://stackoverflow.com/questions/57381430/synonym-of-type-is-deprecated-in-a-future-version-of-numpy-it-will-be-underst) that can be silenced by downgrading numpy:to 1.16.4 BUT this is incompatible with the other package versions, so don't do it

fyi:
(/work3/s222376/MotionDiffuseNew) s222376@n-62-20-1 /work3/s222376/MotionDiffuse/text2motion (train_baseline)$ module list
Currently Loaded Modulefiles:
 1) latex/TeXLive19(default)   3) cudnn/v7.6.5.32-prod-cuda-10.1   5) gcc/5.4.0
 2) cuda/10.1                  4) binutils/2.29(default) <aL>

## TRAINING
* download KIT-ML data from <> and put the zip for it in text2motion/data/
* cd text2motion/data && unzip KIT-ML-20231122T121619Z-001.zip
* cd KIT-ML && unrar x new_joint_vecs.rar
* unrar x new_joints.rar
* unrar x texts.rar
* dirs should look like
```
text2motion/data/KIT-ML
├── new_joint_vecs
│   ├─�
├── new_joints
│   ├─�
└── texts
    ├─�
--all.txt
--<etc>
```
* voltash (dtu hpc command to go to interactive gpu node)
* make train
* verify above works without errors and then kill training because you're on interactive gpu, you will likely run out of memory anyway (can decrease --batchsize but then it's slow)
* to do full training, edit jobscript.sh to use your email and submit job via "make queue"

## INFERENCE with pretrained model
* download...checkpoints?? idk look at their README.md

## Changes I made
* ignore standardization
* tokens are [] empty...
* reusing kit_chain thing lol
* only training on one sequence from grab

TO KEEP IN MIND:
* they specify best way to train in readme somewhere -- follow this when doing real training!
* need to add the emotion text to the caption!!