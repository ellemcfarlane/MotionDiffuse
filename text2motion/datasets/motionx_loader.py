import logging as log
import os
from os.path import join as pjoin
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)



MOCAP_DATASETS = {"egobody", "grab", "humanml", "grab_motion"}
DATA_DIR = "data"
MODELS_DIR = "models"
MOCAP_FACE_DIR = f"{DATA_DIR}/face_motion_data/smplx_322"  # contains face motion data only
MOTION_DIR = f"{DATA_DIR}/motion_data/smplx_322"
ACTION_LABEL_DIR = f"{DATA_DIR}/semantic_labels"
EMOTION_LABEL_DIR = f"{DATA_DIR}/face_texts"


"""
Page 12 of https://arxiv.org/pdf/2307.00818.pdf shows:

smpl-x = {θb, θh, θf , ψ, r} = 3D body pose, 3D hand pose, jaw pose, facial expression, global root orientation, global translation
dims: (22x3, 30x3, 1x3, 1x50, 1x3) = (66, 90, 3, 50, 3, 3)

NOTE: I think they are wrong about n_body_joints though, data indicates it's actually 21x3 = 63, not 22x3 = 66
"""

MY_REPO = os.path.abspath("")
log.info(f"MY_REPO: {MY_REPO}")
NUM_BODY_JOINTS = 23 - 2  # SMPL has hand joints but we're replacing them with more detailed ones by SMLP-X, paper: 22x3 total body dims * not sure why paper says 22
NUM_JAW_JOINTS = 1 # 1x3 total jaw dims
# Motion-X paper says there
NUM_HAND_JOINTS = 15 # x2 for each hand -> 30x3 total hand dims
NUM_JOINTS = NUM_BODY_JOINTS + NUM_HAND_JOINTS * 2 + NUM_JAW_JOINTS
NUM_FACIAL_EXPRESSION_DIMS = 50  # as per Motion-X paper, but why is default 10 in smplx code then?
FACE_SHAPE_DIMS = 100
BODY_SHAPE_DIMS = 10 # betas
ROOT_DIMS = 3
TRANS_DIMS = 3 # same as root, no?

pose_type_to_dims = {
    "pose_body": NUM_BODY_JOINTS * 3,
    "pose_hand": NUM_HAND_JOINTS * 2 * 3, # both hands
    "pose_jaw": NUM_JAW_JOINTS * 3,
    "face_expr": NUM_FACIAL_EXPRESSION_DIMS * 1,  # double check
    "face_shape": FACE_SHAPE_DIMS * 1,  # double check
    "root_orient": ROOT_DIMS * 1,
    "betas": BODY_SHAPE_DIMS * 1,
    "trans": TRANS_DIMS * 1,
}


def get_data_path(dataset_dir: str, seq: str, file: str) -> str:
    # MY_REPO/face_motion_data/smplx_322/GRAB/s1/airplane_fly_1.npy
    top_dir = MOCAP_FACE_DIR if dataset_dir.lower() in MOCAP_DATASETS else MOTION_DIR
    path = f"{os.path.join(MY_REPO, top_dir, dataset_dir, seq, file)}.npy"
    return path


def get_label_paths(dataset_dir: str, seq: str, file: str) -> Dict[str, str]:
    # MY_REPO/MotionDiffuse/face_texts/GRAB/s1/airplane_fly_1.txt
    action_path = f"{os.path.join(MY_REPO, ACTION_LABEL_DIR, dataset_dir, seq, file)}.txt"
    emotion_path = f"{os.path.join(MY_REPO, EMOTION_LABEL_DIR, dataset_dir, seq, file)}.txt"
    paths = {"action": action_path, "emotion": emotion_path}
    return paths

def load_data_as_dict(dataset_dir: str, seq: str, file: str) -> Dict[str, Tensor]:
    path = get_data_path(dataset_dir, seq, file)
    motion = np.load(path)
    motion = torch.tensor(motion).float()
    return {
        "root_orient": motion[:, :3],  # controls the global root orientation
        "pose_body": motion[:, 3 : 3 + 63],  # controls the body
        "pose_hand": motion[:, 66 : 66 + 90],  # controls the finger articulation
        "pose_jaw": motion[:, 66 + 90 : 66 + 93],  # controls the jaw pose
        "face_expr": motion[:, 159 : 159 + 50],  # controls the face expression
        "face_shape": motion[:, 209 : 209 + 100],  # controls the face shape
        "trans": motion[:, 309 : 309 + 3],  # controls the global body position
        "betas": motion[:, 312:],  # controls the body shape. Body shape is static
    }

def motion_arr_to_dict(motion_arr: ArrayLike) -> Dict[str, Tensor]:
    # TODO (elmc): why did I need to convert to tensor again???
    motion_arr = torch.tensor(motion_arr).float()
    return {
        "root_orient": motion_arr[:, :3],  # controls the global root orientation
        "pose_body": motion_arr[:, 3 : 3 + 63],  # controls the body
        "pose_hand": motion_arr[:, 66 : 66 + 90],  # controls the finger articulation
        "pose_jaw": motion_arr[:, 66 + 90 : 66 + 93],  # controls the jaw pose
        "face_expr": motion_arr[:, 159 : 159 + 50],  # controls the face expression
        "face_shape": motion_arr[:, 209 : 209 + 100],  # controls the face shape
        "trans": motion_arr[:, 309 : 309 + 3],  # controls the global body position
        "betas": motion_arr[:, 312:],  # controls the body shape. Body shape is static
    }

def drop_shapes_from_motion_arr(motion_arr: ArrayLike) -> ArrayLike:
    if isinstance(motion_arr, torch.Tensor):
        new_motion_arr = motion_arr.numpy()
    
    # Slice the array to exclude 'face_shape' and 'betas'
    new_motion_arr = np.concatenate((motion_arr[:, :209], motion_arr[:, 309:312]), axis=1)
    
    return new_motion_arr

def load_label_from_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        # Read the contents of the file into a string
        label = file.read()
    return label

def load_label(dataset_dir: str, seq: str, file_path: str) -> Dict[str, str]:
    paths = get_label_paths(dataset_dir, seq, file_path)
    action_path, emotion_path = paths["action"], paths["emotion"]
    log.info(f"loading labels from {action_path} and {emotion_path}")
    paths = {}
    with open(action_path, "r") as file:
        # Read the contents of the file into a string
        action_label = file.read()
    with open(emotion_path, "r") as file:
        # Read the contents of the file into a string
        emotion_label = file.read()
    return {"action": action_label, "emotion": emotion_label}


def to_smplx_dict(motion_dict: Dict[str, Tensor], timestep_range: Optional[Tuple[int, int]] = None) -> Dict[str, Tensor]:
    if timestep_range is None:
        # get all timesteps
        timestep_range = (0, len(motion_dict["pose_body"]))
    smplx_params = {
        "global_orient": motion_dict["root_orient"][
            timestep_range[0] : timestep_range[1]
        ],  # controls the global root orientation
        "body_pose": motion_dict["pose_body"][timestep_range[0] : timestep_range[1]],  # controls the body
        "left_hand_pose": motion_dict["pose_hand"][timestep_range[0] : timestep_range[1]][
            :, : NUM_HAND_JOINTS * 3
        ],  # controls the finger articulation
        "right_hand_pose": motion_dict["pose_hand"][timestep_range[0] : timestep_range[1]][:, NUM_HAND_JOINTS * 3 :],
        "expression": motion_dict["face_expr"][timestep_range[0] : timestep_range[1]],  # controls the face expression
        "jaw_pose": motion_dict["pose_jaw"][timestep_range[0] : timestep_range[1]],  #  controls the jaw pose
        # 'face_shape': motion_dict['face_shape'][timestep],  # controls the face shape, drop since we don't care to train on this
        "transl": motion_dict["trans"][timestep_range[0] : timestep_range[1]],  # controls the global body position
        # "betas": motion["betas"][
        #     timestep_range[0] : timestep_range[1]
        # ],  # controls the body shape. Body shape is static, drop since we don't care to train on this
    }
    return smplx_params

def smplx_dict_to_array(smplx_dict):
    # convert smplx dict to array
    # list keys to ensure known order when iterating over dict
    keys = ["global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "expression", "jaw_pose", "transl"]
    smplx_array = []
    for key in keys:
        smplx_array.append(smplx_dict[key])
    smplx_array = torch.cat(smplx_array, dim=1)
    return smplx_array

if __name__ == "__main__":
    data_dir, seq, file = "kungfu", "subset_0000", "Aerial_Kick_Kungfu_Wushu_clip_13"
    data_dir, seq, file = "idea400", "subset_0000", "Clean_The_Glass,_Clean_The_Windows_And_Sitting_At_The_Same_Time_clip_1"
    data_dir, seq, file = "GRAB_motion", "s1", "airplane_fly_1"

    name = "s1/airplane_fly_1"
    data_root = './data/GRAB'
    motion_dir = pjoin(data_root, 'joints')
    motion_arr = np.load(pjoin(motion_dir, name + '.npy'))
    # directly get smplx dimensionality by dropping body and face shape data
    # motion_arr_smplx_dims = drop_shapes_from_motion_arr(motion_arr)


    motion_dict = motion_arr_to_dict(motion_arr)
    n_points = len(motion_dict["pose_body"])

    log.info(f"POSES: {n_points}")
    # checks data has expected shape
    tot_dims = 0
    for key in motion_dict:
        dims = motion_dict[key].shape[1]
        exp_dims = pose_type_to_dims.get(key)
        tot_dims += motion_dict[key].shape[1]
        log.info(f"{key}: {motion_dict[key].shape}, dims {dims}, exp: {exp_dims}")
    log.info(f"total MOTION-X dims: {tot_dims}\n")

    timestep_range = (0, n_points)
    smplx_params = to_smplx_dict(motion_dict, timestep_range)
    tot_smplx_dims = 0
    for key in smplx_params:
        tot_smplx_dims += smplx_params[key].shape[1]
        log.info(f"{key}: {smplx_params[key].shape}")
    log.info(f"TOTAL SMPLX dims: {tot_smplx_dims}\n")

    action_label_path = pjoin(data_root, 'texts', name + '.txt')
    action_label = load_label_from_file(action_label_path)
    emotion_label_path = pjoin(data_root, 'face_texts', name + '.txt')
    emotion_label = load_label_from_file(emotion_label_path)
    log.info(f"action: {action_label}")
    log.info(f"emotion: {emotion_label}")
