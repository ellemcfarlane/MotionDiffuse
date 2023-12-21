import torch
import numpy as np
from numpy.typing import ArrayLike

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