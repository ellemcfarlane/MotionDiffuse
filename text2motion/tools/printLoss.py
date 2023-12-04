
# ALex funtion
# print the loss function from the original data and the output of the model
import numpy as np
import torch

# Didn't want to be bother with the imports
def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def drop_shapes_from_motion_arr(motion_arr):
    if isinstance(motion_arr, torch.Tensor):
        new_motion_arr = motion_arr.numpy()
    
    # Slice the array to exclude 'face_shape' and 'betas'
    new_motion_arr = np.concatenate((motion_arr[:, :209], motion_arr[:, 309:312]), axis=1)
    
    return new_motion_arr


# target (body + face)
target_file_body = "/dtu/blackhole/13/181395/MotionDiffuse/text2motion/data/GRAB/joints/s3/airplane_pass_1.npy"

# output
output_model_file = "/dtu/blackhole/13/181395/MotionDiffuse/text2motion/checkpoints/grab/md_motiondiffuse/outputs/airplane_pass.npy"

target_body = np.load(target_file_body)


target = drop_shapes_from_motion_arr(target_body)
model_output  = np.load(output_model_file)
print("target", target.shape)
print("output", model_output.shape)

mse = mean_flat((target - model_output) ** 2).view(-1, 1).mean(-1)
print("MSE ", mse)



