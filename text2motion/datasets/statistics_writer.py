from os.path import join as pjoin

import numpy as np
from motionx_explorer import (calc_mean_stddev_pose,
                              drop_shapes_from_motion_arr, get_seq_names)

if __name__ == "__main__":
    # read names from ./data/GRAB/train.txt
    with open(pjoin("./data/GRAB", "train.txt"), "r") as f:
        names = f.readlines()
    names = [name.strip() for name in names]
    print(f"names: {names}")
    all_arrays = []
    for name in names:
        # Load each NumPy array and add it to the list
        array = np.load(pjoin("./data/GRAB/joints", f"{name}.npy"))
        # drop shapes -> 212 dims
        array = drop_shapes_from_motion_arr(array)
        print(f"shape of {name}: {array.shape}")
        all_arrays.append(array)
    mean, stddev = calc_mean_stddev_pose(all_arrays)
    pose_dims = 212
    assert mean.shape[0] == pose_dims
    assert stddev.shape[0] == pose_dims
    # check if stddev has 0's
    stdev_zeros = np.where(stddev == 0)
    n_zeros = len(stdev_zeros[0])
    print(f"idx of stddev where 0: {stdev_zeros}")
    assert n_zeros == 0, "stddev has 0's, but it should not..."
    # save to ./data/GRAB/Mean.npy and ./data/GRAB/Std.npy
    mean_write_path = pjoin("./data/GRAB", "Mean.npy")
    stddev_write_path = pjoin("./data/GRAB", "Std.npy")
    with open(mean_write_path, "wb") as f:
        print(f"saving mean to {mean_write_path}")
        np.save(f, mean)
    with open(stddev_write_path, "wb") as f:
        print(f"saving stddev to {stddev_write_path}")
        np.save(f, stddev)
    
    
    # test calculate_mean_stddev
    # pose_dim = 3
    # arrays_1s = np.full((4, pose_dim), 3)
    # arrays_2s = np.full((2, pose_dim), 2)
    # single_mean = (4*3 + 2*2) / (4+2)
    # std_dev_single = np.sqrt((4*(3-single_mean)**2 + 2*(2-single_mean)**2) / (4+2))
    # exp_mean = np.full((pose_dim), single_mean)
    # exp_stddev = np.full((pose_dim), std_dev_single)
    # all_arrays = [arrays_1s, arrays_2s]
    # mean, stddev = calc_mean_stddev_pose(all_arrays)
    # print(f"mean: {mean}, exp mean: {exp_mean}")
    # print(f"stddev: {stddev}, exp stddev: {exp_stddev}")
    # assert mean.shape == (3,)
    # assert np.all(mean == exp_mean)
    # assert stddev.shape == (3,)
    # assert np.all(stddev == exp_stddev)
