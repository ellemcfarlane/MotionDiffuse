from os.path import join as pjoin

import numpy as np


def calc_mean_stddev_pose(arrays):
    # all_arrays = []
    # for file_path in file_list:
    #     # Load each NumPy array and add it to the list
    #     array = np.load(file_path)
    #     all_arrays.append(array)
    
    # Concatenate all arrays along the first axis (stacking them on top of each other)
    concatenated_arrays = np.concatenate(arrays, axis=0)
    # Calculate the mean and standard deviation across all arrays
    mean = np.mean(concatenated_arrays, axis=0)
    stddev = np.std(concatenated_arrays, axis=0)
    
    return mean, stddev

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
        all_arrays.append(array)
    mean, stddev = calc_mean_stddev_pose(all_arrays)
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
