import numpy as np
import torch
from motionx_explorer import motion_arr_to_dict



if __name__ == "__main__":
    print("Printing the values from a motion sequence")
    
    motionPath = "/dtu/blackhole/13/181395/HumanTOMATO/src/tomato_represenation/data/motion_data/joint/GRAB/GRAB_motion/s1/airplane_fly_1.npy"


    load_motion = np.load(motionPath)

    dicc_322 = motion_arr_to_dict(load_motion, shapes_dropped=False)

    print("---------------------------------------------------------") 
    print("Motionpath: ", motionPath)
    print("---------------------------------------------------------")
    print("ORIGINAL 322 DATA")
    print("root_orient:")
    print(dicc_322["root_orient"][0].size())
    print(dicc_322["root_orient"][0])
    print("pose_body:")
    print(dicc_322["pose_body"][0].size())
    print(dicc_322["pose_body"][0])
    print("pose_hand:")
    print(dicc_322["pose_hand"][0].size())
    print(dicc_322["pose_hand"][0])
    print("pose_jaw:")
    print(dicc_322["pose_jaw"][0].size())
    print(dicc_322["pose_jaw"][0])
    print("face_expr:")
    print(dicc_322["face_expr"][0].size())
    print(dicc_322["face_expr"][0])
    print("face_shape:")
    print(dicc_322["face_shape"][0].size())
    print(dicc_322["face_shape"][0])
    print("trans:")
    print(dicc_322["trans"][0].size())
    print(dicc_322["trans"][0])
    print("betas:")
    print(dicc_322["betas"][0].size())
    print(dicc_322["betas"][0])

    # dicc_212 = motion_arr_to_dict(load_motion, shapes_droped=True)
    # print("root_orient:")
    # print(dicc_212["root_orient"])
    # print("pose_body:")
    # print(dicc_212["pose_body"])
    # print("pose_hand:")
    # print(dicc_212["pose_hand"])
    # print("pose_jaw:")
    # print(dicc_212["pose_jaw"])
    # print("face_expr:")
    # print(dicc_212["face_expr"])
    # print("face_shape:")
    # print(dicc_212["face_shape"])
    # print("trans:")
    # print(dicc_212["trans"])
    # print("betas:")
    # print(dicc_212["betas"])



