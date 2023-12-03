import argparse
import logging as log
import os
from os.path import join as pjoin
from typing import Dict, Optional, Tuple

import imageio
import numpy as np
import pyrender
import smplx
import torch
import trimesh
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

def motion_arr_to_dict(motion_arr: ArrayLike, shapes_droped=False) -> Dict[str, Tensor]:
    # TODO (elmc): why did I need to convert to tensor again???
    motion_arr = torch.tensor(motion_arr).float()
    # if not shapes_droped:
    motion_dict = {
        "root_orient": motion_arr[:, :3],  # controls the global root orientation
        "pose_body": motion_arr[:, 3 : 3 + 63],  # controls the body
        "pose_hand": motion_arr[:, 66 : 66 + 90],  # controls the finger articulation
        "pose_jaw": motion_arr[:, 66 + 90 : 66 + 93],  # controls the jaw pose
        "face_expr": motion_arr[:, 159 : 159 + 50],  # controls the face expression
    }
    if not shapes_droped:
        motion_dict["face_shape"] = motion_arr[:, 209 : 209 + 100] # controls the face shape
        motion_dict["trans"] = motion_arr[:, 309 : 309 + 3] # controls the global body position
        motion_dict["betas"] = motion_arr[:, 312:] # controls the body shape. Body shape is static
    else:
        motion_dict["trans"] = motion_arr[:, 209:] # controls the global body position
    
    return motion_dict
        

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

def render_meshes(output, save_offscreen=False, output_dir="render_output"):
    vertices_list = output.vertices.detach().cpu().numpy().squeeze()
    joints_list = output.joints.detach().cpu().numpy().squeeze()
    if len(vertices_list.shape) == 2:
        vertices_list = [vertices_list]
        joints_list = [joints_list]
    scene = pyrender.Scene()
    if not save_offscreen:
        viewer = pyrender.Viewer(scene, run_in_thread=True)
    mesh_node = None
    joints_node = None
    # Rotation matrix (90 degrees around the X-axis)
    rot = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
    try:
        for i in range(0, len(vertices_list)):
            vertices = vertices_list[i]
            joints = joints_list[i]
            # print("Vertices shape =", vertices.shape)
            # print("Joints shape =", joints.shape)

            # from their demo script
            plotting_module = "pyrender"
            plot_joints = False
            if plotting_module == "pyrender":
                vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
                tri_mesh = trimesh.Trimesh(vertices, model.faces, vertex_colors=vertex_colors)

                # Apply rotation
                tri_mesh.apply_transform(rot)
                # translation_vector = [0, -0.5, 0]  # [x, y, z] - Change in Y-axis
                # # Apply translation
                # tri_mesh.apply_translation(translation_vector)
                # print("Camera pose:")
                # print(viewer.viewer_flags.camera_pose)
                ##### RENDER LOCK #####
                if not save_offscreen:
                    viewer.render_lock.acquire()
                if mesh_node:
                    scene.remove_node(mesh_node)
                mesh = pyrender.Mesh.from_trimesh(tri_mesh)
                mesh_node = scene.add(mesh)

                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
                # # Here we set the camera 1 unit away from the origin and look at the origin
                # cam_pose = np.array([
                #     [1.0,  0,  0,  0],
                #     [0,  1.0,  0,  0],
                #     [0,  0,  1.0,  1],
                #     [0,  0,  0,  1]
                # ])
                # Assuming 'mesh' is your trimesh object
                min_bound, max_bound = mesh.bounds

                # Calculate the center of the bounding box
                center = (min_bound + max_bound) / 2

                # Calculate the extents (the dimensions of the bounding box)
                extents = max_bound - min_bound

                # Estimate a suitable distance
                distance = max(extents) * 2  # Adjust the multiplier as needed

                # Create a camera pose matrix
                # This example places the camera looking towards the center of the bounding box
                # TODO: figure out correct cam_pose so we don't have to manually shift camera
                # and so we can save poses as images, right now camera bose is centered but above the person
                cam_pose = np.array(
                    [
                        [1.0, 0, 0, center[0]],
                        [0, 1.0, 0, center[1]],
                        [0, 0, 1.0, center[2] + distance],
                        [0, 0, 0, 1],
                    ]
                )
                # cam_pose = np.array([
                #     [1,  0,  0,  center[0]],
                #     [0,  0, -1,  center[1]],  # Flipped Z and Y-axis for 'up' to be Y
                #     [0,  1,  0,  center[2] + distance],  # Position camera in front of the person
                #     [0,  0,  0,  1]
                # ])
                # cam_pose = np.array([
                #     [1,  0,  0,  center[0]],
                #     [0,  0, -1,  center[1]],  # Flipped Z and Y-axis for 'up' to be Y
                #     [0,  1,  0,  center[2]],  # Position camera in front of the person
                #     [0,  0,  0,  1]
                # ])
                scene.add(camera, pose=cam_pose)

                # Add light for better visualization
                light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
                scene.add(light, pose=cam_pose)

                # TODO: rotation doesn't work here, so appears sideways
                if plot_joints:
                    sm = trimesh.creation.uv_sphere(radius=0.005)
                    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
                    tfs = np.tile(np.eye(4), (len(joints), 1, 1))
                    # tfs[:, :3, 3] = joints
                    for i, joint in enumerate(joints):
                        tfs[i, :3, :3] = rot[:3, :3]
                        tfs[i, :3, 3] = joint
                    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                    if joints_node:
                        scene.remove_node(joints_node)
                    joints_node = scene.add(joints_pcl)
                ###### RENDER LOCK RELEASE #####
                if not save_offscreen:
                    viewer.render_lock.release()
                if save_offscreen:
                    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
                    color, _ = r.render(scene)
                    output_path = os.path.join(MY_REPO, output_dir, f"mesh_{i}.png")
                    imageio.imsave(output_path, color)  # Save the rendered image as PNG
                    r.delete()  # Free up the resources
    except KeyboardInterrupt:
        viewer.close_external()
        gif_path = os.path.join(MY_REPO, "mesh.gif")
        log.info(f"saving gif to {gif_path}")
        # TODO: save_gif not working
        viewer.save_gif(gif_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-mn",
        "--min_t",
        type=int,
        required=False,
        default=0,
        help="Minimum number of timesteps to render",
    )
    parser.add_argument(
        "-mx",
        "--max_t",
        type=int,
        required=False,
        help="Maximum number of timesteps to render",
    )
    parser.add_argument(
        "-dm",
        "--display_mesh",
        action='store_true',
        required=False,
        default=False,
        help="Display mesh if this flag is present"
    )
    # for now just specifies file name (with spaces) made by inference
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=False,
        default="",
        help="Prompt for inference display",
    )
    parser.add_argument(
        "-sf",
        "--seq_file",
        type=str,
        required=False,
        default="",
        help="file for non-inference display",
    )
    # add model_path arg
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=False,
        default="",
        help="Path to model directory e.g. ./checkpoints/grab/grab_baseline_dp_2gpu_8layers_1000",
    )
    args = parser.parse_args()

    # data_dir, seq, file = "kungfu", "subset_0000", "Aerial_Kick_Kungfu_Wushu_clip_13"
    # data_dir, seq, file = "idea400", "subset_0000", "Clean_The_Glass,_Clean_The_Windows_And_Sitting_At_The_Same_Time_clip_1"
    # data_dir, seq, file = "GRAB_motion", "s1", "airplane_fly_1"

    prompt = args.prompt
    is_inference = len(prompt) > 0

    if args.seq_file != "" and args.prompt != "":
        log.error("cannot provide both prompt and seq_file; if trying to verify model inference, use --prompt, otherwise specify numpy --seq_file name to display")
        exit(1)
    elif args.seq_file == "" and args.prompt == "":
        log.error("must provide either prompt or seq_file; if trying to verify model inference, use --prompt, otherwise specify numpy --seq_file name to display")
        exit(1)
    if not is_inference:
        name = args.seq_file
        data_root = './data/GRAB'
        motion_dir = pjoin(data_root, 'joints')
    else:
        log.info(f"converting prompt into file name")
        name = args.prompt.replace(' ', '_')
        model_type = args.model_path
        motion_dir = pjoin(model_type, 'outputs')
    motion_path = pjoin(motion_dir, name + '.npy')
    log.info(f"loading motion from {motion_path}")
    motion_arr = np.load(motion_path)
    # directly get smplx dimensionality by dropping body and face shape data
    # motion_arr_smplx_dims = drop_shapes_from_motion_arr(motion_arr)

    # our MotionDiffuse predicts motion data that doesn't include face and body shape
    motion_dict = motion_arr_to_dict(motion_arr, shapes_droped=is_inference)
    n_points = len(motion_dict["pose_body"])

    min_t = args.min_t
    max_t = args.max_t or n_points

    timestep_range = (min_t, max_t)

    log.info(f"POSES: {n_points}")
    # checks data has expected shape
    tot_dims = 0
    for key in motion_dict:
        dims = motion_dict[key].shape[1]
        exp_dims = pose_type_to_dims.get(key)
        tot_dims += motion_dict[key].shape[1]
        log.info(f"{key}: {motion_dict[key].shape}, dims {dims}, exp: {exp_dims}")
    log.info(f"total MOTION-X dims: {tot_dims}\n")

    smplx_params = to_smplx_dict(motion_dict, timestep_range)
    tot_smplx_dims = 0
    for key in smplx_params:
        tot_smplx_dims += smplx_params[key].shape[1]
        log.info(f"{key}: {smplx_params[key].shape}")
    log.info(f"TOTAL SMPLX dims: {tot_smplx_dims}\n")

    if not is_inference:
        action_label_path = pjoin(data_root, 'texts', name + '.txt')
        action_label = load_label_from_file(action_label_path)
        emotion_label_path = pjoin(data_root, 'face_texts', name + '.txt')
        emotion_label = load_label_from_file(emotion_label_path)
        log.info(f"action: {action_label}")
        log.info(f"emotion: {emotion_label}")
    
    if args.display_mesh:
        model_folder = os.path.join(MY_REPO, MODELS_DIR, "smplx")
        batch_size = max_t - min_t
        log.info(f"calculating mesh with batch size {batch_size}")
        model = smplx.SMPLX(
            model_folder,
            use_pca=False,  # our joints are not in pca space
            num_expression_coeffs=NUM_FACIAL_EXPRESSION_DIMS,
            batch_size=batch_size,
        )
        output = model.forward(**smplx_params, return_verts=True)
        log.info(f"output size {output.vertices.shape}")
        log.info(f"output size {output.joints.shape}")
        log.info("rendering mesh")
        render_meshes(output)
        log.warning(
            "if you don't see the mesh animation, make sure you are running on graphics compatible DTU machine (vgl xterm)."
        )
