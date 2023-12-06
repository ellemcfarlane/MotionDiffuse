import argparse
import logging
import logging as log
import os
import time
from collections import defaultdict
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
from tqdm import tqdm

from .motionx_explorer import (NUM_FACIAL_EXPRESSION_DIMS,
                               calc_mean_stddev_pose, get_info_from_file,
                               label_code, motion_arr_to_dict, names_to_arrays,
                               to_smplx_dict)

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def save_img(img, save_path):
    imageio.imwrite(save_path, img)
    
# based on https://github.com/vchoutas/smplx/blob/main/examples/demo.py
# used to render one pose (not sequence of poses) e.g. to see the mean pose
def render_mesh(model, output, should_save=False, save_path=None):
    should_display = not should_save
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    # joint points not visualized for now
    # joints = output.joints.detach().cpu().numpy().squeeze()
    scene = pyrender.Scene()
    if should_display:
        viewer = pyrender.Viewer(scene, run_in_thread=True)

    mesh_node = None
    joints_node = None
    # Rotation matrix (90 degrees around the X-axis)
    rot = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
    if should_save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        # print("Vertices shape =", vertices.shape)
        # print("Joints shape =", joints.shape)

        # from their demo script
        plotting_module = "pyrender"
        if plotting_module == "pyrender":
            vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
            tri_mesh = trimesh.Trimesh(vertices, model.faces, vertex_colors=vertex_colors)

            # Apply rotation
            tri_mesh.apply_transform(rot)
            ##### RENDER LOCK #####
            if should_display:
                viewer.render_lock.acquire()
            if mesh_node:
                scene.remove_node(mesh_node)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh)
            mesh_node = scene.add(mesh)

            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
            min_bound, max_bound = mesh.bounds

            # Calculate the center of the bounding box
            center = (min_bound + max_bound) / 2

            # Calculate the extents (the dimensions of the bounding box)
            extents = max_bound - min_bound

            # Estimate a suitable distance
            distance = max(extents) * 2  # Adjust the multiplier as needed

            # Create a camera pose matrix
            cam_pose = np.array(
                [
                    [1.0, 0, 0, center[0]],
                    [0, 1.0, 0, center[1]-1.0],
                    [0, 0, 1.0, center[2] + distance + 0.5],
                    [0, 0, 0, 1],
                ]
            )
            # Rotate around X-axis
            angle = np.radians(90)
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            rot_x = np.array([
                [1, 0,        0,         0],
                [0, cos_angle, -sin_angle, 0],
                [0, sin_angle, cos_angle,  0],
                [0, 0,        0,         1]
            ])
            cam_pose = np.matmul(cam_pose, rot_x)
            # this is great pose, head on, but a bit far from face
            # cam_pose[:3, 3] += np.array([0, 0, -3.5])
            cam_pose[:3, 3] += np.array([-.01, 0.65, -3.3])

            scene.add(camera, pose=cam_pose)

            # Add light for better visualization
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
            scene.add(light, pose=cam_pose)

            if should_save:
                r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
                col_img, _ = r.render(scene)
                save_img(col_img, save_path)
                r.delete()  # Free up the resources
            ###### RENDER LOCK RELEASE #####
            if should_display:
                viewer.render_lock.release()
    except KeyboardInterrupt:
        if should_display:
            viewer.close_external()

# motion_arr is 212 dims (no shapes: aka no betas and no face shapes)
def mesh_and_save(args, motion_arr, seq_name, model_name, emotion):
    motion_dict = motion_arr_to_dict(motion_arr, shapes_dropped=True)
    smplx_params = to_smplx_dict(motion_dict)
    model_folder = "./models/smplx"
    batch_size = 1
    model = smplx.SMPLX(
        model_folder,
        use_pca=False, # our joints are not in pca space
        num_expression_coeffs=NUM_FACIAL_EXPRESSION_DIMS,
        batch_size=batch_size,
    )
    output = model.forward(**smplx_params, return_verts=True)
    log.info(f"output size {output.vertices.shape}")
    log.info(f"output size {output.joints.shape}")
    log.info("rendering mesh")
    base_file = args.file.split('.')[0]
    # add {emotion}_{base_file} as a subfolder if it doesn't exist
    subfolder = f"single_pose_imgs/{model_name}/{emotion}_{base_file}"
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    save_path = f"{subfolder}/{seq_name}_pose.png"
    render_mesh(model, output, should_save=True, save_path=save_path)
    log.warning(
        "if you don't see the mesh animation, make sure you are running on graphics compatible DTU machine (vgl xterm)."
    )
    return subfolder
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--emotion",
        type=str,
        required=True,
        help="emotion to calculate mean, std for",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="file to filter for emotion",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=False,
        default="",
        help="Path to model directory e.g. ./checkpoints/grab/grab_baseline_dp_2gpu_8layers_1000",
    )
    args = parser.parse_args()
    data_root = './data/GRAB'
    motion_label_dir = pjoin(data_root, 'texts')
    emotions_label_dir = pjoin(data_root, 'face_texts')
    args = parser.parse_args()
    seq_list_file = pjoin(data_root, args.file)
    logging.info("aggregating info about sequences...")
    info_dict = get_info_from_file(seq_list_file, emotions_label_dir, motion_label_dir)
    
    # get all files with args.emotion_code
    logging.info("calculating mean pose statistics...")
    emotions = info_dict["unique_emotions"]
    # emotions = [args.emotion]
    for emotion in emotions:
        logging.info(f"render mean mesh for {emotion} in {args.file}...")
        emo_code = label_code(emotion)
        names_with_emo = info_dict["emotion_to_names"][emo_code]
        arrays = names_to_arrays(data_root, names_with_emo)
        
        mean, std = calc_mean_stddev_pose(arrays)
        # add 1 dimension to mean and std
        mean = mean.reshape(1, -1)
        std = std.reshape(1, -1)

        mean_dict = motion_arr_to_dict(mean, shapes_dropped=True)
        std_dict = motion_arr_to_dict(std, shapes_dropped=True)
        logging.info(f"{emotion} mean: {mean_dict['face_expr']}")
        logging.info(f"{emotion} std: {std_dict['face_expr']}")

        logging.info(f"rendering mean mesh for {emotion} in {args.file}...")
        subfolder = mesh_and_save(args, mean, "mean", args.model_path, emotion)

        model_name = args.model_path.split('/')[-1] if args.model_path else "ground_truth"
        # write the sequence names in a metadata folder at subfolder
        metadata_folder = f"{subfolder}/metadata"
        if not os.path.exists(metadata_folder):
            os.makedirs(metadata_folder)
        metadata_path = f"{metadata_folder}/metadata.txt"
        with open(metadata_path, 'w') as f:
            f.write(f"model: {model_name}\n")
            f.write(f"emotion: {emotion}\n")
            f.write(f"file: {args.file}\n")
            f.write(f"mean: {mean_dict}\n")
            f.write(f"std: {std_dict}\n")
            for name in names_with_emo:
                f.write(f"{name}\n")
        
        # now plot mesh for each of the sequences
        for i, arr in enumerate(arrays):
            one_pose = arr[0]
            one_pose = one_pose.reshape(1, -1)
            name = names_with_emo[i]
            # replace / with _
            name = name.replace("/", "_")
            subfolder = mesh_and_save(args, one_pose, name, args.model_path, emotion)
