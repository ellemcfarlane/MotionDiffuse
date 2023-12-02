import argparse
import logging as log
import os
from os.path import join as pjoin

import imageio
import numpy as np
import pyrender
import smplx
import trimesh

from .motionx_explorer import (MODELS_DIR, MY_REPO, NUM_FACIAL_EXPRESSION_DIMS,
                               load_label_from_file, motion_arr_to_dict,
                               pose_type_to_dims, to_smplx_dict)


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
    name = "s1/airplane_fly_1"
    data_root = './data/GRAB'
    motion_dir = pjoin(data_root, 'joints')
    motion_arr = np.load(pjoin(motion_dir, name + '.npy'))
    motion_dict = motion_arr_to_dict(motion_arr)
    n_points = len(motion_dict["pose_body"])

    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        "-mn",
        "--min_t",
        type=int,
        required=False,
        default=0,
        help="Minimum number of timesteps to render",
    )
    # default for min
    parser.add_argument(
        "-mx",
        "--max_t",
        type=int,
        required=False,
        help="Maximum number of timesteps to render",
    )

    # Parse arguments
    args = parser.parse_args()

    # Access the arguments
    min_t = args.min_t
    max_t = args.max_t or n_points

    timestep_range = (min_t, max_t)
    smplx_params = to_smplx_dict(motion_dict, timestep_range)
    tot_smplx_dims = 0
    for key in smplx_params:
        tot_smplx_dims += smplx_params[key].shape[1]
        print(f"{key}: {smplx_params[key].shape}")
    log.info(f"total SMPL-X dims: {tot_smplx_dims}")
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

    log.info(f"POSES: {n_points}")
    # checks data has expected shape
    tot_dims = 0
    for key in motion_dict:
        num_joints = motion_dict[key].shape[1] / 3
        exp_n_joints = pose_type_to_dims.get(key)
        tot_dims += motion_dict[key].shape[1]
        log.info(f"{key}: {motion_dict[key].shape}, joints {num_joints}, exp: {exp_n_joints}")
    log.info(f"total MOTION-X dims: {tot_dims}")

    action_label_path = pjoin(data_root, 'texts', name + '.txt')
    action_label = load_label_from_file(action_label_path)
    emotion_label_path = pjoin(data_root, 'face_texts', name + '.txt')
    emotion_label = load_label_from_file(emotion_label_path)
    log.info(f"action: {action_label}")
    log.info(f"emotion: {emotion_label}")