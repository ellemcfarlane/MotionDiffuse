import pyrender
from tqdm import tqdm
import trimesh
import numpy as np
import os
import imageio


def save_gif(gif_path, gif_frames, duration=0.01):
    if gif_frames:
        print(f"Saving GIF with {len(gif_frames)} frames to {gif_path}")
        imageio.mimsave(uri=gif_path, ims=gif_frames, duration=duration)
    else:
        print("No frames to save.")


def render_meshes(model, output, should_save_gif=False, gif_path=None):
    should_display = not should_save_gif
    vertices_list = output.vertices.detach().cpu().numpy().squeeze()
    joints_list = output.joints.detach().cpu().numpy().squeeze()
    if len(vertices_list.shape) == 2:
        vertices_list = [vertices_list]
        joints_list = [joints_list]
    scene = pyrender.Scene()
    if should_display:
        viewer = pyrender.Viewer(scene, run_in_thread=True)

    mesh_node = None
    joints_node = None
    # Rotation matrix (90 degrees around the X-axis)
    rot = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
    gif_frames = []
    if should_save_gif:
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    try:
        for i in tqdm(range(len(vertices_list))):
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
                angle = np.radians(80)
                cos_angle = np.cos(angle)
                sin_angle = np.sin(angle)
                rot_x_10_deg = np.array([
                    [1, 0,        0,         0],
                    [0, cos_angle, -sin_angle, 0],
                    [0, sin_angle, cos_angle,  0],
                    [0, 0,        0,         1]
                ])
                # rotate cam_pose with rot_x_10_deg
                cam_pose = np.matmul(cam_pose, rot_x_10_deg)
                cam_pose[:3, 3] += np.array([0, -2.2, -3.0])

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
                if should_save_gif:
                    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
                    color, _ = r.render(scene)
                    gif_frames.append(color)
                    r.delete()  # Free up the resources
                ###### RENDER LOCK RELEASE #####
                if should_display:
                    viewer.render_lock.release()
    except KeyboardInterrupt:
        if should_display:
            viewer.close_external()
        save_gif(gif_path, gif_frames)
    finally:
        save_gif(gif_path, gif_frames)
