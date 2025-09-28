import yaml
import pickle
from subprocess import run
import numpy as np
import torch
import torch.nn.functional as F
import smplx
import shutil
import os
from pathlib import Path
import trimesh
import pyrender
import math
import matplotlib.pyplot as plt
import cv2
from contextlib import contextmanager
from PIL import Image

def show_progress_bar(current, total, description="Processing"):
    if total == 0:
        return
    
    progress = (current + 1) / total
    bar_length = 50
    filled_length = int(bar_length * progress)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    percentage = progress * 100
    
    print(f"\r{description}: [{bar}] {percentage:.1f}% ({current + 1}/{total})", end="", flush=True)
    
    if current + 1 == total:
        print()

def copy_with_progress_bar(source_path, dest_path, description="Copying"):
    if not os.path.exists(source_path):
        print(f"Warning: {description} source not found at {source_path}")
        return False
    
    # Remove destination if it exists
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    
    # Create destination directory
    os.makedirs(dest_path, exist_ok=True)
    
    if os.path.isdir(source_path):
        # Copy directory contents
        items = list(Path(source_path).rglob("*"))
        files = [item for item in items if item.is_file()]
        
        print(f"{description} {len(files)} files from {source_path} to {dest_path}")
        
        for i, file_path in enumerate(files):
            rel_path = file_path.relative_to(source_path)
            dest_file = Path(dest_path) / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_file)
            # Update progress bar
            show_progress_bar(i, len(files), description)
        
        return True
    else:
        # Copy single file
        print(f"{description} single file from {source_path} to {dest_path}")
        shutil.copy2(source_path, dest_path)
        return True

def rsync(src, dst):
    run(["rsync", "-a", src, dst], check=True)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_pkl(path):
    with open(path, 'rb') as file:
        return pickle.load(file)
    
def pitch():  # Positive: down
    theta = np.deg2rad(np.random.uniform(-10, 100))
        
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ], dtype=np.float32)
    
    return rotation_matrix, theta > np.deg2rad(50)

def yaw():  # Positive: clockwise
    theta = np.deg2rad(np.random.uniform(-180, 180))
    rotation_matrix = np.array([
        [np.cos(theta), 0, -np.sin(-theta)],
        [0, 1, 0],
        [np.sin(-theta), 0, np.cos(theta)]
    ], dtype=np.float32)
    
    return rotation_matrix

def roll(theta):  # Positive: clockwise
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ], dtype=np.float32)

def rot6d_to_rotmat(rot6d):
    assert rot6d.ndim == 2
    rot6d = rot6d.view(-1, 3, 2)
    a1 = rot6d[:, :, 0]
    a2 = rot6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.linalg.cross(b1, b2)
    rotmat = torch.stack((b1, b2, b3), dim=-1)
    return rotmat

def create_pose_rotmat(pose, orient):
    orient = rot6d_to_rotmat(torch.tensor(orient).reshape(-1, 6)).reshape(1, 3, 3).numpy()
    pose = rot6d_to_rotmat(torch.tensor(pose).reshape(-1, 6)).reshape(-1, 3, 3).numpy()
    return np.concatenate([orient, pose], axis=0)

def get_bbox(proj):
    min_x, min_y = np.min(proj, axis=0)
    max_x, max_y = np.max(proj, axis=0)
    return (min_x, min_y, max_x, max_y)

class SMALLayer(smplx.SMPLLayer):
    NUM_JOINTS = 34
    NUM_BODY_JOINTS = 34
    SHAPE_SPACE_DIM = 145

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vertex_joint_selector.extra_joints_idxs = torch.empty(0, dtype=torch.int32)

def weak_perspective_project(verts, scale, tx, ty):
    proj = verts[:, :2] * scale
    proj[:, 0] += tx
    proj[:, 1] += ty
    return proj

def convert_to_pixel_coords(scale, tx, ty, resolution=1024):
    tx_px = (tx + 0.5) * resolution
    ty_px = (ty + 0.5) * resolution
    s_wp = scale * resolution
    return s_wp, tx_px, ty_px

def save_vertices_obj(vertices, faces, save_path):
    with open(save_path, 'w') as f:
        for v in vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for face in faces + 1: 
            f.write(f'f {face[0]} {face[1]} {face[2]}\n')

def deduce_randomized_weak_perspective_params(verts, img_size=(1024, 1024)):
    min_x, min_y = verts[:, :2].min(axis=0)
    max_x, max_y = verts[:, :2].max(axis=0)

    scale_x = img_size[0] / (max_x - min_x)
    scale_y = img_size[1] / (max_y - min_y)
    scale = min(scale_x, scale_y)

    scale *= np.random.uniform(0.9, 1.0)

    scaled_width = scale * (max_x - min_x)
    scaled_height = scale * (max_y - min_y)

    tx_min = 0 - scale * min_x
    tx_max = img_size[0] - scale * max_x
    ty_min = 0 - scale * min_y
    ty_max = img_size[1] - scale * max_y

    tx = np.random.uniform(tx_min, tx_max)

    ty_mid = (ty_min + ty_max) / 2
    ty = np.random.uniform(ty_mid, ty_max)

    return scale, tx, ty

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self, scale, translation, znear=10.0, zfar=1000.0):
        super().__init__(znear=znear, zfar=zfar)
        self.scale = np.asarray(scale, dtype=float).ravel()[:2]
        self.translation = np.asarray(translation, dtype=float).ravel()[:2]

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        sx, sy = self.scale
        tx, ty = self.translation
        P[0, 0] = sx
        P[1, 1] = sy
        P[0, 3] = tx * sx
        P[1, 3] = -ty * sy
        P[2, 2] = -0.1
        return P


class MeshRenderer:
    def __init__(self, faces, resolution=(1024, 1024)):
        self.faces = faces
        self.resolution = resolution
        self.renderer = pyrender.OffscreenRenderer(*resolution)
        self.scene = pyrender.Scene(
            bg_color=[0, 0, 0, 0],
            ambient_light=(0.5, 0.5, 0.5)
        )
        self._setup_lights()
    
    def _setup_lights(self):
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        for pos in [[0, -1, 1], [0, 1, 1], [1, 1, 2]]:
            pose = np.eye(4)
            pose[:3, 3] = pos
            self.scene.add(light, pose=pose)

    def _calculate_surface_normals(self, vertices, faces):
        normals = np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 1]]
        )
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        return (normals + 1) / 2 * 255

    def _create_mesh(self, vertices, color=None):
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces, process=False)
        mesh = mesh.subdivide_loop(iterations=2)
        transform = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(transform)

        if color is not None:
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[*color, 1.0],
                metallicFactor=0.1,
                alphaMode="OPAQUE"
            )
            return pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
        else:
            normals = self._calculate_surface_normals(mesh.vertices, mesh.faces)
            mesh.visual.face_colors = normals.astype(np.uint8)
            return pyrender.Mesh.from_trimesh(mesh, material=None, smooth=False)
    
    def render(self, vertices, camera_params, color=None, depth_only=False):
        mesh = self._create_mesh(vertices, color)
        mesh_node = self.scene.add(mesh)
    
        cam = WeakPerspectiveCamera(scale=camera_params[:2], translation=camera_params[2:])
        cam_node = self.scene.add(cam, pose=np.eye(4))
    
        if depth_only:
            depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
            return depth
    
        img_rgba, depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
    
        alpha = (depth > 0).astype(np.uint8) * 255
        img_rgba[..., 3] = alpha
    
        self.scene.remove_node(mesh_node); self.scene.remove_node(cam_node)
        return img_rgba

def overlay_rgba_on_rgb(rendered_rgba, background_rgb):
    fg = Image.fromarray(rendered_rgba, mode="RGBA")
    bg = Image.fromarray(background_rgb, mode="RGB")

    if fg.size != bg.size:
        bg = bg.resize(fg.size, resample=Image.BILINEAR)

    out = Image.alpha_composite(bg.convert("RGBA"), fg)
    return np.array(out.convert("RGB"))