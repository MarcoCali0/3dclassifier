import numpy as np
import torch
from scipy.ndimage import affine_transform


def compute_pitch(mesh, object_size=24, pitch_rescale=1.0):
    bbox = mesh.bounding_box_oriented
    bbox_extents = bbox.extents
    max_pitch_x = bbox_extents[0] / object_size
    max_pitch_y = bbox_extents[1] / object_size
    max_pitch_z = bbox_extents[2] / object_size
    pitch = max(max_pitch_x, max_pitch_y, max_pitch_z)
    pitch = pitch_rescale * pitch
    return pitch


def mesh_to_voxel_grid(mesh, object_size=24, pitch_rescale=1.0):
    pitch = compute_pitch(mesh, object_size=object_size, pitch_rescale=pitch_rescale)
    voxel_mat = mesh.voxelized(pitch).matrix.astype(int)
    return voxel_mat


def to_tensor(voxel_grid):
    return torch.tensor(voxel_grid, dtype=torch.float32)


def pad_voxel_grid(voxel_grid, grid_size=32):
    # Calculate padding values
    pad_x = (grid_size - voxel_grid.shape[0]) // 2
    pad_y = (grid_size - voxel_grid.shape[1]) // 2
    pad_z = (grid_size - voxel_grid.shape[2]) // 2

    # Calculate the remaining padding in case the grid size is odd
    pad_x1 = grid_size - voxel_grid.shape[0] - pad_x
    pad_y1 = grid_size - voxel_grid.shape[1] - pad_y
    pad_z1 = grid_size - voxel_grid.shape[2] - pad_z

    # Pad the voxel grid
    padded_voxel_grid = torch.nn.functional.pad(
        voxel_grid,
        (pad_z, pad_z1, pad_y, pad_y1, pad_x, pad_x1),
        mode="constant",
        value=0,
    )
    return padded_voxel_grid


def rotate_voxel_grid(voxel_grid, angle):
    if angle == 0:
        return voxel_grid

    # Create the rotation matrix
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array(
        [[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]]
    )

    # Center of the grid
    center = np.array(voxel_grid.shape) / 2

    # Create an affine transformation matrix that rotates around the center
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = rotation_matrix
    affine_matrix[:3, 3] = center - center @ rotation_matrix.T

    # Apply the affine transformation
    rotated_grid = affine_transform(
        voxel_grid.numpy(), affine_matrix[:3, :3], offset=affine_matrix[:3, 3], order=1
    )

    return torch.tensor(rotated_grid, dtype=voxel_grid.dtype, device=voxel_grid.device)


def normalize_voxel_grid(voxel_grid):
    return 2 * voxel_grid - 1
