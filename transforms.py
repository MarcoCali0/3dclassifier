import numpy as np
import torch
from scipy.ndimage import affine_transform


def compute_pitch(mesh, object_size=24, pitch_rescale=1.0):
    """
    Compute the pitch (voxel size) required to fit the 3D mesh within the desired voxel grid.

    Parameters:
    - mesh: The 3D mesh object.
    - object_size: The target size for the object in the voxel grid (default is 24).
    - pitch_rescale: Scaling factor for adjusting the pitch (default is 1.0).

    Returns:
    - pitch: The computed pitch value that defines the voxel size.
    """
    bbox = mesh.bounding_box_oriented
    bbox_extents = bbox.extents
    max_pitch_x = bbox_extents[0] / object_size
    max_pitch_y = bbox_extents[1] / object_size
    max_pitch_z = bbox_extents[2] / object_size
    pitch = max(max_pitch_x, max_pitch_y, max_pitch_z)
    pitch = pitch_rescale * pitch
    return pitch


def mesh_to_voxel_grid(mesh, object_size=24, pitch_rescale=1.0):
    """
    Convert a 3D mesh into a voxel grid based on the computed pitch.

    Parameters:
    - mesh: The 3D mesh object.
    - object_size: The target size for the object in the voxel grid (default is 24).
    - pitch_rescale: Scaling factor for adjusting the pitch (default is 1.0).

    Returns:
    - voxel_mat: A binary voxel grid representation of the mesh.
    """
    pitch = compute_pitch(mesh, object_size=object_size, pitch_rescale=pitch_rescale)
    voxel_mat = mesh.voxelized(pitch).matrix.astype(int)
    return voxel_mat


def to_tensor(voxel_grid):
    """
    Convert a voxel grid into a PyTorch tensor.

    Parameters:
    - voxel_grid: A binary voxel grid.

    Returns:
    - A PyTorch tensor of the voxel grid in float32 format.
    """
    return torch.tensor(voxel_grid, dtype=torch.float32)


def pad_voxel_grid(voxel_grid, grid_size=32):
    """
    Pad the voxel grid to a given grid size by adding zero-padding.

    Parameters:
    - voxel_grid: A tensor representing the voxel grid.
    - grid_size: The target grid size to pad the voxel grid (default is 32).

    Returns:
    - padded_voxel_grid: A zero-padded voxel grid of size `grid_size x grid_size x grid_size`.
    """
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
    """
    Rotate a voxel grid around the z-axis by a given angle.

    Parameters:
    - voxel_grid: A PyTorch tensor representing the voxel grid.
    - angle: The angle in radians by which to rotate the grid.

    Returns:
    - rotated_grid: The rotated voxel grid as a tensor.
    """
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
    """
    Normalize the voxel grid values to the range [-1, 1].

    Parameters:
    - voxel_grid: A binary voxel grid tensor with values 0 and 1.

    Returns:
    - A normalized voxel grid with values between -1 and 1.
    """
    return 2 * voxel_grid - 1
