import argparse
import glob
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import trimesh
from tqdm import tqdm
from transforms import *

# Set default values for processing parameters
grid_size = 32
object_size = 28
pitch_rescale = 1
no_of_rotations = 4

# Directory for ModelNet10 dataset and output location for voxelized data
DATA_DIR = "ModelNet10"
SAVE_DIR = (
    f"ModelNet10Voxel_{grid_size}_{object_size}_{pitch_rescale}_{no_of_rotations}"
)

# Argument parser to enable customization of parameters via the command line
parser = argparse.ArgumentParser(
    description="Transform the ModelNet10 dataset from meshes to voxel grids."
)

parser.add_argument(
    "--object_size", type=int, help="Size of the object", default=object_size
)
parser.add_argument(
    "--grid_size", type=int, help="Size of the voxel grid", default=grid_size
)
parser.add_argument(
    "--pitch_rescale", type=float, help="Pitch rescale factor", default=pitch_rescale
)
parser.add_argument(
    "--no_of_rotations",
    type=int,
    help="Number of rotations to apply to the training set",
    default=no_of_rotations,
)

# Parse arguments from the command line
args = parser.parse_args()

# Update parameters based on user input
object_size = args.object_size
grid_size = args.grid_size
pitch_rescale = args.pitch_rescale
no_of_rotations = args.no_of_rotations

print(f"object size: {object_size}")
print(f"grid size: {grid_size}")
print(f"pitch rescale: {pitch_rescale}")
print(f"no rot: {no_of_rotations}")

# Sanity checks for the parameters
if grid_size < object_size:
    print("The grid size must be larger than the object size!")
    exit(1)

if no_of_rotations < 1:
    print("The number of rotations must be greater than 0!")
    exit(1)

# Function to voxelize and save a single 3D mesh file
def process_file(file_path, dataset_type, class_name, dataset_save_dir):
    """
    Processes a single mesh file by converting it into a voxel grid, applying transformations, and saving the result.

    Parameters:
    - file_path: Path to the mesh file.
    - dataset_type: The type of dataset ('train' or 'test').
    - class_name: The class label for the object.
    - dataset_save_dir: Directory to save the transformed voxel grid.

    For training data, the voxel grid is rotated multiple times, while test data is stored without rotation.
    """
    try:
        # Load mesh and convert it to a voxel grid
        mesh = trimesh.load(file_path)
        voxel_grid = mesh_to_voxel_grid(
            mesh, object_size=object_size, pitch_rescale=pitch_rescale
        )
        voxel_grid = to_tensor(voxel_grid)
        voxel_grid = pad_voxel_grid(voxel_grid, grid_size=grid_size)

        # For training data, apply multiple rotations to generate more data
        if dataset_type == "train":
            for angle in np.linspace(
                0, 2 * np.pi * (1 - 1 / no_of_rotations), no_of_rotations
            ):
                rotated_voxel_grid = rotate_voxel_grid(voxel_grid, angle)
                rotated_voxel_grid = normalize_voxel_grid(rotated_voxel_grid)
                torch.save(
                    rotated_voxel_grid,
                    os.path.join(
                        dataset_save_dir,
                        (
                            os.path.basename(os.path.splitext(file_path)[0])
                            + (".pt" if angle == 0 else f"rot{round(angle*180/np.pi)}.pt")
                        ),
                    ),
                )
        # For test data, save the voxel grid without rotation
        elif dataset_type == "test":
            voxel_grid = normalize_voxel_grid(voxel_grid)
            torch.save(
                voxel_grid,
                os.path.join(
                    dataset_save_dir,
                    os.path.basename(os.path.splitext(file_path)[0]) + ".pt",
                ),
            )
        else:
            raise NotImplementedError("Invalid dataset type")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


# Main code to iterate through the dataset and apply transformations
if not os.path.isdir(SAVE_DIR):
    # Get all class folders in the dataset
    folders = glob.glob(os.path.join(DATA_DIR, "*"))
    for folder in folders:
        class_name = os.path.basename(folder)
        class_save_dir = os.path.join(SAVE_DIR, class_name)

        # Create class directories if they don't exist
        if not os.path.exists(class_save_dir):
            os.makedirs(class_save_dir)

        # Process both training and test data
        for dataset_type in ["train", "test"]:
            dataset_save_dir = os.path.join(class_save_dir, dataset_type)
            if not os.path.exists(dataset_save_dir):
                os.makedirs(dataset_save_dir)

            files = glob.glob(os.path.join(folder, f"{dataset_type}/*"))

            # Use multiprocessing to process files in parallel
            with Pool(processes=cpu_count()) as pool:
                func = partial(
                    process_file,
                    dataset_type=dataset_type,
                    class_name=class_name,
                    dataset_save_dir=dataset_save_dir,
                )
                list(
                    tqdm(
                        pool.imap(func, files),
                        total=len(files),
                        desc=f"Processing {dataset_type} data for {class_name}",
                    )
                )
else:
    print("Converted dataset already exists")
