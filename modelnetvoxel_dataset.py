import os
import glob
import torch 
from torch.utils.data import Dataset

class ModelNetDatasetVoxel(Dataset):
    def __init__(self, data_dir, train=True):
        self.data_dir = data_dir
        self.train = train
        self.class_map = {}
        self.files, self.labels = self.load_files_and_labels()

    def load_files_and_labels(self):
        folders = glob.glob(os.path.join(self.data_dir, "*"))
        files = []
        labels = []
        for i, folder in enumerate(folders):
            self.class_map[i] = os.path.basename(folder)
            dataset_type = "train" if self.train else "test"
            class_files = glob.glob(os.path.join(folder, f"{dataset_type}/*.pt"))
            files.extend(class_files)
            labels.extend([i] * len(class_files))
        return files, labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        voxel_grid = torch.load(file_path)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return voxel_grid, label

