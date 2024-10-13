import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import ModelNetDatasetVoxel, train_and_evaluate
from cnn_model import *
from torchinfo import summary
import matplotlib.pyplot as plt

seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

num_classes = 10

grid_size = 32
object_size = 28
pitch_rescale = 1
no_of_rotations = 4
DATA_DIR = f"ModelNet10Voxel_{grid_size}_{object_size}_{pitch_rescale}_{no_of_rotations}"

batch_size = 64
dataset = ModelNetDatasetVoxel(DATA_DIR, train=True)
train_datapoints = int(0.85 * len(dataset))
validation_datapoints = len(dataset) - train_datapoints
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_datapoints, validation_datapoints])
test_dataset = ModelNetDatasetVoxel(DATA_DIR, train=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {device}")

train_class_distribution = np.zeros(num_classes)
validation_class_distribution = np.zeros(num_classes)

for index in train_dataset.indices:
    label = train_dataset.dataset[index][1]
    train_class_distribution[label] += 1

for index in validation_dataset.indices:
    label = validation_dataset.dataset[index][1]
    validation_class_distribution[label] += 1

print("Train class distribution (%): \t\t", np.round(train_class_distribution / train_class_distribution.sum() * 100,1))
print("Validation class distribution (%): \t", np.round(validation_class_distribution / validation_class_distribution.sum() * 100,1))

class_weights = torch.tensor(1 / (train_class_distribution / train_class_distribution.sum()), dtype=torch.float32).to(device)
print(class_weights)

# Use the sampler in DataLoader
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=int(os.cpu_count()))
validation_dataloader = DataLoader(validation_dataset, batch_size, shuffle=True, num_workers=int(os.cpu_count()))
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=int(os.cpu_count()))

model = Simple3DCNN(10).to(device)
print(summary(model, input_size=(batch_size, 1, grid_size, grid_size, grid_size)))

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

train_loss_log, val_loss_log, train_acc_log, val_acc_log = train_and_evaluate(
    model, 
    train_dataloader, 
    validation_dataloader,
    loss_fn, 
    opt,
    num_epochs=3,
    patience_epochs=5,
    checkpoint_path="checkpoint.pt",
    resume_training=False,
    device=device
)

epochs = range(len(train_loss_log))
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Plot Training Loss
axs[0, 0].plot(epochs, train_loss_log, label='Training Loss', color='red')
axs[0, 0].set_title('Training Loss')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()
axs[0, 0].grid()

# Plot Validation Loss
axs[0, 1].plot(epochs, val_loss_log, label='Validation Loss', color='blue')
axs[0, 1].set_title('Validation Loss')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()
axs[0, 1].grid()

# Plot Training Accuracy
axs[1, 0].plot(epochs, train_acc_log, label='Training Accuracy', color='green')
axs[1, 0].set_title('Training Accuracy')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Accuracy')
axs[1, 0].legend()
axs[1, 0].grid()

# Plot Validation Accuracy
axs[1, 1].plot(epochs, val_acc_log, label='Validation Accuracy', color='orange')
axs[1, 1].set_title('Validation Accuracy')
axs[1, 1].set_xlabel('Epochs')
axs[1, 1].set_ylabel('Accuracy')
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()
plt.show()
plt.savefig("figures/train_validation_loss_acc.png")