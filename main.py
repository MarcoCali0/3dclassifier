import glob
import itertools
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from cnn_model import Simple3DCNN, Simple3DCNN_BN
from modelnetvoxel_dataset import ModelNetDatasetVoxel

seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

DATA_DIR = "ModelNet10"
SAVE_DIR = "ModelNet10Voxel"


num_classes = 10
grid_size = 32
object_size = 24
pitch_rescale = 1.0
no_of_rotations = 1


batch_size = 64
train_dataset = ModelNetDatasetVoxel(SAVE_DIR, train=True)
test_dataset = ModelNetDatasetVoxel(SAVE_DIR, train=False)
train_dataloader = DataLoader(
    train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True
)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {device}")


model = Simple3DCNN(num_classes).to(device)

# Define your model, optimizer, and loss function
loss_fn = nn.CrossEntropyLoss()  # weight=weights
opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # add weight_decay = 1e-4

# Define the number of epochs and initialize logs
num_epochs = 50
train_loss_log = []
val_loss_log = []
best_val = np.inf
start_epoch = 0

# Check if there's a saved checkpoint
checkpoint_path = "checkpoint.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    opt.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    train_loss_log = checkpoint["train_loss_log"]
    val_loss_log = checkpoint["val_loss_log"]
    best_val = checkpoint["best_val"]
    print(f"Resuming from epoch {start_epoch}")


for epoch_num in range(start_epoch, num_epochs):
    print("#################")
    print(f"# EPOCH {epoch_num}")

    ### TRAIN
    model.train()  # Training mode (e.g. enable dropout, batchnorm updates,...)
    train_losses = []
    iterator = tqdm(train_dataloader)
    for batch_x, batch_y in iterator:
        # Move data to device
        batch_x, batch_y = batch_x.unsqueeze(1).to(device), batch_y.to(device)
        opt.zero_grad()

        # Forward pass
        out = model(batch_x)
        # Compute loss
        loss = loss_fn(out, batch_y)

        # Backpropagation
        loss.backward()

        # Update the weights
        opt.step()

        train_losses.append(loss.item())
        iterator.set_description(f"Train loss: {round(loss.item(), 2)}")

    avg_train_loss = np.mean(train_losses)
    train_loss_log.append(avg_train_loss)

    ### VALIDATION
    model.eval()  # Evaluation mode (e.g. disable dropout, batchnorm,...)
    val_losses = []
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient tracking
        for batch_x, batch_y in tqdm(test_dataloader):
            # Move data to device
            batch_x = batch_x.unsqueeze(1).to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            out = model(batch_x)

            val_losses.append(loss_fn(out, batch_y).item())

            _, predicted = torch.max(out, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    avg_val_loss = np.mean(val_losses)
    val_loss_log.append(avg_val_loss)
    val_acc = correct / total

    print(
        f"Average validation loss: {avg_val_loss:.3f}\tValidation accuracy: {val_acc:.3f}"
    )

    # Save the model and optimizer state at each epoch
    torch.save(
        {
            "epoch": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "train_loss_log": train_loss_log,
            "val_loss_log": val_loss_log,
            "best_val": best_val,
        },
        checkpoint_path,
    )

    if avg_val_loss < best_val:
        print("Update model!!!")
        torch.save(
            model.state_dict(),
            f"best_{model.__class__.__name__}_val_loss{round(avg_val_loss,3)}_val_acc_{round(val_acc,3)}.pt",
        )
        best_val = avg_val_loss
