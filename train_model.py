import os

import numpy as np
import torch
from tqdm import tqdm

def train_and_evaluate(
    model,
    train_dataloader,
    validation_dataloader,
    loss_fn,
    opt,
    num_epochs=50,
    patience_epochs=5,
    checkpoint_path="checkpoint.pt",
    resume_training=False,
    device="cuda",
):
    """
    Train and evaluate a model with early stopping and checkpointing.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to be trained and evaluated.
    - train_dataloader (torch.utils.data.DataLoader): DataLoader for the training set.
    - validation_dataloader (torch.utils.data.DataLoader): DataLoader for the validation set.
    - loss_fn (torch.nn.Module): Loss function (e.g., torch.nn.CrossEntropyLoss).
    - opt (torch.optim.Optimizer): Optimizer (e.g., torch.optim.Adam).
    - num_epochs (int, optional): Maximum number of training epochs. Defaults to 50.
    - patience_epochs (int, optional): Number of epochs without validation loss improvement to wait before early stopping. Defaults to 5.
    - checkpoint_path (str, optional): Path to save and resume training checkpoints. Defaults to "checkpoint.pt".
    - resume_training (bool, optional): Whether to resume training from a saved checkpoint. Defaults to False.
    - device (str, optional): Device to run training on ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
    - train_loss_log (list of float): List of average training losses for each epoch.
    - val_loss_log (list of float): List of average validation losses for each epoch.
    - train_acc_log (list of float): List of training accuracies for each epoch.
    - val_acc_log (list of float): List of validation accuracies for each epoch.

    """
    # Initialize logs and variables
    train_loss_log = []
    val_loss_log = []
    train_acc_log = []
    val_acc_log = []
    
    best_val = np.inf
    patience_epoch_count = 0
    start_epoch = 0

    # Check if there's a saved checkpoint to resume training
    if os.path.exists(checkpoint_path) and resume_training:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        train_loss_log = checkpoint["train_loss_log"]
        val_loss_log = checkpoint["val_loss_log"]
        best_val = checkpoint["best_val"]
        print(f"Resuming from epoch {start_epoch}")

    # Start the training loop
    for epoch_num in range(start_epoch, num_epochs):
        if patience_epoch_count > patience_epochs:
            print(f"Early stopping at {epoch_num} epochs!")
            break

        ### TRAIN
        model.train()  # Set the model to training mode
        train_losses = []
        correct = 0
        total = 0
        iterator = tqdm(train_dataloader)
        for batch_x, batch_y in iterator:
            opt.zero_grad()

            # Move data to device
            batch_x, batch_y = batch_x.unsqueeze(1).to(device), batch_y.to(device)

            # Forward pass
            out = model(batch_x)
            # Compute loss
            loss = loss_fn(out, batch_y)

            # Backpropagation and optimization
            loss.backward()
            opt.step()
            
            _, predicted = torch.max(out, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            train_losses.append(loss.item())
            iterator.set_description(f"# EPOCH {epoch_num}: Train loss: {round(loss.item(), 2)} \t")

        train_acc = correct / total
        train_acc_log.append(train_acc)
        
        avg_train_loss = np.mean(train_losses)
        print(f"Average training loss: {avg_train_loss:.3f}\t Training accuracy: {train_acc:.3f}")
        train_loss_log.append(avg_train_loss)

        ### VALIDATION
        model.eval()  # Set the model to evaluation mode
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient tracking during validation
            for batch_x, batch_y in tqdm(validation_dataloader):
                batch_x = batch_x.unsqueeze(1).to(device)
                batch_y = batch_y.to(device)

                # Forward pass
                out = model(batch_x)
                val_losses.append(loss_fn(out, batch_y).item())

                # Accuracy calculation
                _, predicted = torch.max(out, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        avg_val_loss = np.mean(val_losses)
        val_loss_log.append(avg_val_loss)
        val_acc = correct / total
        
        val_acc_log.append(val_acc)

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

        # Save the best model
        if avg_val_loss < best_val:
            print("Update model!!!")
            torch.save(
                model.state_dict(),
                f"best_{model.__class__.__name__}_val_loss_{round(avg_val_loss, 3)}_val_acc_{round(val_acc, 3)}.pt",
            )
            best_val = avg_val_loss
            patience_epoch_count = 0
        else:
            patience_epoch_count += 1
            print("Incrementing patience epoch count: ", patience_epoch_count)

    return train_loss_log, val_loss_log, train_acc_log, val_acc_log
