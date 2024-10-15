import os
import time
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from cnn_model import *
from utils import ModelNetDatasetVoxel

num_classes = 10
grid_size = 32
object_size = 28
pitch_rescale = 1
no_of_rotations = 4

DATA_DIR = f"ModelNet10Voxel_{grid_size}_{object_size}_{pitch_rescale}_{no_of_rotations}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

test_dataset = ModelNetDatasetVoxel(DATA_DIR, train=False)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=int(os.cpu_count()))

# Load the model
model = Simple3DCNN(num_classes).to(device)
model.load_state_dict(torch.load("best_Simple3DCNN_val_loss_0.088_val_acc_0.97.pt"))
model.eval()

# Lists to store predictions and ground truth labels
all_predictions = []
all_targets = []
top2_correct = 0  # Counter for top-2 accuracy

start = time.time()
iterator = tqdm(test_dataloader)
for inputs, targets in iterator:
    # Move data to device
    inputs = inputs.unsqueeze(1).to(device)
    targets = targets.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)

    # Convert outputs to predicted labels
    _, top2_predicted = torch.topk(outputs, 2, dim=1)

    # Count correct predictions in top-2
    for i in range(targets.size(0)):
        if targets[i].item() in top2_predicted[i].cpu().numpy():
            top2_correct += 1

    # Convert outputs to predicted labels (top-1 for overall accuracy)
    _, predicted = torch.max(outputs, 1)

    # Append to lists
    all_predictions.extend(predicted.cpu().numpy())
    all_targets.extend(targets.cpu().numpy())


end = time.time()

print(f"Whole test dataset inferred in {(end-start):.1f} s")

# Calculate overall accuracy (Top-1 Accuracy)
overall_accuracy = accuracy_score(all_targets, all_predictions)
print(f"Test (Top-1) Accuracy: {(overall_accuracy*100):.1f}%")

# Calculate Top-2 accuracy
top2_accuracy = top2_correct / len(all_targets)
print(f"Test (Top-2) Accuracy: {(top2_accuracy*100):.1f}%")

# Calculate accuracy per class
class_names = [test_dataset.class_map.get(key) for key in range(num_classes)]
class_report = classification_report(
    all_targets, all_predictions, target_names=class_names
)
print("Classification Report:")
print(class_report)

# Compute confusion matrix
cm = confusion_matrix(all_targets, all_predictions)

sorted_indices = np.argsort(class_names)
sorted_class_names = [class_names[i] for i in sorted_indices]
sorted_cm = cm[sorted_indices][:, sorted_indices]

# Plot confusion matrix
plt.figure(figsize=(6, 6))
plt.imshow(sorted_cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
# plt.colorbar()
tick_marks = np.arange(len(sorted_class_names))
plt.xticks(tick_marks, sorted_class_names, rotation=45)
plt.yticks(tick_marks, sorted_class_names)

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()

# Print numerical values in each cell of the matrix
fmt = "d"
thresh = sorted_cm.max() / 2.0
for i, j in itertools.product(range(sorted_cm.shape[0]), range(sorted_cm.shape[1])):
    plt.text(
        j,
        i,
        format(sorted_cm[i, j], fmt),
        horizontalalignment="center",
        color="white" if sorted_cm[i, j] > thresh else "black",
    )

plt.show()
plt.savefig("figures/confusion_matrix.png")

### 

# Choose random samples
num_samples = 4  # Number of random samples to predict
random_indices = random.sample(range(len(test_dataset)), num_samples)

fig = plt.figure(figsize=(6, 7))

for i, idx in enumerate(random_indices):
    # Get a random sample from the test set
    sample, target = test_dataset[idx]
    sample = sample.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Predict with the model
    with torch.no_grad():
        output = model(sample.unsqueeze(0))
        _, predicted = torch.max(output, 1)

    # Convert predicted and target labels to numpy if needed
    predicted_label = predicted.item()
    true_label = target.item()  # Assuming target is a scalar label

    sample_np = sample.cpu().squeeze().numpy()
    ax = fig.add_subplot(2, 2, i+1, projection="3d")
    
    # Voxel plot
    sample_np = (sample_np + 1) / 2
    ax.voxels(sample_np, edgecolor="k")

    ax.set_title(
        f"Pred: {test_dataset.class_map.get(predicted_label)}, True: {test_dataset.class_map.get(true_label)}"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(0, sample_np.shape[0])
    ax.set_ylim(0, sample_np.shape[1])
    ax.set_zlim(0, sample_np.shape[2])

plt.tight_layout()
plt.show()
plt.savefig("figures/predictions.png")