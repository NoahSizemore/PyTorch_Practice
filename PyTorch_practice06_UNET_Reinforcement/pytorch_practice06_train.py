"""
- This is a program created by Noah Sizemore

- This training pipelines is for the UNET model using colorization.py (colorization.py is hidden and will not be posted onto GitHub directly)
- This is to also reinforce the training loop creation process as an addition to building a U-Net
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from pytorch_pracitce06_unet import UNET
from colorization import ColorizationDataset, get_colorization_dataloaders

# Setup device (uses GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup directories for saving outputs
OUTPUT_DIR = ""
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====     Hyperparameters     =====

# Lock the random seed for reproducibility
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 256

# method for reconstructing the images from numpy to torch and save them
def save_image_grid(l_batch, ab_batch, num, filepath):
    reconstructed_list = []
    # repeat for the num amount 
    for i in range(num):
        # form the numpy image
        rgb_numpy = ColorizationDataset.denormalize_lab(l_batch[i], ab_batch[i])
        # reshape to be used for torch
        rgb_tensor = torch.from_numpy(rgb_numpy).permute(2, 0, 1).float()
        reconstructed_list.append(rgb_tensor)
    # stacking the tensors together 
    batch = torch.stack(reconstructed_list)
    # saving four images per row
    save_image(batch, filepath, nrow=num)


# ===== Data Loading =====


# load the CIFAR10 training dataset
trainset = ''
trainloader, testloader = get_colorization_dataloaders(
    dataset="cifar10",
    data_root=trainset,
    batch_size=BATCH_SIZE,
    num_workers=0,  # 0 for debugging
    stl10_use_unlabeled=False,
)

# Grab a fixed batch of images to visually test the model at the end of every epoch
dataiter = iter(trainloader)
l_channel, ab_channels  = next(dataiter)
l_channel = l_channel.to(device)
ab_channels = ab_channels.to(device)

# creating ground truth images
for j in range(4):
    save_image_grid(l_channel, ab_channels, 1, f"{OUTPUT_DIR}/{j:02d}_ground_truth.png")

# Initialize the model (1 in, 2 out for ab image reconstruction)
model = UNET(in_channels=1, out_channels=2).to(device)

# Define loss function (L1Loss/MAE) and the optimizer (Adam)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --------------------------
# =====    TRAINING    =====
# --------------------------

print("=" * 55)
print(f"=== STARTING TRAINING ===")
print("=" * 55)
print(f"Device: {device}\nEpoch(s): {EPOCHS}\nLearning rate: {LEARNING_RATE}\nBatch size: {BATCH_SIZE}")
print(f"Outputs will save to: ./{OUTPUT_DIR}/\n")
print("=" * 55 + "\n")

# Main Epoch Loop
for epoch in range(EPOCHS):
    # Set model to training mode (enables dropout and batch normalization updates)
    model.train()
    epoch_loss = 0.0
    testing_loss = 0.0

    # Loop through the batches of images
    for batch_id, (l_batch, ab_batch) in enumerate(trainloader):
        # moving each batch to the device
        l_batch = l_batch.to(device)
        ab_batch = ab_batch.to(device)
        if epoch == 0 and batch_id == 0:
            print(f"\n=== Sanity checks: L shape: {l_batch.shape} | ab shape: {ab_batch.shape} ===")

        
        # 1. Clear gradients from the previous step
        optimizer.zero_grad()
        
        # 2. Perform the forward pass
        outputs = model(l_batch)
        
        # 3. Calculate the loss
        loss = criterion(outputs, ab_batch)
        
        # 4. Backpropagation (calculate gradients)
        loss.backward()
        
        # 5. Perform an optimizer step (update model weights)
        optimizer.step()

        # Keep track of the total loss for this epoch
        epoch_loss += loss.item()

        # Print an update to the console every 100 batches
        if batch_id % 64 == 0: 
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch {batch_id}/{len(trainloader)} | Loss: {loss.item():.4f}")

    # Epoch summary 
    training_average_loss = epoch_loss / len(trainloader)
    print(f"TRAINING --- Epoch {epoch+1} Complete | Average Loss: {training_average_loss:.4f} ---")

    # --------------------------
    # ====    EVALUATION    ====
    # --------------------------

    # Put model in evaluation mode (freezes batch normalization and dropout)
    model.eval()
    
    # inference_mode disables gradient tracking to save memory and speed up computation
    with torch.inference_mode():
        # iterating through the test data
        for batch_id, (l_batch, ab_batch) in enumerate(testloader):
            # moving items to the device
            l_batch = l_batch.to(device)
            ab_batch = ab_batch.to(device)

            # predicting ab values
            predicted_ab = model(l_batch)
            # calculating the loss
            model_loss = criterion(predicted_ab, ab_batch)
            # Keeping track of total loss
            testing_loss += model_loss.item()

        # testing summary
        testing_average_loss = testing_loss / len(testloader)
        print(f"VALIDATION --- Epoch {epoch+1:02d} Complete | Average Loss: {testing_average_loss:.4f} ---")
        print("=" * 55)

        # running fresh model to compare images
        anchor_prediction = model(l_channel)
        # reconstruct final images
        save_image_grid(l_channel, anchor_prediction, 4, f"{OUTPUT_DIR}/epoch_{epoch+1:02d}_output.png")


    # Save the model weights (overwrites previous epoch to save space)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/unet_model.pth")

print("=" * 55)
print("\n=== TRAINING PIPELINE FINISHED ===")
print("=" * 55)
