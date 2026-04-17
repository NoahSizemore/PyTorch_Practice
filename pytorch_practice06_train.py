"""
- PyTorch_practice06
- Training loop for the unet.py model
- This is to also reinforce the training loop creation process as an addition to building a U-Net
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from pytorch_pracitce06_unet import UNET

# Setup device (uses GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup directories for saving outputs
OUTPUT_DIR = "colorization_unet_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====     Hyperparameters     =====

# Lock the random seed for reproducibility
RANDOM_SEED = torch.manual_seed(42)
if torch.cuda.is_available():
    RANDOM_SEED_CUDA = torch.cuda.manual_seed(42)

EPOCHS = 500
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

# ===== Data Loading =====

# Convert images to PyTorch Tensors (also scales pixels from 0-255 to 0.0-1.0)
transform = transforms.Compose([transforms.ToTensor()])

# Download and load the CIFAR10 training dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Grab a fixed batch of images to visually test the model at the end of every epoch
dataiter = iter(trainloader)
fixed_images, _ = next(dataiter)
fixed_images = fixed_images.to(device)

# Save baseline images to compare against later
save_image(fixed_images, f"{OUTPUT_DIR}/00_ground_truth.png") 

# Initialize the model (3 in, 3 out for RGB image reconstruction)
model = UNET(in_channels=3, out_channels=3).to(device)

# Define loss function (L1Loss/MAE) and the optimizer (Adam)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --------------------------
# =====    TRAINING    =====
# --------------------------

print(f"=== TRAINING ===")
print(f"Device: {device}\nEpoch(s): {EPOCHS}\nLearning rate: {LEARNING_RATE}\nBatch size: {BATCH_SIZE}")
print(f"Outputs will save to: ./{OUTPUT_DIR}/\n")

# Main Epoch Loop
for epoch in range(EPOCHS):
    # Set model to training mode (enables dropout and batch normalization updates)
    model.train()
    epoch_loss = 0.0

    # Loop through the batches of images
    for batch_id, (images, _) in enumerate(trainloader):
        # CIFAR10 provides (images, labels), but we only need the images for reconstruction
        images = images.to(device)
        
        # 1. Clear gradients from the previous step
        optimizer.zero_grad()
        
        # 2. Perform the forward pass
        outputs = model(images)
        
        # 3. Calculate the loss
        loss = criterion(outputs, images)
        
        # 4. Backpropagation (calculate gradients)
        loss.backward()
        
        # 5. Perform an optimizer step (update model weights)
        optimizer.step()

        # Keep track of the total loss for this epoch
        epoch_loss += loss.item()

        # Print an update to the console every 100 batches
        if batch_id % 100 == 0: 
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch {batch_id}/{len(trainloader)} | Loss: {loss.item():.4f}")

    # Epoch summary 
    average_loss = epoch_loss / len(trainloader)
    print(f"--- Epoch {epoch+1} Complete | Average Loss: {average_loss:.4f} ---")

    # --------------------------
    # ====    EVALUATION    ====
    # --------------------------

    # Put model in evaluation mode (freezes batch normalization and dropout)
    model.eval()
    
    # inference_mode disables gradient tracking to save memory and speed up computation
    with torch.inference_mode():
        # Reconstruct our fixed test batch
        reconstructed = model(fixed_images)
        # Save the reconstructed images to the output folder
        save_image(reconstructed, f"{OUTPUT_DIR}/epoch_{epoch+1}_output.png")

    # Save the model weights (overwrites previous epoch to save space)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/unet_model.pth")
    
print("\n=== TRAINING FINISHED ===")