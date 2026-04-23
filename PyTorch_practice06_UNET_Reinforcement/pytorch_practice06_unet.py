"""
- This is a program created by Noah Sizemore

- Creating UNET model that takes 3 in_channels, encodes, decodes, 2_outchannels
- This is to reinforce my knowledge and skill in developing U-Net architecture
"""

import torch
import torch.nn as nn

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Creating a double convolution block used throughout the U-Net
        self.double_conv = nn.Sequential(
            # First convolution layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # Batch normalization to stabilize learning and prevent vanishing gradients
            nn.BatchNorm2d(out_channels),
            # ReLU activation; inplace=True modifies the tensor directly to save memory
            nn.ReLU(inplace=True),
            
            # Second convolution layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    # The forward pass for this specific block
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)
    
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()

        # --------------------------
        # =====     ENCODER    =====
        # --------------------------

        # Initial layer: taking the in_channels and expanding feature maps to 64
        self.down1 = DoubleConvolution(in_channels, 64)
        # Shrinking spatial dimensions by half (kernel 2, stride 2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Features: 64 -> 128
        self.down2 = DoubleConvolution(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Features: 128 -> 256
        self.down3 = DoubleConvolution(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Features: 256 -> 512
        self.down4 = DoubleConvolution(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # The bottleneck: Processes the deepest, most compressed latent features (1024)
        self.bottleneck = DoubleConvolution(512, 1024)

        # --------------------------
        # =====     DECODER    =====
        # --------------------------

        # Upsampling: Doubles the spatial dimension (reversing the MaxPool2d)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # Input is 1024 because the skip connection concatenates 512 + 512
        self.up_conv1 = DoubleConvolution(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # Skip connection: 256 (from upsample) + 256 (from encoder) = 512
        self.up_conv2 = DoubleConvolution(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Skip connection: 128 + 128 = 256
        self.up_conv3 = DoubleConvolution(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Skip connection: 64 + 64 = 128
        self.up_conv4 = DoubleConvolution(128, 64)

        # --------------------------
        # ======     FINAL    ======
        # --------------------------

        # A 1x1 convolution maps the final 64 features to the desired out_channels
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- ENCODER PASS ---
        # Pass through block, save 'x' for skip connection, pool 'p' for next layer
        x1 = self.down1(x)
        p1 = self.pool1(x1)
        
        x2 = self.down2(p1)
        p2 = self.pool2(x2)
        
        x3 = self.down3(p2)
        p3 = self.pool3(x3)
        
        x4 = self.down4(p3)
        p4 = self.pool4(x4)

        # --- BOTTLENECK PASS ---
        b = self.bottleneck(p4)

        # --- DECODER PASS ---
        # Upsample from the bottleneck
        u1 = self.up1(b)
        # Skip connection: concatenate saved x4 features with u1 along the channel dimension (dim=1)
        cat1 = torch.cat([x4, u1], dim=1)
        d1 = self.up_conv1(cat1)
        
        u2 = self.up2(d1)
        cat2 = torch.cat([x3, u2], dim=1)
        d2 = self.up_conv2(cat2)
        
        u3 = self.up3(d2)
        cat3 = torch.cat([x2, u3], dim=1)
        d3 = self.up_conv3(cat3)
        
        u4 = self.up4(d3)
        cat4 = torch.cat([x1, u4], dim=1)
        d4 = self.up_conv4(cat4)

        # --- FINAL PASS ---
        out = self.out_conv(d4)
        
        # transforms.ToTensor() scales image pixels between 0.0 and 1.0.
        # We apply Sigmoid to force the model's predictions into this exact same range.
        return torch.sigmoid(out)