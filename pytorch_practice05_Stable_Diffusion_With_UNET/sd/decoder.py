# creating VAE decoder for image decompression after UNET

# imports
import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

# the residual block to use in the encoder
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # normalize data for better training 
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        # conv layer 1: preserve spatial size with padding=1
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # normalize data for better training
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        # conv layer 2: preserve spatial size with padding=1
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # making x and residue equal so they can be added together at the end of forward()
        # if channels already match, identity pass-through is enough
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            # 1x1 conv projects input channels to match output channels without changing spatial size
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch size, in channels, height, width
        residue = x

        # norm -> activate -> conv -> norm -> activate -> conv, then add residue connection
        return self.conv_2(F.silu(self.groupnorm_2(self.conv_1(F.silu(self.groupnorm_1(x)))))) + self.residual_layer(residue)

# the attention block
class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int,):
        super().__init__()
        # normalize data for better training 
        # normalize data for better training
        self.groupnorm = nn.GroupNorm(32, channels)
        # single-head self-attention over all spatial positions
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: is batchsize, features, height, width
        residue = x

        # batchsize, colorchannels, height, width
        n, c, h, w = x.shape
        # turning x into a view of batchsize, colorchannels, height * width
        x = x.view(n, c, h * w)
        # changing order of x to be batchsize, height * width, colorchannels
        x = x.transpose(-1, -2)
        # apply self-attention across all spatial positions
        x = self.attention(x)
        # changing order of x to be batchsize, colorchannels, height * width
        x = x.transpose(-1, -2)
        # restore original spatial shape: batchsize, colorchannels, height, width
        x = x.view(n, c, h, w)
        # adding residue connection to preserve original information
        x += residue

        return x

# completely inverting the VAE_Encoder (everything is being reverted)
class VAE_Decoder(nn.Sequential):
    def __init__(self, ):
        super().__init__(
            # 1x1 conv: bottleneck that mixes latent channels without changing spatial size
            nn.Conv2d(
                in_channels=4,
                out_channels=4,
                kernel_size=1,
                padding=0
            ),
            # expand latent channels from 4 to 512 to match encoder feature depth
            nn.Conv2d(
                in_channels=4,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            VAE_ResidualBlock(
                512,
                512
            ),
            # attention block: lets every spatial position attend to every other position
            VAE_AttentionBlock(
                512
            ),
            # series of residual blocks at 512 channels to refine features before upsampling
            VAE_ResidualBlock(
                512,
                512
            ),
            VAE_ResidualBlock(
                512,
                512
            ),
            VAE_ResidualBlock(
                512,
                512
            ),
            VAE_ResidualBlock(
                512,
                512
            ),
            # increasing the size of the image (think of resizing an image)
            nn.Upsample(
                scale_factor=2
            ),
            # smooth out checkerboard artifacts that can appear after nearest-neighbor upsample
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            # residual blocks to refine features at this resolution
            VAE_ResidualBlock(
                512,
                512
            ),
            VAE_ResidualBlock(
                512,
                512
            ),
            VAE_ResidualBlock(
                512,
                512
            ),
            # increasing the size of the image (think of resizing an image)
            nn.Upsample(
                scale_factor=2
            ),
            # smooth out checkerboard artifacts after upsample
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            # reducing features
            VAE_ResidualBlock(
                512,
                256
            ),            
            VAE_ResidualBlock(
                256,
                256
            ),  
            VAE_ResidualBlock(
                256,
                256
            ),  
            # increasing the size of the image, now at original size of image
            nn.Upsample(
                scale_factor=2
            ),
            # smooth out checkerboard artifacts after upsample
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            # reducing features
            VAE_ResidualBlock(
                256,
                128
            ),            
            VAE_ResidualBlock(
                128,
                128
            ),  
            VAE_ResidualBlock(
                128,
                128
            ), 
            # normalizing the features into groups of 32
            nn.GroupNorm(
                32, 
                128
            ),
            # activation function that functions similar to ReLU()
            nn.SiLU(),
            # returning image features back into color channels
            nn.Conv2d(
                in_channels=128,
                out_channels=3,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batchsize, 4, height/8, width/8
        # unscale: reverse the scaling applied by the encoder before storing in latent space
        x /= 0.18215

        # pass x through each layer of the decoder sequentially
        for module in self:
            x = module(x)

        return x