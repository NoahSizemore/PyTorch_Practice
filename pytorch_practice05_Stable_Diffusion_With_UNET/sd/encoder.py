# creating VAE encoder for image compression before UNET

# imports
import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_RisidualBlock

# the goal is to reduce the amount of information present in the image and increase the amount of features per pixel
class VAE_Encoder(nn.Sequential):
    def __init__(self, ):
        super().__init__(
            # expand RGB color channels (3) to 128 features while preserving spatial size
            nn.Conv2d(
                in_channels=3,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            # refine features at 128 channels before downsampling
            VAE_RisidualBlock(
                128,
                128
            ),
            # enables the model to learn complex transformations through risiduals
            VAE_RisidualBlock(
                128,
                128
            ),
            # strided conv halves height and width (padding added manually in forward() for stride=2)
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=0
            ),
            # increase features from 128 to 256 as spatial resolution decreases
            VAE_RisidualBlock(
                128,
                256
            ),
            # refine features at 256 channels before next downsample
            VAE_RisidualBlock(
                256,
                256
            ),
            # strided conv halves height and width again
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=0
            ),
            # increase features from 256 to 512 as spatial resolution decreases
            VAE_RisidualBlock(
                256,
                512
            ),
            # refine features at 512 channels before next downsample
            VAE_RisidualBlock(
                512,
                512
            ),
            # strided conv halves height and width a final time (image is now height/8, width/8)
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=0
            ),
            # refine features at full depth before attention
            VAE_RisidualBlock(
                512,
                512
            ),
            VAE_RisidualBlock(
                512,
                512
            ),
            VAE_RisidualBlock(
                512,
                512
            ),
            # attention block: lets every spatial position attend to every other position
            VAE_AttentionBlock(
                512
            ),
            # final residual refinement after attention
            VAE_RisidualBlock(
                512,
                512
            ),
            # normalization to normalize the data (number of groups, number of features)
            nn.GroupNorm(
                32,
                512
            ),
            # activation function that functions similar to ReLU()
            nn.SiLU(),
            # compress 512 features down to 8 (mean and log_var, 4 channels each) for the latent space
            nn.Conv2d(
                in_channels=512,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),
            # 1x1 conv: mixes channels without changing spatial size, finalizing the latent representation
            nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=1,
                padding=0
            ),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: batchsize, channel, height, width
        # noise: batchsize, out channels, height/8, width/8

        for module in self:
            # strided convs use padding=0, so manually pad the right and bottom by 1 pixel
            # this ensures the output size is exactly half (avoids asymmetric cropping)
            if getattr(module, "stride", None) == (2, 2):
                # padding order: (left, right, top, bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # output of the module is not an image but a distribution
        # chunk splits the 8-channel output into two 4-channel tensors: mean and log_var
        mean, log_var = torch.chunk(x, 2, dim=1)

        # clamp log_var to a safe range to prevent numerical instability during exp()
        log_var = torch.clamp(log_var, -30, 20)
        # convert log variance to variance, does not change size
        variance = log_var.exp()
        # standard deviation of the distribution, does not change size
        stdev = variance.sqrt()
        # reparameterization trick: sample from N(mean, stdev) using N(0,1) noise
        # allows gradients to flow through the sampling step during training
        x = mean + stdev * noise
        # scale the output by a constant to normalize the latent to unit variance
        x *= 0.18215

        return x