# creating the pipeline for the data

# imports
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from ddpm import DDPMSampler

# image dimensions and their reduced latent space equivalents (VAE downsamples by 8x)
WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = 512 // 8
LATENTS_HEIGHT = 512 // 8

def generator(
        prompt: str, 
        unconditional_prompt: str, # also known as the negative prompt
        input_image=None,          # if provided, run image-to-image; otherwise text-to-image
        strength=0.8,              # how much noise to add to the input image (higher = more creative)
        do_cfg=True,               # whether to use classifier-free guidance
        cfg_scalar=7.5,            # how much the model show pay attention to condition(s)
        sampler_name="ddpm",
        n_inference_steps=50,      # number of denoising steps
        models={},
        seed=None,
        device=None,
        idle_device=None,          # device to offload models to when not in use
        tokenizer=None
    ):

    with torch.inference_mode():
        if not (0 < strength <= 1):
            raise ValueError("Strength must be between 0 and 1")
        # move models to idle device to free memory when not active
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device = device)

        # seed the generator for reproducibility, or use a random seed
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        # the process of combining two promtps 
        if do_cfg:
            # encode the conditional (positive) prompt into token ids
            cond_tokens = tokenizer.__call__(
                [prompt],
                padding= "max_length",
                max_length=77
            ).input_ids
            # (batch_size, seq_len)
            cond_tokens = torch.tensor(
                cond_tokens,
                dtype=torch.long,
                device=device
            )
            # (batch_size, seq_len, d_embed)
            conditional_context = clip(cond_tokens)

            # encode the unconditional (negative) prompt into token ids
            uncond_tokens = tokenizer.__call__(
                [unconditional_prompt],
                padding= "max_length",
                max_length=77
            ).input_ids
            # (batch_size, seq_len)
            unconditional_tokens = torch.tensor(
                uncond_tokens,
                dtype=torch.long,
                device=device
            )
            # (batch_size, seq_len, d_embed)
            unconditional_context = clip(unconditional_tokens)

            # stack both contexts so the model can process them in a single forward pass
            context = torch.cat([conditional_context, unconditional_context])
        # if only one prompt
        else:
            tokens = tokenizer.__call__(
                [prompt],
                padding= "max_length", 
                max_length=77
            ).input_ids
            tokens = torch.tensor(
                cond_tokens, 
                dtype=torch.long, 
                device=device
            )
            context = clip(tokens)

        # offload clip to idle device now that context embeddings are computed
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            # set the number of timesteps to use during inference (subset of 1000 training steps)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        # (batch, channels, height, width) — 4 latent channels from the VAE
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            # image-to-image: encode the input image into latent space then add noise
            encoder = models["encoder"]
            encoder.to(device)

            # resize to expected dimensions and convert to numpy array
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (height, width, channels) -> float32 tensor
            input_image_tensor = torch.tensor(
                input_image_tensor,
                dtype=torch.float32,
                device=device
                )
            # scale pixel values from [0, 255] to [-1, 1] for the encoder
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # add batch dim: (height, width, channels) -> (1, height, width, channels)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # reorder to (batch, channels, height, width) for PyTorch convolutions
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(
                latents_shape,
                generator=generator,
                device=device
                )
            # encode image to latent space
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to the latents (the encoded input image)
            # strength controls how many timesteps to skip, determining how much the image changes
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # text-to-image: start from pure gaussian noise in latent space
            latents = torch.randn(
                latents_shape,
                generator=generator,
                device=device
                )

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            model_input = latents

            if do_cfg:
                # duplicate the latents so the model processes conditional and unconditional in one pass
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                # split back into conditional and unconditional predictions
                output_cond, output_uncond = model_output.chunk(2)
                # apply classifier-free guidance: interpolate between uncond and cond
                model_output = cfg_scalar * (output_cond - output_uncond) + output_uncond

            # remove predicted noise to get the slightly denoised latent for the next step
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # decode latents back to pixel space: (batch, channels, height, width)
        images = decoder(latents)
        to_idle(decoder)

        # rescale from [-1, 1] to [0, 255] for standard image representation
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # reorder from (batch, channels, height, width) to (batch, height, width, channels)
        images = images.permute(0, 2, 3, 1)
        # move to cpu and convert to uint8 numpy array for output
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    # shift to zero, scale to new range, then shift to new min
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2) — concat cos and sin for a full sinusoidal positional embedding
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)