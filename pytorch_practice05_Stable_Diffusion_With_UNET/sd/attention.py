# creating attention class: ability to relate tokens to each other

# imports
import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        # single linear layer that projects input into Q, K, and V all at once (3x the embed dim)
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
        # projects concatenated head outputs back to the original embed dim
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        # each head works on a slice of the embedding: d_embed split evenly across heads
        self. d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        # x: batchsize, seq_len, d_embed
        input_shape = x.shape

        batch_size, seqeunce_length, d_embed = input_shape

        # intermediate shape used to split embed dim into (n_heads, d_head) for multi-head attention
        intermin_shape = (batch_size, seqeunce_length, self.n_heads, self.d_head)

        # convert batchsize, seq_len, dim to dim * 3 and chunk makes it 3 tensors with the original shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # each token/pizel has their view include number of heads and their dim / number of heads and change view
        q = q.view(intermin_shape).transpose(1, 2)
        k = k.view(intermin_shape).transpose(1, 2)
        v = v.view(intermin_shape).transpose(1, 2)

        # matrix multiplication giving (batchsize, number of heads, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)

        # applying the mask if desired to have two tokens to relate to each other
        if causal_mask:
            # mask where the upper triangle (above the diagonal) is made up of ones
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            # every other value in the matrix is - infinity
            weight.masked_fill_(mask, -torch.inf)

        # scale dot products by sqrt(d_head) to prevent vanishing gradients from large values
        weight /= math.sqrt(self.d_head)

        # softmax over the last dim so each token's attention scores sum to 1
        weight = F.softmax(weight, dim=-1)

        # output: batchsize, number of heads, seq_len, dim / number of heads
        output = weight @ v
        # output: batchsize, seq_len, number of heads, dim / number of heads
        output = output.transpose(1, 2)
        # merge all heads back together by reshaping to the original input shape
        output = output.reshape(input_shape)

        # final linear projection mixes information across heads
        output = self.out_proj(output)

        return output
    
class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embd: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embd, d_embd, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embd, bias=in_proj_bias)        
        self.v_proj = nn.Linear(d_cross, d_embd, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embd, d_embd, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embd // n_heads

    def forward(self, x, y):
        # x is latent, y is context
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)

        return output


        