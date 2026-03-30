# creating the CLIP to understand text prompt and encode them

# imports
import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
        super().__init__()

        # define the embedding (what is the number of embeddings)
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # defining positional encoding (uses learned parameters during training)
        self.posistion_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))

    def forward(self, tokens):
        # converting batch size, seq_len to batch size, seq_len, and dim
        x = self.token_embedding(tokens)
        # add postional encodings to each token
        x += self.posistion_embedding
        # return x
        return x
    
class CLIPLayer(nn.Module):
    def __init__(self, n_heads: int, n_embd: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_heads, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # is batch size, seq_len, and dim
        residue = x

        # self attention 
        x = self.layernorm_1(x)
        x = self.attention(x, casual_mask=True)
        x += residue

        # feed forward layer
        residue = x
        x = self.layernorm_2(x)
        self.linear_1(x)

        # this is the quick GELU activation function, similar to ReLU
        x = x * torch.sigmoid(1.702 * x)

        self.linear_2(x)
        x += residue
        # return x
        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # convert tokens (created by text to numbers) into embeddings (which will be a vector)
        self.embedding = CLIPEmbedding(48408, 768, 77) # in order: vocab size, embedding size, padding
        self.layers = nn.Module([
            # creating 12 CLIP layers
            CLIPLayer(12, 768) for i in range(12)
        ])
        # normalizing each layer
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor: # float tensor because input IDs are numbers 
        tokens = tokens.type(torch.long)
        # converting batch size, seq_len to batch size, seq_len, and dim
        state = self.embedding(tokens)
        # pass layers through each layer of the decoder
        for layer in self.layers:
            state = layer(state)
        # normalize the layers
        output = self.layernorm(state)
        # return output
        return output