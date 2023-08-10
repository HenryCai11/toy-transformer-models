import math
import torch
import torch.nn as nn

def clone(layer, num):
    from copy import deepcopy
    return nn.ModuleList([deepcopy(layer) for _ in range(num)])

def get_subsequent_mask(l):
    all_ones = torch.ones(l, l)
    triu_ones = torch.triu(all_ones, diagonal=1)

    return triu_ones == 0

class PositionalEncoding(nn.Module):
    '''
        PE_(pos, 2i) = sin(pos/10000^(2i/d_model))
        PE_(pos, 2i+1) = cos(pos/10000^(2i/d_model))

        Intuitions behind positional encoding:
        https://kikaben.com/transformers-positional-encoding/
    '''
    def __init__(self, d_model, max_len=25000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(dim=1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("position", pe, persistent=False)

    def forward(self, x):
        x = x + self.position[:x.size(1), :].requires_grad_(False)
        return self.dropout(x)
    
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.a1 = nn.Parameter(torch.ones(d_model))
        self.b1 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # mean = torch.mean(x, dim=-1, keepdim=True)
        # std = torch.std(x, dim=-1, keepdim=True)
        std, mean = torch.std_mean(x, dim=-1, keepdim=True)
        return self.a1 * (x - mean) / (std + self.eps) + self.b1


class SubLayer(nn.Module):
    '''
        Add & Norm
    '''
    def __init__(self, d_model, dropout):
        super().__init__()
        self.layernorm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # return x + self.layernorm(self.dropout(x))
        pass


class Embedding(nn.Module):
    def __init__(self, vocab, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedder = nn.Embedding(vocab, d_model)
        self.size = (vocab, d_model)

    def forward(self, x):
        return self.embedder(x) * math.sqrt(self.d_model)


if __name__ == "__main__":
    pos_enc = PositionalEncoding(d_model=512)