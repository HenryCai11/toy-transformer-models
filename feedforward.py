import torch
import torch.nn as nn


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


class FeedForwardNetwork(SubLayer):
    def __init__(self, d_model, dropout):
        super().__init__(d_model, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=True),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model, bias=True)
        )

    def forward(self, x):
        x = self.ffn(x)
        return x + self.layernorm(self.dropout(x))
