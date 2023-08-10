import torch
import torch.nn as nn
from utils import SubLayer


class FeedForwardNetwork(SubLayer):
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__(d_model, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=True),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model, bias=True)
        )

    def forward(self, x):
        x = self.ffn(x)
        return x + self.layernorm(self.dropout(x))
