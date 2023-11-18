import torch
import torch.nn as nn
from utils import SubLayer


class FeedForwardNetwork(SubLayer):
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__(d_model, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(d_model * 4, d_model, bias=True)
        )

    def forward(self, x):
        return x + self.dropout(self.ffn(self.layernorm(x)))
