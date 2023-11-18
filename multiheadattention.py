import math
import torch
from utils import SubLayer
import torch.nn as nn


def attention(q, k, v, mask=None, dropout=None):
    d_k = k.size(-1)
    attention_score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        # print("mask: ", mask.shape)
        attention_score = attention_score.masked_fill(mask == 0, -1e9)
    attention_score = torch.softmax(attention_score, dim=-1)

    if dropout is not None:
        attention_score = dropout(attention_score)

    output = torch.matmul(attention_score, v)

    return output, attention_score


class MultiHeadAttention(SubLayer):
    def __init__(self, d_model=512, num_head=8, dropout=0.1):
        super().__init__(d_model, dropout)
        if d_model % num_head != 0:
            raise RuntimeError(
                "The hidden size can not be evenly divided by the number\
                    of heads with d_model{} and #head{}".format(d_model, num_head)
            )
        self.head = num_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None, cross=False):
        query = self.layernorm(query)
        if not cross:
            key = self.layernorm(key)
            value = self.layernorm(value)
        if mask is not None:
            mask = mask.unsqueeze(dim=1)
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        q = q.view(q.size(0), -1, self.head, self.d_model
                   // self.head).transpose(-2, -3)
        k = k.view(k.size(0), -1, self.head, self.d_model
                   // self.head).transpose(-2, -3)
        v = v.view(v.size(0), -1, self.head, self.d_model
                   // self.head).transpose(-2, -3)

        output, attention_score = attention(q, k, v, mask, self.dropout)

        output = (
            output.transpose(-2, -3)
            .contiguous()
            .view(output.size(0), -1, self.d_model)
        )

        output = self.w_o(output)

        return query + self.dropout(output)


if __name__ == "__main__":
    attn = MultiHeadAttention()
    q = torch.randn(1, 5, 5)
    k = torch.randn(1, 5, 5)
    v = torch.randn(1, 5, 5)
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    print(attention(q, k, v, mask)[1])
