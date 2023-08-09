import math
import torch
import torch.nn as nn


def attention(q, k, v, mask=None):
    attention_score = torch.matmul(q, k.transpose(-1, -2))
    attention_score = torch.softmax(attention_score / math.sqrt(k.size(-1)),
                                    dim=-1)
    if mask is not None:
        output = torch.matmul(attention_score + mask, v)
    else:
        output = torch.matmul(attention_score, v)

    return output, attention_score


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_head=6, ):
        super().__init__()
        if d_model % num_head != 0:
            raise RuntimeError(
                "The hidden size can not be evenly divided by the number \
                    of heads"
            )
        self.head = num_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(q.size(0), -1, self.head, self.d_model
                   // self.head).transpose(-2, -3)
        k = k.view(k.size(0), -1, self.head, self.d_model
                   // self.head).transpose(-2, -3)
        v = v.view(v.size(0), -1, self.head, self.d_model
                   // self.head).transpose(-2, -3)

        output, attention_score = attention(q, k, v)

        output = (
            output.transpose(-2, -3)
            .contiguous()
            .view(output.size(0), -1, self.d_model)
        )

        return output, attention_score


if __name__ == "__main__":
    attn = MultiHeadAttention()
