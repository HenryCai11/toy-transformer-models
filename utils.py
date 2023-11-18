import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def clone(layer, num):
    from copy import deepcopy
    return nn.ModuleList([deepcopy(layer) for _ in range(num)])


def get_subsequent_mask(l):
    all_ones = torch.ones(l, l)
    triu_ones = torch.triu(all_ones, diagonal=1)

    return triu_ones == 0


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.model.decode(
            memory, src_mask, ys, get_subsequent_mask(ys.size(1)).unsqueeze(0).
            type_as(src.data)
        )
        prob = model.generate(out[:, -1])
        id, next_word = torch.max(prob, dim=1)
        print(id)
        print(next_word)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


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
    
class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
    
class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & get_subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)

if __name__ == "__main__":
    pos_enc = PositionalEncoding(d_model=512)