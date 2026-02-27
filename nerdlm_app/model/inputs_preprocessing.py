import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 256):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_length = max_length

    def forward(self, x):
        pe = torch.zeros(self.max_length, self.d_model)
        pos = torch.arange(0, self.max_length, dtype=torch.float32).unsqueeze(1).to(torch.int64)
        i = torch.arange(0, self.d_model, 2, dtype=torch.float32).unsqueeze(0).to(torch.int64)
        pe[pos, i] = torch.sin(pos / 10000 ** (i / self.d_model))
        pe[pos, i + 1] = torch.cos(pos / 10000 ** ((i + 1) / self.d_model))

        seq_len = x.size(-1)
        x = x + pe[:seq_len, :].unsqueeze(0)

        return x

class InputPreprocessing(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, max_length: int = 256):
        super(InputPreprocessing, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length)
    
    def subsequent_mask(self, size):
        return torch.triu(torch.ones(size, size)).transpose(0, 1).type(torch.uint8).unsqueeze(0)

    def create_mask(self, token_ids):
        x_mask = token_ids != 0
        seq_len = token_ids.size(-1)
        x_mask = x_mask.unsqueeze(1) & self.subsequent_mask(seq_len).type_as(x_mask.data)
        return x_mask

    def forward(self, token_ids):
        x_embedded = self.embedding(token_ids)
        x = self.positional_encoding(x_embedded)
        x_mask = self.create_mask(token_ids)
        return x, x_mask
