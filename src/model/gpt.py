import torch
import torch.nn as nn
from config.config import Config
from torch.nn import functional as F


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, config: Config, head_size: int):
        super().__init__()
        self.key = nn.Linear(config.model.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.model.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.model.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.data.block_size, config.data.block_size)))

        self.dropout = nn.Dropout(config.model.dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """
    In Multi-Head Attention the input sequence is first transformed into three kinds of vectors:

    1. Query vectors: Calculates the similarity between the current input token and other tokens in the sequence.
    2. Key vectors: Represent the input sequence by multiplying the input sequence by a learned weight matrix.
    3. Value vectors: Value vectors are used to compute the final output of Multi-Head Attention.

    Multi-head attention allows the model to jointly attend to information from different representation subspaces
    at different positions.
    """

    def __init__(self, config: Config):
        super().__init__()
        head_size = config.model.n_embd // config.model.n_head
        self.heads = nn.ModuleList([Head(config=config, head_size=head_size) for _ in range(config.model.n_head)])
        self.proj = nn.Linear(head_size * config.model.n_head, config.model.n_embd, device=config.train.device)
        self.dropout = nn.Dropout(config.model.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, config: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.model.n_embd, 4 * config.model.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.model.n_embd, config.model.n_embd),
            nn.Dropout(config.model.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Block is the Transformer component performs the communication using the muti-head Attention
    followed by computation runs foward pass in the NN.
    """

    def __init__(self, config: Config):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedFoward(config)
        self.ln1 = nn.LayerNorm(config.model.n_embd)
        self.ln2 = nn.LayerNorm(config.model.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size: int, config: Config):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.config = config

        self.token_embedding_table = nn.Embedding(vocab_size, config.model.n_embd, device=config.train.device)
        self.position_embedding_table = nn.Embedding(config.data.block_size, config.model.n_embd, device=config.train.device)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.model.n_layer)])
        self.ln_f = nn.LayerNorm(config.model.n_embd)  # final layer norm
        self.lm_head = nn.Linear(config.model.n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.config.train.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.model.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
