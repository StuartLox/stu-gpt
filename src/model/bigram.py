import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLangaugeModel(nn.Module):
    """
    Bigram Langauge Model uses a sequence of embeddings to predict
    a future embedding.

    :param vocab_size: number of unique tokens from the training set
    :param token_embedding_table: Lookup table which stores embeddings
    """
    def __init__(self, vocab_size: int, n_embed: int = 32, block_size: int = 32):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx: int, targets: torch.Tensor = None):
        """
        Computation to be executed each forward pass of the neural network

        :param: idx: index in batch size
        :param: targets: vector of predictions

        :returns: logits: Batch, Time, Channel (B,T,C)
        :returns: loss: Cross Entropy Loss between input logits and target
        """
        B, T = idx.shape

        # idx  and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (T,C)
        pos_emb = self.token_embedding_table(torch.arange(T))  # (T,C)
        x = tok_emb + pos_emb(B, T)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: int, max_new_tokens: int) -> int:
        """
        Generate the required index

        :param: idx: is (B, T) array of indices in required context
        :param: max_new_tokens: token length to iterate over
        """
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
