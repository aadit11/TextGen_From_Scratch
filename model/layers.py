"""
Transformer model layer implementations.

This module implements the core layers used in the transformer architecture,
including multi-head attention which allows the model to jointly attend to
information from different representation subspaces.
"""

import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention mechanism for transformer models.

    This layer performs attention in parallel across multiple heads, where each head
    can focus on different aspects of the input. The outputs from all heads are
    concatenated and linearly transformed to produce the final output.

    The multi-head attention allows the model to:
    1. Learn different aspects of the input in different representation subspaces
    2. Attend to different positions simultaneously
    3. Capture different types of dependencies in the data

    Args:
        d_model (int): The dimension of the model
        n_heads (int): Number of attention heads. Must divide d_model evenly

    Raises:
        AssertionError: If d_model is not divisible by n_heads
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        """
        Compute multi-head attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, d_model)
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, d_model)
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len_v, d_model)
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, n_heads, seq_len_q, seq_len_k).
                Values of 0 in the mask are replaced with -inf, values of 1 are kept as is.
                Defaults to None.

        Returns:
            torch.Tensor: The attention output of shape (batch_size, seq_len_q, d_model)
        """
        batch_size = query.shape[0]

        Q = self.query(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, V).transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.head_dim)
        return self.fc_out(out)
