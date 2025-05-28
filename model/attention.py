"""
Attention mechanism implementation for the transformer model.

This module implements the scaled dot-product attention mechanism as described in
the "Attention Is All You Need" paper. It includes the core attention computation
that allows the model to focus on different parts of the input sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Implements scaled dot-product attention mechanism.

    This attention mechanism computes attention scores between queries and keys,
    scales them by the square root of the dimension, applies optional masking,
    and uses softmax to get attention weights. The attention weights are then
    used to compute a weighted sum of the values.

    The scaling factor (1/sqrt(d_k)) helps prevent the dot products from growing
    too large in magnitude, which would push the softmax function into regions
    with small gradients.

    Args:
        d_model (int): The dimension of the model, used to scale the attention scores
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_k = d_model

    def forward(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.

        Args:
            Q (torch.Tensor): Query tensor of shape (..., seq_len_q, d_k)
            K (torch.Tensor): Key tensor of shape (..., seq_len_k, d_k)
            V (torch.Tensor): Value tensor of shape (..., seq_len_v, d_v)
            mask (torch.Tensor, optional): Mask tensor of shape (..., seq_len_q, seq_len_k).
                Values of 0 in the mask are replaced with -inf, values of 1 are kept as is.
                Defaults to None.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The attention output of shape (..., seq_len_q, d_v)
                - torch.Tensor: The attention weights of shape (..., seq_len_q, seq_len_k)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        return torch.matmul(attention, V), attention
