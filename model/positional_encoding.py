"""
Positional encoding implementation for transformer models.

This module implements sinusoidal positional encoding as described in the
"Attention Is All You Need" paper. Positional encoding is crucial for transformer
models as they have no inherent notion of token order, and this encoding
provides information about the relative or absolute position of tokens in the sequence.
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding for transformer models.

    This layer adds positional information to the input embeddings using
    sine and cosine functions of different frequencies. The encoding is
    added to the input embeddings to give the model information about
    the relative or absolute position of tokens in the sequence.

    The positional encoding is computed as:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    where pos is the position and i is the dimension.

    Args:
        d_model (int): The dimension of the model
        max_len (int, optional): Maximum sequence length. Defaults to 5000
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
                with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]