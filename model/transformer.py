"""
Transformer model implementation for text generation.

This module implements a transformer model architecture as described in the
"Attention Is All You Need" paper, adapted for text generation tasks. It includes
the core transformer block with multi-head attention and feed-forward networks,
as well as the complete transformer model with embedding and positional encoding layers.
"""

import torch
import math
import torch.nn as nn
from model.layers import MultiHeadAttention
from model.positional_encoding import PositionalEncoding
from model.attention import ScaledDotProductAttention

class TransformerBlock(nn.Module):
    """
    Implements a single transformer block with self-attention and feed-forward network.

    A transformer block consists of:
    1. Multi-head self-attention layer
    2. Layer normalization and residual connection
    3. Feed-forward network (two linear layers with ReLU activation)
    4. Layer normalization and residual connection

    Args:
        d_model (int): The dimension of the model
        n_heads (int): Number of attention heads
        d_ff (int): Dimension of the feed-forward network
    """

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Process input through the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        return self.norm2(x + ffn_output)

class Transformer(nn.Module):
    """
    Implements a complete transformer model for text generation.

    The model consists of:
    1. Token embeddings
    2. Positional encoding
    3. Stack of transformer blocks
    4. Final linear layer for vocabulary prediction

    Args:
        vocab_size (int): Size of the vocabulary
        d_model (int): Dimension of the model
        n_heads (int): Number of attention heads
        d_ff (int): Dimension of the feed-forward network
        n_layers (int): Number of transformer blocks
    """

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x):
        """
        Process input through the transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len)
                containing token indices

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size)
                containing logits for each token in the vocabulary
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)