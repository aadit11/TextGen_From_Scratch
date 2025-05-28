"""
Training module for the transformer model.

This module implements the training loop for the transformer model, handling
data loading, model initialization, optimization, and checkpointing. It includes
utilities for training the model on preprocessed text data and saving the
trained model weights.
"""

import torch
import torch.optim as optim
import os
from model.transformer import Transformer
from utils.tokenizer import detokenize

def train_model(data_path, vocab_size, config):
    """
    Train the transformer model on the provided data.

    This function implements the training loop for the transformer model, which:
    1. Loads preprocessed data and vocabulary
    2. Initializes the transformer model
    3. Sets up the optimizer and loss function
    4. Trains the model for the specified number of epochs
    5. Saves the model checkpoint after training

    The training process:
    - Processes data in batches
    - Uses teacher forcing (input sequence is shifted by one position for targets)
    - Computes cross-entropy loss between model predictions and targets
    - Updates model parameters using backpropagation
    - Tracks and reports average loss per epoch

    Args:
        data_path (str): Path to the preprocessed training data (.pt file)
        vocab_size (int): Size of the vocabulary
        config (dict): Configuration dictionary containing:
            - d_model (int): Model dimension
            - n_heads (int): Number of attention heads
            - d_ff (int): Feed-forward network dimension
            - n_layers (int): Number of transformer layers
            - lr (float): Learning rate
            - epochs (int): Number of training epochs
            - checkpoint_path (str): Path to save the model checkpoint

    Returns:
        None: The trained model is saved to the checkpoint path specified in config
    """
    data, vocab = torch.load(data_path)
    model = Transformer(vocab_size, config['d_model'], config['n_heads'], config['d_ff'], config['n_layers'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(config['checkpoint_path']), exist_ok=True)

    for epoch in range(config['epochs']):
        total_loss = 0
        batch_count = 0
        
        for batch in data:
            input_seq = torch.tensor(batch[:-1])
            target_seq = torch.tensor(batch[1:])
            output = model(input_seq.unsqueeze(0)).squeeze(0)

            loss = criterion(output.view(-1, vocab_size), target_seq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{config['epochs']}: Loss {avg_loss:.4f}")

    torch.save(model.state_dict(), config['checkpoint_path'])
    print(f"Model saved to {config['checkpoint_path']}")

if __name__ == "__main__":
    from config.config import config
    train_model('data/processed/train_data.pt', config['vocab_size'], config)