"""
Text generation module using a trained transformer model.

This module provides functionality to generate text using a trained transformer model.
It includes utilities for text generation with temperature-based sampling and
handles the loading and initialization of the model and vocabulary.
"""

import torch
from model.transformer import Transformer
from utils.tokenizer import tokenize, detokenize

def generate_text(model, vocab, seed_text, max_length=50, temperature=1.0):
    """
    Generates text using a trained transformer model with temperature-based sampling.

    The function takes a seed text and generates a sequence of tokens by:
    1. Tokenizing the input seed text
    2. Iteratively predicting the next token using the model
    3. Sampling from the probability distribution with temperature scaling
    4. Converting the generated tokens back to text

    Args:
        model (Transformer): The trained transformer model
        vocab (dict): Vocabulary mapping tokens to indices
        seed_text (str): Initial text to start generation from
        max_length (int, optional): Maximum number of tokens to generate. Defaults to 50
        temperature (float, optional): Controls randomness in generation. 
            Higher values (e.g., 1.0) make output more random, 
            lower values (e.g., 0.5) make it more focused. Defaults to 1.0

    Returns:
        str: The generated text, including the seed text
    """
    model.eval()
    tokens = tokenize(seed_text, vocab)
    input_seq = torch.tensor(tokens).unsqueeze(0)

    generated_tokens = tokens.copy()

    with torch.no_grad():
        for _ in range(max_length):
            input_seq = input_seq[:, -min(len(input_seq[0]), 64):]
            
            output = model(input_seq)
            logits = output[0, -1, :] / temperature
            
            probabilities = torch.softmax(logits, dim=-1)
            
            next_token = torch.multinomial(probabilities, 1).item()
            
            generated_tokens.append(next_token)
            input_seq = torch.tensor(generated_tokens).unsqueeze(0)

    return detokenize(generated_tokens, vocab)

if __name__ == "__main__":
    from config.config import config
    
    model = Transformer(config['vocab_size'], config['d_model'], config['n_heads'], config['d_ff'], config['n_layers'])
    model.load_state_dict(torch.load(config['checkpoint_path']))
    model.eval()

    # Load vocabulary
    with open('data/processed/train_data.pt', 'rb') as f:
        _, vocab = torch.load(f)

    seed_texts = [
        "Once upon a time",
    ]

    for seed in seed_texts:
        print(f"\nSeed: {seed}")
        generated = generate_text(model, vocab, seed, max_length=100)
        print(generated)