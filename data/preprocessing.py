import torch
from utils.tokenizer import build_vocab, tokenize
import os

def preprocess_text(input_file, output_file, vocab_size=50000, seq_length=64, encoding='utf-8'):
    """
    Preprocesses text data for training a language model.

    This function performs the following steps:
    1. Reads text from the input file, attempting UTF-8 encoding first, then falling back to latin-1
    2. Converts text to lowercase
    3. Builds a vocabulary from the text
    4. Tokenizes the text using the built vocabulary
    5. Creates sequences of specified length for training
    6. Saves the processed data and vocabulary as a PyTorch file

    Args:
        input_file (str): Path to the input text file
        output_file (str): Path where the processed data will be saved
        vocab_size (int, optional): Maximum size of the vocabulary. Defaults to 50000
        seq_length (int, optional): Length of sequences to create for training. Defaults to 64
        encoding (str, optional): Initial encoding to try when reading the file. Defaults to 'utf-8'

    Raises:
        Exception: If the file cannot be read with either UTF-8 or latin-1 encoding

    Returns:
        None: The function saves the processed data to disk and prints statistics
    """
    try:
        with open(input_file, 'r', encoding=encoding) as file:
            text = file.read().lower()
    except UnicodeDecodeError:
        try:
            with open(input_file, 'r', encoding='latin-1') as file:
                text = file.read().lower()
        except Exception as e:
            print(f"Error reading file: {e}")
            raise

    vocab = build_vocab(text, vocab_size)
    tokenized_text = tokenize(text, vocab)

    data = [tokenized_text[i:i+seq_length] for i in range(0, len(tokenized_text) - seq_length, seq_length)]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    torch.save((data, vocab), output_file)
    print(f"Data saved to {output_file}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Total tokens: {len(tokenized_text)}")
    print(f"Number of sequences: {len(data)}")

if __name__ == "__main__":
    preprocess_text('data/data.txt', 'data/processed/train_data.pt')