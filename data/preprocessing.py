import torch
from utils.tokenizer import build_vocab, tokenize
import os

def preprocess_text(input_file, output_file, vocab_size=5000, seq_length=64, encoding='utf-8'):
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
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    torch.save((data, vocab), output_file)
    print(f"Data saved to {output_file}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Total tokens: {len(tokenized_text)}")
    print(f"Number of sequences: {len(data)}")

if __name__ == "__main__":
    preprocess_text('data/data.txt', 'data/processed/train_data.pt')