"""
Text tokenization utilities for the transformer model.

This module provides functions for text tokenization, detokenization, and vocabulary
building. It implements a simple word-based tokenization scheme that:
- Converts text to lowercase
- Splits on word boundaries
- Handles unknown words with a special <UNK> token
- Includes special tokens for padding (<PAD>) and unknown words (<UNK>)
"""

from collections import Counter
import re

def tokenize(text, vocab):
    """
    Convert text into a sequence of token IDs using the provided vocabulary.

    This function:
    1. Converts text to lowercase
    2. Splits text into words using word boundaries
    3. Maps each word to its corresponding token ID from the vocabulary
    4. Uses <UNK> token ID for words not in the vocabulary

    Args:
        text (str): Input text to tokenize
        vocab (dict): Vocabulary mapping words to token IDs

    Returns:
        list: List of token IDs corresponding to the input text
    """
    words = re.findall(r'\b\w+\b', text.lower())
    return [vocab.get(word, vocab["<UNK>"]) for word in words]

def detokenize(tokens, vocab):
    """
    Convert a sequence of token IDs back into text.

    This function:
    1. Creates a reverse mapping from token IDs to words
    2. Converts each token ID to its corresponding word
    3. Uses <UNK> for token IDs not in the vocabulary
    4. Joins the words with spaces

    Args:
        tokens (list): List of token IDs to convert
        vocab (dict): Vocabulary mapping words to token IDs

    Returns:
        str: The reconstructed text
    """
    reverse_vocab = {v: k for k, v in vocab.items()}
    return " ".join([reverse_vocab.get(token, "<UNK>") for token in tokens])

def build_vocab(text, vocab_size=5000):
    """
    Build a vocabulary from the input text.

    This function:
    1. Converts text to lowercase and splits into words
    2. Counts word frequencies
    3. Creates a vocabulary with the most common words
    4. Adds special tokens (<PAD> and <UNK>)
    5. Assigns token IDs (0 for <PAD>, 1 for <UNK>, 2+ for words)

    Args:
        text (str): Input text to build vocabulary from
        vocab_size (int, optional): Maximum size of the vocabulary, including
            special tokens. Defaults to 5000. The actual vocabulary will contain
            vocab_size - 2 words plus the two special tokens.

    Returns:
        dict: Vocabulary mapping words to token IDs, with the following structure:
            - "<PAD>": 0
            - "<UNK>": 1
            - word: idx + 2 (for each word in the vocabulary)
    """
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    most_common = word_counts.most_common(vocab_size - 2)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab.update({word: idx + 2 for idx, (word, _) in enumerate(most_common)})

    return vocab
