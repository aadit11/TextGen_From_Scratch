from collections import Counter
import re

def tokenize(text, vocab):
    words = re.findall(r'\b\w+\b', text.lower())
    return [vocab.get(word, vocab["<UNK>"]) for word in words]

def detokenize(tokens, vocab):
    reverse_vocab = {v: k for k, v in vocab.items()}
    return " ".join([reverse_vocab.get(token, "<UNK>") for token in tokens])

def build_vocab(text, vocab_size=5000):
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    most_common = word_counts.most_common(vocab_size - 2)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab.update({word: idx + 2 for idx, (word, _) in enumerate(most_common)})

    return vocab
