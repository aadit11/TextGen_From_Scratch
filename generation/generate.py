import torch
from model.transformer import Transformer
from utils.tokenizer import tokenize, detokenize

def generate_text(model, vocab, seed_text, max_length=50, temperature=1.0):
    model.eval()
    tokens = tokenize(seed_text, vocab)
    input_seq = torch.tensor(tokens).unsqueeze(0)

    generated_tokens = tokens.copy()

    with torch.no_grad():
        for _ in range(max_length):
            # Ensure input doesn't exceed model's expected sequence length
            input_seq = input_seq[:, -min(len(input_seq[0]), 64):]
            
            output = model(input_seq)
            # Get the last token's logits
            logits = output[0, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits, dim=-1)
            
            # Sample from the distribution instead of taking argmax
            next_token = torch.multinomial(probabilities, 1).item()
            
            generated_tokens.append(next_token)
            input_seq = torch.tensor(generated_tokens).unsqueeze(0)

    return detokenize(generated_tokens, vocab)

if __name__ == "__main__":
    from config.config import config
    
    # Load model
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