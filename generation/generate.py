import torch
from model.transformer import Transformer
from utils.tokenizer import tokenize, detokenize

def generate_text(model, vocab, seed_text, max_length=50):
    model.eval()
    tokens = tokenize(seed_text, vocab)
    input_seq = torch.tensor(tokens).unsqueeze(0)

    for _ in range(max_length):
        with torch.no_grad():
            output = model(input_seq).squeeze(0)
            next_token = output[-1].argmax().item()
            tokens.append(next_token)
            input_seq = torch.tensor(tokens).unsqueeze(0)

    return detokenize(tokens, vocab)

if __name__ == "__main__":
    from config.config import config
    model = Transformer(config['vocab_size'], config['d_model'], config['n_heads'], config['d_ff'], config['n_layers'])
    model.load_state_dict(torch.load(config['checkpoint_path']))
    
    with open('data/processed/train_data.pt', 'rb') as f:
        _, vocab = torch.load(f)

    print(generate_text(model, vocab, "Once upon a time"))