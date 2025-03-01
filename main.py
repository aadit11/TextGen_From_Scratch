import torch
from training.train import train_model
from generation.generate import generate_text
from config.config import config
from model.transformer import Transformer

def main():
    # Train the model
    print("🔹 Starting Training...")
    train_model('data/processed/train_data.pt', config['vocab_size'], config)
    
    # Load trained model
    print("\n🔹 Loading Trained Model for Generation...")
    model = Transformer(config['vocab_size'], config['d_model'], config['n_heads'], config['d_ff'], config['n_layers'])
    model.load_state_dict(torch.load(config['checkpoint_path']))
    model.eval()

    # Load vocabulary
    with open('data/processed/train_data.pt', 'rb') as f:
        _, vocab = torch.load(f)

    # Generate text
    seed_text = "Once upon a time"
    generated_text = generate_text(model, vocab, seed_text)
    
    print("\n🔹 Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
