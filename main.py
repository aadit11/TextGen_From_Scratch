import torch
import os
import sys
from training.train import train_model
from generation.generate import generate_text
from config.config import config
from model.transformer import Transformer
from data.preprocessing import preprocess_text

def main():
    try:
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
        
        if not os.path.exists('data/processed/train_data.pt'):
            print("🔹 Preprocessing Data...")
            preprocess_text('data/data.txt', 'data/processed/train_data.pt', config['vocab_size'])
        
        print("🔹 Starting Training...")
        train_model('data/processed/train_data.pt', config['vocab_size'], config)
        
        print("\n🔹 Loading Trained Model for Generation...")
        model = Transformer(config['vocab_size'], config['d_model'], config['n_heads'], config['d_ff'], config['n_layers'])
        
        if not os.path.exists(config['checkpoint_path']):
            raise FileNotFoundError(f"Checkpoint not found at {config['checkpoint_path']}")
        
        model.load_state_dict(torch.load(config['checkpoint_path']))
        model.eval()

        with open('data/processed/train_data.pt', 'rb') as f:
            _, vocab = torch.load(f)

        seed_texts = [
            "Once upon a time",
           
        ]

        print("\n🔹 Generated Texts:")
        for seed in seed_texts:
            generated_text = generate_text(model, vocab, seed)
            print(f"\nSeed: {seed}")
            print(generated_text)

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("Detailed traceback:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()