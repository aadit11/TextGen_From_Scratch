import torch
import torch.optim as optim
import os
from model.transformer import Transformer
from utils.tokenizer import detokenize

def train_model(data_path, vocab_size, config):
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