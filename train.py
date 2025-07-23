import torch
import torch.nn as nn
import torch.optim as optim
from model import GPTMiniModel
from prepare_dataset import get_batch
from tokenizer import CharTokenizer
import config
import os

# Load tokenizer and set vocab size in config
with torch.serialization.safe_globals({"tokenizer.CharTokenizer": CharTokenizer}):
    tokenizer: CharTokenizer = torch.load("data/tokenizer.pt", weights_only=False)

# Update config with the actual vocabulary size from tokenizer
config.vocab_size = tokenizer.vocab_size

# Initialize the model
model = GPTMiniModel(
    vocabulary_size=config.vocab_size,
    sequence_length=config.block_size,
    embedding_dim=config.embed_dim,
    number_of_heads=config.num_heads,
    number_of_layers=config.num_layers
).to(config.device)

# Optimizer and loss
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

# Create checkpoint directory if not exists
os.makedirs("checkpoints", exist_ok=True)

print("🚀 Training started...")
step = 0
model.train()

for iteration in range(config.max_iters):
    xb, yb = get_batch('train')
    xb, yb = xb.to(config.device), yb.to(config.device)

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Evaluation
    if iteration % config.eval_interval == 0 or iteration == config.max_iters - 1:
        model.eval()
        with torch.no_grad():
            val_x, val_y = get_batch('val')
            val_x, val_y = val_x.to(config.device), val_y.to(config.device)
            _, val_loss = model(val_x, val_y)
        model.train()

        print(f"Step {iteration:5d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "checkpoints/mini_gpt.pt")
print("✅ Training complete. Model saved to checkpoints/mini_gpt.pt")
