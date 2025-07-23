# Quick Training Test Script
import torch
import torch.nn as nn
import torch.optim as optim
from model import GPTMiniModel
from prepare_dataset import get_batch
from tokenizer import CharTokenizer
import config
import os

print("🧪 Quick Training Test for Chat Model V2")
print("=" * 50)

# Load tokenizer and set vocab size in config
with torch.serialization.safe_globals({"tokenizer.CharTokenizer": CharTokenizer}):
    tokenizer: CharTokenizer = torch.load("data/tokenizer.pt", weights_only=False)

# Update config with the actual vocabulary size from tokenizer
config.vocab_size = tokenizer.vocab_size

print(f"Vocabulary size: {config.vocab_size}")
print(f"Block size: {config.block_size}")
print(f"Model architecture: {config.num_layers} layers, {config.num_heads} heads, {config.embed_dim} embed_dim")

# Initialize the model
model = GPTMiniModel(
    vocabulary_size=config.vocab_size,
    sequence_length=config.block_size,
    embedding_dim=config.embed_dim,
    number_of_heads=config.num_heads,
    number_of_layers=config.num_layers
).to(config.device)

print(f"Model initialized on {config.device}")

# Optimizer and loss
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

# Create checkpoint directory if not exists
os.makedirs("checkpoints", exist_ok=True)

print("🚀 Quick training started...")
model.train()

# Quick training - just 100 iterations to test
quick_iters = 100
eval_interval = 25

for iteration in range(quick_iters):
    xb, yb = get_batch('train')
    xb, yb = xb.to(config.device), yb.to(config.device)

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Evaluation
    if iteration % eval_interval == 0 or iteration == quick_iters - 1:
        model.eval()
        with torch.no_grad():
            val_x, val_y = get_batch('val')
            val_x, val_y = val_x.to(config.device), val_y.to(config.device)
            _, val_loss = model(val_x, val_y)
        model.train()

        print(f"Step {iteration:3d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

# Save quick test model
torch.save(model.state_dict(), "checkpoints/mini_gpt_quick.pt")
print("✅ Quick training complete. Model saved to checkpoints/mini_gpt_quick.pt")

# Quick generation test
print("\n🎯 Quick Generation Test:")
print("-" * 30)

model.eval()
prompt = "Human: Hi!\nAssistant:"
context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=config.device)
context = context.unsqueeze(0)

with torch.no_grad():
    for _ in range(50):
        context_cropped = context[:, -config.block_size:]
        logits = model(context_cropped)
        logits = logits[:, -1, :] / 0.8
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token), dim=1)

generated_text = tokenizer.decode(context[0].tolist())
print(f"Generated: {generated_text}")
print("\n" + "=" * 50)
