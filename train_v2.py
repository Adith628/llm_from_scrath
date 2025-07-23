import torch
import torch.nn as nn
import torch.optim as optim
from model import GPTMiniModel
from prepare_dataset import get_batch
from tokenizer import CharTokenizer
import config
import os

print("🚀 Starting training for ChatGPT Mini v2...")

# Update the input file for conversation data
def prepare_conversation_dataset():
    """Prepare the conversation dataset"""
    with open("data/conversation_input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"Loaded conversation dataset with {len(text)} characters")
    
    # Initialize the tokenizer with conversation data
    tokenizer = CharTokenizer(text)
    
    # Encode the entire text to integer tokens
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # Split into train and validation sets
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Save the train and validation data
    torch.save(train_data, "data/train_v2.pt")
    torch.save(val_data, "data/val_v2.pt")
    torch.save(tokenizer, "data/tokenizer_v2.pt")
    
    print(f"Dataset prepared with {len(train_data)} training tokens and {len(val_data)} validation tokens.")
    return tokenizer

# Function to get batch for v2
def get_batch_v2(split):
    """Modified get_batch function for v2 data"""
    train_data = torch.load("data/train_v2.pt")
    val_data = torch.load("data/val_v2.pt")
    
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])
    return x, y

# Prepare conversation dataset
tokenizer = prepare_conversation_dataset()

# Update config with the actual vocabulary size from tokenizer
config.vocab_size = tokenizer.vocab_size

print(f"Vocabulary size: {config.vocab_size}")
print(f"Training device: {config.device}")

# Initialize the model
model = GPTMiniModel(
    vocabulary_size=config.vocab_size,
    sequence_length=config.block_size,
    embedding_dim=config.embed_dim,
    number_of_heads=config.num_heads,
    number_of_layers=config.num_layers
).to(config.device)

print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

# Optimizer and loss
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

# Create checkpoint directory if not exists
os.makedirs("checkpoints", exist_ok=True)

print("🚀 Training started...")
model.train()

best_val_loss = float('inf')

for iteration in range(config.max_iters):
    xb, yb = get_batch_v2('train')
    xb, yb = xb.to(config.device), yb.to(config.device)

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Evaluation
    if iteration % config.eval_interval == 0 or iteration == config.max_iters - 1:
        model.eval()
        with torch.no_grad():
            val_x, val_y = get_batch_v2('val')
            val_x, val_y = val_x.to(config.device), val_y.to(config.device)
            _, val_loss = model(val_x, val_y)
        model.train()

        print(f"Step {iteration:5d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
        
        # Save best model
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), "checkpoints/mini_gpt_v2_best.pt")

# Save final model
torch.save(model.state_dict(), "checkpoints/mini_gpt_v2.pt")
print("✅ Training complete!")
print(f"Final model saved to checkpoints/mini_gpt_v2.pt")
print(f"Best model (val_loss: {best_val_loss:.4f}) saved to checkpoints/mini_gpt_v2_best.pt")
