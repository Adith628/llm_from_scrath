import torch
from tokenizer import CharTokenizer

with open("data/input.txt","r",encoding="utf-8") as f:
    text = f.read()
    
# Initialize the tokenizer
tokenizer = CharTokenizer(text)

# Encode the entire text to integer tokens
data = torch.tensor(tokenizer.encode(text),dtype=torch.long)

# Split into train and validation sets

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Save the train and validation data
torch.save(train_data, "data/train.pt")
torch.save(val_data, "data/val.pt")
torch.save(tokenizer, "data/tokenizer.pt") # Save the tokenizer for later use
print(f"Dataset prepared with {len(train_data)} training tokens and {len(val_data)} validation tokens.")

block_size = 64 # Length of context window
batch_size = 32 # Number of sequences in a batch

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y