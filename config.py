# config.py

import torch

# Training parameters
batch_size = 32         # Number of sequences per batch
block_size = 64         # Length of each input sequence
max_iters = 5000        # Number of training steps
eval_interval = 500     # Steps between logging loss
learning_rate = 1e-3    # AdamW learning rate

# Model architecture
vocab_size = None       # Set later in prepare_dataset
embed_dim = 64          # Embedding dimension
num_heads = 4           # Number of attention heads
num_layers = 2          # Number of transformer layers

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
