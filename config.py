# config.py

import torch

# Training parameters
batch_size = 32         # Number of sequences per batch
block_size = 128        # Length of each input sequence (increased for longer conversations)
max_iters = 8000        # Number of training steps (increased for better chat training)
eval_interval = 500     # Steps between logging loss
learning_rate = 1e-3    # AdamW learning rate

# Model architecture
vocab_size = None       # Set later in prepare_dataset
embed_dim = 128         # Embedding dimension (increased for better representation)
num_heads = 8           # Number of attention heads (increased)
num_layers = 4          # Number of transformer layers (increased for better understanding)

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
