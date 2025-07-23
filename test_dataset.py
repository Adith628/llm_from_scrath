from prepare_dataset import get_batch

x, y = get_batch('train')
print(f"Batch x shape: {x.shape}, Batch y shape: {y.shape}")
print(f"First batch x: {x[0]}")
print(f"First batch y: {y[0]}")