##Model Architecture

Input Token IDs
↓
Token Embeddings + Position Embeddings
↓
[ Transformer Block × N ]
↓
LayerNorm + Linear → Vocab Size
↓
Softmax (internally during loss)
↓
Predicted Next Token
