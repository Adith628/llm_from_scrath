# ChatBot Project - Version History

## V1 (Shakespeare Model) - COMPLETED ✅

- **Location**: `v1_shakespeare/`
- **Dataset**: Shakespeare texts
- **Model**: `v1_shakespeare/mini_gpt_v1.pt`
- **Tokenizer**: `v1_shakespeare/tokenizer_v1.pt`
- **Characteristics**: Generates Shakespearean-style text
- **Performance**: Good at mimicking archaic language patterns

## V2 (Chat Model) - IN PROGRESS 🔄

- **Location**: Current directory
- **Dataset**: Conversational dataset (41 conversations, 9,335 characters)
- **Model**: Enhanced architecture for better conversation
- **Improvements over V1**:
  - Increased block_size: 64 → 128 (longer context)
  - Increased embed_dim: 64 → 128 (better representations)
  - Increased num_heads: 4 → 8 (more attention)
  - Increased num_layers: 2 → 4 (deeper understanding)
  - Increased max_iters: 5000 → 8000 (longer training)

## Architecture Comparison

### V1 Configuration:

```
batch_size = 32
block_size = 64
embed_dim = 64
num_heads = 4
num_layers = 2
max_iters = 5000
```

### V2 Configuration:

```
batch_size = 32
block_size = 128
embed_dim = 128
num_heads = 8
num_layers = 4
max_iters = 8000
```

## Dataset Details

### V1 Dataset:

- Shakespeare text
- ~1M training tokens
- ~111K validation tokens
- Character-level tokenization

### V2 Dataset:

- Conversational data
- 8,401 training tokens
- 934 validation tokens
- Human-Assistant format conversations
- Covers: greetings, casual chat, support, questions, philosophy

## Files Structure:

```
custom_llm/
├── v1_shakespeare/          # V1 backup
│   ├── input.txt
│   ├── mini_gpt_v1.pt
│   └── tokenizer_v1.pt
├── data/                    # Current V2 data
│   ├── input.txt           # Conversational dataset
│   ├── tokenizer.pt
│   ├── train.pt
│   └── val.pt
├── checkpoints/
│   └── mini_gpt.pt         # V2 model (in training)
├── chatbot_v2.py           # Chat interface for V2
├── download_chat_dataset.py # Dataset creation script
└── test_model.py           # V1 testing script
```

## Usage Instructions:

### To use V1 (Shakespeare):

1. Copy files from `v1_shakespeare/` back to main directories
2. Run `python test_model.py`

### To use V2 (Chat) - Once training completes:

1. Run `python chatbot_v2.py` for interactive chat
2. Run `python chatbot_v2.py test` for quick test

## Next Steps:

1. ⏳ Wait for V2 training to complete
2. 🧪 Test V2 chat capabilities
3. 🔧 Fine-tune if needed
4. 🚀 Deploy chat interface
