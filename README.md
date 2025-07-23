# 🤖 Custom LLM from Scratch

A mini GPT implementation built from scratch using PyTorch, featuring both Shakespeare text generation and conversational AI capabilities.

## 🌟 Features

- **Character-level tokenization** with custom tokenizer
- **Transformer architecture** with multi-head attention
- **Two model versions**:
  - **V1**: Shakespeare text generation 
  - **V2**: Conversational AI chatbot
- **Interactive chat interface**
- **Configurable model architecture**
- **Training and evaluation scripts**

## 📁 Project Structure

```
custom_llm/
├── 📊 Data & Models
│   ├── data/                    # Training data and tokenizers
│   ├── checkpoints/             # Trained model weights
│   └── v1_shakespeare/          # V1 model backup
├── 🧠 Core Components
│   ├── model.py                 # GPT model architecture
│   ├── tokenizer.py             # Character-level tokenizer
│   ├── config.py                # Model and training configuration
│   └── prepare_dataset.py       # Dataset preparation
├── 🚀 Training & Testing
│   ├── train.py                 # Main training script
│   ├── test_model.py            # Model testing utilities
│   └── quick_train_test.py      # Quick training validation
├── 💬 Chat Interface
│   ├── chatbot_v2.py            # Interactive chatbot (V2)
│   └── simple_chat.py           # Simple chat interface
├── 📋 Utilities
│   ├── download_chat_dataset.py # Conversation dataset creation
│   └── VERSION_HISTORY.md       # Version tracking
└── 📚 Documentation
    ├── README.md                # This file
    └── requirements.txt         # Python dependencies
```

## 🏗️ Model Architecture

```
Input Token IDs
↓
Token Embeddings + Position Embeddings
↓
[ Transformer Block × N ]
  ├── Multi-Head Self-Attention
  ├── Layer Normalization
  ├── Feed Forward Network
  └── Residual Connections
↓
Final Layer Normalization
↓
Linear Layer → Vocabulary Size
↓
Softmax (for probability distribution)
↓
Predicted Next Token
```

### Architecture Specifications

| Component | V1 (Shakespeare) | V2 (Chat) |
|-----------|------------------|-----------|
| **Context Length** | 64 tokens | 128 tokens |
| **Embedding Dimension** | 64 | 128 |
| **Attention Heads** | 4 | 8 |
| **Transformer Layers** | 2 | 4 |
| **Parameters** | ~85K | ~680K |
| **Training Iterations** | 5,000 | 8,000 |

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
PyTorch 1.9+
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd custom_llm
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Training a Model

#### Option 1: Train Shakespeare Model (V1)
```bash
# Use Shakespeare text (restore from backup if needed)
python prepare_dataset.py
python train.py
```

#### Option 2: Train Chat Model (V2)
```bash
# Create conversational dataset
python download_chat_dataset.py

# Prepare and train
python prepare_dataset.py
python train.py
```

### Testing the Model

#### Shakespeare Model (V1)
```bash
python test_model.py
```

#### Chat Model (V2)
```bash
# Interactive chat
python chatbot_v2.py

# Quick test
python chatbot_v2.py test
```

## 💻 Usage Examples

### 1. Shakespeare Text Generation (V1)
```python
from test_model import load_trained_model, generate_text

model, tokenizer = load_trained_model()
text = generate_text(model, tokenizer, "To be or not to be", max_new_tokens=100)
print(text)
```

**Sample Output:**
```
To be or not to be, that in the senses of the bear in the world,
And such a man as you have done with me...
```

### 2. Conversational AI (V2)
```python
from chatbot_v2 import ChatBot

bot = ChatBot()
response = bot.generate_response("How are you today?")
print(response)
```

**Sample Conversation:**
```
🧑 You: Hi, how are you today?
🤖 Assistant: Hello! I'm doing great, thank you for asking. How about you? What's been going on in your day?

🧑 You: I'm feeling a bit stressed about work
🤖 Assistant: I hear you. Work stress can be really overwhelming. What's been weighing on your mind lately? Sometimes just talking about it can help lighten the load.
```

## 📊 Model Performance

### V1 (Shakespeare Model)
- **Training Data**: ~1M tokens (Shakespeare corpus)
- **Final Loss**: Train: 1.71 | Validation: 1.83
- **Characteristics**: Excellent at archaic language patterns, character names, dialogue structure

### V2 (Chat Model) 
- **Training Data**: ~8.4K tokens (41 conversations)
- **Dataset**: Human-Assistant conversational pairs
- **Capabilities**: Casual conversation, emotional support, question answering

## 🔧 Configuration

Edit `config.py` to customize model architecture:

```python
# Training parameters
batch_size = 32         # Batch size
block_size = 128        # Context window length
max_iters = 8000        # Training iterations
learning_rate = 1e-3    # Learning rate

# Model architecture
embed_dim = 128         # Embedding dimension
num_heads = 8           # Attention heads
num_layers = 4          # Transformer layers
```

## 🎯 Key Components

### 1. **Tokenizer** (`tokenizer.py`)
- Character-level tokenization
- Handles encoding/decoding of text
- Vocabulary built from training data

### 2. **Model** (`model.py`)
- GPT-style transformer architecture
- Self-attention mechanisms
- Feed-forward networks
- Layer normalization

### 3. **Training** (`train.py`)
- AdamW optimizer
- Cross-entropy loss
- Periodic validation
- Model checkpointing

### 4. **Chat Interface** (`chatbot_v2.py`)
- Interactive conversation
- Temperature control
- Response generation
- Context management

## 📈 Training Tips

1. **Start Small**: Begin with smaller models to validate your setup
2. **Monitor Loss**: Watch both training and validation loss curves
3. **Adjust Temperature**: Use 0.8-1.2 for generation (lower = more conservative)
4. **Context Length**: Longer contexts require more memory but better coherence
5. **Dataset Quality**: Clean, diverse data leads to better performance

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config.py
   - Reduce `block_size` for shorter contexts

2. **Poor Generation Quality**
   - Train for more iterations
   - Increase model size (embed_dim, num_layers)
   - Improve dataset quality

3. **Import Errors**
   - Ensure virtual environment is activated
   - Install requirements: `pip install -r requirements.txt`

## 🔮 Future Improvements

- [ ] **Byte-pair encoding** for better tokenization
- [ ] **Beam search** for better generation
- [ ] **Fine-tuning** capabilities
- [ ] **Web interface** for easier interaction
- [ ] **Model quantization** for efficiency
- [ ] **Multi-turn conversation** memory
- [ ] **Reinforcement learning** from human feedback

## 📝 Version History

- **V1.0**: Shakespeare text generation model
- **V2.0**: Conversational AI with enhanced architecture
- **V2.1**: Improved chat interface and dataset

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Inspired by the original GPT architecture
- Built with PyTorch framework
- Character-level tokenization approach
- Educational implementation for learning purposes

---

**⭐ Star this repository if you found it helpful!**

For questions or issues, please open an issue in the repository.
