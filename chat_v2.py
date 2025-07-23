import torch
import torch.nn.functional as F
from model import GPTMiniModel
from tokenizer import CharTokenizer
import config

def load_chat_model(version="v2"):
    """Load the conversational model and tokenizer"""
    print(f"Loading conversational model v{version}...")
    
    # Load tokenizer
    tokenizer_file = f"data/tokenizer_{version}.pt" if version != "v1" else "data/tokenizer.pt"
    with torch.serialization.safe_globals({"tokenizer.CharTokenizer": CharTokenizer}):
        tokenizer = torch.load(tokenizer_file, weights_only=False)
    
    # Update config with vocabulary size
    config.vocab_size = tokenizer.vocab_size
    
    # Initialize model
    model = GPTMiniModel(
        vocabulary_size=config.vocab_size,
        sequence_length=config.block_size,
        embedding_dim=config.embed_dim,
        number_of_heads=config.num_heads,
        number_of_layers=config.num_layers
    ).to(config.device)
    
    # Load weights
    model_file = f"checkpoints/mini_gpt_{version}.pt"
    try:
        model.load_state_dict(torch.load(model_file, map_location=config.device))
    except FileNotFoundError:
        model_file = f"checkpoints/mini_gpt_{version}_best.pt"
        model.load_state_dict(torch.load(model_file, map_location=config.device))
    
    model.eval()
    print(f"Model loaded from {model_file}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    return model, tokenizer

def generate_response(model, tokenizer, user_input, max_tokens=100, temperature=0.8):
    """Generate a conversational response"""
    model.eval()
    
    # Format the input as a conversation
    prompt = f"Human: {user_input}\nAssistant:"
    
    # Encode the prompt
    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=config.device)
    context = context.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Crop context if too long
            context_cropped = context[:, -config.block_size:]
            logits = model(context_cropped)
            
            # Focus on the last token
            logits = logits[:, -1, :] / temperature
            
            # Apply softmax
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check if we hit a natural stopping point (end of assistant response)
            next_char = tokenizer.decode([next_token.item()])
            
            # Append to sequence
            context = torch.cat((context, next_token), dim=1)
            
            # Stop if we encounter a human turn or double newline
            generated_text = tokenizer.decode(context[0].tolist())
            if "\nHuman:" in generated_text[len(prompt):] or generated_text.endswith("\n\n"):
                break
    
    # Extract just the assistant's response
    generated_text = tokenizer.decode(context[0].tolist())
    assistant_response = generated_text[len(prompt):].strip()
    
    # Clean up the response - remove any "Human:" that might have been generated
    if "\nHuman:" in assistant_response:
        assistant_response = assistant_response.split("\nHuman:")[0].strip()
    
    return assistant_response

def chat_interface():
    """Interactive chat interface"""
    print("🤖 ChatGPT Mini v2 - Conversational Mode")
    print("=" * 50)
    print("Type 'quit' to exit, 'clear' to clear history")
    print("Type 'switch v1' to use the Shakespeare model")
    print("=" * 50)
    
    current_version = "v2"
    model, tokenizer = load_chat_model(current_version)
    
    conversation_history = []
    
    while True:
        user_input = input("\n🧑 You: ").strip()
        
        if user_input.lower() == 'quit':
            print("\n👋 Goodbye! Thanks for chatting!")
            break
        elif user_input.lower() == 'clear':
            conversation_history = []
            print("\n🗑️ Conversation history cleared!")
            continue
        elif user_input.lower().startswith('switch'):
            if 'v1' in user_input.lower():
                current_version = "v1"
                print("\n🎭 Switching to Shakespeare model (v1)...")
            else:
                current_version = "v2" 
                print("\n💬 Switching to conversational model (v2)...")
            
            try:
                model, tokenizer = load_chat_model(current_version)
                print(f"✅ Successfully switched to version {current_version}")
            except Exception as e:
                print(f"❌ Error switching models: {e}")
            continue
        elif not user_input:
            continue
        
        # Generate response
        try:
            print("\n🤖 Assistant: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input, 
                                       max_tokens=80, temperature=0.8)
            print(response)
            
            # Add to conversation history
            conversation_history.append(f"Human: {user_input}")
            conversation_history.append(f"Assistant: {response}")
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")

def test_chat_model():
    """Test the chat model with predefined prompts"""
    print("🧪 Testing ChatGPT Mini v2...")
    print("=" * 50)
    
    model, tokenizer = load_chat_model("v2")
    
    test_prompts = [
        "Hello! How are you?",
        "What's your favorite hobby?",
        "I'm feeling stressed today.",
        "Can you help me with something?",
        "Tell me a joke.",
        "What should I have for dinner?",
        "I'm learning to code.",
        "Good morning!",
        "I had a great day today!",
        "What do you think about artificial intelligence?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🧑 Test {i}: {prompt}")
        response = generate_response(model, tokenizer, prompt, max_tokens=60, temperature=0.8)
        print(f"🤖 Assistant: {response}")
        print("-" * 40)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_chat_model()
    else:
        chat_interface()
