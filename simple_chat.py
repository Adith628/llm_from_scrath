import torch
import torch.nn.functional as F
from model import GPTMiniModel
from tokenizer import CharTokenizer
import config

def load_best_chat_model():
    """Load the best conversational model"""
    print("Loading best conversational model...")
    
    # Load tokenizer
    with torch.serialization.safe_globals({"tokenizer.CharTokenizer": CharTokenizer}):
        tokenizer = torch.load("data/tokenizer_v2.pt", weights_only=False)
    
    # Update config
    config.vocab_size = tokenizer.vocab_size
    
    # Initialize model
    model = GPTMiniModel(
        vocabulary_size=config.vocab_size,
        sequence_length=config.block_size,
        embedding_dim=config.embed_dim,
        number_of_heads=config.num_heads,
        number_of_layers=config.num_layers
    ).to(config.device)
    
    # Load best weights
    model.load_state_dict(torch.load("checkpoints/mini_gpt_v2_best.pt", map_location=config.device))
    model.eval()
    
    print(f"Best model loaded (vocabulary size: {tokenizer.vocab_size})")
    return model, tokenizer

def simple_chat():
    """Simple chat interface with improved response generation"""
    print("🤖 ChatGPT Mini v2 - Simple Chat Mode")
    print("=" * 50)
    print("Type 'quit' to exit")
    print("=" * 50)
    
    model, tokenizer = load_best_chat_model()
    
    while True:
        user_input = input("\n🧑 You: ").strip()
        
        if user_input.lower() == 'quit':
            print("\n👋 Goodbye!")
            break
        elif not user_input:
            continue
        
        # Simple prompt format
        prompt = f"Human: {user_input}\nAssistant:"
        
        # Encode
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=config.device)
        context = context.unsqueeze(0)
        
        print("🤖 Assistant: ", end="", flush=True)
        
        with torch.no_grad():
            generated_tokens = []
            for _ in range(50):  # Shorter responses
                context_cropped = context[:, -config.block_size:]
                logits = model(context_cropped)
                logits = logits[:, -1, :] / 0.7  # Lower temperature
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for stopping conditions
                next_char = tokenizer.decode([next_token.item()])
                if next_char == '\n':
                    # Check if we have a complete response
                    current_text = tokenizer.decode(context[0].tolist())
                    if current_text.count('\n') >= 2:  # Found end of assistant response
                        break
                
                context = torch.cat((context, next_token), dim=1)
                generated_tokens.append(next_token.item())
                
                # Print character by character for better UX
                print(next_char, end="", flush=True)
                
                # Stop if we hit another "Human:" or "Assistant:"
                current_text = tokenizer.decode(context[0].tolist())
                if "\nHuman:" in current_text[len(prompt):] or "Assistant:" in current_text[len(prompt):]:
                    break
        
        print()  # New line after response

if __name__ == "__main__":
    simple_chat()
