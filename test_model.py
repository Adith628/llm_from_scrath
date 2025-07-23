import torch
import torch.nn.functional as F
from model import GPTMiniModel
from tokenizer import CharTokenizer
import config

def load_trained_model():
    """Load the trained model and tokenizer"""
    print("Loading tokenizer...")
    with torch.serialization.safe_globals({"tokenizer.CharTokenizer": CharTokenizer}):
        tokenizer = torch.load("data/tokenizer.pt", weights_only=False)
    
    print("Loading trained model...")
    # Update config with vocabulary size
    config.vocab_size = tokenizer.vocab_size
    
    # Initialize model with same architecture as training
    model = GPTMiniModel(
        vocabulary_size=config.vocab_size,
        sequence_length=config.block_size,
        embedding_dim=config.embed_dim,
        number_of_heads=config.num_heads,
        number_of_layers=config.num_layers
    ).to(config.device)
    
    # Load the trained weights
    model.load_state_dict(torch.load("checkpoints/mini_gpt.pt", map_location=config.device))
    model.eval()
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0):
    """Generate text from the model given a prompt"""
    model.eval()
    
    # Encode the prompt
    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=config.device)
    context = context.unsqueeze(0)  # Add batch dimension
    
    print(f"Generating text with prompt: '{prompt}'")
    print(f"Max new tokens: {max_new_tokens}, Temperature: {temperature}")
    print("-" * 50)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get the logits for the last token
            # Crop context to the last block_size tokens if it's too long
            context_cropped = context[:, -config.block_size:]
            logits = model(context_cropped)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the running sequence
            context = torch.cat((context, next_token), dim=1)
    
    # Decode the generated sequence
    generated_tokens = context[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text

def test_model():
    """Test the trained model with various prompts"""
    print("🧪 Testing the trained GPT Mini model...")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = load_trained_model()
    
    print(f"Model loaded successfully!")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Device: {config.device}")
    print("=" * 60)
    
    # Test prompts
    test_prompts = [
        "The",
        "Hello",
        "Once upon a time",
        "In the beginning",
        "Shakespeare wrote"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n🎯 Test {i+1}: Prompt = '{prompt}'")
        print("-" * 40)
        
        # Generate with different temperatures
        for temp in [0.8, 1.0, 1.2]:
            print(f"\nTemperature: {temp}")
            generated = generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=temp)
            print(f"Generated: {generated}")
            print()
    
    print("=" * 60)
    print("✅ Model testing complete!")

def interactive_mode():
    """Interactive mode for testing custom prompts"""
    print("\n🎮 Interactive Mode - Enter your own prompts!")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    model, tokenizer = load_trained_model()
    
    while True:
        prompt = input("\nEnter prompt: ").strip()
        if prompt.lower() == 'quit':
            break
        
        if not prompt:
            continue
            
        try:
            temp = float(input("Enter temperature (0.1-2.0, default 1.0): ") or "1.0")
            max_tokens = int(input("Enter max new tokens (default 100): ") or "100")
        except ValueError:
            temp = 1.0
            max_tokens = 100
        
        generated = generate_text(model, tokenizer, prompt, max_new_tokens=max_tokens, temperature=temp)
        print(f"\nGenerated text:\n{generated}")

if __name__ == "__main__":
    # Run basic tests
    test_model()
    
    # Ask if user wants interactive mode
    while True:
        choice = input("\nWould you like to try interactive mode? (y/n): ").strip().lower()
        if choice == 'y':
            interactive_mode()
            break
        elif choice == 'n':
            break
        else:
            print("Please enter 'y' or 'n'")
    
    print("\n👋 Goodbye!")
