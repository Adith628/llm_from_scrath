import torch
import torch.nn.functional as F
from model import GPTMiniModel
from tokenizer import CharTokenizer
import config
import re

class ChatBot:
    def __init__(self, model_path="checkpoints/mini_gpt.pt"):
        """Initialize the chatbot with trained model"""
        print("🤖 Loading ChatBot V2...")
        
        # Load tokenizer
        with torch.serialization.safe_globals({"tokenizer.CharTokenizer": CharTokenizer}):
            self.tokenizer = torch.load("data/tokenizer.pt", weights_only=False)
        
        # Update config
        config.vocab_size = self.tokenizer.vocab_size
        
        # Load model
        self.model = GPTMiniModel(
            vocabulary_size=config.vocab_size,
            sequence_length=config.block_size,
            embedding_dim=config.embed_dim,
            number_of_heads=config.num_heads,
            number_of_layers=config.num_layers
        ).to(config.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=config.device))
        self.model.eval()
        
        print(f"✅ ChatBot loaded! Vocabulary size: {self.tokenizer.vocab_size}")
        print(f"🧠 Model architecture: {config.num_layers} layers, {config.num_heads} heads, {config.embed_dim} embed_dim")
    
    def generate_response(self, user_input, max_new_tokens=150, temperature=0.8):
        """Generate a response to user input"""
        # Format the input as a conversation
        prompt = f"Human: {user_input}\nAssistant:"
        
        # Encode the prompt
        context = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long, device=config.device)
        context = context.unsqueeze(0)  # Add batch dimension
        
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context to fit block size
                context_cropped = context[:, -config.block_size:]
                logits = self.model(context_cropped)
                
                # Get logits for the last token
                logits = logits[:, -1, :] / temperature
                
                # Apply softmax and sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check if we've hit a natural stopping point
                next_char = self.tokenizer.decode([next_token.item()])
                generated_tokens.append(next_token.item())
                
                # Stop if we encounter a human prompt or double newline (conversation boundary)
                generated_so_far = self.tokenizer.decode(generated_tokens)
                if "\nHuman:" in generated_so_far or "\n\n" in generated_so_far:
                    break
                
                # Append to context for next iteration
                context = torch.cat((context, next_token), dim=1)
        
        # Decode the full generated sequence
        full_response = self.tokenizer.decode(context[0].tolist())
        
        # Extract just the assistant's response
        if "Assistant:" in full_response:
            response = full_response.split("Assistant:")[-1].strip()
            # Clean up the response
            response = response.split("\nHuman:")[0].strip()
            response = response.split("\n\n")[0].strip()
        else:
            response = "I'm not sure how to respond to that."
        
        return response
    
    def chat(self):
        """Start an interactive chat session"""
        print("\n💬 ChatBot V2 is ready! Type 'quit' to exit, 'help' for commands.")
        print("=" * 60)
        
        while True:
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye! Thanks for chatting!")
                break
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("- 'quit': Exit the chat")
                print("- 'help': Show this help message")
                print("- 'temp <value>': Change temperature (e.g., 'temp 1.2')")
                print("- Just type normally to chat!")
                continue
            elif user_input.lower().startswith('temp '):
                try:
                    new_temp = float(user_input.split()[1])
                    self.temperature = new_temp
                    print(f"🌡️ Temperature set to {new_temp}")
                    continue
                except:
                    print("❌ Invalid temperature. Use format: 'temp 0.8'")
                    continue
            elif not user_input:
                continue
            
            print("🤖 Assistant: ", end="", flush=True)
            response = self.generate_response(user_input)
            print(response)

def quick_test():
    """Quick test of the chatbot with preset questions"""
    print("🧪 Quick Test Mode")
    print("=" * 40)
    
    bot = ChatBot()
    
    test_questions = [
        "Hi, how are you?",
        "What's your favorite color?",
        "Tell me something interesting",
        "I'm feeling sad today",
        "What do you like to do for fun?"
    ]
    
    for question in test_questions:
        print(f"\n🧑 Human: {question}")
        response = bot.generate_response(question)
        print(f"🤖 Assistant: {response}")
    
    print("\n" + "=" * 40)
    print("✅ Quick test complete!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        bot = ChatBot()
        bot.chat()
