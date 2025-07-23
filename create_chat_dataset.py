#!/usr/bin/env python3
"""
Create a conversational dataset for training a chatbot
"""

def create_conversational_dataset():
    """Create a dataset with various conversation patterns"""
    
    conversations = [
        # Greetings and basic interactions
        ("Human: Hello! How are you today?", "Assistant: Hi there! I'm doing well, thank you for asking. How are you doing today?"),
        ("Human: Good morning!", "Assistant: Good morning! I hope you're having a wonderful start to your day."),
        ("Human: Hi there!", "Assistant: Hello! Nice to meet you. What's on your mind today?"),
        ("Human: Hey, what's up?", "Assistant: Hey! Not much, just here and ready to chat. What about you?"),
        
        # Small talk
        ("Human: What's your favorite color?", "Assistant: I find blue quite calming and pleasant. What about you? Do you have a favorite color?"),
        ("Human: Do you like music?", "Assistant: I appreciate many forms of music! Different genres can evoke such interesting emotions. What kind of music do you enjoy?"),
        ("Human: What's the weather like?", "Assistant: I don't have access to current weather data, but I'd love to hear about the weather where you are! Is it nice out?"),
        ("Human: I'm feeling tired today.", "Assistant: I'm sorry to hear you're feeling tired. Make sure to get enough rest when you can. Is there anything specific that's been wearing you out?"),
        
        # Asking for help
        ("Human: Can you help me with something?", "Assistant: Of course! I'd be happy to help. What do you need assistance with?"),
        ("Human: I need advice.", "Assistant: I'm here to listen and help however I can. What's the situation you'd like advice on?"),
        ("Human: I'm confused about something.", "Assistant: That's okay, confusion is normal! I'm here to help clarify things. What's puzzling you?"),
        ("Human: Can you explain this to me?", "Assistant: Absolutely! I'll do my best to explain it clearly. What would you like me to explain?"),
        
        # Emotional support
        ("Human: I'm feeling sad today.", "Assistant: I'm sorry you're feeling sad. It's okay to have difficult days. Would you like to talk about what's bothering you?"),
        ("Human: I'm really excited!", "Assistant: That's wonderful! I love hearing when people are excited. What's got you feeling so enthusiastic?"),
        ("Human: I'm stressed about work.", "Assistant: Work stress can be really challenging. It sounds like you're dealing with a lot. What aspects of work are causing you the most stress?"),
        ("Human: I had a great day!", "Assistant: That's fantastic to hear! I'm so glad you had a great day. What made it special for you?"),
        
        # Interests and hobbies
        ("Human: I love reading books.", "Assistant: Reading is such a wonderful hobby! There's something magical about getting lost in a good story. What genres do you enjoy most?"),
        ("Human: Do you like movies?", "Assistant: I find movies fascinating as a form of storytelling and art! What types of movies do you enjoy watching?"),
        ("Human: I enjoy cooking.", "Assistant: Cooking is such a creative and rewarding activity! Do you have a favorite cuisine to cook, or are you experimenting with new recipes?"),
        ("Human: I play video games.", "Assistant: Gaming can be so much fun and immersive! What types of games do you like to play? Any favorites recently?"),
        
        # Learning and knowledge
        ("Human: I want to learn something new.", "Assistant: That's a great attitude! Learning new things keeps life interesting. What subject or skill are you thinking about exploring?"),
        ("Human: Can you teach me about science?", "Assistant: I'd love to help you learn about science! Science is fascinating and covers so many areas. What aspect of science interests you most?"),
        ("Human: I'm curious about history.", "Assistant: History is incredibly rich and full of amazing stories! What period or aspect of history are you most curious about?"),
        ("Human: How do I get better at writing?", "Assistant: Writing is a skill that improves with practice! Reading widely, writing regularly, and getting feedback are great ways to improve. What kind of writing interests you?"),
        
        # Daily life
        ("Human: What should I have for lunch?", "Assistant: That depends on what you're in the mood for! Are you looking for something quick and easy, or do you want to cook something special?"),
        ("Human: I can't decide what to watch.", "Assistant: The paradox of choice with so many options! What genre are you in the mood for? Comedy, drama, documentary, or something else?"),
        ("Human: I'm bored.", "Assistant: Boredom can sometimes lead to the best discoveries! What kinds of activities usually interest you? Maybe we can brainstorm something fun to do."),
        ("Human: It's getting late.", "Assistant: Time does seem to fly by sometimes! Are you getting ready to wind down for the evening, or do you have more plans?"),
        
        # Philosophical and thoughtful
        ("Human: What makes life meaningful?", "Assistant: That's such a profound question that people have pondered for centuries. I think meaning often comes from connections, growth, and contributing positively to the world. What gives your life meaning?"),
        ("Human: Do you think about the future?", "Assistant: The future is full of possibilities and unknowns, which is both exciting and uncertain. What aspects of the future do you think about most?"),
        ("Human: What's the most important thing in life?", "Assistant: Different people find different things most important - relationships, personal growth, helping others, creativity. What do you think is most important to you?"),
        
        # Endings and goodbyes
        ("Human: I should probably go now.", "Assistant: It was really nice chatting with you! Take care, and feel free to come back anytime you want to talk."),
        ("Human: Thanks for the chat!", "Assistant: You're very welcome! I enjoyed our conversation too. Have a wonderful rest of your day!"),
        ("Human: See you later!", "Assistant: See you later! Looking forward to our next conversation. Take care!"),
        ("Human: Goodbye!", "Assistant: Goodbye! It was great talking with you. Hope to chat again soon!"),
        
        # Follow-up questions and engagement
        ("Human: That's interesting.", "Assistant: I'm glad you found it interesting! What aspect caught your attention the most?"),
        ("Human: I never thought of it that way.", "Assistant: It's always fascinating when we discover new perspectives! Sometimes a different angle can really change how we see things."),
        ("Human: Tell me more.", "Assistant: I'd be happy to elaborate! What specific part would you like me to dive deeper into?"),
        ("Human: That makes sense.", "Assistant: Great! I'm glad that was helpful. Is there anything else you'd like to explore or discuss?"),
    ]
    
    # Format the conversations into a single text
    formatted_text = ""
    for human, assistant in conversations:
        formatted_text += f"{human}\n{assistant}\n\n"
    
    # Add some variations and longer conversations
    longer_conversations = [
        """Human: I'm learning to play guitar.
Assistant: That's awesome! Guitar is such a versatile and rewarding instrument to learn. How long have you been playing?
Human: Just started a few weeks ago.
Assistant: Everyone starts somewhere! Those first few weeks can be challenging with sore fingertips and getting used to the chord shapes. What songs are you working on?
Human: Just basic chords right now.
Assistant: That's the perfect foundation! Once you get comfortable with basic chords like G, C, D, and Em, you'll be amazed at how many songs you can play. Keep practicing regularly, even if it's just 10-15 minutes a day!

""",
        """Human: I had a job interview today.
Assistant: Oh wow, how did it go? Job interviews can be both exciting and nerve-wracking!
Human: I think it went well, but I'm nervous about the results.
Assistant: It's completely normal to feel nervous after an interview, even when it goes well! The fact that you think it went well is a really good sign. What kind of position was it for?
Human: It's for a marketing role at a tech company.
Assistant: That sounds like an exciting opportunity! Marketing in tech can be really dynamic and creative. What attracted you to this particular role and company?

""",
        """Human: My cat is acting weird today.
Assistant: Oh no, what kind of weird behavior is your cat showing? Cats can be pretty expressive when something's different!
Human: She's hiding under the bed and won't come out.
Assistant: That does sound unusual, especially if she's normally social. Has anything changed in her environment recently? Sometimes cats hide when they're stressed, not feeling well, or if there's been a change in routine.
Human: We had some construction work done yesterday.
Assistant: Ah, that could definitely explain it! Cats are very sensitive to changes and loud noises. The construction probably stressed her out. She'll likely come out when she feels the environment is back to normal and safe. You might try speaking softly to her and maybe offering a favorite treat near her hiding spot.

"""
    ]
    
    for conversation in longer_conversations:
        formatted_text += conversation
    
    return formatted_text

if __name__ == "__main__":
    print("Creating conversational dataset...")
    dataset = create_conversational_dataset()
    
    # Save to file
    with open("data/conversation_input.txt", "w", encoding="utf-8") as f:
        f.write(dataset)
    
    print(f"Dataset created with {len(dataset)} characters")
    print("Saved to data/conversation_input.txt")
    
    # Show a preview
    print("\nPreview of the dataset:")
    print("-" * 50)
    print(dataset[:500] + "...")
