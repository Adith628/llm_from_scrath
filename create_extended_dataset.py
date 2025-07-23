import os

def download_conversational_dataset():
    """Download a larger conversational dataset from the internet"""
    
    # Let's create a much larger synthetic dataset
    print("Creating a larger conversational dataset...")
    
    # Extended conversations with more variety
    conversations = []
    
    # Greetings and introductions
    greetings = [
        ("Human: Hello there!", "Assistant: Hello! It's nice to meet you. How can I help you today?"),
        ("Human: Hi, how are you doing?", "Assistant: I'm doing well, thank you for asking! How are you doing today?"),
        ("Human: Good morning!", "Assistant: Good morning! I hope you're having a wonderful start to your day."),
        ("Human: Hey!", "Assistant: Hey there! What's on your mind?"),
        ("Human: What's up?", "Assistant: Not much, just here and ready to chat! What about you?"),
        ("Human: How's it going?", "Assistant: It's going well! Thanks for asking. How are things with you?"),
    ]
    
    # Small talk and casual conversation
    smalltalk = [
        ("Human: What's your favorite color?", "Assistant: I find blue quite calming and peaceful. What about you? Do you have a favorite color?"),
        ("Human: Do you like music?", "Assistant: I enjoy many different types of music! Music can be so expressive and moving. What kind of music do you like?"),
        ("Human: What's the weather like today?", "Assistant: I don't have access to current weather information, but I'd love to hear about the weather where you are! Is it nice out?"),
        ("Human: I love pizza.", "Assistant: Pizza is amazing! There are so many delicious varieties. What's your favorite type of pizza?"),
        ("Human: Do you have any hobbies?", "Assistant: I find learning about new topics and having conversations really engaging! What hobbies do you enjoy?"),
        ("Human: What do you think about books?", "Assistant: Books are wonderful! They can transport you to different worlds and teach you so much. Do you enjoy reading?"),
    ]
    
    # Emotional support and empathy
    emotions = [
        ("Human: I'm feeling sad today.", "Assistant: I'm sorry to hear you're feeling sad. It's completely normal to have difficult days. Would you like to talk about what's bothering you?"),
        ("Human: I'm really excited about something.", "Assistant: That's wonderful! I love hearing when people are excited. What's got you feeling so enthusiastic?"),
        ("Human: I'm stressed about work.", "Assistant: Work stress can be really challenging to deal with. It sounds like you have a lot on your plate. What aspects of work are causing you the most stress?"),
        ("Human: I had a great day!", "Assistant: That's fantastic to hear! I'm so glad you had a great day. What made it particularly special for you?"),
        ("Human: I'm feeling lonely.", "Assistant: I'm sorry you're feeling lonely. That can be really difficult. Sometimes talking about it can help. Is there anything specific that's making you feel this way?"),
        ("Human: I'm worried about something.", "Assistant: I understand that worry can be overwhelming. It's natural to feel concerned about things that matter to us. What's been on your mind?"),
    ]
    
    # Help and assistance
    help_requests = [
        ("Human: Can you help me with something?", "Assistant: Of course! I'd be happy to help however I can. What do you need assistance with?"),
        ("Human: I need some advice.", "Assistant: I'm here to listen and help. What situation would you like advice about?"),
        ("Human: I'm confused about something.", "Assistant: That's perfectly okay! Confusion is a normal part of learning and growing. What's puzzling you?"),
        ("Human: Can you explain this to me?", "Assistant: Absolutely! I'll do my best to explain it clearly. What would you like me to help you understand?"),
        ("Human: I don't know what to do.", "Assistant: It can be tough when you're facing uncertainty. Sometimes talking through the situation can help clarify things. What's the situation you're dealing with?"),
    ]
    
    # Learning and knowledge
    learning = [
        ("Human: I want to learn something new.", "Assistant: That's a great attitude! Learning keeps life interesting and helps us grow. What subject or skill are you thinking about exploring?"),
        ("Human: How do I get better at writing?", "Assistant: Writing improves with practice and feedback! Reading widely, writing regularly, and getting input from others are all great ways to develop your skills. What type of writing interests you?"),
        ("Human: I'm interested in science.", "Assistant: Science is fascinating! There's so much to discover about how the world works. What area of science interests you most?"),
        ("Human: Can you teach me about history?", "Assistant: History is full of incredible stories and lessons! What period or aspect of history would you like to explore?"),
        ("Human: I'm trying to learn a new language.", "Assistant: That's awesome! Learning a new language opens up so many opportunities. Which language are you studying, and how's it going so far?"),
    ]
    
    # Daily life and decisions
    daily_life = [
        ("Human: What should I have for lunch?", "Assistant: That depends on what you're in the mood for! Are you looking for something quick and easy, or do you want to cook something special?"),
        ("Human: I can't decide what to watch.", "Assistant: The paradox of choice with streaming! What genre are you in the mood for? Comedy, drama, documentary, or something else entirely?"),
        ("Human: I'm bored.", "Assistant: Boredom can sometimes lead to great discoveries! What kinds of activities usually interest you? Maybe we can brainstorm something fun to do."),
        ("Human: Should I go out or stay in tonight?", "Assistant: Both have their merits! Going out can be energizing and social, while staying in can be relaxing and cozy. What are you leaning towards?"),
        ("Human: I'm trying to decide on a career change.", "Assistant: That's a big decision! Career changes can be both exciting and scary. What's motivating you to consider a change?"),
    ]
    
    # Philosophical and deeper topics
    philosophical = [
        ("Human: What makes life meaningful?", "Assistant: That's such a profound question that philosophers have pondered for centuries. I think meaning often comes from our relationships, personal growth, and positive impact on others. What gives your life meaning?"),
        ("Human: Do you think about the future?", "Assistant: The future is full of possibilities, which is both exciting and uncertain. What aspects of the future do you think about most?"),
        ("Human: What's the most important thing in life?", "Assistant: Different people find different things most important - relationships, growth, creativity, helping others. What do you think matters most to you?"),
        ("Human: How do you define happiness?", "Assistant: Happiness seems to come from a combination of contentment, meaningful connections, and pursuing things we care about. How would you define happiness?"),
    ]
    
    # Goodbyes and closings
    goodbyes = [
        ("Human: I should go now.", "Assistant: It was really nice chatting with you! Take care, and feel free to come back anytime you want to talk."),
        ("Human: Thanks for the chat!", "Assistant: You're very welcome! I enjoyed our conversation too. Have a wonderful rest of your day!"),
        ("Human: See you later!", "Assistant: See you later! Looking forward to our next conversation. Take care!"),
        ("Human: Goodbye!", "Assistant: Goodbye! It was great talking with you. Hope to chat again soon!"),
        ("Human: Talk to you soon.", "Assistant: Sounds great! Looking forward to it. Have a good day!"),
    ]
    
    # Combine all conversation types
    all_conversations = greetings + smalltalk + emotions + help_requests + learning + daily_life + philosophical + goodbyes
    
    # Create longer multi-turn conversations
    longer_conversations = [
        """Human: I just started learning guitar.
Assistant: That's awesome! Guitar is such a rewarding instrument to learn. How long have you been playing?
Human: Just a couple of weeks now.
Assistant: Everyone starts somewhere! Those first few weeks can be challenging with sore fingertips and getting used to chord shapes. What songs are you working on?
Human: Just basic chords like G, C, and D.
Assistant: Perfect! Those are foundational chords. Once you master those, you'll be amazed at how many songs you can play. Are you taking lessons or learning on your own?
Human: I'm using online tutorials.
Assistant: Online tutorials are great! There are so many good resources available. Just remember to practice regularly, even if it's just 10-15 minutes a day. Consistency is key!

""",
        """Human: I had a job interview today.
Assistant: How did it go? Job interviews can be both exciting and nerve-wracking!
Human: I think it went well, but I'm nervous about the results.
Assistant: It's completely normal to feel nervous after an interview, even when it goes well! The fact that you think it went well is a really positive sign. What position was it for?
Human: It's a marketing role at a tech startup.
Assistant: That sounds like an exciting opportunity! Marketing at a tech startup can be really dynamic and creative. What attracted you to this particular role?
Human: I love the innovative culture and the chance to make a real impact.
Assistant: That's fantastic! Startups can offer incredible opportunities for growth and impact. I hope you hear back with good news soon!

""",
        """Human: My dog has been acting strange lately.
Assistant: Oh, what kind of strange behavior is your dog showing? Dogs can be pretty expressive when something's different.
Human: He's been hiding under the bed and won't come out to play.
Assistant: That does sound unusual, especially if he's normally playful. Has anything changed in his environment recently? Sometimes dogs hide when they're stressed or not feeling well.
Human: We moved to a new house last week.
Assistant: Ah, that could definitely explain it! Dogs can be very sensitive to changes in their environment. Moving is a big adjustment for them. He's probably just getting used to all the new smells and sounds.
Human: How long does it usually take for them to adjust?
Assistant: It varies by dog, but typically a few days to a couple of weeks. You can help by keeping his routine consistent and maybe placing some of his favorite toys or blankets near his hiding spot to make him feel more secure.

""",
        """Human: I'm thinking about starting a garden.
Assistant: That's wonderful! Gardening can be so rewarding and relaxing. Do you have any experience with gardening, or would this be your first time?
Human: This would be my first time. I don't know where to start.
Assistant: Starting a garden for the first time is exciting! The best place to start is figuring out what you want to grow and what space you have available. Do you have a yard, or are you thinking about container gardening?
Human: I have a small backyard with some sunny spots.
Assistant: Perfect! Sunny spots are great for most vegetables and flowers. For beginners, I'd recommend starting with easy-to-grow plants like tomatoes, lettuce, or herbs like basil. They're forgiving and give you quick results to stay motivated.
Human: When should I start planting?
Assistant: That depends on your location and what you want to grow! Generally, spring is a great time to start most gardens. You might want to check when your area's last frost date typically occurs, as that's usually when it's safe to plant most things outdoors.

"""
    ]
    
    # Format everything into a single text
    formatted_text = ""
    
    # Add the basic conversations
    for human, assistant in all_conversations:
        formatted_text += f"{human}\n{assistant}\n\n"
    
    # Add longer conversations
    for conversation in longer_conversations:
        formatted_text += conversation
    
    # Repeat the dataset to make it larger
    final_text = formatted_text * 3  # Triple the size
    
    return final_text

if __name__ == "__main__":
    print("Creating extended conversational dataset...")
    dataset = download_conversational_dataset()
    
    # Save to file
    with open("data/conversation_input.txt", "w", encoding="utf-8") as f:
        f.write(dataset)
    
    print(f"Extended dataset created with {len(dataset)} characters")
    print("Saved to data/conversation_input.txt")
    
    # Show stats
    lines = dataset.split('\n')
    human_lines = len([line for line in lines if line.startswith('Human:')])
    assistant_lines = len([line for line in lines if line.startswith('Assistant:')])
    
    print(f"Statistics:")
    print(f"- Total characters: {len(dataset)}")
    print(f"- Human turns: {human_lines}")
    print(f"- Assistant turns: {assistant_lines}")
    print(f"- Total conversation turns: {human_lines + assistant_lines}")
