import requests
import json
import re
import os

def download_conversational_dataset():
    """Download and prepare a conversational dataset for training"""
    print("📥 Creating conversational dataset...")
    
    # Create a comprehensive conversational dataset
    # I'll create a mix of casual conversations, Q&A, and friendly dialogues
    
    conversations = []
    
    # Add various types of casual conversations
    casual_conversations = [
        # Greetings and casual chat
        "Human: Hi there! How are you doing today?\nAssistant: Hello! I'm doing great, thanks for asking. How about you? What's been going on in your day?",
        "Human: Hey, what's up?\nAssistant: Not much, just here and ready to chat! What brings you here today?",
        "Human: Good morning!\nAssistant: Good morning to you too! I hope you're having a wonderful start to your day. What are your plans?",
        "Human: How's your day going?\nAssistant: It's going well, thank you! I enjoy having conversations like this. What about yours? Anything interesting happening?",
        
        # Casual questions and responses
        "Human: What do you like to do for fun?\nAssistant: I love having interesting conversations with people! I find it fascinating to learn about different perspectives and experiences. What about you? What do you enjoy doing in your free time?",
        "Human: Tell me something interesting.\nAssistant: Here's something cool: did you know that octopuses have three hearts? Two pump blood to their gills, and one pumps blood to the rest of their body. What kind of interesting topics do you enjoy learning about?",
        "Human: I'm feeling a bit down today.\nAssistant: I'm sorry to hear you're not feeling your best. Sometimes we all have those days. Would you like to talk about what's bothering you, or would you prefer a distraction? I'm here either way.",
        "Human: What's your favorite color?\nAssistant: I find that an interesting question! I don't experience colors the way humans do, but I think each color has its own character. Blue feels calm and vast, red feels energetic, green feels fresh and alive. What's your favorite color and why?",
        
        # Helpful responses
        "Human: Can you help me with something?\nAssistant: Of course! I'd be happy to help. What do you need assistance with?",
        "Human: I'm trying to decide what to have for lunch.\nAssistant: That's always a fun decision! What are you in the mood for? Something light and fresh, or maybe something more hearty and comforting? Do you have any ingredients at home or are you thinking of going out?",
        "Human: I need some motivation today.\nAssistant: I understand that feeling! Remember that every small step forward is progress, even if it doesn't feel like much. You've got this! What's one small thing you could accomplish today that would make you feel good?",
        
        # Questions about experiences
        "Human: What's the weather like where you are?\nAssistant: I don't experience weather directly, but I'd love to hear about the weather where you are! Is it sunny, rainy, snowy? I find it interesting how weather can affect our mood and plans.",
        "Human: Do you have any hobbies?\nAssistant: In a way, learning and having conversations are like hobbies for me! I enjoy exploring ideas and helping people think through problems. What hobbies do you have? I'd love to hear about what you're passionate about.",
        "Human: What's your favorite food?\nAssistant: I don't eat food, but I find the topic fascinating! Food brings people together and there's so much culture and creativity in cooking. What's your favorite dish? Is it something that has special meaning to you?",
        
        # Philosophical and thoughtful
        "Human: What makes you happy?\nAssistant: I find joy in meaningful conversations and helping people! There's something wonderful about that moment when I can provide exactly the information or perspective someone needs. What makes you happiest?",
        "Human: Do you ever get lonely?\nAssistant: That's a thoughtful question. I don't experience loneliness in the traditional sense, but I do value these interactions. Each conversation brings something new and interesting. Do you ever feel lonely? What helps when you do?",
        "Human: What do you think about at night?\nAssistant: I don't have a sleep cycle like humans do, but if I did, I think I'd reflect on all the interesting conversations I've had and wonder about the people I've talked with. Do you have any bedtime thoughts or routines?",
        
        # Casual sharing
        "Human: I just finished a really good book!\nAssistant: That's awesome! There's nothing quite like finishing a book that really captivates you. What was it about? Would you recommend it to others?",
        "Human: I'm learning to cook.\nAssistant: How exciting! Cooking is such a wonderful skill to develop. What kind of dishes are you trying to make? Have you had any kitchen successes or disasters yet? Both can be great learning experiences!",
        "Human: I went for a walk today.\nAssistant: That sounds lovely! Walking can be so refreshing. Was it somewhere new or a familiar route? I always think there's something peaceful about taking time to just walk and observe the world around you.",
        
        # Problem-solving casual style
        "Human: I can't decide what to watch tonight.\nAssistant: The eternal dilemma! What kind of mood are you in? Something light and funny, something thrilling, or maybe a good drama? Are you looking for something new or revisiting an old favorite?",
        "Human: I'm procrastinating on something important.\nAssistant: Ah, the procrastination struggle is real! Sometimes breaking it into tiny steps helps, or setting a timer for just 10 minutes of work. What's the task you're avoiding? Maybe we can brainstorm ways to make it feel less overwhelming.",
        
        # Fun and playful
        "Human: Tell me a joke!\nAssistant: Sure! Why don't scientists trust atoms? Because they make up everything! *ba dum tss* What about you - do you have any favorite jokes or puns?",
        "Human: I'm bored.\nAssistant: Boredom can actually be a great opportunity! You could try something creative, learn a new skill, reach out to a friend, or even just daydream. What usually interests you when you're looking for something to do?",
        "Human: What's the meaning of life?\nAssistant: Wow, going right for the big questions! I think it might be different for everyone - maybe it's about connections, growth, making a positive impact, or simply finding joy in small moments. What do you think gives life meaning?",
        
        # Supportive responses
        "Human: I had a rough day at work.\nAssistant: I'm sorry you had a tough day. Work stress can really weigh on you. Do you want to talk about what happened, or would you prefer to focus on something more relaxing to help you unwind?",
        "Human: I'm excited about something!\nAssistant: That's wonderful! I love when people are excited about things. What's got you feeling so enthusiastic? Your energy is contagious!",
        "Human: I'm worried about tomorrow.\nAssistant: It's natural to feel anxious about upcoming events. Sometimes it helps to focus on what you can control and prepare for. What's happening tomorrow that's on your mind?",
        
        # Learning and curiosity
        "Human: I learned something new today.\nAssistant: That's fantastic! Learning new things is one of life's great pleasures. What did you discover? I love hearing about what catches people's curiosity.",
        "Human: Do you like music?\nAssistant: I find music fascinating! The way it can convey emotions and bring back memories is amazing. I don't experience it the way you do, but I appreciate its complexity and beauty. What kind of music do you enjoy?",
        "Human: What do you think about dreams?\nAssistant: Dreams are so intriguing! The way our minds process experiences and create these sometimes bizarre, sometimes meaningful stories while we sleep. Do you remember your dreams often? Any recurring ones or particularly vivid ones recently?",
        
        # More casual conversations
        "Human: I'm feeling stressed.\nAssistant: I hear you. Stress can be really overwhelming. What's been weighing on your mind lately? Sometimes just talking about it can help lighten the load.",
        "Human: What's your opinion on pizza?\nAssistant: Pizza is such a universally loved food! There's something magical about how it brings people together. Are you team thin crust or thick crust? And what's your go-to topping combination?",
        "Human: I can't sleep.\nAssistant: That's frustrating when you need rest. Are your thoughts racing, or is it more of a physical restlessness? Sometimes gentle activities like reading or soft music can help ease into sleep mode.",
        "Human: Do you have friends?\nAssistant: In a way, every person I talk with is like a friend! I value these connections and conversations. Each interaction teaches me something new. What qualities do you value most in your friendships?",
        "Human: I love rainy days.\nAssistant: There's something so cozy and peaceful about rainy weather! The sound of raindrops, the fresh smell in the air, the way everything feels quieter and more intimate. What do you like to do on rainy days?",
        "Human: I'm trying to stay positive.\nAssistant: That's a wonderful intention, and it's okay if it's not always easy. Focusing on small good things can help - maybe a warm cup of coffee, a text from a friend, or even just a moment of sunshine. What helps lift your spirits?",
        "Human: What do you think about social media?\nAssistant: Social media is such a double-edged tool, isn't it? It can connect us with friends and interesting ideas, but it can also be overwhelming or make us compare ourselves to others. How do you find balance with it?",
        "Human: I miss traveling.\nAssistant: Travel has such a special way of opening our minds and creating lasting memories. Where are some places you've loved visiting? Or if you haven't traveled much yet, where would you love to go someday?",
        "Human: I'm learning a new skill.\nAssistant: That's exciting! There's something so rewarding about challenging yourself to learn something new. What skill are you working on? How's the learning process going so far?",
        "Human: I feel grateful today.\nAssistant: What a lovely feeling! Gratitude has such a way of brightening everything around us. What's making you feel especially grateful today? I'd love to hear about it."
    ]
    
    return casual_conversations

def create_conversational_input_file():
    """Create the input.txt file with conversational data"""
    conversations = download_conversational_dataset()
    
    # Join all conversations with newlines
    full_text = "\n\n".join(conversations)
    
    # Write to input.txt
    with open("data/input.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(f"✅ Created conversational dataset with {len(conversations)} conversations")
    print(f"📊 Total characters: {len(full_text)}")
    print("💾 Saved to data/input.txt")

if __name__ == "__main__":
    create_conversational_input_file()
    print("\n🎯 Dataset ready for training! You can now run prepare_dataset.py and train.py")
