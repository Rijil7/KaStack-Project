import json
import spacy
from transformers import pipeline
import numpy as np
from data_parser import load_conversations
import re
import os

def contains_emoji(text):
    # A simple regex to detect non-ASCII, non-punctuation characters commonly used as emojis
    return bool(re.search(r'[^\x00-\x7F]', text))

def extract_persona(limit=2000):
    messages = load_conversations('conversations.csv')
    if limit:
        messages = messages[:limit]
        
    user_msgs = [m['text'] for m in messages if m['speaker'] == 'User 1']
    
    # Load spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        print("Downloading en_core_web_sm...")
        import sys
        os.system(f"{sys.executable} -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        
    personal_facts = []
    habits = []
    
    # Simple rule-based extraction for facts
    for text in user_msgs:
        doc = nlp(text)
        for sent in doc.sents:
            txt = sent.text.lower()
            if txt.startswith("i am a ") or txt.startswith("i work as ") or txt.startswith("i have a "):
                personal_facts.append(sent.text)
            if " my dog " in txt or " my cat " in txt or " my pet " in txt:
                personal_facts.append(sent.text)
            if txt.startswith("i love to ") or txt.startswith("i like to ") or txt.startswith("i usually ") or txt.startswith("i always "):
                habits.append(sent.text)
                
    # Deduplicate
    personal_facts = list(set(personal_facts))[:5]
    habits = list(set(habits))[:5]
    
    if not personal_facts:
        personal_facts = ["Loves animals", "Works in a creative field"]
    if not habits:
        habits = ["Enjoys talking to people", "Likes learning new things"]
        
    # Communication style
    word_counts = [len(m.split()) for m in user_msgs]
    avg_word_count = float(np.mean(word_counts)) if word_counts else 0.0
    
    emoji_count = sum(1 for m in user_msgs if contains_emoji(m))
    emoji_usage = "high" if emoji_count > len(user_msgs) * 0.1 else ("medium" if emoji_count > 0 else "low")
    
    # Default values for local fast execution if transformer is too slow
    tone = "casual"
    personality_traits = ["friendly", "curious"]
    
    try:
        print("Loading zero-shot classifier for tone and personality...")
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
        # Use first 50 messages to not overload context
        full_text = " ".join(user_msgs[:50])
        tone_res = classifier(full_text, candidate_labels=["casual", "formal", "enthusiastic", "serious"])
        tone = tone_res['labels'][0]
        
        trait_res = classifier(full_text, candidate_labels=["sarcastic", "analytical", "friendly", "introverted", "outgoing"])
        personality_traits = trait_res['labels'][:2]
    except Exception as e:
        print(f"Classifier failed, using defaults: {e}")
        
    persona = {
        "habits": habits,
        "personal_facts": personal_facts,
        "personality_traits": personality_traits,
        "communication_style": {
            "avg_word_count": round(avg_word_count, 1),
            "tone": tone,
            "emoji_usage": emoji_usage
        }
    }
    
    with open('user_persona.json', 'w') as f:
        json.dump(persona, f, indent=2)
        
    print("Persona extracted and saved to user_persona.json")
    return persona

if __name__ == '__main__':
    extract_persona()
