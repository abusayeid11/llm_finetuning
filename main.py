from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from difflib import get_close_matches
import torch

# 1. Initialize Small Model
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# 2. Career Knowledge Base (Expandable!)
CAREER_KNOWLEDGE = {
    "software engineering": {
        "core_skills": ["Data Structures & Algorithms", "Version Control (Git)", "OOP Principles"],
        "languages": ["Python", "Java/JavaScript", "C++ (for systems programming)"],
        "learning_path": [
            "1. Master one language deeply",
            "2. Build 2-3 portfolio projects",
            "3. Practice Leetcode-style problems",
            "4. Contribute to open source"
        ]
    },
    "data science": {
        "essentials": ["Python/R", "SQL", "Pandas/Numpy", "Statistics"],
        "ml_tools": ["Scikit-learn", "TensorFlow/PyTorch", "MLflow"],
        "roadmap": [
            "1. Learn data wrangling with Pandas",
            "2. Study statistical modeling",
            "3. Explore ML algorithms",
            "4. Build end-to-end projects"
        ]
    }
}

# 3. Smart Response Generator
def generate_response(user_input):
    # Step 1: Check Knowledge Base
    matches = get_close_matches(
        user_input.lower(), 
        CAREER_KNOWLEDGE.keys(), 
        n=1, 
        cutoff=0.6
    )
    
    if matches:
        topic = matches[0]
        response = f"🚀 Here's career guidance for {topic}:\n\n"
        for category, items in CAREER_KNOWLEDGE[topic].items():
            response += f"• {category.replace('_', ' ').title()}:\n   - " + "\n   - ".join(items) + "\n\n"
        return response.strip()
    
    # Step 2: Fallback to Model Generation
    prompt = f"""As a career advisor, provide specific, actionable advice. 
    Question: {user_input}
    Advice:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        # temperature=0.7,
        # top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id = 50256
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Advice:")[-1].strip()

# 4. Test Interface
def test_bot():
    print("\n💬 Career Guide Bot (Type 'quit' to exit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        print("\nBot:", generate_response(user_input))

if __name__ == "__main__":
    test_bot()