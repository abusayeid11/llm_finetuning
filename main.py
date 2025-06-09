from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.file_loader import load_knowledge_base, get_best_match, detect_subtopic
import torch

# 1. Initialize Model
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

def fallback_to_model(user_input):
    """Handles queries when no knowledge base match is found"""
    prompt = f"""As a career advisor, answer the following question with concise, actionable advice.
    If you don't know the answer, say "I specialize in career advice for tech fields like software engineering and data science. Could you clarify your question?"

    Question: {user_input}
    Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()

# 2. Load Knowledge Base
KNOWLEDGE = load_knowledge_base()
# bot.py
def generate_response(user_input):
    # Step 1: Detect main topic (e.g., "software engineering")
    topic = get_best_match(user_input, KNOWLEDGE)
    if not topic:
        return fallback_to_model(user_input)  # Your existing DialoGPT fallback
    
    # Step 2: Detect sub-topic (e.g., "pros")
    subtopic = detect_subtopic(user_input)
    
    # Step 3: Generate targeted response
    response = f"🔍 {topic.replace('_', ' ').title()}: "
    if subtopic:
        data = KNOWLEDGE[topic].get(subtopic)
        if data:
            if isinstance(data, dict):  # Nested resources
                response += f"Top {subtopic}:\n"
                for category, items in data.items():
                    response += f"• {category.title()}: {', '.join(items)}\n"
            else:  # Simple lists (pros/cons/steps)
                response += "\n• " + "\n• ".join(data)
        else:
            response += f"I don't have {subtopic} info yet. Ask about: overview/pros/cons/resources/steps."
    else:
        # Default to overview if no subtopic detected
        response += KNOWLEDGE[topic].get("overview", "Ask me about: pros/cons/resources/steps.")
    
    return response

# 4. Test Interface (unchanged)
def test_bot():
    print("\n💬 Career Guide Bot (Type 'quit' to exit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        print("\nBot:", generate_response(user_input))

if __name__ == "__main__":
    test_bot()