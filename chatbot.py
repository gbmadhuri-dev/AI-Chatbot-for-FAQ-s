import os
import uuid
import threading
import sqlite3
from datetime import datetime
from flask import Flask, request, render_template_string, session
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import string
from rapidfuzz import process  # For fuzzy matching

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'default-secret-key-change-in-production')  # From env var

# Load model (configurable via env)
model_name = os.environ.get('MODEL_NAME', 'microsoft/DialoGPT-small')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Threading lock for concurrency safety on model.generate()
generation_lock = threading.Lock()

# Load FAQs
FAQS_FILE = 'faqs.json'
faqs = {}
if os.path.exists(FAQS_FILE):
    try:
        with open(FAQS_FILE, 'r', encoding='utf-8') as f:
            faqs = json.load(f)
    except Exception as e:
        print(f"Error loading FAQs: {e}")

# DB setup with session_id column
def init_db():
    conn = sqlite3.connect('chatbot_logs.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_input TEXT,
                    bot_response TEXT,
                    timestamp TEXT
                )''')
    # Add session_id column if not exists (backward-compatible)
    try:
        c.execute("ALTER TABLE logs ADD COLUMN session_id TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()
    conn.close()

def log_interaction(session_id, user_input, bot_response):
    try:
        conn = sqlite3.connect('chatbot_logs.db')
        c = conn.cursor()
        c.execute("INSERT INTO logs (session_id, user_input, bot_response, timestamp) VALUES (?, ?, ?, ?)",
                  (session_id, user_input[:512], bot_response, datetime.now().isoformat()))  # Sanitize input
        conn.commit()
        print(f"Successfully logged: session_id={session_id}, user_input='{user_input}', bot_response='{bot_response}'")  # Debug
    except Exception as e:
        print(f"DB logging error: {e}")
    finally:
        conn.close()

# Fuzzy FAQ matching with threshold
FUZZY_THRESHOLD = 75
def rule_based_response(user_input):
    if not faqs:
        return None
    input_clean = user_input.lower().translate(str.maketrans('', '', string.punctuation))
    best_match = process.extractOne(input_clean, [k.lower().translate(str.maketrans('', '', string.punctuation)) for k in faqs.keys()])
    if best_match and best_match[1] >= FUZZY_THRESHOLD:
        original_key = list(faqs.keys())[best_match[2]]
        return faqs[original_key]
    return None

# Generate response with context and safety
def generate_response(user_input, conversation_history):
    # Check FAQ first
    rule_response = rule_based_response(user_input)
    if rule_response:
        return rule_response
    
    # AI response with lock
    try:
        with generation_lock:
            # Build input with history (last 10 turns to avoid OOM)
            history_str = ' '.join(conversation_history[-10:]) + ' ' + user_input
            inputs = tokenizer.encode(history_str + tokenizer.eos_token, return_tensors='pt')
            outputs = model.generate(
                inputs,
                max_length=100,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=30,
                top_p=0.8,
                temperature=0.4
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).split(user_input)[-1].strip()
            return response or "I'm sorry, I couldn't generate a response."
    except Exception as e:
        return f"Error generating response: {str(e)}. Please try again."

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>AI Chatbot</title></head>
<body>
    <h1>AI-Powered Chatbot</h1>
    <form method="POST">
        <label>Your Message:</label><br>
        <input type="text" name="user_input" required><br><br>
        <button type="submit">Send</button>
        <button type="submit" name="reset">Reset Conversation</button>
    </form>
    {% if response %}<h2>Bot Response:</h2><p>{{ response }}</p>{% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def chat():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    if 'conversation' not in session:
        session['conversation'] = []
    
    response = None
    if request.method == 'POST':
        user_input = request.form.get('user_input', '').strip()
        if 'reset' in request.form:
            session['conversation'] = []
            response = "Conversation reset."
        elif len(user_input) > 512:
            response = "Input too long (max 512 characters). Please shorten your message."
        else:
            session['conversation'].append(f"User: {user_input}")
            response = generate_response(user_input, session['conversation'])
            session['conversation'].append(f"Bot: {response}")
            # Ensure we have data before logging
            if user_input and response:
                log_interaction(session['session_id'], user_input, response)
            else:
                print(f"Skipped logging: user_input='{user_input}', response='{response}'")  # Debug
    return render_template_string(HTML_TEMPLATE, response=response)

if __name__ == '__main__':
    init_db()
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=debug_mode)