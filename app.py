import random
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from preprocess import tokenize, stem, bag_of_words

# 1. Page Config
st.set_page_config(page_title="MindEase AI", page_icon="🧠", layout="centered")

st.title("🧠 MindEase AI")
st.caption("Empathetic, instantaneous mental health awareness support.")
st.divider()

# 2. Lightning-Fast Resource Loading
@st.cache_resource
def load_bot_resources():
    # Load model and configurations
    model = tf.keras.models.load_model('mental_health_model.keras', compile=False)
    with open("data_mappings.json", "r") as f:
        mappings = json.load(f)
    with open("data/intents.json", "r") as f:
        intents = json.load(f)
    return model, mappings["all_words"], mappings["tags"], intents

model, all_words, tags, intents = load_bot_resources()

# 3. Session Memory State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am MindEase. How are you holding up today?"}
    ]

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 4. Optimized Chat Processing Pipeline
if user_input := st.chat_input("Type something here..."):
    # Render user bubble immediately
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Debug print to terminal to see if the script is awake
    print(f"\n[DEBUG] User Typed: '{user_input}'")
    
    # Run text preprocessing
    tokenized = tokenize(user_input)
    bow_vector = bag_of_words(tokenized, all_words)
    input_data = np.array([bow_vector], dtype=np.float32)
    
    # Instant prediction execution
    prediction = model(input_data, training=False).numpy()
    highest_idx = np.argmax(prediction[0])
    confidence = prediction[0][highest_idx]
    predicted_tag = tags[highest_idx]
    
    # Debug results in your terminal window
    print(f"[DEBUG] Predicted Tag: '{predicted_tag}' | Confidence: {confidence:.2f}")
    
    # Routing Engine
    if predicted_tag == "crisis":
        reply = intents['intents'][7]['responses'][0] # Direct fast pull
    elif confidence < 0.35:  # Lowered threshold to guarantee it replies more easily
        reply = "I hear you, but I want to make sure I understand perfectly. Could you share a bit more about what's on your mind?"
    else:
        # Match tag quickly
        reply = "I'm listening. Please go on."  # Quick fallback assignment
        for intent in intents['intents']:
            if intent['tag'] == predicted_tag:
                reply = random.choice(intent['responses'])
                break

    # Display assistant response
    with st.chat_message("assistant"):
        st.write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})