import random
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from preprocess import tokenize, stem, bag_of_words

# ==========================================
# 1. CLEAN STABLE NATIVE PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Neura AI - Wellness Space", 
    page_icon="🌸", 
    layout="centered"
)

# App Titles using standard layout blocks
st.title("🌸 Neura AI")
st.write("Your empathetic, context-aware mental wellness reflection companion.")
st.divider()

# Simple native sidebar implementation
with st.sidebar:
    st.header("📌 Support Directory")
    st.write("If you are under stress or in immediate distress, helpful and confidential spaces are open:")
    
    st.markdown("- **Umang Helpline (Pakistan):** 📞 [0311-7786264](tel:03117786264) (Available 24/7)")
    st.markdown("- **Crisis Lifeline:** 📞 Call or Text **988**")
    
    st.divider()
    if st.button("🔄 Clear Chat Session Memory"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello. I'm Neura. I'm here to listen without judging you. What's been on your mind lately?"}
        ]
        st.session_state.active_context = None
        st.session_state.detected_topic = None
        st.rerun()

# ==========================================
# 2. CACHED MODEL RESOURCE LOADING
# ==========================================
@st.cache_resource
def load_bot_resources():
    model = tf.keras.models.load_model('mental_health_model.keras', compile=False)
    with open("data_mappings.json", "r") as f:
        mappings = json.load(f)
    with open("data/intents.json", "r") as f:
        intents = json.load(f)
    return model, mappings["all_words"], mappings["tags"], intents

model, all_words, tags, intents = load_bot_resources()

# ==========================================
# 3. STATE MACHINE CONVERSATION HISTORY
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello. I'm Neura. I'm here to listen without judging you. What's been on your mind lately?"}
    ]

if "active_context" not in st.session_state:
    st.session_state.active_context = None

if "detected_topic" not in st.session_state:
    st.session_state.detected_topic = None

# Render existing conversation via official chat components
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ==========================================
# 4. EXPLICIT STRATIFIED DIALOGUE ANSWERS
# ==========================================
CONTEXT_ANSWERS = {
    "awaiting_sleep_confirmation": {
        "yes": "**🌸 Coping Strategy:** Since your sleep schedule has been irregular, let's focus on a tiny recovery technique: try the *20-Minute Out-of-Bed Rule*. If you can't fall asleep after 20 minutes, step out of bed and rest in a dimly lit room reading a physical book until you feel tired.\n\n**✨ Reassurance:** Navigating exhaustion is highly draining. Remember that resting quietly still helps your mind settle, and you don't need to be perfectly productive tomorrow.",
        "no": "**🌸 Coping Strategy:** I'm glad your schedule isn't completely broken! If your mind is still racing with thoughts before sleeping, try doing a 2-minute 'brain dump' on a physical pad of paper to visually park your worries.\n\n**✨ Reassurance:** Tonight is just one night out of many. Your fundamental worth is never tied to how perfectly your body rests."
    },
    "awaiting_panic_grounding": {
        "yes": "**🌸 Let's walk through the 5-4-3-2-1 Grounding Method right now:**\n\n1. 👀 Name **5** items you can see around your room.\n2. 🖐️ Focus on **4** physical sensations you can feel right now.\n3. 👂 Acknowledge **3** sounds in your background environment.\n4. 👃 Breathe and locate **2** elements you can smell.\n5. 👅 Identify **1** thing you can taste.\n\n**✨ Reassurance:** This physical surge is a wave of pure adrenaline. It is deeply uncomfortable, but it is temporary. You are fundamentally safe, and your nervous system will bring you back down to calm shortly.",
        "no": "**✨ Reassurance:** I am incredibly relieved to hear that. Let's focus on keeping things steady and quiet. Give your body permission to release tension by dropping your shoulders and taking a long, gentle exhale."
    },
    "awaiting_burnout_reflection": {
        "yes": "**🌸 Coping Strategy:** Stepping away from studying is a huge step! Let's reinforce that with *Absolute Disconnection*: shut down your laptop lid completely and turn off university group chats for at least 30 minutes tonight.\n\n**✨ Reassurance:** Taking a deliberate break isn't slacking off—it is routine maintenance for your mind so you can function cleanly later.",
        "no": "**🌸 Coping Strategy:** Pushing through severe fatigue without a break can quickly lead to cognitive burnout. Let's implement a strict *Pomodoro interval*: set a timer to focus for 25 minutes, followed by a mandatory 5-minute break away from your screen.\n\n**✨ Reassurance:** Your value as a person is not defined by your project velocity, a broken script, or your grades. You are allowed to take up space and rest."
    }
}

CONTEXT_SETTERS = {
    "sleep_problems": "awaiting_sleep_confirmation",
    "panic_attack": "awaiting_panic_grounding",
    "burnout": "awaiting_burnout_reflection",
    "academic_pressure": "awaiting_burnout_reflection"
}

TOPIC_REMINDERS = {
    "academic_pressure": "🎓 University Constraints",
    "burnout": "🎓 University Constraints",
    "family_pressure": "🏡 Domestic/Family Dynamics",
    "relationship_stress": "🤝 Interpersonal Bonds",
    "panic_attack": "🚨 High-Anxiety Overload"
}

# ==========================================
# 5. INPUT DISPATCHER & MAIN PROCESSING LOOP
# ==========================================
user_input = st.chat_input("Talk to Neura...")

if st.session_state.active_context is not None:
    st.write("💡 Quick reply choices:")
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("✨ Yes"):
            user_input = "yes"
    with col2:
        if st.button("🍃 No"):
            user_input = "no"

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    cleaned_input = user_input.strip().lower()
    reply = None
    
    # Process Context Tree Loops
    if st.session_state.active_context in CONTEXT_ANSWERS:
        current_context = st.session_state.active_context
        if cleaned_input in ["yes", "yeah", "yup", "haan", "ji", "true"]:
            reply = CONTEXT_ANSWERS[current_context]["yes"]
            st.session_state.active_context = None 
        elif cleaned_input in ["no", "nah", "nope", "nahi", "false"]:
            reply = CONTEXT_ANSWERS[current_context]["no"]
            st.session_state.active_context = None 

    # Process Fallback Memory Hooks
    if reply is None:
        if cleaned_input in ["tell me more", "what should i do", "help me", "i am struggling", "what do i do"]:
            if st.session_state.detected_topic == "🎓 University Constraints":
                reply = "Since we were talking about academic pressure, try breaking your current tracking task or project into small milestones. Pick one item that takes less than ten minutes to handle. Don't worry about the whole script right now."
            elif st.session_state.detected_topic == "🚨 High-Anxiety Overload":
                reply = "Since you brought up feeling overwhelmed with panic symptoms earlier, let's look out for your body. Drop your shoulders, let your hands relax, and take a long, slow exhale out through your mouth."

    # TensorFlow Machine Learning Classifier Pipeline
    if reply is None:
        tokenized = tokenize(user_input)
        bow_vector = bag_of_words(tokenized, all_words)
        input_data = np.array([bow_vector], dtype=np.float32)
        
        prediction = model(input_data, training=False).numpy()
        highest_idx = np.argmax(prediction[0])
        confidence = prediction[0][highest_idx]
        predicted_tag = tags[highest_idx]
        
        if predicted_tag in TOPIC_REMINDERS:
            st.session_state.detected_topic = TOPIC_REMINDERS[predicted_tag]
            
        if confidence < 0.45:
            reply = random.choice(intents.get("fallback_responses", ["I'm listening closely. Can you tell me a little more about that?"]))
        else:
            for intent in intents['intents']:
                if intent['tag'] == predicted_tag:
                    reply = random.choice(intent['responses'])
                    break
            
            if predicted_tag in CONTEXT_SETTERS:
                st.session_state.active_context = CONTEXT_SETTERS[predicted_tag]

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()