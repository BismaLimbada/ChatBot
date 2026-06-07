import random
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from preprocess import tokenize, stem, bag_of_words
import nltk

def download_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

download_nltk()

# ==========================================
# 1. Page Configuration & Custom Styling
# ==========================================
st.set_page_config(page_title="Neura AI", page_icon="🌸", layout="centered")

st.html("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 5rem; }
        h1 { text-align: center; margin-bottom: 0px; }
        div[data-testid="stCaptionContainer"] { text-align: center; }
        [data-testid="stChatMessageAvatarAssistant"] { background-color: black !important; color: #ffb6c1 !important; }
        [data-testid="stChatMessageAvatarUser"] { background-color: black !important; color: #ffb6c1 !important; }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p { color: black !important; }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background-color: #FFC5D3; border-radius: 15px; padding: 10px; }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { border-radius: 15px; padding: 10px; }
        .stButton > button { background-color: #FFC5D3 !important; color: black !important; border: 3px solid #ff9fbc !important; font-weight: 600; }
        .stButton > button:hover { background-color: #FFC5D3 !important; color: black !important; }
        .stProgress > div > div > div > div { background-color: #ffb6c1 !important; }
        [data-testid="stChatInput"] { background-color: #FFC5D3 !important; border-radius: 15px !important; border: 2px solid #ff9fbc !important; padding: 8px !important; }
        [data-testid="stChatInput"] > div { background-color: #FFC5D3 !important; border-radius: 15px !important; }
        [data-testid="stChatInput"] textarea { background-color: #FFC5D3 !important; color: black !important; }
        [data-testid="stChatInput"] textarea::placeholder { color: black !important; opacity: 1 !important; }
    </style>
""")

st.markdown("""
<div style="
    background-color:#FFC5D3;
    padding:20px;
    border:3px solid #ff9fbc;
    border-radius:20px;
    text-align:center;
    margin-bottom:20px;
    color:black;
">
    <a href="https://neuraai.streamlit.app/" target="_blank" style="text-decoration: none; color: black;">
        <h1 style="margin-bottom:5px; color:black; cursor:pointer;">Neura AI</h1>
    </a>
    <p style="font-size:16px;">
        Your empathetic, context-aware framework for mental health awareness & support.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. Optimized Resource Loading
# ==========================================
@st.cache_resource
def load_bot_resources():
    model = tf.keras.models.load_model("mental_health_model.keras", compile=False)
    with open("data_mappings.json", "r") as f:
        mappings = json.load(f)
    with open("data/intents.json", "r") as f:
        intents = json.load(f)
    return model, mappings["all_words"], mappings["tags"], intents

model, all_words, tags, intents = load_bot_resources()

# ==========================================
# 3. Agent State Architecture Setup
# ==========================================
if "agent_internal_state" not in st.session_state:
    st.session_state.agent_internal_state = {
        "emotion_counters": {"anxious": 0, "depressed": 0, "exhausted": 0, "frustrated": 0},
        "crisis_mode_active": False,
        "total_conversation_turns": 0,
        "has_greeted": False 
    }

if "active_context" not in st.session_state:
    st.session_state.active_context = None

if "messages" not in st.session_state:
    st.session_state.messages = []

EMOTION_TAG_ROUTING = {
    "panic_attack": "anxious", "overthinking": "anxious", "night_anxiety": "anxious",
    "social_anxiety": "anxious", "health_anxiety": "anxious", "sleep_problems": "anxious",
    "loneliness": "depressed", "low_self_esteem": "depressed", "grief": "depressed", "self_isolation": "depressed",
    "motivation_issues": "exhausted", "burnout": "exhausted", "academic_pressure": "exhausted",
    "anger_frustration": "frustrated", "family_pressure": "frustrated", "relationship_stress": "frustrated"
}

CONTEXT_SETTERS = {
    "panic_attack": "awaiting_panic_followup", "overthinking": "awaiting_overthinking_followup",
    "academic_pressure": "awaiting_academic_followup", "family_pressure": "awaiting_family_followup",
    "loneliness": "awaiting_loneliness_followup", "motivation_issues": "awaiting_motivation_followup",
    "sleep_problems": "awaiting_sleep_followup", "burnout": "awaiting_burnout_followup"
}

with st.sidebar:
    st.subheader("🤖 Agent Internal Model State")
    st.write("This panel tracks how the Model-Based Reflex Agent builds its perception of the conversation state:")
    state = st.session_state.agent_internal_state
    
    st.progress(min(state["emotion_counters"]["anxious"] * 25, 100), text=f"Anxiety Scale: {state['emotion_counters']['anxious']}")
    st.progress(min(state["emotion_counters"]["depressed"] * 25, 100), text=f"Depression Scale: {state['emotion_counters']['depressed']}")
    st.progress(min(state["emotion_counters"]["exhausted"] * 25, 100), text=f"Exhaustion Scale: {state['emotion_counters']['exhausted']}")
    st.progress(min(state["emotion_counters"]["frustrated"] * 25, 100), text=f"Frustration Scale: {state['emotion_counters']['frustrated']}")
    st.divider()
    
    st.caption(f"Active Context Token: `{st.session_state.active_context}`")
    st.caption(f"Global Dialogue Index: {state['total_conversation_turns']}")

    if st.button("Reset Session History & Internal State", use_container_width=True):
        st.session_state.messages = []
        st.session_state.active_context = None
        st.session_state.agent_internal_state = {
            "emotion_counters": {"anxious": 0, "depressed": 0, "exhausted": 0, "frustrated": 0},
            "crisis_mode_active": False, "total_conversation_turns": 0, "has_greeted": False 
        }
        st.rerun()

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 4. Input Layer & Core Logic
# ==========================================
if user_input := st.chat_input("Type something here..."):
    
    with st.chat_message("user"):
        st.markdown(user_input)
        
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.agent_internal_state["total_conversation_turns"] += 1

    reply = None
    cleaned_choice = user_input.strip().lower().replace(".", "").replace("!", "")

    with st.spinner("Processing..."):
        
        # PHASE 1: CRISIS MODE
        if st.session_state.agent_internal_state["crisis_mode_active"]:
            reply = "Please reach out to a trusted professional or contact the Umang Helpline at 0311-7786264 immediately. For safety purposes, active chatbot configurations are suspended."

        if reply is None:
            # PHASE 2: ML INFERENCE FIRST
            current_tokens = tokenize(user_input)
            current_bow = bag_of_words(current_tokens, all_words)

            historical_bow = np.zeros(len(all_words), dtype=np.float32)
            user_history = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
            if len(user_history) > 1:
                prior_tokens = tokenize(user_history[-2])
                historical_bow = bag_of_words(prior_tokens, all_words)

            combined_bow = np.maximum(current_bow, historical_bow)
            input_tensor = np.array([combined_bow], dtype=np.float32)

            prediction = model(input_tensor, training=False).numpy()
            highest_idx = np.argmax(prediction[0])
            confidence = prediction[0][highest_idx]
            predicted_tag = tags[highest_idx]

            print(f"[DEBUG] Input: '{user_input}' | Prior Context: {st.session_state.active_context}")
            print(f"[DEBUG] ML Predicted Tag: '{predicted_tag}' | Confidence: {confidence:.2f}")

            # Low confidence override for non-explicit goodbyes
            is_explicit_farewell = any(word in cleaned_choice for word in ["bye", "goodbye", "leave", "going", "allah hafiz", "khuda hafiz", "exit"])
            if predicted_tag == "goodbye" and not is_explicit_farewell:
                confidence = 0.10

            if confidence < 0.50:
                reply = random.choice([
                    "I'm not sure I understand that. Could you tell me a little more about what's on your mind?",
                    "I want to make sure I understand you properly, but I didn't quite catch that. What's been going on?",
                    "I'm not sure I understand. If things are feeling heavy or complicated, take your time expressing them."
                ])
                st.session_state.active_context = None
                
            else:
                # PHASE 3: CONTEXT ROUTING (Using ML output + safety nets)
                is_affirmation = predicted_tag == "affirmation" or cleaned_choice in ["yes", "yeah", "yup", "haan", "ji", "okay", "sure"]
                is_rejection = predicted_tag == "rejection" or cleaned_choice in ["no", "nah", "nope", "nahi", "stop"]

                if (is_affirmation or is_rejection) and st.session_state.active_context is not None:
                    
                    context = st.session_state.active_context
                    
                    if context == "awaiting_panic_followup":
                        reply = "Alright, let's step through a grounding exercise together. Press both feet firmly into the ground. Breathe in slowly for 4 seconds... hold... now breathe out for 4 seconds." if is_affirmation else "I'm glad it has not escalated further. Let's take things slowly. What thoughts are bothering you most right now?"
                    
                    elif context == "awaiting_overthinking_followup":
                        reply = "Since it's taking up a lot of space, try writing those thoughts down to get them out of your head, or break your focus with a physical change of environment." if is_affirmation else "I'm glad it's not completely taking over your headspace right now. Remember to take things one step at a time."
                    
                    elif context == "awaiting_academic_followup":
                        reply = "Balancing studies can be overwhelming. Try breaking assignments into small blocks and taking a short break every 25 minutes." if is_affirmation else "I'm glad you've found a rhythm that lets you manage the load without burning out completely."
                    
                    elif context == "awaiting_family_followup":
                        reply = "Navigating those expectations can feel incredibly heavy. Remember that it's okay to protect your peace and take things one day at a time." if is_affirmation else "I'm glad you have spaces where you feel free to express yourself outside of those expectations."
                    
                    elif context == "awaiting_loneliness_followup":
                        reply = "Keeping things bottled up can make isolation feel twice as heavy. Thank you for sharing that with me. Remember, you don't have to navigate everything entirely on your own." if is_affirmation else "I'm glad to hear you're able to open up sometimes. Sharing with even one trusted connection can change how you carry this weight."
                    
                    elif context == "awaiting_motivation_followup":
                        reply = "When stress completely drains your batteries, what looks like lack of motivation is often your mind demanding real rest. Be gentle with yourself today." if is_affirmation else "If it isn't deep stress, sometimes routines just get stagnant. Try picking one microscopic, 2-minute task today to break the friction."
                    
                    elif context == "awaiting_sleep_followup":
                        reply = "When sleep routines break down, it directly intensifies daily emotional exhaustion. Try setting a screen-free winding window tonight." if is_affirmation else "I'm glad your schedule isn't completely fractured, but struggling to clear your head at night is still an exhausting cycle."
                    
                    elif context == "awaiting_burnout_followup":
                        reply = "Taking a physical break is amazing, but your mind might still be processing heavy background tabs. Practice letting go of tasks completely for an hour." if is_affirmation else "Pushing through without pausing is exactly how burnout seals itself in place. Try implementing a hard stop time for work tonight."
                    
                    st.session_state.active_context = None

                else:
                    # PHASE 4: STANDARD INTENT MATCHING & CONTEXT SETTING
                    if predicted_tag in ["greeting", "islamic_greeting"]:
                        if st.session_state.agent_internal_state["has_greeted"]:
                            reply = random.choice([
                                "I'm right here with you. What's on your mind?",
                                "I'm listening. Tell me more about how you've been doing."
                            ])
                        else:
                            for intent in intents['intents']:
                                if intent['tag'] == predicted_tag:
                                    reply = random.choice(intent['responses'])
                                    break
                            st.session_state.agent_internal_state["has_greeted"] = True
                    else:
                        for intent in intents['intents']:
                            if intent['tag'] == predicted_tag:
                                reply = random.choice(intent['responses'])
                                break

                    # Set new context if applicable
                    if predicted_tag in CONTEXT_SETTERS:
                        st.session_state.active_context = CONTEXT_SETTERS[predicted_tag]
                        print(f"[DEBUG] Context switched to: {st.session_state.active_context}")
                    else:
                        st.session_state.active_context = None
                        print("[DEBUG] Context Cleared (Topic Shifted Successfully)")

                    # UPDATE EMOTION TRACKERS
                    explicit_depressed_keywords = ["depressed", "depression", "sad", "hopeless", "lonely", "isolated"]
                    if any(word in cleaned_choice.split() for word in explicit_depressed_keywords):
                        target_dimension = "depressed"
                    elif predicted_tag in EMOTION_TAG_ROUTING:
                        target_dimension = EMOTION_TAG_ROUTING[predicted_tag]
                    else:
                        target_dimension = None

                    if target_dimension:
                        st.session_state.agent_internal_state["emotion_counters"][target_dimension] += 1

                    if predicted_tag == "crisis_support":
                        st.session_state.agent_internal_state["crisis_mode_active"] = True

    # Render Assistant Response
    with st.chat_message("assistant"):
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()