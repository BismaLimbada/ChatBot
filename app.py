import random
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from preprocess import tokenize, stem, bag_of_words

# 1. Page Configuration & Custom Styling
st.set_page_config(page_title="Neura AI", page_icon="🌸", layout="centered")

# Enforces a clean, minimalist layout and changes the assistant avatar to pastel pink
st.html("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 5rem; }
        h1 { margin-bottom: 0px; }
        /* Style the native avatar container to be a beautiful pastel pink */
        [data-testid="stChatMessageAvatarAssistant"] {
            background-color: #F7D6DB !important;
            color: #000000 !important;
        }
    </style>
""")

# Main Header Design
st.title("🌸 Neura AI")
st.caption("Your empathetic, context-aware framework for mental health awareness & support.")
st.write("---")

# 2. Optimized Resource Loading
@st.cache_resource
def load_bot_resources():
    model = tf.keras.models.load_model('mental_health_model.keras', compile=False)
    with open("data_mappings.json", "r") as f:
        mappings = json.load(f)
    with open("data/intents.json", "r") as f:
        intents = json.load(f)
    return model, mappings["all_words"], mappings["tags"], intents

model, all_words, tags, intents = load_bot_resources()

# --- MODEL-BASED REFLEX AGENT: STATE ARCHITECTURE SETUP ---
# To satisfy the criteria of a model-based reflex agent, the agent must keep an 
# internal state tracking how the environment updates over time independent of single inputs.
if "agent_internal_state" not in st.session_state:
    st.session_state.agent_internal_state = {
        "emotion_counters": {
            "anxious": 0,
            "depressed": 0,
            "exhausted": 0,
            "frustrated": 0
        },
        "crisis_mode_active": False,
        "total_conversation_turns": 0
    }

if "active_context" not in st.session_state:
    st.session_state.active_context = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Map tags explicitly to emotional dimensions within the agent's tracking model
EMOTION_TAG_ROUTING = {
    "panic_attack": "anxious", "overthinking": "anxious", "night_anxiety": "anxious", 
    "social_anxiety": "anxious", "health_anxiety": "anxious", "sleep_problems": "anxious",
    "loneliness": "depressed", "low_self_esteem": "depressed", "grief": "depressed", "self_isolation": "depressed",
    "motivation_issues": "exhausted", "burnout": "exhausted", "academic_pressure": "exhausted",
    "anger_frustration": "frustrated", "family_pressure": "frustrated", "relationship_stress": "frustrated"
}

# Mapping conditions where contexts must be applied based on your classification layout
CONTEXT_SETTERS = {
    "panic_attack": "awaiting_panic_followup",
    "overthinking": "awaiting_overthinking_followup",
    "academic_pressure": "awaiting_academic_followup",
    "family_pressure": "awaiting_family_followup"
}

# --- SIDEBAR MONITOR: VISUALIZING THE AGENT'S INTERNAL MODEL ---
with st.sidebar:
    st.subheader("🤖 Agent Internal Model State")
    st.write("This panel tracks how the Model-Based Reflex Agent builds its perception of the conversation state over multiple turns:")
    
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
            "crisis_mode_active": False,
            "total_conversation_turns": 0
        }
        st.rerun()

# 3. Chat History Rendering Loop
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Active Pipeline Input Layer
if user_input := st.chat_input("Type something here..."):
    
    # Render User's chat bubble and append to session memory logs
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Update multi-turn system attributes inside our internal agent model state tracker
    st.session_state.agent_internal_state["total_conversation_turns"] += 1
    
    reply = None
    
    with st.spinner("Processing..."):
        
        # --- PHASE 1: AGENT EMOTIONAL PROFILE ASSESSMENT ---
        # The agent uses input sequences to extract keywords to feed into its state counter
        raw_tokenized = tokenize(user_input)
        
        # --- PHASE 2: CRITICAL EMERGENCY ENFORCEMENT GUARD ---
        if st.session_state.agent_internal_state["crisis_mode_active"]:
            reply = "Please reach out to a trusted professional or contact the Umang Helpline at 0311-7786264 immediately. For safety purposes, active chatbot configurations are suspended."

        # --- PHASE 3: STATE MACHINE TREE ANALYSIS (CONTEXT HANDLING) ---
        if reply is None and st.session_state.active_context is not None:
            cleaned_choice = user_input.strip().lower()
            
            if st.session_state.active_context == "awaiting_panic_followup":
                if cleaned_choice in ["yes", "yeah", "yup", "haan", "ji"]:
                    reply = "Alright, let's step through a physical grounding rhythm right now. Focus entirely on my words. Press both your feet firmly into the ground. Breathe in slowly for 4 seconds... hold it... now breathe out for 4 seconds. Let's repeat this until your heartbeat slows down."
                    st.session_state.active_context = None
                elif cleaned_choice in ["no", "nah", "nope", "nahi"]:
                    reply = "I'm glad to hear it hasn't escalated to that point. Let's just focus on taking things slow. What's currently occupying your thoughts?"
                    st.session_state.active_context = None
                    
            elif st.session_state.active_context == "awaiting_overthinking_followup":
                if "study" in cleaned_choice or "exam" in cleaned_choice or "university" in cleaned_choice:
                    reply = "Academic stress can cause real thought spirals. Remember, you don't have to resolve the entire syllabus tonight. Focus on one small section at a time."
                    st.session_state.active_context = None
                elif "family" in cleaned_choice or "friend" in cleaned_choice:
                    reply = "Interpersonal friction takes an immense amount of emotional energy. It makes total sense why your thoughts are racing right now."
                    st.session_state.active_context = None

        # --- PHASE 4: MACHINE LEARNING & REFLEX MATRICES INFERENCE ---
        if reply is None:
            tokenized_seq = tokenize(user_input)
            bow_vector = bag_of_words(tokenized_seq, all_words)
            input_tensor = np.array([bow_vector], dtype=np.float32)
            
            # Predict intent tag using model
            prediction = model(input_tensor, training=False).numpy()
            highest_idx = np.argmax(prediction[0])
            confidence = prediction[0][highest_idx]
            predicted_tag = tags[highest_idx]
            
            # Diagnostics console routing outputs
            print(f"[DEBUG] Input Sequence: '{user_input}' | State Stack Memory: {st.session_state.active_context}")
            print(f"[DEBUG] Model Tag Classification Result: '{predicted_tag}' | Vector Confidence Mapping score: {confidence:.2f}")
            
            # Confidence Threshold Routing Handler
            if confidence < 0.40:
                reply = random.choice(intents.get("fallback_responses", ["I am listening closely. Tell me more about what's going on."]))
            else:
                # Find matching response block array
                for intent in intents['intents']:
                    if intent['tag'] == predicted_tag:
                        reply = random.choice(intent['responses'])
                        break
                
                # --- PHASE 5: UPDATE AGENT STATE VECTORS ---
                if predicted_tag in EMOTION_TAG_ROUTING:
                    target_dimension = EMOTION_TAG_ROUTING[predicted_tag]
                    st.session_state.agent_internal_state["emotion_counters"][target_dimension] += 1
                
                if predicted_tag == "crisis_support":
                    st.session_state.agent_internal_state["crisis_mode_active"] = True
                
                if predicted_tag in CONTEXT_SETTERS:
                    st.session_state.active_context = CONTEXT_SETTERS[predicted_tag]
                    print(f"[DEBUG] Agent State Changed: Active Context State raised to '{st.session_state.active_context}'")

    # Render Assistant's computed chat bubble and commit to persistent state
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    
    # Forces layout cycle update to clear input widgets cleanly
    st.rerun()