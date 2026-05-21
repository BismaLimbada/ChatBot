# 🌸 Neura AI

Neura AI is an empathetic, context-aware chatbot framework built to support mental health awareness. Powered by a Model-Based Reflex Agent architecture implemented via TensorFlow, NLTK, and Streamlit, the system actively monitors and logs conversational state dynamics, tracked emotional progressions, and strict context-driven verification loops over multi-turn exchanges.

> **Status:** 🚀 Live Deployment Active

An empathetic, context-aware Model-Based Reflex Agent framework designed for mental health awareness, emotional scaling, and supportive dialogue.

### 🔗 Quick Links
* **Live Web Application:** [neuraai.streamlit.app](https://neuraai.streamlit.app/)
* **Development Repository:** [GitHub Root](https://github.com/BismaLimbada/ChatBot)

---

## 🚀 Key Features

* **Model-Based Reflex Agent State:** Tracks conversational contexts dynamically across user exchanges using localized historical context vectors.
* **Greeting Duplication Prevention:** Tracks state variables under-the-hood to identify initial greetings, ensuring the agent provides supportive, conversational replies instead of generic introductory messages if a greeting tag is triggered mid-dialogue.
* **Context State Machine Loops:** Employs explicit guard structures to seamlessly process confirmations (e.g., grounding rules, sleep advice, or burnout recovery tasks) before moving to generic fallback or deep model inference.
* **Smart Fallback Management:** Gracefully flags unrecognized inputs or text matching below the neural network's confidence limit, keeping conversations grounded and safe.

---

## 🛠️ Pipeline Architecture

The core interactive processing script operates sequentially:

1. **Phase 1: Context Memory Validation** – Before feeding input arrays to the neural model, the pipeline evaluates if an active `st.session_state.active_context` is open. If the user replies with a clear confirmation keyword (`yes`/`no`), it bypasses standard ML parsing entirely to route straight to your custom solution steps.
2. **Phase 2: Text Preprocessing** – Cleans text parameters by standardizing common contractions, applying sentence tokenization, stripping away lone floating punctuation noise, and reducing tokens to their base roots via a Porter Stemmer.
3. **Phase 3: Deep Inference Implementation** – Translates stemmed arrays into fixed-length binary multi-hot Bag-of-Words vectors. Predictions are fed to the pre-compiled Keras neural network model to map user input against distinct tags.
4. **Phase 4: Multi-Turn State Management** – Based on matching confidence thresholds, it triggers empathetic intent templates, updates runtime memory logs, or sets new tracking milestones inside `st.session_state.active_context`.

---

## 📂 Project Structure

```text
├── data/
│   └── intents.json           # Structural dataset containing tags, patterns, and responses
├── .gitignore                 # Specifies intentionally untracked build files to ignore
├── README.md                  # Project documentation overview
├── app.py                     # Main Streamlit user interface, core processing, & state tracking
├── data_mappings.json         # Compiled vocabulary arrays (all_words) and target model tags
├── preprocess.py              # Text tokenization, Porter stemming, and BoW array compilation
├── requirements.txt           # Project environment dependencies
├── train_data.py              # Auxiliary dataset preprocessing utilities
└── train_model.py             # Neural network compilation and training routine script