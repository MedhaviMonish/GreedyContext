# GreedyContext
**GreedyContext** is a lightweight, dependency-free module that intelligently filters long conversation histories using embedding similarity and greedy path traversal. It minimizes token usage and latency when feeding context to LLMs — especially in chatbots and retrieval-based systems.

---

## 🔍 What It Does
- Encodes all messages using `SentenceTransformers`
- Builds a **cosine similarity graph** from message embeddings
- Uses a **greedy path algorithm** to traverse the most semantically relevant message chain
- Outputs only the most relevant subset of messages — sometimes as few as one!

This avoids blindly passing the last 100 messages to your LLM, saving tokens and improving latency.

---

## 🧠 Ideal Use Cases

- LLM-powered chatbots with long conversation history
- Memory-efficient assistants
- No need for summarization at every few steps
- Token-aware prompt construction tools

---

## 🚀 Quick Demo (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
