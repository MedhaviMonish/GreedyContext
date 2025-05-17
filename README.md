# GreedyContext

**GreedyContext** is a simple, self-contained solution to reduce token usage and latency in LLM-based applications.  
It identifies only the most semantically relevant messages from a conversation history using cosine similarity and a greedy traversal path over a graph of sentence embeddings.

---

## 🔍 What It Does

- Encodes conversation messages using `SentenceTransformers`
- Builds a **cosine similarity matrix** between all messages
- Converts it into a **strict upper triangular graph** — so messages only link to **previous** ones
- Traverses the graph using a **greedy algorithm** to extract the most relevant backward message chain
- Outputs a reduced set of messages to feed into your LLM

---

## ✅ Why Use It

- No need to pass all past messages — pass just the semantically important ones
- Reduces token count and speeds up LLM responses
- No summarization step required
- Works without vector DBs or third-party memory libraries

---

## ⚙️ Practical Notes

- Suggested similarity **threshold**: `0.2`  
  (Helps ignore unrelated or noisy messages)
- **Pyvis** is used only for visualization. You don’t need it in your actual chatbot pipeline.
- This is not a Python library or package — it’s a **working solution** you can adapt to your use case.

---

## 💡 Use Cases

- LLM-powered chatbots with large conversation histories
- Token-aware context filtering (e.g., for OpenAI, Claude, LLaMA, etc.)
- Systems where summarization is not ideal or would degrade fidelity
- Fast pre-processing before calling LLMs

---

## 📷 Demo (Optional)

Run the app to see how the graph and greedy traversal work:

```bash
pip install -r requirements.txt
python app.py
