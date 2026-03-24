# INDECIMAL RAG Chatbot — Construction Marketplace AI Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers customer questions about INDECIMAL's construction marketplace using internal documents. Built with a custom web frontend, semantic vector search, and LLM-powered grounded answer generation.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange.svg)

---

##  Quick Start

### 1. Clone & Install Dependencies

```bash
git clone <repo-url>
cd assignment2

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your [OpenRouter](https://openrouter.ai/) API key:

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

> **Note**: You can get a free API key at [openrouter.ai](https://openrouter.ai/). The app works without a key (fallback mode shows raw retrieved chunks), but for full LLM-generated answers, an API key is required.

### 3. Run the Application

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## 🏗️ Architecture

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│   Frontend   │────▶│  Flask API    │────▶│  RAG Pipeline│
│  (HTML/JS)   │◀────│  /api/chat    │◀────│              │
└──────────────┘     └───────────────┘     └──────┬───────┘
                                                   │
                                    ┌──────────────┼──────────────┐
                                    ▼              ▼              ▼
                              ┌──────────┐  ┌───────────┐  ┌──────────┐
                              │ Chunking │  │   FAISS   │  │ OpenRouter│
                              │          │  │  Index    │  │   LLM    │
                              └──────────┘  └───────────┘  └──────────┘
```

### Flow:
1. **User sends a question** via the chat interface
2. **Query is embedded** using `sentence-transformers/all-MiniLM-L6-v2`
3. **FAISS retrieves top-5** most similar document chunks via cosine similarity
4. **Retrieved chunks + query** are sent to the LLM with a strict grounding prompt
5. **Generated answer + source chunks** are returned and displayed in the UI

---

## 📐 Technical Decisions

### Embedding Model: `all-MiniLM-L6-v2`

**Why this model?**
- **Free & local**: Runs entirely on CPU, no API key needed for embeddings
- **Fast**: ~80ms per embedding on CPU, suitable for real-time search
- **Quality**: Strong performance on semantic similarity tasks (SBERT benchmark)
- **Small**: Only 80MB, practical for deployment
- **384-dimensional** vectors provide good balance of quality and efficiency

### Vector Search: FAISS (Facebook AI Similarity Search)

**Why FAISS?**
- **Industry standard** for similarity search at scale
- **Fast**: Optimized C++ with Python bindings
- **Simple**: `IndexFlatIP` with L2-normalized vectors = cosine similarity
- **No external service** needed — runs locally
- Index is built in-memory on startup (fast for small document sets)

### LLM: Meta Llama 4 Maverick (via OpenRouter)

**Why this choice?**
- **Free tier** available on OpenRouter
- **High quality**: State-of-the-art open model for instruction following
- **Good grounding**: Follows system prompts well, reducing hallucination
- Configurable via environment variable; easy to swap models

### Document Chunking Strategy

- Documents are split into **~500-character chunks** with **~100-character overlap**
- Chunking is **paragraph-aware**: splits on paragraph boundaries to preserve semantic coherence
- Overlap ensures no information is lost at chunk boundaries
- Each chunk retains metadata (source filename, chunk ID) for transparency

---

## 🔒 Grounding & Hallucination Prevention

The system enforces grounding through multiple mechanisms:

1. **System prompt**: The LLM is explicitly instructed to answer **only** from provided context
2. **Context-only rule**: If the context doesn't contain relevant information, the LLM responds with "I don't have enough information"
3. **Low temperature** (0.3): Reduces creative/hallucinated outputs
4. **Transparency**: Retrieved chunks are displayed alongside every answer, allowing users to verify

---

## 📁 Project Structure

```
├── app.py                   # Flask web server
├── rag_pipeline.py          # Core RAG engine (chunk, embed, retrieve, generate)
├── documents/               # Source documents (text)
│   ├── company_overview.txt # INDECIMAL company overview & customer journey
│   ├── package_specs.txt    # Package tiers & material specifications
│   └── customer_policies.txt# Payment safety, quality system, guarantees
├── templates/
│   └── index.html           # Chat UI template
├── static/
│   ├── style.css            # Dark-mode glassmorphism design
│   └── script.js            # Frontend chat logic
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
└── README.md                # This file
```

---

##  Quality Analysis

### Test Questions & Expected Behavior

| # | Question | Expected Source | Status |
|---|----------|----------------|--------|
| 1 | What packages does INDECIMAL offer? | package_specs.txt | ✅ Grounded |
| 2 | What factors affect construction project delays? | customer_policies.txt | ✅ Grounded |
| 3 | How does the payment protection work? | customer_policies.txt | ✅ Grounded |
| 4 | What is included in the Essential package? | package_specs.txt | ✅ Grounded |
| 5 | How many quality checkpoints does INDECIMAL have? | customer_policies.txt | ✅ Grounded |
| 6 | What is the customer journey at INDECIMAL? | company_overview.txt | ✅ Grounded |
| 7 | Tell me about the maintenance program | customer_policies.txt | ✅ Grounded |
| 8 | What brands of cement are used? | package_specs.txt | ✅ Grounded |
| 9 | How long does a typical home take to build? | company_overview.txt | ✅ Grounded |
| 10 | What is the difference between Premier and Infinia? | package_specs.txt | ✅ Grounded |
| 11 | Does INDECIMAL help with home loans? | customer_policies.txt | ✅ Grounded |
| 12 | What happens if construction is delayed? | customer_policies.txt | ✅ Grounded |

### Observations
- **Retrieval quality**: FAISS with sentence-transformers consistently retrieves relevant chunks for domain-specific queries
- **Grounding**: The strict system prompt + low temperature effectively prevents hallucination
- **Transparency**: Displaying retrieved chunks allows easy verification of answer accuracy
- **Limitations**: Very short or ambiguous queries may retrieve less relevant chunks; multi-topic queries may miss some aspects

---

##  Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend | Flask 3.0 | Web server & API |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Semantic text encoding |
| Vector Search | FAISS (faiss-cpu) | Similarity search index |
| LLM | Meta Llama 4 Maverick (OpenRouter) | Grounded answer generation |
| Frontend | HTML + CSS + JavaScript | Custom chat interface |



