import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests
import json


class RAGPipeline:
    """Core RAG pipeline: document loading, chunking, embedding, retrieval, and generation."""

    def __init__(self, documents_dir="documents", model_name="all-MiniLM-L6-v2"):
        self.documents_dir = documents_dir
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.chunk_metadata = []
        self.index = None
        self.embeddings = None

    # ── Document Loading ──────────────────────────────────────────────────

    def load_documents(self):
        """Load all .txt files from the documents directory."""
        documents = []
        for filename in sorted(os.listdir(self.documents_dir)):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.documents_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append({"filename": filename, "content": content})
        return documents

    # ── Chunking ──────────────────────────────────────────────────────────

    def chunk_document(self, text, filename, chunk_size=500, overlap=100):
        """Split a document into overlapping chunks by character count with sentence awareness."""
        sentences = []
        for para in text.split("\n"):
            para = para.strip()
            if para:
                sentences.append(para)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "source": filename,
                    "chunk_id": len(chunks)
                })
                # Keep overlap: find the last portion of current_chunk
                words = current_chunk.split()
                overlap_words = words[-min(len(words), overlap // 5):]
                current_chunk = " ".join(overlap_words) + "\n" + sentence
            else:
                current_chunk += "\n" + sentence if current_chunk else sentence

        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "source": filename,
                "chunk_id": len(chunks)
            })

        return chunks

    def chunk_all_documents(self):
        """Load and chunk all documents."""
        documents = self.load_documents()
        self.chunks = []
        for doc in documents:
            doc_chunks = self.chunk_document(doc["content"], doc["filename"])
            self.chunks.extend(doc_chunks)
        print(f"[OK] Created {len(self.chunks)} chunks from {len(documents)} documents")
        return self.chunks

    # ── Embedding & Indexing ──────────────────────────────────────────────

    def build_index(self):
        """Generate embeddings for all chunks and build a FAISS index."""
        if not self.chunks:
            self.chunk_all_documents()

        texts = [chunk["text"] for chunk in self.chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)

        # Build FAISS index (Inner Product on normalized vectors = cosine similarity)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

        print(f"[OK] FAISS index built with {self.index.ntotal} vectors (dim={dimension})")

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve(self, query, top_k=5):
        """Retrieve top-k most relevant chunks for a given query."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx]["text"],
                    "source": self.chunks[idx]["source"],
                    "chunk_id": self.chunks[idx]["chunk_id"],
                    "score": float(scores[0][i])
                })
        return results

    # ── LLM Generation ────────────────────────────────────────────────────

    def generate_answer(self, query, retrieved_chunks, api_key=None):
        """Generate an answer using OpenRouter LLM grounded on retrieved chunks."""
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY", "")

        if not api_key:
            return self._fallback_answer(query, retrieved_chunks)

        # Build context from retrieved chunks
        context = "\n\n---\n\n".join(
            [f"[Source: {c['source']} | Chunk {c['chunk_id']}]\n{c['text']}" for c in retrieved_chunks]
        )

        system_prompt = """You are the INDECIMAL AI Assistant, helping customers with questions about INDECIMAL's construction marketplace.

STRICT RULES:
1. Answer ONLY based on the provided context below. Do NOT use any external or general knowledge.
2. If the context does not contain enough information to answer the question, say: "I don't have enough information in the available documents to answer that question."
3. Be clear, concise, and helpful.
4. When mentioning specific details (prices, percentages, timelines), cite them accurately from the context.
5. Do NOT make up or hallucinate any information."""

        user_prompt = f"""Context (retrieved from internal documents):

{context}

---

Customer Question: {query}

Provide a helpful, grounded answer based strictly on the context above."""

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:5000",
                    "X-Title": "INDECIMAL RAG Chatbot"
                },
                json={
                    "model": "meta-llama/llama-4-maverick:free",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1024
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            return answer
        except Exception as e:
            print(f"[WARN] OpenRouter API error: {e}")
            return self._fallback_answer(query, retrieved_chunks)

    def _fallback_answer(self, query, retrieved_chunks):
        """Fallback: return top retrieved chunks as the answer when no API key is set."""
        answer = "⚠️ **No OpenRouter API key configured.** Showing the most relevant document excerpts:\n\n"
        for i, chunk in enumerate(retrieved_chunks[:3], 1):
            answer += f"**Excerpt {i}** (from `{chunk['source']}`, similarity: {chunk['score']:.2f}):\n"
            answer += f"> {chunk['text'][:500]}\n\n"
        return answer

    # ── Full Pipeline ─────────────────────────────────────────────────────

    def query(self, user_query, top_k=5, api_key=None):
        """Full RAG pipeline: retrieve + generate."""
        retrieved = self.retrieve(user_query, top_k=top_k)
        answer = self.generate_answer(user_query, retrieved, api_key=api_key)
        return {
            "query": user_query,
            "answer": answer,
            "sources": retrieved
        }
