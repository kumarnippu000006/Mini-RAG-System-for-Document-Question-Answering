import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

load_dotenv()

app = Flask(__name__)

# ── Initialize RAG Pipeline on startup ────────────────────────────────────
print("[*] Initializing RAG Pipeline...")
rag = RAGPipeline(documents_dir="documents")
rag.build_index()
print("[OK] RAG Pipeline ready!\n")


@app.route("/")
def index():
    """Serve the chatbot frontend."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat requests: retrieve + generate."""
    data = request.get_json()
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    result = rag.query(user_query, top_k=5, api_key=api_key)

    return jsonify({
        "answer": result["answer"],
        "sources": [
            {
                "text": s["text"],
                "source": s["source"],
                "score": round(s["score"], 4)
            }
            for s in result["sources"]
        ]
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "chunks_indexed": len(rag.chunks),
        "api_key_configured": bool(os.environ.get("OPENROUTER_API_KEY"))
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
