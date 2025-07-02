import os, uuid, gc
from functools import lru_cache

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import PyPDF2

# ── LangChain & HF ──────────────────────────────────────────────────────────────
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain.chains import RetrievalQA

# ── ENV & CONFIG ────────────────────────────────────────────────────────────────
load_dotenv()
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# Simple in‑memory store: file_id ➜ RetrievalQA
qa_chains: dict[str, RetrievalQA] = {}

# ── Singleton helpers ───────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@lru_cache(maxsize=1)
def get_llm():
    return ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.3,
            max_new_tokens=512,
        )
    )

# ── Utility to read PDF ─────────────────────────────────────────────────────────
def extract_text_from_pdf(path: str) -> str:
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "".join(p.extract_text() or "" for p in reader.pages)

def build_qa_for_pdf(path: str) -> RetrievalQA:
    text = extract_text_from_pdf(path)
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)\
             .split_text(text)

    store = FAISS.from_texts(chunks, get_embedder())
    retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    return RetrievalQA.from_chain_type(llm=get_llm(), retriever=retriever)

# ── ROUTES ───────────────────────────────────────────────────────────────────────
@app.route("/healthz")
def healthz():
    return "ok", 200

# Landing page (serves index.html template)
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# ---------- Upload PDF ----------------------------------------------------------
@app.post("/upload")
def upload_file():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "❌ Please upload a valid PDF."}), 400

    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    try:
        qa_chains[filename] = build_qa_for_pdf(path)
        return jsonify({"file_id": filename}), 200
    except Exception as e:
        return jsonify({"error": f"❌ Error building chain: {e}"}), 500

# ---------- Ask a question ------------------------------------------------------
@app.post("/ask")
def ask_question():
    data = request.get_json(force=True)
    file_id  = data.get("file_id")
    question = (data.get("question") or "").strip()

    if not file_id or not question:
        return jsonify({"error": "❌ Missing file_id or question."}), 400

    qa_chain = qa_chains.get(file_id)
    if not qa_chain:
        return jsonify({"error": "❌ File not found or expired."}), 404

    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"] if isinstance(result, dict) else result
        gc.collect()
        return jsonify({"answer": answer}), 200
    except Exception as e:
        return jsonify({"error": f"❌ Error: {e}"}), 500

# ── ENTRY ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Remove debug=True in production
    app.run(debug=True)
