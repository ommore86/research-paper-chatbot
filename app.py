import os, uuid
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import PyPDF2
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain.chains import RetrievalQA

# TOP OF app.py — ADD THIS
from functools import lru_cache
import gc

@lru_cache(maxsize=1)
def get_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def get_llm():
    return ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.3,
            max_new_tokens=512,
        )
    )


load_dotenv()

# ── CONFIG ───────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ── HELPERS ──────────────────────────────────────────────────────────────────────
def extract_text_from_pdf(path: str) -> str:
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "".join(p.extract_text() or "" for p in reader.pages)

def build_qa_for_pdf(path: str) -> RetrievalQA:
    text = extract_text_from_pdf(path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = get_embedder()  # ✅ use shared singleton
    store = FAISS.from_texts(chunks, embeddings)

    llm = get_llm()  # ✅ use shared singleton
    retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ── ROUTES ───────────────────────────────────────────────────────────────────────
@app.route("/healthz")
def healthz():
    return "ok", 200

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename.lower().endswith(".pdf"):
            return render_template("index.html", results="❌ Please upload a valid PDF file.")

        # Save file
        path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            f"{uuid.uuid4()}_{secure_filename(file.filename)}",
        )
        file.save(path)

        # Build chain & get question
        question = request.form.get("question", "").strip()
        if not question:
            question = "Summarise the paper in a paragraph."

        try:
            qa_chain = build_qa_for_pdf(path)

            # Optional: Debug retrieved chunks
            retrieved_docs = qa_chain.retriever.get_relevant_documents(question)
            print("\n=== Retrieved Chunks ===")
            for i, doc in enumerate(retrieved_docs):
                print(f"\nChunk {i+1}:\n{doc.page_content[:300]}")

            # Proper invoke method
            response = qa_chain.invoke({"query": question})
            answer = response.get("result", response) if isinstance(response, dict) else response

            gc.collect()  
        except Exception as e:
            answer = f"❌ Error during processing: {e}"

    return render_template("index.html", results=answer)

# ── ENTRY ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)