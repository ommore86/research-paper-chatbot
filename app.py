import os, uuid
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import PyPDF2

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain.chains import RetrievalQA

# ── CONFIG ───────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

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

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    store = FAISS.from_texts(chunks, embeddings)

    # LLM
    llm = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.3,
            max_new_tokens=512,  # More tokens for detailed answers
        )
    )

    retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 5})  # Optional boost
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ── ROUTES ───────────────────────────────────────────────────────────────────────
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

        except Exception as e:
            answer = f"❌ Error during processing: {e}"

    return render_template("index.html", results=answer)

# ── ENTRY ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)