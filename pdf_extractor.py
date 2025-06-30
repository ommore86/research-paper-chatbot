import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import PyPDF2

# Set your Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load and split PDF
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text

pdf_text = extract_text_from_pdf("IEEEpaper.pdf")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(pdf_text)

from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.from_texts(chunks, embedding)

# ✅ Use HuggingFaceEndpoint correctly
hf_llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    temperature=0.5,
    max_new_tokens=512,
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
)

llm = ChatHuggingFace(llm=hf_llm)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
response = qa.run("Who is the author?")

# print("Total Chunks:", len(chunks))
# print("First Chunk:", chunks[0] if chunks else "No chunks extracted")