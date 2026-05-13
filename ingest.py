import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ── Load CSV data ─────────────────────────────────────────────────────────────
documents = []
df = pd.read_csv("data/court_data.csv")

for i, row in df.iterrows():
    text = f"Judgment: {row['Judgment']}\nSummary: {row['Summary']}"
    documents.append(
        Document(
            page_content=text,
            metadata={"row": i}
        )
    )

# ── Split into chunks ─────────────────────────────────────────────────────────
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# ── Create embeddings and save FAISS index ────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local("embeddings")

print("Embeddings created successfully")
