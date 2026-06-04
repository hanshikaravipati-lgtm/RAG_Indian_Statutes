import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

print("Loading dataset...")

csv_path = "data/court_data.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError("court_data.csv not found inside data folder")

df = pd.read_csv(csv_path)

required_columns = ["ID", "Judgment", "Summary"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column in CSV: {col}")

documents = []

for _, row in df.iterrows():
    text = f"""
Judgment:
{str(row["Judgment"])[:2000]}

Summary:
{str(row["Summary"])[:1000]}
"""

    documents.append(
        Document(
            page_content=text,
            metadata={
                "id": str(row["ID"])
            }
        )
    )

print(f"Loaded {len(documents)} documents")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    documents,
    embeddings
)

vectorstore.save_local("embeddings")

print("Embeddings created successfully!")