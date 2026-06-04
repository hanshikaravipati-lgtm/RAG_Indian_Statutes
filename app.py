import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


st.set_page_config(
    page_title="Legal Intelligence Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Load CSS
with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("⚖️ Legal Intelligence Assistant")
st.markdown("### RAG-based Question Answering over Indian Court Judgements")

st.sidebar.header("Project Details")
st.sidebar.info("""
Dataset: Indian Court Judgements

Embedding Model:
all-MiniLM-L6-v2

Vector Database:
FAISS

LLM:
Llama 3.3 70B (Groq)

Evaluation:
LLM-as-a-Judge
""")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()

    if not os.path.exists("embeddings"):
        st.error("Embeddings folder not found. Please create FAISS embeddings first.")
        st.stop()

    return FAISS.load_local(
        "embeddings",
        embeddings,
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def load_groq_client():
    load_dotenv(override=True)
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        st.error("GROQ_API_KEY not found")
        st.stop()
    return Groq(api_key=api_key)

client = load_groq_client()

vectorstore = load_vectorstore()
st.success("✅ Vector Database Loaded Successfully!")

question = st.text_input("🔍 Enter your legal question:")

if question:
    with st.spinner("Searching legal database..."):
        docs = vectorstore.similarity_search(question, k=4)

        if not docs:
            st.warning("No relevant documents found.")
            st.stop()

        context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a legal assistant specializing in Indian law.
Use the following legal context if relevant, otherwise answer from your own knowledge.
If answering from your own knowledge, end your answer with:
"Legal References: [list the relevant articles, acts, or case names you used]"

Context:
{context}

Question:
{question}

Answer:
"""

    with st.spinner("Generating answer..."):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512
        )

        answer = response.choices[0].message.content

    st.markdown("## ⚖️ Answer")
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    st.markdown("## 📊 Evaluation Metrics")

    confidence = min(len(docs) * 25, 100)
    st.metric("Retrieval Confidence", f"{confidence}%")

    judge_prompt = f"""
You are an AI evaluator.

Question:
{question}

Answer:
{answer}

Evaluate:

1. Relevance (1-10)
2. Correctness (1-10)
3. Completeness (1-10)

Give a brief justification.
"""

    judge_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": judge_prompt}
        ],
        temperature=0,
        max_tokens=300
    )

    st.markdown("## 🧑‍⚖️ LLM-as-a-Judge Evaluation")
    st.info(judge_response.choices[0].message.content)

    with st.expander("📂 View Retrieved Legal Context"):
        for i, doc in enumerate(docs, 1):
            st.markdown(f"### Document {i}")
            st.write(doc.page_content[:1000])
            st.markdown("---")