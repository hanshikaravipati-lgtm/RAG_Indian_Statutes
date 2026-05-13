import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="RAG QA over Indian Legal Dataset",
    page_icon="⚖️",
    layout="centered"
)

with open("style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1>⚖️ RAG-based QA over Indian Legal Dataset</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Ask questions from Indian Court Judgements and Summaries dataset.</p>",
    unsafe_allow_html=True
)

# ── Load embeddings ──────────────────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── Load FAISS vectorstore ───────────────────────────────────────────────────
@st.cache_resource
def load_vectorstore(_embeddings):
    return FAISS.load_local(
        "embeddings",
        _embeddings,
        allow_dangerous_deserialization=True
    )

# ── Load OpenAI client ───────────────────────────────────────────────────────
@st.cache_resource
def load_llm():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

embeddings   = load_embeddings()
vectorstore  = load_vectorstore(embeddings)
retriever    = vectorstore.as_retriever(search_kwargs={"k": 4})
client       = load_llm()

st.success("✅ Vector database & LLM loaded successfully!")

# ── Query interface ───────────────────────────────────────────────────────────
question = st.text_input("Enter your legal question:")

if question:
    with st.spinner("Retrieving relevant documents and generating answer…"):

        # Step 1 – Retrieve top-k relevant chunks
        docs = retriever.invoke(question)

        # Step 2 – Build context from retrieved chunks
        context = "\n\n".join([doc.page_content for doc in docs[:3]])

        # Step 3 – Generate answer using OpenAI GPT-3.5
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a legal assistant specializing in Indian law. "
                        "Use the provided court judgement excerpts to answer the question accurately."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Context from Indian Court Judgements:\n{context}\n\n"
                        f"Question: {question}\n\n"
                        f"Please provide a clear and concise answer based on the context above."
                    )
                }
            ],
            max_tokens=512,
            temperature=0.3,
        )
        answer_text = response.choices[0].message.content

    # ── Display answer ────────────────────────────────────────────────────────
    st.markdown("## Answer")
    st.markdown(
        f"<div class='answer-box'>{answer_text}</div>",
        unsafe_allow_html=True
    )

    # ── Display sources ───────────────────────────────────────────────────────
    st.markdown("## Sources")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Indian Court Judgements Dataset")
        row    = doc.metadata.get("row", "Unknown")
        st.markdown(
            f"<div class='source-box'>📄 Source {i}: {source} | Dataset Row: {row}</div>",
            unsafe_allow_html=True
        )