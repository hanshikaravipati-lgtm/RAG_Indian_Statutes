import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

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

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_vectorstore(_embeddings):
    return FAISS.load_local(
        "embeddings",
        _embeddings,
        allow_dangerous_deserialization=True
    )

embeddings = load_embeddings()
vectorstore = load_vectorstore(embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

st.success("Vector database loaded successfully!")

question = st.text_input("Enter your question:")

if question:
    docs = retriever.invoke(question)

    answer_text = docs[0].page_content

    st.markdown("## Answer")
    st.markdown(
        f"<div class='answer-box'>{answer_text}</div>",
        unsafe_allow_html=True
    )

    st.markdown("## Sources")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Indian Court Judgements Dataset")
        row = doc.metadata.get("row", "Unknown")

        st.markdown(
            f"<div class='source-box'>Source {i}: {source} | Dataset Row: {row}</div>",
            unsafe_allow_html=True
        )