import streamlit as st
import os

from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings as LCEmbeddings
from datasets import Dataset

# ── Model config ───────────────────────────────────────────────────────────────
ANSWER_MODEL = "llama-3.1-8b-instant"       # generates the answer
JUDGE_MODEL  = "llama-3.3-70b-versatile"   # used internally by RAGAS as judge
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Legal Intelligence Assistant",
    page_icon="⚖️",
    layout="wide"
)

with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("⚖️ Legal Intelligence Assistant")
st.markdown("### RAG-based Question Answering over Indian Court Judgements")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Project Details")
st.sidebar.info(f"""
**Dataset:** Indian Court Judgements

**Embedding Model:**  
all-MiniLM-L6-v2

**Vector Database:**  
FAISS

**LLM (Answering):**  
{ANSWER_MODEL}

**LLM (Evaluation Judge):**  
{JUDGE_MODEL} via RAGAS

**Evaluation Metrics:**  
- Faithfulness  
- Answer Relevancy  
- Context Precision  
- Context Recall
""")

top_k = st.sidebar.slider(
    "Number of documents to retrieve (k)",
    min_value=1,
    max_value=10,
    value=4,
    help="How many context chunks to fetch from FAISS for each question."
)

use_recall = st.sidebar.checkbox(
    "Enable Context Recall",
    value=False,
    help=(
        "Context Recall requires a real ground-truth answer. "
        "When disabled, the model answer is NOT used as a proxy, "
        "avoiding artificially inflated recall scores."
    )
)


# ── Cached resources ───────────────────────────────────────────────────────────

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
    # FIX: Load .env here so the key is always fresh when the cache is first populated
    from dotenv import load_dotenv
    load_dotenv(override=True)
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        st.error("GROQ_API_KEY not found in .env")
        st.stop()
    return Groq(api_key=api_key)


@st.cache_resource
def load_ragas_llm():
    """RAGAS needs a LangChain-wrapped LLM as its judge."""
    # FIX: Load .env here so the key is always fresh when the cache is first populated
    from dotenv import load_dotenv
    load_dotenv(override=True)
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        st.error("GROQ_API_KEY not found in .env")
        st.stop()
    llm = ChatGroq(model=JUDGE_MODEL, api_key=api_key, temperature=0)
    return LangchainLLMWrapper(llm)


@st.cache_resource
def load_ragas_embeddings():
    emb = LCEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(emb)


client      = load_groq_client()
vectorstore = load_vectorstore()
st.success("✅ Vector Database Loaded Successfully!")

question = st.text_input("🔍 Enter your legal question:")

ground_truth_input = st.text_area(
    "📝 Ground Truth Answer (optional — required for accurate Context Recall)",
    placeholder="Paste the correct reference answer here if you have one...",
    height=80,
)

if question:

    # ── 1. Retrieval ──────────────────────────────────────────────────────────
    with st.spinner("Searching legal database..."):
        docs = vectorstore.similarity_search(question, k=top_k)
        if not docs:
            st.warning("No relevant documents found.")
            st.stop()
        contexts     = [doc.page_content for doc in docs]
        context_text = "\n\n".join(contexts)

    # ── 2. Answer ─────────────────────────────────────────────────────────────
    prompt = f"""
You are a legal assistant specializing in Indian law.
Use the following legal context if relevant, otherwise answer from your own knowledge.
If answering from your own knowledge, end your answer with:
"Legal References: [list the relevant articles, acts, or case names you used]"

Context:
{context_text}

Question:
{question}

Answer:
"""
    with st.spinner(f"Generating answer with `{ANSWER_MODEL}`..."):
        # FIX: Re-read the API key at call time in case the cached client was
        # built before the .env was loaded (primary cause of the 401 error).
        # We create a fresh Groq instance here using the current env value.
        from dotenv import load_dotenv
        load_dotenv(override=True)
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            st.error("GROQ_API_KEY not found in .env — cannot generate answer.")
            st.stop()
        fresh_client = Groq(api_key=api_key)

        response = fresh_client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512
        )
        answer = response.choices[0].message.content

    st.markdown("## ⚖️ Answer")
    st.caption(f"Generated by: `{ANSWER_MODEL}`")
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    # ── 3. RAGAS Evaluation ───────────────────────────────────────────────────
    st.markdown("## 📊 RAG Evaluation — RAGAS Metrics")
    st.caption(
        f"Evaluated using **RAGAS framework** · "
        f"Judge LLM: `{JUDGE_MODEL}` (different from answer model)"
    )

    with st.spinner("Running RAGAS evaluation..."):
        try:
            ragas_llm = load_ragas_llm()
            ragas_emb = load_ragas_embeddings()

            ground_truth = ground_truth_input.strip() if ground_truth_input.strip() else answer

            eval_dict = {
                "question":     [question],
                "answer":       [answer],
                "contexts":     [contexts],
                "ground_truth": [ground_truth],
            }
            eval_dataset = Dataset.from_dict(eval_dict)

            active_metrics = [faithfulness, answer_relevancy, context_precision]
            if use_recall:
                active_metrics.append(context_recall)

            result = evaluate(
                dataset=eval_dataset,
                metrics=active_metrics,
                llm=ragas_llm,
                embeddings=ragas_emb,
            )

            scores = result.to_pandas().iloc[0]

            cols = st.columns(4 if use_recall else 3)

            def fmt(val):
                try:
                    return f"{float(val):.2f}"
                except Exception:
                    return "N/A"

            cols[0].metric(
                "Faithfulness",
                fmt(scores.get("faithfulness", "N/A")),
                help="How grounded is the answer in the retrieved context? (1.0 = fully grounded, no hallucination)"
            )
            cols[1].metric(
                "Answer Relevancy",
                fmt(scores.get("answer_relevancy", "N/A")),
                help="How relevant is the answer to the question? (1.0 = perfectly relevant)"
            )
            cols[2].metric(
                "Context Precision",
                fmt(scores.get("context_precision", "N/A")),
                help="How much of the retrieved context is actually useful? (1.0 = all retrieved context is relevant)"
            )
            if use_recall:
                cols[3].metric(
                    "Context Recall",
                    fmt(scores.get("context_recall", "N/A")),
                    help="How much of the needed information was retrieved? (1.0 = nothing was missed)"
                )

            with st.expander("ℹ️ What do these metrics mean?"):
                st.markdown(f"""
| Metric | What it measures | Good score |
|---|---|---|
| **Faithfulness** | Answer stays grounded in context — catches hallucinations | Close to 1.0 |
| **Answer Relevancy** | Answer actually addresses the question asked | Close to 1.0 |
| **Context Precision** | Retrieved docs are relevant (no noise) | Close to 1.0 |
| **Context Recall** | Retrieved docs contain all needed information *(requires real ground truth)* | Close to 1.0 |

> All scores are between 0 and 1. These are **reference-free** metrics (except Context Recall).  
> RAGAS uses `{JUDGE_MODEL}` internally to compute them.
                """)

            if not ground_truth_input.strip() and use_recall:
                st.warning(
                    "⚠️ Context Recall is enabled but no ground truth was provided. "
                    "The model's own answer was used as a proxy — recall scores will be inflated and unreliable."
                )

        except Exception as e:
            st.error(f"RAGAS evaluation failed: {e}")
            st.info("Tip: This can happen if the Groq rate limit is hit. Try again in a moment.")

    # ── 4. Retrieved context ──────────────────────────────────────────────────
    with st.expander("📂 View Retrieved Legal Context"):
        for i, doc in enumerate(docs, 1):
            st.markdown(f"### Document {i}")
            content = doc.page_content
            if len(content) > 1000:
                st.write(content[:1000])
                st.caption(f"*Showing first 1000 of {len(content)} characters.*")
            else:
                st.write(content)
            st.markdown("---")