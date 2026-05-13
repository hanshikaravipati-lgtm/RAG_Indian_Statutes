# RAG-based Question Answering over Indian Legal Data

**Course:** CS5202 - Generative AI & LLMs  
**Team:** 12  
**Deployed App:** [https://ragindianstatutes-qbeqfy4786mr8b3bh5ght2.streamlit.app/](https://ragindianstatutes-qbeqfy4786mr8b3bh5ght2.streamlit.app/)  
**GitHub:** [https://github.com/hanshikaravipati-lgtm/RAG_Indian_Statutes](https://github.com/hanshikaravipati-lgtm/RAG_Indian_Statutes)

---

## Team Members

| Roll Number | Name |
|-------------|------|
| SE23UCSE129 | Nossam Nithin Reddy |
| SE23UCSE130 | Amulya Oruganti |
| SE23UCSE152 | Hanshika Ravipati |
| SE23UCSE156 | Sanikommu Vaishnavi Reddy |
| SE23UARI068 | Venkata Subba Rao |

---

## 1. Project Overview

This project focuses on building a **Retrieval-Augmented Generation (RAG)** system for answering questions from Indian legal datasets. The domain of interest is **Legal AI**, specifically question-answering over court judgments and statutory documents.

Legal documents are large, complex, and difficult to navigate manually. By using modern NLP techniques, we enable efficient retrieval and summarization of relevant legal information through a conversational interface.

---

## 2. Objectives

- Build a RAG pipeline over Indian Court Judgements dataset
- Enable natural language question answering over legal text
- Use pre-trained LLMs for accurate answer generation
- Deploy the system as an accessible web application

---

## 3. Dataset

- **Name:** Indian Court Judgements and Summaries
- **Format:** CSV (`court_data.csv`)
- **Columns:**
  - `Judgment` — Full text of the court's decision
  - `Summary` — Concise summary of the judgment
- **Domain:** Indian legal cases covering criminal, civil, and constitutional matters

---

## 4. System Architecture

```
User Question
      ↓
Convert to Vector Embedding (all-MiniLM-L6-v2)
      ↓
FAISS Similarity Search (Top 4 Chunks)
      ↓
Build Prompt (Context + Question)
      ↓
LLaMA 3.3 70B via Groq API
      ↓
Display Answer + Sources (Streamlit UI)
```

---

## 5. Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Database | FAISS (Facebook AI Similarity Search) |
| LLM | Meta LLaMA 3.3 70B via Groq API |
| Frontend | Streamlit |
| Language | Python 3.11 |

---

## 6. Pre-trained Models Used

### 6.1 all-MiniLM-L6-v2 (Embeddings)
- Made by **Microsoft**, hosted on HuggingFace
- Lightweight sentence embedding model with 6 transformer layers
- Produces 384-dimensional vectors capturing semantic meaning
- Used to convert legal text chunks and user queries into vectors

### 6.2 LLaMA 3.3 70B (Answer Generation)
- Made by **Meta (Facebook)**
- Pre-trained Large Language Model with 70 billion parameters
- Accessed via **Groq API** for free, fast cloud inference
- Generates accurate, context-grounded legal answers

---

## 7. Implementation

### 7.1 Data Ingestion (`ingest.py`)
- Reads `court_data.csv` using pandas
- Combines `Judgment` and `Summary` columns into one text per row
- Creates LangChain `Document` objects with row metadata
- Splits documents into chunks of **1000 characters** with **200 character overlap**
- Generates embeddings using `all-MiniLM-L6-v2`
- Saves FAISS index locally (`embeddings/index.faiss`, `embeddings/index.pkl`)

### 7.2 Application (`app.py`)
- Loads FAISS index and embedding model at startup
- Accepts user question via Streamlit text input
- Retrieves top-4 most similar chunks using FAISS similarity search
- Builds a structured prompt with system role and retrieved context
- Sends prompt to LLaMA 3.3 70B via Groq API
- Displays generated answer and source document references

### 7.3 Prompt Engineering
The system uses a two-part prompt:
- **System prompt:** Assigns the LLM the role of an Indian legal assistant
- **User prompt:** Combines retrieved context chunks with the user question

```
System: You are a legal assistant specializing in Indian law.
        Use the provided court judgement excerpts to answer accurately.

User:   Context: [top 4 retrieved legal chunks]
        Question: [user question]
        Please provide a clear and concise answer.
```

---

## 8. RAG vs Pure LLM

| Aspect | Pure LLM | RAG (Our Approach) |
|--------|----------|-------------------|
| Data source | Training data only | Retrieved real documents |
| Hallucination | High risk | Significantly reduced |
| Domain accuracy | General | Legal domain specific |
| Updatable | No | Yes (update vector store) |

---

## 9. Team Contributions

| Member | Contribution |
|--------|-------------|
| **Hanshika Ravipati** | Full system integration, LLM pipeline, app.py, ingest.py, deployment |
| **Nossam Nithin Reddy** | Data collection, preprocessing, court_data.csv preparation |
| **Amulya Oruganti** | Embeddings research, FAISS index creation and storage |
| **Vaishnavi Reddy** | LLM research, prompt engineering, RAG vs LLM analysis |
| **Venkata Subba Rao** | UI/UX design, style.css, project documentation |

---

## 10. Results

The system successfully:
- Retrieves relevant legal document chunks for any legal question
- Generates accurate, context-grounded answers using LLaMA 3.3 70B
- Displays source references for transparency
- Runs as a live web application accessible via public URL

---

## 11. How to Run Locally

```bash
# Clone the repository
git clone https://github.com/hanshikaravipati-lgtm/RAG_Indian_Statutes.git
cd RAG_Indian_Statutes

# Create virtual environment
py -V:3.11-64 -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Add Groq API key
# Create .streamlit/secrets.toml and add:
# GROQ_API_KEY = "your_key_here"

# Run the app
streamlit run app.py
```

---

## 12. Deployed Application

**Live URL:** [https://ragindianstatutes-qbeqfy4786mr8b3bh5ght2.streamlit.app/](https://ragindianstatutes-qbeqfy4786mr8b3bh5ght2.streamlit.app/)
