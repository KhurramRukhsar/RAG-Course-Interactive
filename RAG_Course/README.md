# 🎓 RAG Interactive Course: Pakistani News Edition

Welcome to the **RAG Interactive Course**. This project is a comprehensive implementation of the 5-lecture RAG curriculum, applied to real-world Pakistani news data.

It is designed as a "living laboratory" where you can interactively experiment with the core components of a Retrieval-Augmented Generation (RAG) system.

## 🚀 Interactive Labs

The application is structured into four educational labs:

1.  **📦 Data & Chunking**: Compare document-level vs. fixed-size character chunking.
2.  **🔍 Retrieval Lab**: Side-by-side comparison of **BM25**, **Vector**, and **Hybrid (RRF)** retrieval.
3.  **🎯 Advanced Retrieval**: Hands-on tuning of BM25 `k1` and `b` parameters, plus **LLM-based Reranking**.
4.  **🤖 Generation Lab**: Control LLM sampling parameters (**Temperature**, **Top-P**) to see their effect on grounding and hallucinations.

## 🌟 Detailed Features

*   **Custom Chunking**: Experiment with how splitting text affects retrieval precision (Lecture 2/5).
*   **RRF Hybrid Search**: Combine lexical and semantic search to get the best of both worlds (Lecture 3).
*   **LLM-as-a-Judge**: Use LLM-based reranking to achieve higher retrieval relevance (Lecture 3).
*   **Sampling Control**: Real-time control over non-deterministic LLM behaviors (Lecture 4).

## ⚙️ Technology Stack

*   **LLM**: Google Gemini 1.5 Flash
*   **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
*   **Vector Store**: FAISS
*   **Keyword Search**: Rank-BM25
*   **UI Framework**: Streamlit

## 🏃 How to Run

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/KhurramRukhsar/RAG-Course-Interactive.git
    cd RAG-Course-Interactive
    ```

2.  **Navigate to the project directory**:
    ```bash
    cd RAG_Course
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the Interactive Lab**:
    ```bash
    streamlit run main.py
    ```

## 📂 Project Structure

*   `main.py`: The interactive Streamlit dashboard.
*   `src/data_processor.py`: Multi-strategy chunking logic.
*   `src/retrievers.py`: Implementations of BM25, Vector, and RRF Hybrid search.
*   `src/reranker.py`: LLM-as-a-judge reranking logic.
*   `src/generator.py`: Tunable LLM generation wrapper.
*   `src/config.py`: Central hub for all architectural parameters.

## ✅ Verification Results

The system has been verified using `verify_pipeline.py`. Key observations include:
*   **Hybrid Search**: RRF effectively surface documents that match both the "intent" (Vector) and "exact terms" (BM25).
*   **Tuning Effects**: Low temperature (0.1) significantly reduces hallucination and ensures responses are strictly grounded in the Pakistani news context.
