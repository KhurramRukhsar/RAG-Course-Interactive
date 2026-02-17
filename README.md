# � RAG Interactive Course: Pakistani News Edition 🇵🇰🤖

Welcome to the **RAG Interactive Course**. This repository contains a "living laboratory" implementation of a Retrieval-Augmented Generation (RAG) system, applied to real-world Pakistani news data.

It is designed as an educational resource to explore how different parameters, chunking strategies, and retrieval techniques affect the performance and accuracy of AI systems.

## 🚀 The Interactive Lab

The heart of this project is a **Streamlit Dashboard** divided into four educational labs:

1.  **📦 Data & Chunking**: Compare document-level vs. fixed-size character chunking and observe the impact on retrieval granularity (Lecture 2/5).
2.  **🔍 Retrieval Lab**: See side-by-side results for **BM25 (Keyword)**, **Vector (Semantic)**, and **Hybrid (RRF)** search (Lecture 2/3).
3.  **🎯 Advanced Retrieval**: Tune BM25 saturation (`k1`) and normalization (`b`) parameters. Toggle **LLM-based Reranking** for high-precision results (Lecture 3).
4.  **🤖 Generation Lab**: Experiment with LLM sampling parameters (**Temperature**, **Top-P**) to see how they affect grounding and potential hallucinations (Lecture 4).

## ✅ Verification Results

This system has been verified using a standalone pipeline test. Observations include:
*   **Hybrid RRF**: Successfully surface documents that match both the "intent" (Vector) and "exact terms" (BM25).
*   **Grounding Control**: Low temperature (0.1) significantly reduces hallucinations, ensuring responses are strictly tethered to the news context.
*   **Reranking**: Higher precision results were observed when using the "LLM-as-a-judge" pattern for final document scoring.

## ⚙️ Technology Stack

*   **LLM**: Google Gemini 1.5 Flash
*   **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
*   **Vector Store**: FAISS
*   **Ranking**: Rank-BM25 & Reciprocal Rank Fusion (RRF)
*   **Dashboard**: Streamlit

## 🏃 Quick Start

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/KhurramRukhsar/RAG-Course-Interactive.git
    cd RAG-Course-Interactive
    ```

2.  **Navigate to the project and install requirements**:
    ```bash
    cd RAG_Course
    pip install -r requirements.txt
    ```

2.  **Configure API Key**:
    Add your `GOOGLE_API_KEY` to a `.env` file in the root.

3.  **Launch the Lab**:
    ```bash
    streamlit run main.py
    ```

## 📂 Repository Structure

The interactive course logic is contained within the `RAG_Course` directory:
*   `RAG_Course/main.py`: Interactive Dashboard.
*   `RAG_Course/src/`: Core logic for retrievers, rerankers, and generators.
*   `RAG_Course/TECHNICAL_GUIDE.md`: Deep dive into the math and architecture.
