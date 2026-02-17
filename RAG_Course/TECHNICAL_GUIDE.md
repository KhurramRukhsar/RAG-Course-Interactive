# RAG Course: Technical Deep Dive

This document explains the architectural decisions and mathematical foundations of the RAG techniques implemented in this project.

## 1. Retrieval Foundations (Lecture 2 & 3)

### BM25 (Best Matching 25)
Standardized keyword retrieval algorithm.
*   **k1**: Controls term frequency saturation.
*   **b**: Controls document length normalization.

### Vector Search
Semantic retrieval using dense embeddings.
*   **Model**: `all-MiniLM-L6-v2` (384 dimensions).
*   **Distance**: Euclidean (L2) distance via FAISS.

### Hybrid Search (RRF)
Reciprocal Rank Fusion (RRF) combines the rankings of BM25 and Vector search.
*   **Formula**: $Score = \sum_{r \in R} \frac{1}{k + rank(r)}$
*   **k**: Default 60, prevents a single top result from dominating.

## 2. Advanced Retrieval (Lecture 3)

### LLM Reranking
Uses the "LLM-as-a-judge" pattern. A secondary LLM call evaluates the top-N retrieved documents for specific relevance to the query, providing a more nuanced score than vector distance.

## 3. Large Language Models (Lecture 4)

### Sampling Parameters
*   **Temperature**: Affects the distribution of token probabilities. Lower = deterministic; Higher = creative.
*   **Top-P (Nucleus Sampling)**: Selects from tokens whose cumulative probability is $\le P$.
*   **Grounding**: The system prompt is engineered to enforce strict adherence to the retrieved context.

## 5. Verification & Performance

The pipeline logic was validated through automated and manual trace analysis:

### Retrieval Performance
*   **Keyword (BM25)**: Reliable for entity-heavy queries (names, dates, specific figures).
*   **Semantic (Vector)**: Handles thematic queries where exact wording varies (e.g., "economic crisis" vs "financial instability").
*   **Hybrid (RRF)**: Provides the most robust results by assigning high reciprocal scores to documents that perform well in both categories.

### Generation Quality
*   **Grounding**: Verified that the system prompt effectively restricts the LLM to the provided context.
*   **Hallucination Check**: Using a high temperature (e.g., 1.2) can lead to more creative but potentially untethered answers, validating the need for sampling control in production RAG systems.
