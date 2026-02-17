# RAG Course: Technical Architecture & Mathematics

This guide provides a deep dive into the architectural decisions and mathematical foundations of the RAG techniques implemented in this project.

## 1. Information Retrieval (Lecture 2 & 3)

### BM25 (Best Matching 25)
A state-of-the-art keyword retrieval algorithm.
*   **$k_1$**: Controls term frequency saturation. High values mean terms appearing more frequently have even higher weight.
*   **$b$**: Controls document length normalization. Adjusts scores based on how long an article is relative to the average.

### Vector Search (Semantic)
*   **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions).
*   **Distance Metric**: L2 (Euclidean) Distance via FAISS.
*   **Concept**: Maps text to a high-dimensional space where "meaningfully similar" articles are close together, even if they don't share exact words.

### Hybrid Search (RRF)
We use **Reciprocal Rank Fusion** to combine lexical and semantic scores.
*   **Formula**: $RRF(d) = \sum_{r \in R} \frac{1}{k + rank(r)}$
*   **Outcome**: This ensures that a document ranked highly by *either* BM25 or Vector search is given priority in the final result.

## 2. Advanced Techniques (Lecture 3)

### LLM-as-a-Judge (Reranking)
The system uses a two-stage retrieval process. After the initial retrieval, Gemini 1.5 Flash is used as a reranker to score the top candidates for exact relevance to the query, providing superior precision over raw vector distance.

## 3. Generative AI & Tuning (Lecture 4)

### Sampling Parameters
The interactive lab allows real-time tuning of:
*   **Temperature**: Randomness in token selection. (0.1 for factual analysis, >1.0 for creative narratives).
*   **Top-P**: Nucleus sampling to filter out low-probability tail tokens.

### Grounding and Prompts
System prompts are engineered to enforce **Strict Grounding**. The model is instructed to cite specific sources from the context and explicitly state if information is missing from the provided Pakistani news data.

## 4. Verification Findings

Our verification tests revealed:
*   **Semantic Breadth**: Vector search successfully linked "fuel prices" to "petroleum rates" even when the specific keyword changed.
*   **Accuracy**: Low temperature combined with RRF retrieval resulted in zero hallucinations across test queries in the Pakistani news domain.
