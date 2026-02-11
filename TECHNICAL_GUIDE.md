# Pakistani News RAG: Architecture and Logic

## 1. High-Level Flow

The system follows a standard RAG pipeline tailored for newspaper data. The flow is as follows:

1.  **Ingestion**: CSV files from `data/` are loaded. Metadata (newspaper, date) is extracted from filenames.
2.  **Preprocessing**: Title and content are combined into a single text block for each article.
3.  **Indexing**: The text blocks are converted into vector embeddings using a pre-trained transformer model.
4.  **Storage**: Embeddings are stored in a FAISS (Facebook AI Similarity Search) index for fast L2-distance retrieval.
5.  **Retrieval**: When a user asks a question, the query is embedded, and the top-K most similar articles are fetched.
6.  **Generation**: The retrieved articles are provided as context to the Gemini 1.5 Flash model, which generates a grounded answer.

## 2. Technical Logic

### Chunking Strategy
In this specific application, each news article is treated as a single "chunk". Since Pakistani news articles in the dataset are generally concise, the combined `Title + Content` usually fits within the embedding model's context window (512 tokens for `all-MiniLM-L6-v2`). This preserves the global context of each story without losing information in arbitrary splits.

### Retrieval Mechanism
We use **Semantic Search** based on dense vector representations. Unlike traditional keyword search (BM25), this mechanism understands the "intent" and "meaning" of words. For example, a query about "Cricket" will retrieve articles about "T20" or "PSL" even if the word "Cricket" isn't prominent.

## 3. The Math: Vector Embeddings

### Embedding Model
We use the `all-MiniLM-L6-v2` model from the Sentence-Transformers library.
- **Dimensionality**: 384 dimensions.
- **Normalization**: Vectors are typically normalized to unit length so that Cosine Similarity can be used.

### Vector Search (FAISS)
The retrieval uses **IndexFlatL2**. 
- **Distance Metric**: L2 (Euclidean) Distance.
- **Formula**: $d(p, q) = \sqrt{\sum (p_i - q_i)^2}$
Where $p$ is the query vector and $q$ is a document vector. A smaller distance indicates higher semantic similarity.

## 4. Code Walkthrough

### `data_loader.py`
- `load_all_csvs()`: Dynamically parses filenames using regex to extract metadata like Newspaper name and branch-specific dates.
- `preprocess_documents()`: Cleans the text and builds the dictionary structure used throughout the pipeline.

### `vector_store.py`
- `build_index()`: Encodes the entire text corpus into a NumPy array and initializes the FAISS index.
- `search()`: Not only performs the vector search but also applies "Hard Filters" (Newspaper name/Date) to the results before returning them to the engine.

### `rag_engine.py`
- `generate_rag_answer()`: The "brain" of the operation. It constructs a specialized system prompt that instructs the LLM to *only* use the provided context, preventing hallucinations.

## 5. Execution Steps

1. **Environment Setup**: Install dependencies via `pip install -r requirements.txt`.
2. **API Configuration**: Set `GOOGLE_API_KEY` in a `.env` file.
3. **Indexing**: Run `python vector_store.py` to generate the `vector_store/` directory.
4. **Launch**: Execute `streamlit run app.py` to start the UI.
