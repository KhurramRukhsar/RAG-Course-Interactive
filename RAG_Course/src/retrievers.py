import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from .config import EMBEDDING_MODEL_NAME, DEFAULT_BM25_K1, DEFAULT_BM25_B, DEFAULT_RRF_K

class BaseRetriever:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.corpus = [c['text'] for c in chunks]

class BM25Retriever(BaseRetriever):
    def __init__(self, chunks: List[Dict], k1=DEFAULT_BM25_K1, b=DEFAULT_BM25_B):
        super().__init__(chunks)
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]

class VectorRetriever(BaseRetriever):
    def __init__(self, chunks: List[Dict], model_name=EMBEDDING_MODEL_NAME):
        super().__init__(chunks)
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(self.corpus, show_progress_bar=False)
        self.dimension = self.embeddings.shape[1]
        
        # L2 Distance (Euclidean)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings.astype('float32'))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        query_vector = self.model.encode([query], show_progress_bar=False).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        
        # Convert L2 distance to a "similarity" score (1 / (1 + d))
        results = []
        for d, i in zip(distances[0], indices[0]):
            if i != -1:
                score = 1 / (1 + d)
                results.append((self.chunks[i], float(score)))
        return results

class HybridRetriever:
    def __init__(self, bm25_retriever: BM25Retriever, vector_retriever: VectorRetriever):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever

    def search_rrf(self, query: str, top_k: int = 5, k=DEFAULT_RRF_K) -> List[Tuple[Dict, float]]:
        """Reciprocal Rank Fusion (RRF) implementation."""
        bm25_results = self.bm25.search(query, top_k=50) # Search more to fuse
        vector_results = self.vector.search(query, top_k=50)

        scores = {} # Map chunk text to performance score
        
        # Helper to get unique key for a chunk
        def get_key(chunk): return chunk['text']

        # RRF formula: Score = sum(1 / (k + rank))
        for rank, (chunk, _) in enumerate(bm25_results, 1):
            key = get_key(chunk)
            scores[key] = scores.get(key, 0) + (1.0 / (k + rank))
            
        for rank, (chunk, _) in enumerate(vector_results, 1):
            key = get_key(chunk)
            scores[key] = scores.get(key, 0) + (1.0 / (k + rank))

        # Sort by RRF score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Map back to chunk objects
        text_to_chunk = {chunk['text']: chunk for chunk, score in (bm25_results + vector_results)}
        return [(text_to_chunk[text], score) for text, score in sorted_results]
