import google.generativeai as genai
from typing import List, Dict, Tuple
from .config import GOOGLE_API_KEY, GEMINI_MODEL_NAME

class Reranker:
    def __init__(self):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    def rerank_with_llm(self, query: str, results: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        """
        Uses LLM to score the relevance of retrieved documents.
        This demonstrates the 'LLM-as-a-judge' and reranking concepts from Lecture 3.
        """
        reranked_results = []
        for chunk, initial_score in results:
            prompt = f"""
            Score the relevance of the following news snippet to the user query.
            Query: {query}
            Snippet: {chunk['text']}
            
            Return ONLY a single number between 0 and 1, where 1 is highly relevant and 0 is not relevant at all.
            """
            try:
                response = self.model.generate_content(prompt)
                llm_score = float(response.text.strip())
                reranked_results.append((chunk, llm_score))
            except:
                # Fallback to initial score if LLM scoring fails
                reranked_results.append((chunk, initial_score))
        
        # Sort by the new LLM score
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return reranked_results
