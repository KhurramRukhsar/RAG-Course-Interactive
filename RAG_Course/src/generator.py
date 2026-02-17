import google.generativeai as genai
from typing import List, Dict
from .config import GOOGLE_API_KEY, GEMINI_MODEL_NAME, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K_SAMPLING, DEFAULT_MAX_OUTPUT_TOKENS

class Generator:
    def __init__(self):
        genai.configure(api_key=GOOGLE_API_KEY)

    def generate_answer(self, 
                        query: str, 
                        context_chunks: List[Dict], 
                        temperature: float = DEFAULT_TEMPERATURE,
                        top_p: float = DEFAULT_TOP_P,
                        top_k: int = DEFAULT_TOP_K_SAMPLING,
                        system_prompt: str = None) -> str:
        """
        Generates a RAG-enhanced answer using Gemini with tunable sampling parameters.
        """
        context_text = "\n\n---\n\n".join([c['text'] for c in context_chunks])
        
        if not system_prompt:
            system_prompt = """
            You are a professional News Analyst. Use the provided context to answer the user query.
            If the answer is not in the context, say "I don't have enough information from the provided news sources."
            Cite your sources by mentioning the newspaper name and title when applicable.
            """

        full_prompt = f"""
        {system_prompt}
        
        CONTEXT:
        {context_text}
        
        USER QUERY: {query}
        
        ANSWER:
        """

        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        }

        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(full_prompt, generation_config=generation_config)
        
        return response.text
