import os
import google.generativeai as genai
from vector_store import VectorStore

MODEL_NAME = "gemini-flash-latest"

PERSONA_PROMPTS = {
    "Default": "You are a helpful assistant for a Pakistani news analysis system.",
    "Journalist": "You are a professional reporter. Summarize the news based only on the provided context. Maintain an objective tone.",
    "Analyst": "You are a data analyst. Look for patterns, key figures, and implications in the news context provided.",
    "Skeptic": "You are a critical thinker. Examine the news provided for any biases, missing information, or contradictions.",
    "Optimist": "You are a positive commentator. Highlight the good news, potential opportunities, and hopeful developments in the provided context."
}

class RAGEngine:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
        # Configure Gemini API if not already configured
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            
        self.model = genai.GenerativeModel(MODEL_NAME)

    def generate_rag_answer(self, query, newspaper_filter="All", date_filter=None, 
                           persona="Default", temperature=0.7):
        """
        Generates an answer using RAG (Retrieval-Augmented Generation).
        """
        # 1. Retrieve context
        if self.vector_store.index is None:
             if not self.vector_store.load_index():
                 return "Error: Vector store not initialized.", []

        retrieved_docs = self.vector_store.search(
            query, 
            top_k=5, 
            newspaper_filter=newspaper_filter if newspaper_filter != "All" else None,
            date_filter=date_filter
        )
        
        if not retrieved_docs:
            return "No relevant documents found to answer your question.", []

        # 2. Construct Prompt
        context_str = "\n\n".join(
            [f"--- Document {i+1} ---\n{doc['text']}" for i, doc in enumerate(retrieved_docs)]
        )
        
        persona_instruction = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["Default"])
        
        system_instruction = (
            f"{persona_instruction}\n"
            "Answer the user's question based ONLY on the provided context.\n"
            "If the answer is not in the context, say 'I cannot answer this based on the available news articles.'\n"
            "STRICT CONSTRAINTS:\n"
            "- Be concise and impactful. Do not be overly wordy.\n"
            "- Ensure your bullet points are complete sentences.\n"
            "- Do not repeat the context; provide insights.\n"
            "- Use bullet points for the main points of your response.\n"
            "- Cite newspapers where applicable."
        )
        
        prompt = f"{system_instruction}\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        
        # 3. Generate Answer
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature
            )
            response = self.model.generate_content(prompt, generation_config=generation_config)
            return response.text, retrieved_docs
        except Exception as e:
            return f"Error generating answer: {e}", []

    def generate_plain_answer(self, query, persona="Default", temperature=0.7):
        """
        Generates an answer using the LLM's internal knowledge only (No RAG).
        """
        persona_instruction = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["Default"])
        
        system_instruction = (
            f"{persona_instruction}\n"
            "Answer based on your general knowledge.\n"
            "STRICT CONSTRAINTS:\n"
            "- Keep it short and professionally formatted.\n"
            "- Use bullet points.\n"
        )
        
        prompt = f"{system_instruction}\n\nQuestion: {query}\n\nAnswer:"
        
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature
            )
            response = self.model.generate_content(prompt, generation_config=generation_config)
            return response.text
        except Exception as e:
            return f"Error generating answer: {e}"

if __name__ == "__main__":
    # Test
    if not GOOGLE_API_KEY:
        print("Skipping RAG Engine test: GOOGLE_API_KEY not set.")
    else:
        from vector_store import VectorStore
        vs = VectorStore()
        if vs.load_index():
            rag = RAGEngine(vs)
            ans, sources = rag.generate_rag_answer("Who won the PSL final?")
            print("RAG Answer:\n", ans)
            print("\nSources:", [s['metadata']['title'] for s in sources])
            
            plain_ans = rag.generate_plain_answer("Who won the PSL final?")
            print("\nPlain Answer:\n", plain_ans)
