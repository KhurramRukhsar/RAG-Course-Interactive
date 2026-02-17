import streamlit as st
import pandas as pd
from src.config import DATA_PATH, DEFAULT_BM25_K1, DEFAULT_BM25_B, DEFAULT_ALPHA, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from src.data_processor import DataProcessor
from src.retrievers import BM25Retriever, VectorRetriever, HybridRetriever
from src.reranker import Reranker
from src.generator import Generator

st.set_page_config(page_title="RAG Interactive Course", layout="wide")

st.title("🇵🇰 RAG Interactive Course & Lab")
st.markdown("""
This app is a hands-on implementation of the RAG Course. 
Explore how different parameters and techniques affect the performance and accuracy of a RAG system using Pakistani News data.
""")

# --- Sidebar: Global Settings ---
st.sidebar.header("Global Settings")
if 'processor' not in st.session_state:
    st.session_state.processor = DataProcessor(DATA_PATH)
    st.session_state.raw_df = st.session_state.processor.load_csvs()

# --- Tabs: Interactive Labs ---
tab1, tab2, tab3, tab4 = st.tabs(["📦 Data & Chunking", "🔍 Retrieval Lab", "🎯 Advanced Retrieval", "🤖 Generation Lab"])

with tab1:
    st.header("1. Data & Chunking")
    st.write(f"Loaded {len(st.session_state.raw_df)} articles from Pakistani news sources.")
    
    col1, col2 = st.columns(2)
    with col1:
        chunk_strategy = st.selectbox("Chunking Strategy", ["document", "fixed"], help="Lecture 2: Strategy for splitting data.")
        chunk_size = st.slider("Chunk Size", 100, 1000, 500) if chunk_strategy == "fixed" else 500
        overlap = st.slider("Overlap", 0, 100, 50) if chunk_strategy == "fixed" else 0
        
    if st.button("Process & Chunk"):
        with st.spinner("Processing..."):
            st.session_state.chunks = st.session_state.processor.chunk_documents(
                st.session_state.raw_df, strategy=chunk_strategy, chunk_size=chunk_size, overlap=overlap
            )
            st.success(f"Created {len(st.session_state.chunks)} chunks!")
            st.session_state.retriever_bm25 = BM25Retriever(st.session_state.chunks)
            st.session_state.retriever_vector = VectorRetriever(st.session_state.chunks)
            st.session_state.hybrid = HybridRetriever(st.session_state.retriever_bm25, st.session_state.retriever_vector)

    if 'chunks' in st.session_state:
        st.subheader("Sample Chunks")
        st.write(st.session_state.chunks[:3])

with tab2:
    st.header("2. Retrieval Lab")
    query = st.text_input("Enter a query (e.g., 'Cricket match in Karachi', 'Fuel prices in Pakistan')")
    
    if query and 'chunks' in st.session_state:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("BM25 (Keyword)")
            bm25_res = st.session_state.retriever_bm25.search(query)
            for chunk, score in bm25_res:
                st.info(f"**Score: {score:.2f}**\n\n{chunk['text'][:200]}...")
        
        with col2:
            st.subheader("Vector (Semantic)")
            vec_res = st.session_state.retriever_vector.search(query)
            for chunk, score in vec_res:
                st.success(f"**Score: {score:.4f}**\n\n{chunk['text'][:200]}...")
                
        with col3:
            st.subheader("Hybrid (RRF)")
            hyb_res = st.session_state.hybrid.search_rrf(query)
            for chunk, score in hyb_res:
                st.warning(f"**Score: {score:.4f}**\n\n{chunk['text'][:200]}...")

with tab3:
    st.header("3. Advanced Retrieval & Reranking")
    st.markdown("Experiment with BM25 tuning and LLM-based reranking (Lecture 3).")
    
    col1, col2 = st.columns(2)
    with col1:
        k1 = st.slider("BM25 k1 (Saturation)", 0.0, 3.0, DEFAULT_BM25_K1)
        b = st.slider("BM25 b (Normalization)", 0.0, 1.0, DEFAULT_BM25_B)
        use_reranker = st.checkbox("Use LLM Reranker")
    
    if st.button("Run Advanced Search"):
        # Update BM25 with new params
        temp_bm25 = BM25Retriever(st.session_state.chunks, k1=k1, b=b)
        results = temp_bm25.search(query)
        
        if use_reranker:
            st.info("Reranking top 5 results using Gemini...")
            reranker = Reranker()
            results = reranker.rerank_with_llm(query, results)
        
        for chunk, score in results:
            st.write(f"---")
            st.write(f"**Final Score: {score:.4f}**")
            st.write(chunk['text'])

with tab4:
    st.header("4. Generation Lab")
    st.markdown("Tune the LLM sampling parameters (Lecture 4) and observe the effect on the RAG answer.")
    
    col1, col2 = st.columns(2)
    with col1:
        temp = st.slider("Temperature", 0.0, 1.5, DEFAULT_TEMPERATURE)
        top_p = st.slider("Top-P", 0.0, 1.0, DEFAULT_TOP_P)
    
    if st.button("Generate RAG Answer"):
        generator = Generator()
        # Using hybrid results for the final answer
        context = st.session_state.hybrid.search_rrf(query, top_k=3)
        answer = generator.generate_answer(query, [c for c, s in context], temperature=temp, top_p=top_p)
        
        st.subheader("RAG Answer")
        st.markdown(answer)
        
        with st.expander("Show Context Chunks Used"):
            for c, s in context:
                st.write(c['text'])
