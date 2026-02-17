import os
from dotenv import load_dotenv

load_dotenv()

# Model Configurations
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-1.5-flash"

# Default Parameters
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_ALPHA = 0.5  # Hybrid search weight (1 = Vector, 0 = Keyword)
DEFAULT_TOP_K = 5
DEFAULT_RRF_K = 60

# LLM Sampling Defaults
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K_SAMPLING = 40
DEFAULT_MAX_OUTPUT_TOKENS = 1024

# Path Configurations
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data")
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_store")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
