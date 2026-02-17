import os
import pandas as pd
import glob
import re
from typing import List, Dict

class DataProcessor:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_csvs(self) -> pd.DataFrame:
        """Loads all Pakistani news CSVs from the data directory."""
        all_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        df_list = []
        for filename in all_files:
            try:
                df = pd.read_csv(filename)
                # Extract source from filename (e.g., 'The News' or 'Express Tribune')
                source = "Unknown"
                if "thenews" in filename.lower():
                    source = "The News"
                elif "tribune" in filename.lower():
                    source = "The Express Tribune"
                
                df['source'] = source
                df_list.append(df)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        if not df_list:
            return pd.DataFrame()
            
        return pd.concat(df_list, ignore_index=True)

    def chunk_documents(self, df: pd.DataFrame, strategy: str = "document", chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """
        Chunks documents based on selected strategy.
        Strategies: 'document' (one chunk per article), 'fixed' (fixed character size).
        """
        chunks = []
        for idx, row in df.iterrows():
            title = str(row.get('title', ''))
            content = str(row.get('content', ''))
            full_text = f"Title: {title}\n\nContent: {content}"
            metadata = {
                "source": row.get('source', 'Unknown'),
                "index": idx,
                "title": title
            }

            if strategy == "document":
                chunks.append({
                    "text": full_text,
                    "metadata": metadata
                })
            elif strategy == "fixed":
                # Simple fixed-size character chunking
                start = 0
                while start < len(full_text):
                    end = start + chunk_size
                    chunk_text = full_text[start:end]
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {**metadata, "chunk_id": start}
                    })
                    start += (chunk_size - overlap)
        
        return chunks

if __name__ == "__main__":
    from config import DATA_PATH
    processor = DataProcessor(DATA_PATH)
    raw_df = processor.load_csvs()
    print(f"Loaded {len(raw_df)} articles.")
    doc_chunks = processor.chunk_documents(raw_df, strategy="document")
    print(f"Created {len(doc_chunks)} document-level chunks.")
    fixed_chunks = processor.chunk_documents(raw_df, strategy="fixed", chunk_size=300, overlap=50)
    print(f"Created {len(fixed_chunks)} fixed-size chunks.")
