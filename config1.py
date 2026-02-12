# config.py
import os
FEEDBACK_DIR = "feedback"
FEEDBACK_FILE = "feedback/feedback_log.csv"

class Config:
    PDF_SOURCE_DIRECTORY: str = "uploaded_docs"
    CHROMA_PERSIST_DIRECTORY: str = "docs/chroma"
    FEEDBACK_FILE = "feedback/feedback_log.csv" 
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    # CHUNK_SIZE = 2028
    # CHUNK_OVERLAP = 250
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    def __init__(self):
        os.makedirs(self.PDF_SOURCE_DIRECTORY, exist_ok=True)
        os.makedirs(self.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        os.makedirs("feedback", exist_ok=True)  


config = Config()
