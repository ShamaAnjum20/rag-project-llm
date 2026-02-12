import os

class Config:
    PDF_SOURCE_DIRECTORY = "uploaded_docs"
    CHROMA_PERSIST_DIRECTORY = "docs/chroma"
    FEEDBACK_DIR = "feedback"
    FEEDBACK_FILE = "feedback/feedback_log.csv"
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    def __init__(self):
        os.makedirs(self.PDF_SOURCE_DIRECTORY, exist_ok=True)
        os.makedirs(self.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        os.makedirs(self.FEEDBACK_DIR, exist_ok=True)

config = Config()
