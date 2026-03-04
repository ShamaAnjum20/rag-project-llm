import os

class Config:
    def __init__(self):
        # Project root directory
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # 📂 Uploaded documents
        self.PDF_SOURCE_DIRECTORY = os.path.join(self.BASE_DIR, "uploaded_docs")

        # 📦 Chroma persistence
        self.CHROMA_PERSIST_DIRECTORY = os.path.join(self.BASE_DIR, "docs", "chroma")

        # 📝 Feedback storage
        self.FEEDBACK_DIR = os.path.join(self.BASE_DIR, "feedback")
        self.FEEDBACK_FILE = os.path.join(self.FEEDBACK_DIR, "feedback_log.csv")

        # 🤖 Embedding model
        self.EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

        # Chunking config
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200

        # Create directories
        os.makedirs(self.PDF_SOURCE_DIRECTORY, exist_ok=True)
        os.makedirs(self.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        os.makedirs(self.FEEDBACK_DIR, exist_ok=True)


# Initialize config
config = Config()
