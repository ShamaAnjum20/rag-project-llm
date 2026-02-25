import os
import psycopg2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from docx import Document
from config import config

# PostgreSQL
pg_conn = psycopg2.connect(
    from dotenv import load_dotenv
import os
import psycopg2

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
)
pg_conn.autocommit = True
pg = pg_conn.cursor()

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

# ---------- TEXT EXTRACT ----------
def read_file(file_path):
    ext = file_path.split(".")[-1].lower()

    if ext == "pdf":
        reader = PdfReader(file_path)
        return "\n".join([p.extract_text() or "" for p in reader.pages])

    elif ext == "docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    else:  # txt, md, csv, json
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

# ---------- MAIN INGEST ----------
def process_and_store(file_path):
    filename = os.path.basename(file_path)
    file_type = filename.split(".")[-1]
    size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # Save document metadata
    pg.execute("""
        INSERT INTO documents (filename, file_type, size_mb)
        VALUES (%s,%s,%s) RETURNING id
    """, (filename, file_type, size_mb))
    doc_id = pg.fetchone()[0]

    # Read text
    text = read_file(file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Save chunks to PostgreSQL
    for c in chunks:
        pg.execute("INSERT INTO chunks (doc_id, content) VALUES (%s,%s)", (doc_id, c))

    # Save embeddings to Chroma
    vectordb = Chroma(
        persist_directory=config.CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    vectordb.add_texts(
        texts=chunks,
        metadatas=[{"source": filename}] * len(chunks)
    )

    vectordb.persist()   # ✅ IMPORTANT FIX

    return len(chunks)
