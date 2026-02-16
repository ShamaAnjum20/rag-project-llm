import os
import psycopg2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import config

# PostgreSQL connection
pg_conn = psycopg2.connect(
    host="localhost",
    database="academic_rag",
    user="postgres",
    password="shama"
)
pg_conn.autocommit = True
pg = pg_conn.cursor()

embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

def process_and_store(file_path):
    filename = os.path.basename(file_path)
    file_type = filename.split(".")[-1]
    size_mb = os.path.getsize(file_path) / (1024*1024)

    # Save document metadata
    pg.execute("""
        INSERT INTO documents (filename, file_type, size_mb)
        VALUES (%s,%s,%s) RETURNING id
    """, (filename, file_type, size_mb))
    doc_id = pg.fetchone()[0]

    # Read text
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Save chunks in PG
    for c in chunks:
        pg.execute("INSERT INTO chunks (doc_id, content) VALUES (%s,%s)", (doc_id, c))

    # Store embeddings in Chroma
    vectordb = Chroma(
        persist_directory=config.CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    vectordb.add_texts(chunks, metadatas=[{"source": filename}] * len(chunks))
    vectordb.persist()
