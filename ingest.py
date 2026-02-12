# ingest.py
import os
import json
import csv
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import config

# ------------------ HELPERS ------------------

def filename_only(path):
    return os.path.basename(path)

# ------------------ LOADERS ------------------

def load_scanned_pdf(file_path: str):
    images = convert_from_path(file_path)
    docs = []

    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": filename_only(file_path),
                    "type": "scanned_pdf",
                    "page": i + 1
                }
            )
        )
    return docs

def load_html(file_path: str):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    text = soup.get_text(separator="\n")

    return [
        Document(
            page_content=text,
            metadata={
                "source": filename_only(file_path),
                "type": "html",
                "section": "Full Document"
            }
        )
    ]

def load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for i, item in enumerate(data):
        text = (
            f"Student {item['name']} from {item['department']} department "
            f"studying year {item['year']} has a CGPA of {item['cgpa']}. "
            f"Skills include {', '.join(item.get('skills', []))}."
        )
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": filename_only(file_path),
                    "row": i + 1
                }
            )
        )
    return docs

def load_csv(file_path: str):
    docs = []
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            text = (
                f"Student {row.get('name')} from {row.get('department')} department "
                f"studying year {row.get('year')} has a CGPA of {row.get('cgpa')}. "
                f"Skills include {row.get('skills')}. "
                f"Teacher: {row.get('teacher', 'N/A')}."
            )
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename_only(file_path),
                        "row": i + 1
                    }
                )
            )
    return docs

# ------------------ MAIN LOADER ------------------

def load_document(file_path: str):
    ext = file_path.split(".")[-1].lower()

    if ext == "pdf":
        try:
            docs = PyPDFLoader(file_path).load()
            for d in docs:
                d.metadata["source"] = filename_only(file_path)
                if "page" not in d.metadata:
                    d.metadata["page"] = None
            return docs
        except Exception:
            return load_scanned_pdf(file_path)

    elif ext in ["txt", "md"]:
        docs = TextLoader(file_path, encoding="utf-8").load()
        for d in docs:
            d.metadata["source"] = filename_only(file_path)
            d.metadata["section"] = "Full Document"
        return docs

    elif ext == "docx":
        docs = Docx2txtLoader(file_path).load()
        for d in docs:
            d.metadata["source"] = filename_only(file_path)
            d.metadata["section"] = "Full Document"
        return docs

    elif ext in ["html", "htm"]:
        return load_html(file_path)

    elif ext == "json":
        return load_json(file_path)

    elif ext == "csv":
        return load_csv(file_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ------------------ PROCESS & STORE ------------------

def process_and_store(file_path: str):
    documents = load_document(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    # Keep original metadata in chunks
    for chunk in chunks:
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = filename_only(file_path)

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIRECTORY
    )

    vectordb.persist()
    return len(chunks)
