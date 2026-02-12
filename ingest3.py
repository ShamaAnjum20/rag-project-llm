# ingest.py
import os
import json
import pandas as pd
import pytesseract
import csv
from PIL import Image
from bs4 import BeautifulSoup
from pdf2image import convert_from_path

from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import config

def load_scanned_pdf(file_path: str):
    images = convert_from_path(file_path)
    extracted_text = ""

    for image in images:
        extracted_text += pytesseract.image_to_string(image)

    return [
        Document(
            page_content=extracted_text,
            metadata={"source": file_path, "type": "scanned_pdf"}
        )
    ]

# def load_html(file_path: str):
#     with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#         soup = BeautifulSoup(f.read(), "html.parser")

#     text = soup.get_text(separator="\n")

#     return [
#         Document(
#             page_content=text,
#             metadata={"source": file_path, "type": "html"}
#         )
#     ]



def load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []

    for item in data:
        text = (
            f"Student {item['name']} from {item['department']} department "
            f"studying year {item['year']} has a CGPA of {item['cgpa']}. "
            f"Skills include {', '.join(item.get('skills', []))}."
        )

        docs.append(
            Document(
                page_content=text,
                metadata={"source": file_path}
            )
        )
    return docs



def load_csv(file_path: str):
    docs = []

    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = (
                f"Student {row.get('name')} from {row.get('department')} department "
                f"studying year {row.get('year')} has a CGPA of {row.get('cgpa')}. "
                f"Skills include {row.get('skills')}. "
                f"Teacher: {row.get('teacher', 'N/A')}."
            )
            docs.append(
                Document(page_content=text, metadata={"source": file_path})
            )

    return docs



def load_document(file_path: str):
    ext = file_path.split(".")[-1].lower()

    if ext == "pdf":
        try:
            # Try normal PDF first
            return PyPDFLoader(file_path).load()
        except Exception:
            # Fallback to OCR for scanned PDFs
            return load_scanned_pdf(file_path)

    elif ext == "txt":
        return TextLoader(file_path, encoding="utf-8").load()

    elif ext == "md":
        return TextLoader(file_path, encoding="utf-8").load()

    elif ext == "docx":
        return Docx2txtLoader(file_path).load()

    elif ext in ["html", "htm"]:
        return load_html(file_path)

    elif ext == "json":
        return load_json(file_path)

    elif ext == "csv":
        return load_csv(file_path)

  

    else:
        raise ValueError(f"Unsupported file type: {ext}")

def process_and_store(file_path: str):
    documents = load_document(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

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

