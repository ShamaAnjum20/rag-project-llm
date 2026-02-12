# app.py
import json
import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

from ingest import process_and_store
from config import config

import psycopg2

# ------------------ ENV ------------------
load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ------------------ POSTGRES ------------------
pg_conn = psycopg2.connect(
    host=os.getenv("PG_HOST", "localhost"),
    database=os.getenv("PG_DB", "chatdb"),
    user=os.getenv("PG_USER", "shakirahamedk"),
    password=os.getenv("PG_PASSWORD", "postgres"),
)
pg_conn.autocommit = True
pg_cur = pg_conn.cursor()

# ------------------ HELPERS (DB) ------------------

def new_session_id():
    return str(uuid.uuid4())

def save_message(session_id, role, content):
    pg_cur.execute(
        """
        INSERT INTO conversations (id, session_id, role, content, created_at)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (str(uuid.uuid4()), session_id, role, content, datetime.now())
    )

def get_all_sessions():
    pg_cur.execute(
        """
        SELECT session_id, MIN(created_at)
        FROM conversations
        GROUP BY session_id
        ORDER BY MIN(created_at) DESC
        """
    )
    return [row[0] for row in pg_cur.fetchall()]

def load_chat(session_id):
    pg_cur.execute(
        """
        SELECT role, content
        FROM conversations
        WHERE session_id = %s
        ORDER BY created_at
        """,
        (session_id,)
    )
    return pg_cur.fetchall()

# ------------------ SESSION STATE ------------------

if "current_session" not in st.session_state:
    st.session_state.current_session = new_session_id()

# ------------------ UI ------------------
st.set_page_config(page_title="Academic QA", layout="wide")

# ------------------ SIDEBAR ------------------
st.sidebar.title("💬 Chats")

if st.sidebar.button("➕ New Chat"):
    st.session_state.current_session = new_session_id()
    st.rerun()

sessions = get_all_sessions()

for s in sessions:
    label = f"Chat {s[:8]}"
    if st.sidebar.button(label, key=s):
        st.session_state.current_session = s
        st.rerun()

# ------------------ MAIN TITLE ------------------
st.title("📄 Academic RAG System")

# ------------------ EMBEDDINGS & VECTOR DB ------------------
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"}
)

def load_vectordb():
    return Chroma(
        persist_directory=config.CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

vectordb = load_vectordb()

# ------------------ LLM ------------------
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_tokens=400
)

# ------------------ UPLOAD ------------------
uploaded_file = st.file_uploader(
    "Upload document",
    type=["pdf","docx","txt","md","html","htm","json","csv"]
)

if uploaded_file:
    save_path = os.path.join(config.PDF_SOURCE_DIRECTORY, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("Processing document…")
    chunks = process_and_store(save_path)
    st.success(f"{chunks} chunks added")
    vectordb = load_vectordb()

st.divider()

# ------------------ LOAD CHAT HISTORY ------------------
chat = load_chat(st.session_state.current_session)

for role, content in chat:
    with st.chat_message(role):
        st.write(content)

# ------------------ QUESTION ------------------
query = st.chat_input("Ask a question from your uploaded documents")

if query:
    # USER
    with st.chat_message("user"):
        st.write(query)

    save_message(st.session_state.current_session, "user", query)

    # ASSISTANT
    with st.chat_message("assistant"):
        with st.spinner("Retrieving answer..."):
            docs = vectordb.similarity_search_with_score(query, k=3)
            docs = [d for d in docs if d[1] < 2.0]

            citations = []
            used_general_knowledge = False

            if not docs:
                used_general_knowledge = True
                answer = model.invoke([
                    SystemMessage(content="Answer clearly using your general knowledge."),
                    HumanMessage(content=query)
                ]).content
            else:
                context = "\n\n".join(d[0].page_content for d in docs)
                answer = model.invoke([
                    SystemMessage(content="Answer ONLY from the given context."),
                    HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{query}")
                ]).content.strip()

                if answer.lower().startswith("i don't know"):
                    used_general_knowledge = True
                    answer = model.invoke([
                        SystemMessage(content="Answer clearly using your general knowledge."),
                        HumanMessage(content=query)
                    ]).content
                else:
                    for d, _ in docs:
                        meta = d.metadata
                        src = meta.get("source", "Unknown")
                        page = meta.get("page")
                        citations.append(f"{src} (page {page})" if page else src)

            if used_general_knowledge:
                final_answer = (
                    "⚠️ Uploaded document doesn’t contain this information.\n\n"
                    + answer
                )
            else:
                final_answer = answer
                if citations:
                    final_answer += "\n\n📚 Sources:\n" + "\n".join(
                        f"- {c}" for c in set(citations)
                    )

            st.write(final_answer)
            save_message(st.session_state.current_session, "assistant", final_answer)

    st.rerun()
