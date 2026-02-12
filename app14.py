import json
import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from ingest import process_and_store
from config import config
from feedback import save_feedback

import psycopg2

# ------------------ ENV ------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
load_dotenv()

# ------------------ POSTGRESQL ------------------
pg_conn = psycopg2.connect(
        host="localhost",
        database="academic_rag",
        user="postgres",
        password="shama"
)
pg_conn.autocommit = True
pg_cur = pg_conn.cursor()

# ------------------ DB FUNCTIONS ------------------

def new_session_id():
    return str(uuid.uuid4())

def save_message(session_id, role, content):
    pg_cur.execute("""
        INSERT INTO conversations (id, session_id, role, content, created_at)
        VALUES (%s, %s, %s, %s, %s)
    """, (str(uuid.uuid4()), session_id, role, content, datetime.now()))

def get_sessions():
    pg_cur.execute("""
        SELECT DISTINCT session_id, MIN(created_at)
        FROM conversations
        GROUP BY session_id
        ORDER BY MIN(created_at) DESC
    """)
    return [row[0] for row in pg_cur.fetchall()]

def load_chat(session_id):
    pg_cur.execute("""
        SELECT role, content
        FROM conversations
        WHERE session_id = %s
        ORDER BY created_at
    """, (session_id,))
    return pg_cur.fetchall()

# ------------------ STREAMLIT SESSION ------------------

if "current_session" not in st.session_state:
    st.session_state.current_session = new_session_id()

# ------------------ PAGE ------------------
st.set_page_config(page_title="Academic RAG", layout="wide")
st.title("📄 Academic RAG System")

# ------------------ SIDEBAR ------------------
st.sidebar.header("💬 Chats")

if st.sidebar.button("➕ New Chat"):
    st.session_state.current_session = new_session_id()
    st.rerun()

sessions = get_sessions()

for s in sessions:
    label = f"Chat {s[:8]}"
    if st.sidebar.button(label, key=s):
        st.session_state.current_session = s
        st.rerun()

# ------------------ EMBEDDINGS ------------------
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
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_tokens=400
)

# ------------------ UPLOAD ------------------
uploaded_file = st.file_uploader(
    "Upload document",
    type=["pdf","docx","txt","md","html","json","csv"]
)

if uploaded_file:
    save_path = os.path.join(config.PDF_SOURCE_DIRECTORY, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing document..."):
        chunks = process_and_store(save_path)

    vectordb = load_vectordb()
    st.success(f"{chunks} chunks added")

st.divider()

# ------------------ LOAD PREVIOUS CHAT HISTORY ------------------
chat_history = load_chat(st.session_state.current_session)

for role, content in chat_history:
    with st.chat_message(role):
        st.write(content)

# ------------------ QUESTION ------------------
final_answer = None
citations = []

query = st.chat_input("Ask a question")

if query:
    # show user
    with st.chat_message("user"):
        st.write(query)
    save_message(st.session_state.current_session, "user", query)

    # assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            docs = vectordb.similarity_search_with_score(query, k=3)
            docs = [d for d in docs if d[1] < 2.0]

            citations = []
            used_general_knowledge = False

            if not docs:
                used_general_knowledge = True
                answer = llm.invoke([
                    SystemMessage(content="Answer using general knowledge."),
                    HumanMessage(content=query)
                ]).content
            else:
                context = "\n\n".join(d[0].page_content for d in docs)

                answer = llm.invoke([
                    SystemMessage(content="Answer ONLY from context."),
                    HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{query}")
                ]).content.strip()

                if answer.lower().startswith("i don't know"):
                    used_general_knowledge = True
                    answer = llm.invoke([
                        SystemMessage(content="Answer using general knowledge."),
                        HumanMessage(content=query)
                    ]).content
                else:
                    for d, _ in docs:
                        meta = d.metadata
                        src = os.path.basename(meta.get("source", "Unknown"))
                        page = meta.get("page")
                        citations.append(f"{src} (page {page})" if page else src)

            citations = list(set(citations))

            if used_general_knowledge:
                final_answer = "⚠️ Not found in documents.\n\n" + answer
            else:
                final_answer = answer
                if citations:
                    final_answer += "\n\n📚 Sources:\n" + "\n".join(f"- {c}" for c in citations)

            st.write(final_answer)
            save_message(st.session_state.current_session, "assistant", final_answer)

    st.rerun()

# ------------------ FEEDBACK (FIXED) ------------------
st.divider()
st.subheader("📝 Feedback")

rating = st.radio("Rate answer", [1,2,3,4,5], horizontal=True)
issue = st.selectbox("Issue", ["None","Incorrect","Incomplete","Hallucinated","Formatting"])

if st.button("Submit Feedback"):
    if chat_history:
        last_question = chat_history[-2][1] if len(chat_history) >= 2 else ""
        last_answer = chat_history[-1][1] if len(chat_history) >= 1 else ""
        save_feedback(last_question, last_answer, rating, issue, citations)
        st.success("Feedback saved")
    else:
        st.warning("No chat history found")
