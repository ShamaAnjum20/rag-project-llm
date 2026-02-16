import json, os, uuid
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from ingest1 import process_and_store
from config import config
from feedback import save_feedback

import psycopg2
import redis
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- ENV ----------------
load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ---------------- POSTGRES ----------------
pg_conn = psycopg2.connect(
    host="localhost",
    database="academic_rag",
    user="postgres",
    password="shama"
)
pg_conn.autocommit = True
pg = pg_conn.cursor()

# ---------------- REDIS CACHE ----------------
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# ---------------- DB FUNCTIONS ----------------
def new_session():
    sid = str(uuid.uuid4())
    pg.execute("INSERT INTO chat_sessions (session_id, title) VALUES (%s,%s)", (sid, "New Chat"))
    return sid

def set_title(session_id, title):
    pg.execute("UPDATE chat_sessions SET title=%s WHERE session_id=%s", (title, session_id))

def save_message(session_id, role, content):
    pg.execute("""
        INSERT INTO conversations (id, session_id, role, content)
        VALUES (%s,%s,%s,%s)
    """, (str(uuid.uuid4()), session_id, role, content))
    redis_client.delete(session_id)

def get_sessions():
    pg.execute("SELECT session_id, title FROM chat_sessions ORDER BY created_at DESC")
    return pg.fetchall()

def load_chat(session_id):
    cached = redis_client.get(session_id)
    if cached:
        return json.loads(cached)

    pg.execute("""
        SELECT role, content FROM conversations
        WHERE session_id=%s ORDER BY created_at
    """, (session_id,))
    data = pg.fetchall()
    data_json = json.dumps(data)
    redis_client.set(session_id, data_json)
    return data

def clear_chat(session_id):
    pg.execute("DELETE FROM conversations WHERE session_id=%s", (session_id,))
    redis_client.delete(session_id)

# ---------------- RETRIEVAL ACCURACY ----------------
def calc_retrieval_accuracy(docs):
    if len(docs) == 0:
        return 0
    relevant = [d for d, score in docs if score < 1.5]
    return len(relevant) / len(docs) * 100

# ---------------- STREAMLIT STATE ----------------
if "current_session" not in st.session_state:
    st.session_state.current_session = new_session()

# ---------------- PAGE ----------------
st.set_page_config(page_title="Academic RAG", layout="wide")
st.title("📄 Academic RAG System (Advanced)")

# ---------------- SIDEBAR ----------------
st.sidebar.header("💬 Chats")

if st.sidebar.button("➕ New Chat"):
    st.session_state.current_session = new_session()
    st.rerun()

sessions = get_sessions()
for sid, title in sessions:
    if st.sidebar.button(title, key=sid):
        st.session_state.current_session = sid
        st.rerun()

if st.sidebar.button("🗑️ Clear Chat"):
    clear_chat(st.session_state.current_session)
    st.rerun()

# Monitoring Dashboard Button
if st.sidebar.button("📊 Monitoring Dashboard"):
    st.switch_page("monitoring_dashboard.py")

# ---------------- EMBEDDINGS ----------------
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

# ---------------- LLM ----------------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, max_tokens=400)

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload document", type=["pdf","docx","txt","md","csv","json"])

if uploaded_file:
    save_path = os.path.join(config.PDF_SOURCE_DIRECTORY, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing document..."):
        process_and_store(save_path)

    vectordb = load_vectordb()
    st.success("Document added to knowledge base")

st.divider()

# ---------------- LOAD CHAT ----------------
history = load_chat(st.session_state.current_session)

for role, content in history:
    with st.chat_message(role):
        st.write(content)

# ---------------- CHAT INPUT ----------------
query = st.chat_input("Ask a question")

final_answer = None
citations = []

if query:
    if len(history) == 0:
        set_title(st.session_state.current_session, query[:30])

    with st.chat_message("user"):
        st.write(query)

    save_message(st.session_state.current_session, "user", query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            docs = vectordb.similarity_search_with_score(query, k=3)
            docs = [d for d in docs if d[1] < 2.0]

            used_gk = False
            citations = []

            # ---------- NO DOCS ----------
            if not docs:
                used_gk = True
                answer = llm.invoke([
                    SystemMessage(content="Answer using general knowledge."),
                    HumanMessage(content=query)
                ]).content

            # ---------- DOCS FOUND ----------
            else:
                context = "\n".join(d[0].page_content for d in docs)

                answer = llm.invoke([
                    SystemMessage(content="Answer ONLY from context."),
                    HumanMessage(content=f"{context}\nQuestion:{query}")
                ]).content

                # fallback to GK
                if answer.lower().startswith("i don't know"):
                    used_gk = True
                    answer = llm.invoke([
                        SystemMessage(content="Answer using general knowledge."),
                        HumanMessage(content=query)
                    ]).content
                else:
                    # Save citations + log references
                    for d, _ in docs:
                        src = os.path.basename(d.metadata.get("source", "Unknown"))
                        citations.append(src)
                        pg.execute("INSERT INTO chunk_references (source) VALUES (%s)", (src,))

            citations = list(set(citations))

            # ---------- FINAL ANSWER ----------
            if used_gk:
                final_answer = "⚠️ From general knowledge\n\n" + answer
            else:
                final_answer = answer
                if citations:
                    final_answer += "\n\n📚 Sources:\n" + "\n".join(citations)

            st.write(final_answer)
            save_message(st.session_state.current_session, "assistant", final_answer)

            # Retrieval Accuracy Metric
            accuracy = calc_retrieval_accuracy(docs)
            st.sidebar.metric("Retrieval Accuracy", f"{accuracy:.2f}%")

    st.rerun()

# ---------------- EXPORT CHAT TO PDF ----------------
st.divider()
st.subheader("📤 Export Chat")

if st.button("Download Chat as PDF"):
    history = load_chat(st.session_state.current_session)
    pdf_file = f"/tmp/chat_{st.session_state.current_session}.pdf"

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    content = []

    for role, msg in history:
        text = f"{role.upper()}: {msg}"
        content.append(Paragraph(text, styles["Normal"]))

    doc.build(content)

    with open(pdf_file, "rb") as f:
        st.download_button("Download PDF", f, file_name="chat.pdf")

# ---------------- FEEDBACK ----------------
st.divider()
st.subheader("📝 Feedback")

rating = st.radio("Rate", [1,2,3,4,5], horizontal=True)
issue = st.selectbox("Issue", ["None","Incorrect","Incomplete","Hallucinated","Formatting"])

if st.button("Submit Feedback"):
    history = load_chat(st.session_state.current_session)
    if len(history) >= 2:
        last_q = history[-2][1]
        last_a = history[-1][1]
        save_feedback(last_q, last_a, rating, issue, citations)
        st.success("Feedback saved")
