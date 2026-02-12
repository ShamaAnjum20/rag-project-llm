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
import redis
import psycopg2

# ------------------ ENV ------------------
load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ------------------ REDIS ------------------
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

# ------------------ POSTGRES ------------------
pg_conn = psycopg2.connect(
    host=os.getenv("PG_HOST", "localhost"),
    database=os.getenv("PG_DB", "chatdb"),
    user=os.getenv("PG_USER", "shakirahamedk"),
    password=os.getenv("PG_PASSWORD", "postgres"),
)
pg_conn.autocommit = True
pg_cur = pg_conn.cursor()
pg_cur.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    role TEXT,
    content TEXT,
    created_at TIMESTAMP
)
""")

# ------------------ SESSION ID ------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
SESSION_ID = st.session_state.session_id

# ------------------ SESSION STATE ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_pairs" not in st.session_state:
    st.session_state.qa_pairs = []
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# ------------------ HELPERS ------------------
def save_to_redis(role, content):
    key = f"chat:{SESSION_ID}"
    redis_client.rpush(key, json.dumps({
        "role": role,
        "content": content,
        "time": datetime.now().isoformat()
    }))


def save_to_postgres(role, content):
    pg_cur.execute(
        """
        INSERT INTO conversations (id, session_id, role, content, created_at)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            str(uuid.uuid4()),   # MUST be here
            SESSION_ID,
            role,
            content,
            datetime.now()
        )
    )


def load_history_from_redis():
    key = f"chat:{SESSION_ID}"
    data = redis_client.lrange(key, 0, -1)
    return [json.loads(x) for x in data]

# ------------------ UI ------------------
st.set_page_config(page_title="Academic QA", layout="centered")
st.title("📄 Academic RAG System")

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
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, max_tokens=400)

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

# ------------------ LOAD HISTORY ------------------
history = load_history_from_redis()
for msg in history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ------------------ QUESTION ------------------
query = st.chat_input("Ask a question from your uploaded documents")

if query:
    st.session_state.messages.append({"role":"user","content":query})
    save_to_redis("user", query)
    save_to_postgres("user", query)

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving answer..."):
            docs = vectordb.similarity_search_with_score(query, k=3)
            docs = [d for d in docs if d[1] < 2.0]
            citations = []
            used_general_knowledge = False

            if not docs:
                used_general_knowledge = True
                gk_prompt = [
                    SystemMessage(content="Answer clearly using your general knowledge."),
                    HumanMessage(content=query)
                ]
                answer = model.invoke(gk_prompt).content
            else:
                context = "\n\n".join(f"[SOURCE]\n{d[0].page_content}" for d in docs)
                system_prompt = (
                    "Answer ONLY from the given context. "
                    "If not present, say 'I don't know'. "
                    "Max 3 sentences."
                )
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{query}")
                ]
                answer = model.invoke(messages).content.strip()
                if answer.lower().startswith("i don't know"):
                    used_general_knowledge = True
                    gk_prompt = [
                        SystemMessage(content="Answer clearly using your general knowledge."),
                        HumanMessage(content=query)
                    ]
                    answer = model.invoke(gk_prompt).content
                else:
                    for d,_ in docs:
                        meta = d.metadata
                        doc = meta.get("source","Unknown")
                        page = meta.get("page")
                        section = meta.get("section")
                        if page:
                            ref = f"{doc} (page {page})"
                        elif section:
                            ref = f"{doc} (section: {section})"
                        else:
                            ref = doc
                        citations.append(ref)

            citations = list(set(citations))

            if used_general_knowledge:
                answer_with_cite = (
                    "⚠️ Uploaded document doesn’t contain this information, "
                    "so I’m answering from my general knowledge.\n\n" + answer
                )
            else:
                answer_with_cite = answer
                if citations:
                    answer_with_cite += "\n\n📚 Sources:\n" + "\n".join(f"- {c}" for c in citations)

            st.write(answer_with_cite)
            save_to_redis("assistant", answer_with_cite)
            save_to_postgres("assistant", answer_with_cite)

            st.session_state.last_question = query
            st.session_state.last_answer = answer
            st.session_state.last_sources = citations

# ------------------ FEEDBACK ------------------
if st.session_state.last_question:
    st.divider()
    st.subheader("📝 Feedback for Last Answer")
    rating = st.radio("Rate this answer", [1,2,3,4,5], horizontal=True)
    issue = st.selectbox("Issue", ["None","Incorrect","Incomplete","Hallucinated","Formatting"])

    if st.button("Submit Feedback"):
        from feedback import save_feedback
        save_feedback(
            st.session_state.last_question,
            st.session_state.last_answer,
            rating,
            issue,
            st.session_state.last_sources
        )
        st.success("Feedback saved!")

# ------------------ EXPORT ------------------
st.subheader("📤 Export Knowledge")
st.download_button(
    "Download Q&A as JSON",
    data=json.dumps(st.session_state.qa_pairs, indent=2),
    file_name="knowledge_base.json",
    mime="application/json"
)
