import json
import os
import uuid
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from ingest import process_and_store
from config import config
from feedback import save_feedback

# ------------------ ENV ------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
load_dotenv()

# ------------------ CONSTANTS ------------------
CHAT_STORE_FILE = "chat_store.json"

# ------------------ HELPERS ------------------
def load_chats():
    if os.path.exists(CHAT_STORE_FILE):
        with open(CHAT_STORE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_chats():
    with open(CHAT_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chats, f, indent=2)

# ------------------ PAGE ------------------
st.set_page_config(page_title="Academic RAG", layout="wide")
st.title("📄 Academic RAG System")

# ------------------ SESSION STATE ------------------
if "chats" not in st.session_state:
    st.session_state.chats = load_chats()

if "active_chat" not in st.session_state:
    st.session_state.active_chat = None

def new_chat():
    chat_id = str(uuid.uuid4())[:8]
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "qa_pairs": [],
        "last_question": None,
        "last_answer": None,
        "last_sources": []
    }
    st.session_state.active_chat = chat_id
    save_chats()

if not st.session_state.chats:
    new_chat()

if st.session_state.active_chat is None:
    st.session_state.active_chat = list(st.session_state.chats.keys())[0]

chat = st.session_state.chats[st.session_state.active_chat]

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("💬 Chats")

    for cid, cdata in st.session_state.chats.items():
        label = cdata["title"]
        if st.button(label, key=cid):
            st.session_state.active_chat = cid
            st.rerun()

    st.divider()

    if st.button("➕ New Chat"):
        new_chat()
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
    type=["pdf", "docx", "txt", "md", "html", "json", "csv"]
)

if uploaded_file:
    save_path = os.path.join(config.PDF_SOURCE_DIRECTORY, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing document..."):
        chunks = process_and_store(save_path)

    vectordb = load_vectordb()
    st.success(f"{chunks} chunks added to knowledge base")

st.divider()

# ------------------ CHAT HISTORY ------------------
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ------------------ QUESTION ------------------
query = st.chat_input("Ask a question from your documents")

if query:
    if chat["title"] == "New Chat":
        chat["title"] = query[:30] + ("..." if len(query) > 30 else "")

    chat["messages"].append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = vectordb.similarity_search_with_score(query, k=3)
            docs = [d for d in docs if d[1] < 2.0]

            citations = []
            used_general_knowledge = False

            if not docs:
                used_general_knowledge = True
                response = llm.invoke([
                    SystemMessage(content="Answer clearly using your general knowledge."),
                    HumanMessage(content=query)
                ])
                answer = response.content
            else:
                context = "\n\n".join(
                    f"[SOURCE]\n{d[0].page_content}" for d in docs
                )

                response = llm.invoke([
                    SystemMessage(
                        content="Answer ONLY from the context. If not found, say 'I don't know'."
                    ),
                    HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{query}")
                ])

                answer = response.content.strip()

                if answer.lower().startswith("i don't know"):
                    used_general_knowledge = True
                    response = llm.invoke([
                        SystemMessage(content="Answer clearly using your general knowledge."),
                        HumanMessage(content=query)
                    ])
                    answer = response.content
                else:
                    for d, _ in docs:
                        meta = d.metadata
                        src = meta.get("source", "Unknown")
                        page = meta.get("page")
                        citations.append(f"{src} (page {page})" if page else src)

            citations = list(set(citations))

            if used_general_knowledge:
                final_answer = (
                    "⚠️ *Not found in uploaded documents. "
                    "Answering from general knowledge.*\n\n" + answer
                )
            else:
                final_answer = answer
                if citations:
                    final_answer += "\n\n📚 **Sources:**\n" + "\n".join(
                        f"- {c}" for c in citations
                    )

            st.write(final_answer)

    chat["messages"].append({"role": "assistant", "content": final_answer})
    chat["qa_pairs"].append({
        "question": query,
        "answer": answer,
        "sources": citations,
        "used_general_knowledge": used_general_knowledge
    })

    chat["last_question"] = query
    chat["last_answer"] = answer
    chat["last_sources"] = citations

    save_chats()

# ------------------ FEEDBACK ------------------
if chat["last_question"]:
    st.divider()
    st.subheader("📝 Feedback")

    rating = st.radio("Rate answer", [1,2,3,4,5], horizontal=True)
    issue = st.selectbox(
        "Issue",
        ["None", "Incorrect", "Incomplete", "Hallucinated", "Formatting"]
    )

    if st.button("Submit Feedback"):
        save_feedback(
            chat["last_question"],
            chat["last_answer"],
            rating,
            issue,
            chat["last_sources"]
        )
        st.success("Feedback saved")

# ------------------ EXPORT ------------------
st.divider()
st.subheader("📤 Export")

st.download_button(
    "Download Q&A as JSON",
    data=json.dumps(chat["qa_pairs"], indent=2),
    file_name="knowledge_base.json",
    mime="application/json"
)
