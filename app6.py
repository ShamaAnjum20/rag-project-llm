import json
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from ingest import process_and_store
from config import config

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

# ------------------ ENV ------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
load_dotenv()

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
    "Upload document (PDF / DOCX / TXT / MD / HTML / JSON / CSV)",
    type=["pdf", "docx", "txt", "md", "html", "htm", "json", "csv"]
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

# ------------------ CHAT HISTORY ------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ------------------ QUESTION ------------------
query = st.chat_input("Ask a question from your uploaded documents")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving answer..."):
            docs = vectordb.similarity_search_with_score(query, k=3)
            docs = [d for d in docs if d[1] < 2.0]

            citations = []
            used_general_knowledge = False

            # ---------- CASE 1: No relevant docs ----------
            if not docs:
                used_general_knowledge = True
                gk_prompt = [
                    SystemMessage(content="Answer clearly using your general knowledge."),
                    HumanMessage(content=query)
                ]
                response = model.invoke(gk_prompt)
                answer = response.content

            # ---------- CASE 2: Docs found ----------
            else:
                context = "\n\n".join(
                    f"[SOURCE]\n{d[0].page_content}"
                    for d in docs
                )

                system_prompt = (
                    "Answer ONLY from the given context. "
                    "If not present, say 'I don't know'. "
                    "Max 3 sentences."
                )

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{query}")
                ]

                response = model.invoke(messages)
                answer = response.content.strip()

                # If model still says "I don't know", switch to GK
                if answer.lower().startswith("i don't know"):
                    used_general_knowledge = True
                    gk_prompt = [
                        SystemMessage(content="Answer clearly using your general knowledge."),
                        HumanMessage(content=query)
                    ]
                    response = model.invoke(gk_prompt)
                    answer = response.content
                else:
                    # Build citations
                    for d, _ in docs:
                        meta = d.metadata
                        doc = meta.get("source", "Unknown document")
                        page = meta.get("page")
                        section = meta.get("section")

                        if page is not None:
                            ref = f"{doc} (page {page})"
                        elif section:
                            ref = f"{doc} (section: {section})"
                        else:
                            ref = f"{doc}"

                        citations.append(ref)

            citations = list(set(citations))

            # ---------- FINAL DISPLAY ----------
            if used_general_knowledge:
                answer_with_cite = (
                    "⚠️ *Uploaded document doesn’t contain this information, "
                    "so I’m answering from my general knowledge.*\n\n"
                    + answer
                )
            else:
                if citations:
                    answer_with_cite = answer + "\n\n📚 **Sources:**\n" + "\n".join(
                        f"- {c}" for c in citations
                    )
                else:
                    answer_with_cite = answer

            st.write(answer_with_cite)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer_with_cite}
            )

            st.session_state.qa_pairs.append({
                "question": query,
                "answer": answer,
                "sources": citations,
                "used_general_knowledge": used_general_knowledge
            })

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
