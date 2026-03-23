import streamlit as st
import psycopg2
import redis
import os
import pandas as pd
import time
from dotenv import load_dotenv
from config import config
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "plotly_white"

custom_colors = px.colors.qualitative.Set3
# ================= PAGE CONFIG =================
st.set_page_config(page_title="RAG Monitoring", layout="wide")
load_dotenv()

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.big-font {font-size:28px;font-weight:bold;}
.small-font {font-size:14px;color:gray;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Intelligent Document Q&A – Monitoring Dashboard")

# ================= DB CONNECTION =================
pg_conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
pg_conn.autocommit = True
pg = pg_conn.cursor()

redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# ================= KPI METRICS =================
pg.execute("SELECT COUNT(DISTINCT filename) FROM documents")
doc_count = pg.fetchone()[0]

pg.execute("SELECT COUNT(*) FROM chunks")
chunk_count = pg.fetchone()[0]

pg.execute("SELECT COUNT(*) FROM conversations WHERE role='user'")
query_count = pg.fetchone()[0]

pg.execute("SELECT COUNT(*) FROM feedback")
feedback_count = pg.fetchone()[0]

# ================= TOP KPI ROW =================
st.header("System Performance Metrics")
col1, col2, col3 = st.columns(3)

col1.markdown(f"<div class='card'><div class='small-font'>Documents</div><div class='big-font'>{doc_count}</div></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='card'><div class='small-font'>Chunks</div><div class='big-font'>{chunk_count}</div></div>", unsafe_allow_html=True)
#col3.markdown(f"<div class='card'><div class='small-font'>User Queries</div><div class='big-font'>{query_count}</div></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='card'><div class='small-font'>Feedback</div><div class='big-font'>{feedback_count}</div></div>", unsafe_allow_html=True)

# ================= VECTOR DB SIZE =================
db_size = 0
for root, dirs, files in os.walk(config.CHROMA_PERSIST_DIRECTORY):
    for f in files:
        db_size += os.path.getsize(os.path.join(root, f))

st.markdown(
    f"<div class='card'><div class='small-font'>Vector DB Size (MB)</div>"
    f"<div class='big-font'>{round(db_size/1024/1024,2)}</div></div>",
    unsafe_allow_html=True
)
## ================= SECOND ROW =================
st.header("Query Analytics")

st.markdown(f"<div class='card'><div class='small-font'>User Queries</div><div class='big-font'>{query_count}</div></div>", unsafe_allow_html=True)

# ================= MOST FREQUENT QUESTION =================
st.markdown("📑 Most Recently Asked Question")

pg.execute("""
    SELECT content,created_at
    FROM conversations
    WHERE role='user'
    ORDER BY created_at DESC
    LIMIT 10
""")

rows = pg.fetchall()

if rows:
    df_docss = pd.DataFrame(
        rows,
        columns=[ "content","created_at"]
    )

    st.dataframe(df_docss, use_container_width=True)

else:
    st.info("No documents found in the database.")

# ---- USER RATINGS ----
pg.execute("""
    SELECT rating, COUNT(*) 
    FROM feedback 
    GROUP BY rating 
    ORDER BY rating
    """)
ratings = pg.fetchall()
df_ratings = pd.DataFrame(ratings, columns=["Rating", "Count"])
with st.container(border=True):
    st.subheader("User Rating Distribution")

    if not df_ratings.empty:
        fig_rating = px.bar(
            df_ratings,
            x="Rating",
            y="Count",
            color="Rating",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_rating, use_container_width=True,key="rating_distribution")
    else:
        st.info("No feedback available.")




# ================= TOTAL QUERIES PROCESSED =================
pg.execute("""
    SELECT COUNT(*) 
    FROM conversations 
    WHERE role = 'user'
""")
total_queries = pg.fetchone()[0]

st.markdown(
    f"<div class='card'><div class='small-font'>Total Queries Processed</div>"
    f"<div class='big-font'>{total_queries}</div></div>",
    unsafe_allow_html=True
)
# ================= THIRD ROW =================
st.header("Document Analytics")
col1, col2 = st.columns(2)

# ---- MOST REFERENCED DOCS ----
with col1:
    pg.execute("""
        SELECT source, COUNT(*) 
        FROM chunk_references
        GROUP BY source 
        ORDER BY COUNT(*) DESC 
        LIMIT 5
    """)
    top_docs = pg.fetchall()
    df_top = pd.DataFrame(top_docs, columns=["Document", "References"])

    with st.container(border=True):
        st.subheader("Most Referenced Documents")

        if not df_top.empty:
            fig_bar = px.bar(
                df_top,
                x="Document",
                y="References",
                color="Document",
                color_discrete_sequence=custom_colors
)
            st.plotly_chart(fig_bar, use_container_width=True,key="top_documents_bar")
        else:
            st.info("No references found.")


# ---- DONUT CHART (Document Types) ----
with col2:
    pg.execute("SELECT file_type, COUNT(*) FROM documents GROUP BY file_type")
    doc_types = pg.fetchall()
    df_types = pd.DataFrame(doc_types, columns=["File Type", "Count"])

    with st.container(border=True):
        st.subheader("Documents by Type")

        if not df_types.empty:
            fig_donut = px.pie(
                df_types,
                names="File Type",
                values="Count",
                hole=0.6,
                color="File Type",
                color_discrete_sequence=custom_colors
            )
            st.plotly_chart(fig_donut, use_container_width=True,key="document_type_donut")
        else:
            st.info("No documents available.")


#DOCUMENT TABLE
import pandas as pd
import streamlit as st

st.markdown("## 📄 Documents Table")

# Fetch data from PostgreSQL
pg.execute("""
  SELECT DISTINCT ON (filename) filename, file_type, size_mb, uploaded_at
  FROM documents
  ORDER BY filename, uploaded_at DESC;
""")

rows = pg.fetchall()

if rows:
    df_docs = pd.DataFrame(
        rows,
        columns=[ "Filename", "File Type", "Size (MB)", "Uploaded At"]
    )

    st.dataframe(df_docs, use_container_width=True)

else:
    st.info("No documents found in the database.")
# ================= AVG ANSWER LENGTH =================
st.header("Answer Quality Metrics")
pg.execute("""
    SELECT AVG(LENGTH(content))
    FROM conversations
    WHERE role='assistant'
""")

avg_len = pg.fetchone()[0] or 0

st.markdown(
    f"<div class='card'><div class='small-font'>Avg Answer Length (chars)</div>"
    f"<div class='big-font'>{int(avg_len)}</div></div>",
    unsafe_allow_html=True
)

# ================= HALLUCINATION FLAGS =================

st.markdown("## 🚨 Hallucination Reports")

pg.execute("""
    SELECT COUNT(*)
    FROM feedback
    WHERE issue ILIKE '%hallucinated%'
""")

halluc_count = pg.fetchone()[0]

if halluc_count > 0:
    st.error(f"{halluc_count} hallucination cases reported!")
else:
    st.success("No hallucinations reported 🎉")


col1, col2 = st.columns(2)

# ----citation rate ----
with col1:
    pg.execute("SELECT COUNT(*) FROM conversations WHERE role='assistant' AND content LIKE '%Sources:%'")
    with_citations = pg.fetchone()[0]

    pg.execute("SELECT COUNT(*) FROM conversations WHERE role='assistant'")
    assistant_count = pg.fetchone()[0]

    citation_rate = (with_citations / max(assistant_count, 1)) * 100

    with st.container(border=True):
        st.subheader("Citation Rate %")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=citation_rate,
            gauge={'axis': {'range': [0,100]}}
        ))

        st.plotly_chart(fig_gauge, use_container_width=True,key="citation_rate")


# ---- CONFIDENCE SCORE DISTRIBUTION ----
with col2:

    with st.container(border=True):
        st.subheader("Confidence Score Distribution")

        pg.execute("""
            SELECT CAST(
                SUBSTRING(content FROM 'Confidence Score: ([0-9.]+)')
                AS FLOAT
            )
            FROM conversations
            WHERE role='assistant'
              AND content LIKE '%Confidence Score:%'
        """)

        scores = [row[0] for row in pg.fetchall() if row[0] is not None]

        if scores:
            df_conf = pd.DataFrame(scores, columns=["Confidence"])

            fig_conf = px.histogram(
                df_conf,
                x="Confidence",
                nbins=10,
                color_discrete_sequence=custom_colors
            )

            fig_conf.update_layout(
                xaxis_title="Confidence %",
                yaxis_title="Number of Responses"
            )

            st.plotly_chart(fig_conf, use_container_width=True,key="confidence_distribution_chart")
        else:
            st.info("No confidence scores found.")


st.caption("🔄 Auto refresh every 10 seconds")
time.sleep(10)
st.rerun()