
import streamlit as st
import psycopg2
import redis
import os
import pandas as pd
import time
from dotenv import load_dotenv
from config import config
import plotly.express as px

# ================== ENV ==================
load_dotenv()

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="RAG Monitoring Dashboard", layout="wide")

# ================== CUSTOM CSS FOR COLORFUL CARDS ==================
st.markdown("""
<style>
.metric-box {
    padding: 20px;
    border-radius: 15px;
    color: white;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}
.bg-blue {background: linear-gradient(135deg,#1f77b4,#17becf);}
.bg-green {background: linear-gradient(135deg,#2ca02c,#98df8a);}
.bg-orange {background: linear-gradient(135deg,#ff7f0e,#ffbb78);}
.bg-red {background: linear-gradient(135deg,#d62728,#ff9896);}
.bg-purple {background: linear-gradient(135deg,#9467bd,#c5b0d5);}
.bg-black {background: linear-gradient(135deg,#111,#444);}
</style>
""", unsafe_allow_html=True)

st.title("📊 Intelligent Document Q&A System – Monitoring Dashboard")

# ================== DATABASE CONNECTIONS ==================
pg_conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
pg_conn.autocommit = True
pg = pg_conn.cursor()

redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# ================== KPI METRICS ==================
pg.execute("SELECT COUNT(*) FROM documents")
doc_count = pg.fetchone()[0]

pg.execute("SELECT COUNT(*) FROM chunks")
chunk_count = pg.fetchone()[0]

pg.execute("SELECT COUNT(*) FROM conversations WHERE role='user'")
query_count = pg.fetchone()[0]

pg.execute("SELECT COUNT(*) FROM feedback")
feedback_count = pg.fetchone()[0]

# ================== COLORFUL METRIC CARDS ==================
st.markdown("## ⚙️ System Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"<div class='metric-box bg-blue'>📄 Documents<br>{doc_count}</div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<div class='metric-box bg-green'>🧩 Chunks<br>{chunk_count}</div>", unsafe_allow_html=True)

with col3:
    st.markdown(f"<div class='metric-box bg-orange'>❓ Queries<br>{query_count}</div>", unsafe_allow_html=True)

with col4:
    st.markdown(f"<div class='metric-box bg-red'>⭐ Feedback<br>{feedback_count}</div>", unsafe_allow_html=True)

# ================== VECTOR DB SIZE ==================
db_size = 0
for root, dirs, files in os.walk(config.CHROMA_PERSIST_DIRECTORY):
    for f in files:
        db_size += os.path.getsize(os.path.join(root, f))

st.markdown(f"<div class='metric-box bg-purple'>📦 Vector DB Size (MB)<br>{round(db_size/1024/1024,2)}</div>", unsafe_allow_html=True)

# ================== REDIS STATUS ==================
redis_info = redis_client.info()
redis_keys = redis_info.get("db0", {}).get("keys", 0)
st.markdown(f"<div class='metric-box bg-black'>⚡ Redis Cached Keys<br>{redis_keys}</div>", unsafe_allow_html=True)

# ================== DOCUMENT ANALYTICS ==================
st.markdown("## 📚 Document Analytics")

pg.execute("SELECT file_type, COUNT(*) FROM documents GROUP BY file_type")
doc_types = pg.fetchall()
df_types = pd.DataFrame(doc_types, columns=["File Type", "Count"])

if not df_types.empty:
    fig_pie = px.pie(df_types, names="File Type", values="Count", title="Documents by File Type")
    st.plotly_chart(fig_pie, use_container_width=True)

# ================== MOST REFERENCED DOCUMENTS ==================
pg.execute("""
SELECT source, COUNT(*) FROM chunk_references 
GROUP BY source ORDER BY COUNT(*) DESC LIMIT 5
""")
top_docs = pg.fetchall()
df_top = pd.DataFrame(top_docs, columns=["Document", "References"])

if not df_top.empty:
    fig_bar = px.bar(df_top, x="Document", y="References", title="Most Referenced Documents", color="References")
    st.plotly_chart(fig_bar, use_container_width=True)

# ================== USER RATING DISTRIBUTION ==================
pg.execute("SELECT rating, COUNT(*) FROM feedback GROUP BY rating ORDER BY rating")
ratings = pg.fetchall()
df_ratings = pd.DataFrame(ratings, columns=["Rating", "Count"])

if not df_ratings.empty:
    fig_rating = px.bar(df_ratings, x="Rating", y="Count", title="User Feedback Ratings", color="Count")
    st.plotly_chart(fig_rating, use_container_width=True)

# ================== ANSWER QUALITY ==================
st.markdown("## 🧠 Answer Quality Metrics")

pg.execute("SELECT AVG(LENGTH(content)) FROM conversations WHERE role='assistant'")
avg_len = pg.fetchone()[0] or 0

pg.execute("SELECT COUNT(*) FROM conversations WHERE role='assistant' AND content LIKE '%Sources:%'")
with_citations = pg.fetchone()[0]
citation_rate = (with_citations / max(query_count, 1)) * 100

col1, col2 = st.columns(2)
col1.markdown(f"<div class='metric-box bg-green'>📏 Avg Answer Length<br>{int(avg_len)}</div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-box bg-blue'>📚 Citation Rate (%)<br>{round(citation_rate,2)}</div>", unsafe_allow_html=True)

# ================== HALLUCINATION REPORTS ==================
pg.execute("SELECT COUNT(*) FROM feedback WHERE issue='Hallucinated'")
hallucinations = pg.fetchone()[0]
st.markdown(f"<div class='metric-box bg-red'>🚨 Hallucinations<br>{hallucinations}</div>", unsafe_allow_html=True)

# ================== LIVE SYSTEM HEALTH ==================
st.markdown("## 💻 Live System Health")

# Redis latency
start = time.time()
redis_client.ping()
redis_latency = (time.time() - start) * 1000

# PostgreSQL latency
start = time.time()
pg.execute("SELECT 1")
pg.fetchone()
pg_latency = (time.time() - start) * 1000

col1, col2 = st.columns(2)
col1.markdown(f"<div class='metric-box bg-purple'>⚡ Redis Latency (ms)<br>{round(redis_latency,2)}</div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-box bg-orange'>🐘 PostgreSQL Latency (ms)<br>{round(pg_latency,2)}</div>", unsafe_allow_html=True)

# ================== AUTO REFRESH ==================
st.caption("🔄 Dashboard refreshes every 10 seconds")
time.sleep(10)
st.rerun()