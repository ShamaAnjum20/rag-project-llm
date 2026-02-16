# monitoring_dashboard.py
import streamlit as st
import psycopg2
import redis
import time
import pandas as pd
import os
from dotenv import load_dotenv

# ---------------- ENV ----------------
load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Monitoring Dashboard", layout="wide")
st.title("📊 Academic RAG Monitoring Dashboard")

# ---------------- POSTGRES ----------------
pg_conn = psycopg2.connect(
    host="localhost",
    database="academic_rag",
    user="postgres",
    password="shama"
)
pg_conn.autocommit = True
pg = pg_conn.cursor()

# ---------------- REDIS ----------------
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# =========================================================
# 📌 SYSTEM PERFORMANCE METRICS
# =========================================================
st.header("⚙️ System Performance Metrics")

# Number of documents and chunks
pg.execute("SELECT COUNT(*) FROM documents")
doc_count = pg.fetchone()[0]

pg.execute("SELECT COUNT(*) FROM chunks")
chunk_count = pg.fetchone()[0]

st.metric("📄 Total Documents", doc_count)
st.metric("🧩 Total Chunks", chunk_count)

# Vector DB size
chroma_path = "chroma_db"
db_size = 0
for root, dirs, files in os.walk(chroma_path):
    for f in files:
        db_size += os.path.getsize(os.path.join(root, f))
st.metric("📦 Vector DB Size (MB)", round(db_size / 1024 / 1024, 2))

# Redis Cache Stats
redis_info = redis_client.info()
st.metric("⚡ Redis Keys Cached", redis_info.get("db0", {}).get("keys", 0))

# =========================================================
# 📌 QUERY ANALYTICS
# =========================================================
st.header("📈 Query Analytics")

pg.execute("SELECT COUNT(*) FROM conversations WHERE role='user'")
total_queries = pg.fetchone()[0]
st.metric("Total Queries", total_queries)

# Most frequent questions
pg.execute("""
SELECT content, COUNT(*) 
FROM conversations 
WHERE role='user'
GROUP BY content ORDER BY COUNT(*) DESC LIMIT 5
""")
faq = pg.fetchall()
df_faq = pd.DataFrame(faq, columns=["Question", "Frequency"])
st.subheader("🔥 Most Frequent Questions")
st.dataframe(df_faq)

# Low confidence answers (general knowledge flagged)
pg.execute("""
SELECT COUNT(*) FROM conversations 
WHERE role='assistant' AND content LIKE '%From general knowledge%'
""")
low_conf = pg.fetchone()[0]
st.metric("⚠️ Low Confidence Answers", low_conf)

# =========================================================
# 📌 DOCUMENT ANALYTICS
# =========================================================
st.header("📚 Document Analytics")

pg.execute("""
SELECT file_type, COUNT(*) 
FROM documents GROUP BY file_type
""")
docs_type = pg.fetchall()
df_type = pd.DataFrame(docs_type, columns=["Type", "Count"])
st.subheader("Documents by Type")
st.bar_chart(df_type.set_index("Type"))

# Most referenced documents
pg.execute("""
SELECT source, COUNT(*) 
FROM chunk_references 
GROUP BY source ORDER BY COUNT(*) DESC LIMIT 5
""")
top_docs = pg.fetchall()
st.subheader("Most Referenced Documents")
st.dataframe(pd.DataFrame(top_docs, columns=["Document", "References"]))

# =========================================================
# 📌 ANSWER QUALITY METRICS
# =========================================================
st.header("🧠 Answer Quality Metrics")

# Average answer length
pg.execute("""
SELECT AVG(LENGTH(content)) FROM conversations WHERE role='assistant'
""")
avg_len = pg.fetchone()[0]
st.metric("📏 Avg Answer Length (chars)", int(avg_len or 0))

# Citation rate
pg.execute("""
SELECT COUNT(*) FROM conversations 
WHERE role='assistant' AND content LIKE '%Sources:%'
""")
with_citations = pg.fetchone()[0]
citation_rate = (with_citations / max(total_queries, 1)) * 100
st.metric("📚 Citation Rate (%)", round(citation_rate, 2))

# User feedback stats
pg.execute("""
SELECT rating, COUNT(*) FROM feedback GROUP BY rating
""")
ratings = pg.fetchall()
df_ratings = pd.DataFrame(ratings, columns=["Rating", "Count"])
st.subheader("⭐ User Ratings")
st.bar_chart(df_ratings.set_index("Rating"))

# Hallucination detection flags
pg.execute("""
SELECT COUNT(*) FROM feedback WHERE issue='Hallucinated'
""")
hallucinations = pg.fetchone()[0]
st.metric("🚨 Hallucination Reports", hallucinations)

# =========================================================
# 📌 LIVE SYSTEM HEALTH
# =========================================================
st.header("💻 Live System Health")

start = time.time()
redis_client.ping()
redis_latency = (time.time() - start) * 1000

st.metric("Redis Latency (ms)", round(redis_latency, 2))

start = time.time()
pg.execute("SELECT 1")
pg.fetchone()
pg_latency = (time.time() - start) * 1000
st.metric("PostgreSQL Latency (ms)", round(pg_latency, 2))

# =========================================================
# 📌 AUTO REFRESH
# =========================================================
st.caption("Dashboard refreshes every 10 seconds")
st.rerun()
