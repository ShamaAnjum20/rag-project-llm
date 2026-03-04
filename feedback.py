import psycopg2
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
pg_conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
pg_conn.autocommit = True
pg = pg_conn.cursor()

def save_feedback(question, answer, rating, issue, citations):
    pg.execute("""
        INSERT INTO feedback (rating, issue, created_at)
        VALUES (%s,%s,%s)
    """, (rating, issue, datetime.now()))
