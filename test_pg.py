import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="academic_rag",
    user="postgres",
    password="shama"
)

print("Connected successfully!")
