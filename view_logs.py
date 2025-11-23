import sqlite3
import os

db_path = 'chatbot_logs.db'
if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM logs")
        count = c.fetchone()[0]
        print(f"Total logs: {count}")
        c.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 5")  # Last 5 entries
        recent = c.fetchall()
        print("Recent logs:")
        for row in recent:
            print(row)
        conn.close()
    except Exception as e:
        print(f"DB error: {e}")
else:
    print("DB file not found.")