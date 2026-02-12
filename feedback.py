import csv
import os
from datetime import datetime
from config import config

def save_feedback(question, answer, rating, issue, sources):
    file_exists = os.path.isfile(config.FEEDBACK_FILE)

    with open(config.FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header only once
        if not file_exists:
            writer.writerow([
                "timestamp", "question", "answer", "rating", "issue", "sources"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            question,
            answer,
            rating,
            issue,
            ", ".join(sources)
        ])
