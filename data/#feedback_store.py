# feedback_store.py
import json
import os

FEEDBACK_PATH = "data/feedback.json"

def append_feedback(item):
    data = []
    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    data.append(item)
    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
