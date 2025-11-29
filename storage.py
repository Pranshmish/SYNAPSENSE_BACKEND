import sqlite3
import csv
import os
import json
from datetime import datetime
from typing import List, Dict, Any

DB_PATH = "db/samples.db"
DATASET_DIR = "dataset"

class StorageManager:
    def __init__(self):
        self._init_db()
        self._init_dirs()

    def _init_dirs(self):
        os.makedirs("db", exist_ok=True)
        os.makedirs(DATASET_DIR, exist_ok=True)

    def _init_db(self):
        os.makedirs("db", exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person TEXT NOT NULL,
                features_json TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def save_sample(self, person: str, features: Dict[str, Any]):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Save to SQLite
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO samples (person, features_json, timestamp) VALUES (?, ?, ?)",
                  (person, json.dumps(features), timestamp))
        conn.commit()
        conn.close()
        
        # 2. Save to CSV
        person_dir = os.path.join(DATASET_DIR, person)
        os.makedirs(person_dir, exist_ok=True)
        csv_path = os.path.join(person_dir, f"features_{person}.csv")
        
        file_exists = os.path.exists(csv_path)
        
        # Ensure consistent field order
        fieldnames = list(features.keys())
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(features)

    def get_sample_counts(self) -> Dict[str, int]:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT person, COUNT(*) FROM samples GROUP BY person")
        rows = c.fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}

    def get_all_samples(self):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT person, features_json FROM samples")
        rows = c.fetchall()
        conn.close()
        
        data = []
        labels = []
        for person, features_json in rows:
            features = json.loads(features_json)
            # Convert dict values to list in consistent order
            # We assume the order is consistent or we enforce it.
            # Let's enforce the order based on a known list of keys from features.py logic
            # But here we might just return the dict and let ML handle it.
            # However, ML needs arrays.
            # Let's rely on the keys being consistent from the extractor.
            # Or better, extract values sorted by key or specific list.
            
            # We'll return dicts and let ML module handle conversion to array
            data.append(features)
            labels.append(person)
            
        return data, labels
