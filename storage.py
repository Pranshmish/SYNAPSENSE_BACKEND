import sqlite3
import csv
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple

DB_PATH = "db/samples.db"
DATASET_DIR = "dataset"

# Main aggregated files for HOME/INTRUDER classification
HOME_CSV_PATH = os.path.join(DATASET_DIR, "HOME.csv")
INTRUDER_CSV_PATH = os.path.join(DATASET_DIR, "INTRUDER.csv")

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

    def save_sample_dual(self, person: str, features: Dict[str, Any]) -> Dict[str, bool]:
        """
        DUAL DATASET SAVING:
        1. Saves to main HOME.csv (for HOME/INTRUDER classification)
        2. Saves to individual HOME_{person}.csv (for person identification)
        
        ALL data is treated as HOME (INTRUDER is detected by MLP, not stored).
        
        Example: "Apurv" or "HOME_Apurv" → saves to:
          - HOME.csv (label='HOME')
          - HOME_Apurv/features_HOME_Apurv.csv (label='Apurv', class='HOME')
        
        Returns dict with save status for each file.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = {"main_csv": False, "individual_csv": False, "sqlite": False}
        
        # Extract filtered waveform if present (don't save to CSV)
        filtered_waveform = features.pop('_filtered_waveform', None)
        
        # ALWAYS treat as HOME - INTRUDER is detected by MLP, not stored
        main_class = 'HOME'
        
        # Normalize person label - extract name from any format
        # "INTRUDER_Apurv" -> "Apurv", "HOME_Apurv" -> "Apurv", "Apurv" -> "Apurv"
        individual_name = person
        if '_' in person:
            individual_name = person.split('_', 1)[1]
        
        # Final person label is always HOME_name format
        normalized_person = f"HOME_{individual_name}" if individual_name and individual_name.upper() != 'HOME' else 'HOME'
        
        # 1. Save to main aggregated CSV (always HOME.csv)
        main_csv_path = HOME_CSV_PATH
        try:
            features_with_meta = features.copy()
            features_with_meta['_label'] = main_class
            features_with_meta['_person'] = individual_name
            features_with_meta['_timestamp'] = timestamp
            
            file_exists = os.path.exists(main_csv_path)
            fieldnames = list(features_with_meta.keys())
            
            with open(main_csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(features_with_meta)
            result["main_csv"] = True
        except Exception as e:
            print(f"[STORAGE] Error saving to main CSV: {e}")
        
        # 2. Save to individual person CSV
        try:
            person_dir = os.path.join(DATASET_DIR, normalized_person)
            os.makedirs(person_dir, exist_ok=True)
            csv_path = os.path.join(person_dir, f"features_{normalized_person}.csv")
            
            features_individual = features.copy()
            features_individual['_label'] = individual_name
            features_individual['_class'] = main_class
            features_individual['_timestamp'] = timestamp
            
            file_exists = os.path.exists(csv_path)
            fieldnames = list(features_individual.keys())
            
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(features_individual)
            result["individual_csv"] = True
        except Exception as e:
            print(f"[STORAGE] Error saving individual CSV: {e}")
        
        # 3. Save to SQLite (always with HOME_ prefix)
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO samples (person, features_json, timestamp) VALUES (?, ?, ?)",
                      (normalized_person, json.dumps(features), timestamp))
            conn.commit()
            conn.close()
            result["sqlite"] = True
        except Exception as e:
            print(f"[STORAGE] Error saving to SQLite: {e}")
        
        # 4. Save waveform if present
        if filtered_waveform:
            self._save_waveform(person, filtered_waveform, timestamp)
        
        return result

    def save_sample(self, person: str, features: Dict[str, Any]):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_safe = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Extract filtered waveform if present
        filtered_waveform = features.pop('_filtered_waveform', None)
        
        # 1. Save to SQLite
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO samples (person, features_json, timestamp) VALUES (?, ?, ?)",
                  (person, json.dumps(features), timestamp))
        conn.commit()
        conn.close()
        
        # 2. Save to CSV (Features)
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
            
        # 3. Save Waveform (CSV + PNG)
        if filtered_waveform:
            # Save Waveform CSV
            wave_dir = os.path.join(person_dir, "waveforms")
            os.makedirs(wave_dir, exist_ok=True)
            wave_csv_path = os.path.join(wave_dir, f"wave_{timestamp_safe}.csv")
            
            try:
                import pandas as pd
                df_wave = pd.DataFrame({'amplitude': filtered_waveform})
                df_wave.to_csv(wave_csv_path, index=False)
            except Exception as e:
                print(f"[STORAGE] Error saving waveform CSV: {e}")
                
            # Save Waveform PNG
            plot_dir = os.path.join(person_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"plot_{timestamp_safe}.png")
            
            try:
                import matplotlib
                matplotlib.use('Agg') # Non-interactive backend
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(10, 4))
                plt.plot(filtered_waveform, color='green', linewidth=1.5)
                plt.title(f"Filtered Footstep Event - {person} ({timestamp})")
                plt.xlabel("Sample")
                plt.ylabel("Amplitude (Filtered)")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
            except Exception as e:
                print(f"[STORAGE] Error saving waveform PNG: {e}")

    def _save_waveform(self, person: str, filtered_waveform: List, timestamp: str):
        """Helper to save waveform data"""
        timestamp_safe = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        person_dir = os.path.join(DATASET_DIR, person)
        os.makedirs(person_dir, exist_ok=True)
        
        # Save Waveform CSV
        wave_dir = os.path.join(person_dir, "waveforms")
        os.makedirs(wave_dir, exist_ok=True)
        wave_csv_path = os.path.join(wave_dir, f"wave_{timestamp_safe}.csv")
        
        try:
            import pandas as pd
            df_wave = pd.DataFrame({'amplitude': filtered_waveform})
            df_wave.to_csv(wave_csv_path, index=False)
        except Exception as e:
            print(f"[STORAGE] Error saving waveform CSV: {e}")
        
        # Save Waveform PNG
        plot_dir = os.path.join(person_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"plot_{timestamp_safe}.png")
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 4))
            plt.plot(filtered_waveform, color='green', linewidth=1.5)
            plt.title(f"Filtered Footstep Event - {person} ({timestamp})")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude (Filtered)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
        except Exception as e:
            print(f"[STORAGE] Error saving waveform PNG: {e}")

    def get_dual_dataset_status(self) -> Dict[str, Any]:
        """Get status of dual dataset (HOME.csv and individual files)"""
        status = {
            "home_csv": {"exists": False, "samples": 0, "persons": []},
            "intruder_csv": {"exists": False, "samples": 0},
            "individual_files": [],
            "total_samples": 0,
            "target_samples": 150,
            "progress_percent": 0
        }
        
        # Check HOME.csv
        if os.path.exists(HOME_CSV_PATH):
            try:
                import pandas as pd
                df = pd.read_csv(HOME_CSV_PATH)
                status["home_csv"]["exists"] = True
                status["home_csv"]["samples"] = len(df)
                if '_person' in df.columns:
                    status["home_csv"]["persons"] = df['_person'].unique().tolist()
            except Exception as e:
                print(f"[STORAGE] Error reading HOME.csv: {e}")
        
        # Check INTRUDER.csv
        if os.path.exists(INTRUDER_CSV_PATH):
            try:
                import pandas as pd
                df = pd.read_csv(INTRUDER_CSV_PATH)
                status["intruder_csv"]["exists"] = True
                status["intruder_csv"]["samples"] = len(df)
            except Exception as e:
                print(f"[STORAGE] Error reading INTRUDER.csv: {e}")
        
        # Check individual files
        if os.path.exists(DATASET_DIR):
            for person_dir in os.listdir(DATASET_DIR):
                person_path = os.path.join(DATASET_DIR, person_dir)
                if os.path.isdir(person_path):
                    csv_path = os.path.join(person_path, f"features_{person_dir}.csv")
                    if os.path.exists(csv_path):
                        try:
                            import pandas as pd
                            df = pd.read_csv(csv_path)
                            status["individual_files"].append({
                                "name": person_dir,
                                "samples": len(df)
                            })
                        except:
                            pass
        
        status["total_samples"] = status["home_csv"]["samples"] + status["intruder_csv"]["samples"]
        if status["target_samples"] > 0:
            status["progress_percent"] = min(100, int(status["total_samples"] / status["target_samples"] * 100))
        
        return status

    def get_sample_counts(self) -> Dict[str, int]:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT person, COUNT(*) FROM samples GROUP BY person")
        rows = c.fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}

    def get_all_samples(self):
        """Get all samples with data and labels"""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT person, features_json FROM samples")
        rows = c.fetchall()
        conn.close()
        
        data = []
        labels = []
        for person, features_json in rows:
            features = json.loads(features_json)
            data.append(features)
            labels.append(person)
            
        return data, labels

    def get_all_samples_with_details(self):
        """
        Get all samples with detailed dataset information.
        Returns: (data, labels, dataset_details)
        """
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get samples
        c.execute("SELECT person, features_json FROM samples")
        rows = c.fetchall()
        
        # Get per-person counts
        c.execute("SELECT person, COUNT(*) FROM samples GROUP BY person ORDER BY person")
        person_counts = c.fetchall()
        
        conn.close()
        
        data = []
        labels = []
        for person, features_json in rows:
            features = json.loads(features_json)
            data.append(features)
            labels.append(person)
        
        # Build detailed dataset info
        dataset_details = {
            "total_samples": len(data),
            "datasets": [
                {"name": person, "samples": count} 
                for person, count in person_counts
            ],
            "dataset_names": [person for person, _ in person_counts]
        }
            
        return data, labels, dataset_details

    def get_samples_by_datasets(self, selected_datasets):
        """
        Get samples filtered by selected dataset names.
        
        Args:
            selected_datasets: List of dataset names to include (e.g., ['HOME_Apurv', 'HOME_Samir'])
        
        Returns: (data, labels, dataset_details)
        """
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Build SQL query with placeholders
        placeholders = ','.join('?' * len(selected_datasets))
        
        # Get samples for selected datasets only
        c.execute(f"SELECT person, features_json FROM samples WHERE person IN ({placeholders})", selected_datasets)
        rows = c.fetchall()
        
        # Get per-person counts for selected datasets
        c.execute(f"SELECT person, COUNT(*) FROM samples WHERE person IN ({placeholders}) GROUP BY person ORDER BY person", selected_datasets)
        person_counts = c.fetchall()
        
        conn.close()
        
        data = []
        labels = []
        for person, features_json in rows:
            features = json.loads(features_json)
            data.append(features)
            labels.append(person)
        
        # Build detailed dataset info
        dataset_details = {
            "total_samples": len(data),
            "datasets": [
                {"name": person, "samples": count} 
                for person, count in person_counts
            ],
            "dataset_names": [person for person, _ in person_counts],
            "selected_datasets": selected_datasets
        }
            
        return data, labels, dataset_details
