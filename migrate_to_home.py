"""
Migration script to convert all dataset labels to HOME_label format.
This script:
1. Renames all person labels in SQLite from INTRUDER_x to HOME_x
2. Renames dataset directories from INTRUDER_x to HOME_x
3. Renames CSV files inside directories
4. Updates HOME.csv to include all samples

Run this script once to fix existing data.
"""

import sqlite3
import os
import shutil
import csv
import json
from datetime import datetime

# Paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BACKEND_DIR, 'db', 'samples.db')
DATASET_DIR = os.path.join(BACKEND_DIR, 'dataset')
HOME_CSV_PATH = os.path.join(DATASET_DIR, 'HOME.csv')
INTRUDER_CSV_PATH = os.path.join(DATASET_DIR, 'INTRUDER.csv')

def migrate_sqlite():
    """Update all person labels in SQLite to HOME_name format."""
    print("\n=== Migrating SQLite Database ===")
    
    if not os.path.exists(DB_PATH):
        print("Database not found, skipping SQLite migration.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get all unique person labels
    c.execute("SELECT DISTINCT person FROM samples")
    persons = [row[0] for row in c.fetchall()]
    print(f"Found {len(persons)} unique labels: {persons}")
    
    changes = 0
    for person in persons:
        # Normalize: extract name and add HOME_ prefix
        if '_' in person:
            name = person.split('_', 1)[1]
        else:
            name = person
        
        # Skip if already correct format
        new_label = f"HOME_{name}" if name.upper() != 'HOME' else 'HOME'
        
        if person != new_label:
            print(f"  Renaming: '{person}' → '{new_label}'")
            c.execute("UPDATE samples SET person = ? WHERE person = ?", (new_label, person))
            changes += 1
    
    conn.commit()
    
    # Verify changes
    c.execute("SELECT DISTINCT person FROM samples")
    new_persons = [row[0] for row in c.fetchall()]
    print(f"After migration: {new_persons}")
    print(f"Total label changes: {changes}")
    
    conn.close()

def migrate_directories():
    """Rename INTRUDER_x directories to HOME_x."""
    print("\n=== Migrating Dataset Directories ===")
    
    if not os.path.exists(DATASET_DIR):
        print("Dataset directory not found, skipping.")
        return
    
    changes = 0
    for item in os.listdir(DATASET_DIR):
        item_path = os.path.join(DATASET_DIR, item)
        
        if not os.path.isdir(item_path):
            continue
        
        # Check if it's an INTRUDER directory
        if item.upper().startswith('INTRUDER_'):
            name = item.split('_', 1)[1]
            new_name = f"HOME_{name}"
            new_path = os.path.join(DATASET_DIR, new_name)
            
            if os.path.exists(new_path):
                # Merge contents
                print(f"  Merging '{item}' into existing '{new_name}'")
                for file in os.listdir(item_path):
                    src = os.path.join(item_path, file)
                    dst = os.path.join(new_path, file.replace('INTRUDER', 'HOME'))
                    if os.path.isfile(src):
                        # Append CSV contents instead of overwriting
                        if file.endswith('.csv') and os.path.exists(dst):
                            with open(src, 'r') as f_src:
                                reader = csv.DictReader(f_src)
                                rows = list(reader)
                            with open(dst, 'a', newline='') as f_dst:
                                if rows:
                                    writer = csv.DictWriter(f_dst, fieldnames=rows[0].keys())
                                    writer.writerows(rows)
                        else:
                            shutil.copy2(src, dst)
                shutil.rmtree(item_path)
            else:
                print(f"  Renaming '{item}' → '{new_name}'")
                os.rename(item_path, new_path)
                # Also rename CSV file inside
                old_csv = os.path.join(new_path, f"features_{item}.csv")
                new_csv = os.path.join(new_path, f"features_{new_name}.csv")
                if os.path.exists(old_csv):
                    os.rename(old_csv, new_csv)
            
            changes += 1
    
    print(f"Total directory changes: {changes}")

def merge_intruder_csv():
    """Merge INTRUDER.csv into HOME.csv and delete it."""
    print("\n=== Merging INTRUDER.csv into HOME.csv ===")
    
    if not os.path.exists(INTRUDER_CSV_PATH):
        print("No INTRUDER.csv found, skipping.")
        return
    
    # Read INTRUDER.csv
    intruder_rows = []
    with open(INTRUDER_CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Update labels
            if '_label' in row:
                row['_label'] = 'HOME'
            if '_class' in row:
                row['_class'] = 'HOME'
            intruder_rows.append(row)
    
    print(f"Found {len(intruder_rows)} samples in INTRUDER.csv")
    
    if intruder_rows:
        # Append to HOME.csv
        file_exists = os.path.exists(HOME_CSV_PATH)
        
        with open(HOME_CSV_PATH, 'a', newline='') as f:
            fieldnames = intruder_rows[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(intruder_rows)
        
        print(f"Appended {len(intruder_rows)} samples to HOME.csv")
    
    # Backup and remove INTRUDER.csv
    backup_path = INTRUDER_CSV_PATH + '.backup'
    shutil.move(INTRUDER_CSV_PATH, backup_path)
    print(f"Moved INTRUDER.csv to {backup_path}")

def show_summary():
    """Show final data summary."""
    print("\n=== Final Summary ===")
    
    # SQLite summary
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT person, COUNT(*) FROM samples GROUP BY person ORDER BY person")
        rows = c.fetchall()
        print("\nSQLite samples per person:")
        for person, count in rows:
            print(f"  {person}: {count} samples")
        conn.close()
    
    # Directory summary
    print("\nDataset directories:")
    if os.path.exists(DATASET_DIR):
        for item in sorted(os.listdir(DATASET_DIR)):
            item_path = os.path.join(DATASET_DIR, item)
            if os.path.isdir(item_path):
                csv_count = len([f for f in os.listdir(item_path) if f.endswith('.csv')])
                print(f"  {item}/ ({csv_count} CSV files)")

def main():
    print("=" * 60)
    print("MIGRATION: Converting all datasets to HOME_label format")
    print("=" * 60)
    print(f"\nDatabase: {DB_PATH}")
    print(f"Dataset dir: {DATASET_DIR}")
    
    # Confirm
    response = input("\nProceed with migration? (y/n): ")
    if response.lower() != 'y':
        print("Migration cancelled.")
        return
    
    # Run migrations
    migrate_sqlite()
    migrate_directories()
    merge_intruder_csv()
    show_summary()
    
    print("\n" + "=" * 60)
    print("Migration complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
