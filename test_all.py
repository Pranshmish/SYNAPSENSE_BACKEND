import requests
import numpy as np
import random
import time

BASE_URL = "http://localhost:8000"
PERSONS = ['Pranshul', 'Aditi', 'Apurv', 'Samir', 'Intruder']

def generate_weak_chunk():
    # Generate a weak chunk that should pass with reduced thresholds
    # Energy > 1540 (e.g. 1800)
    # Std > 4.2
    t = np.linspace(0, 1, 200)
    freq = random.uniform(10, 50)
    amp = 150 # Reduced amplitude
    noise = np.random.normal(0, 2, 200)
    chunk = amp * np.sin(2 * np.pi * freq * t) + noise
    return chunk.tolist()

def test_full_pipeline():
    print("=== TESTING FULL PIPELINE WITH REDUCED THRESHOLDS ===")
    
    # 1. Reset
    print("\n1. Resetting model...")
    resp = requests.post(f"{BASE_URL}/reset_model")
    if not resp.ok:
        print(f"Reset Failed: {resp.text}")
        return
    print("Reset OK")

    # 2. Add Data (Weak Signals)
    print("\n2. Adding WEAK samples (should be accepted)...")
    total_added = 0
    for person in PERSONS:
        print(f"  Adding data for {person}...")
        chunks = [{"raw_time_series": generate_weak_chunk()} for _ in range(5)]
        payload = {
            "data": chunks,
            "label": person,
            "train_model": False
        }
        resp = requests.post(f"{BASE_URL}/train_data", json=payload)
        if resp.ok:
            data = resp.json()
            count = data['samples_per_person'].get(person, 0)
            print(f"  -> Success! Count: {count}")
            total_added += count
        else:
            print(f"  -> Failed: {resp.text}")

    # 3. Verify Dataset Endpoint
    print("\n3. Verifying /dataset endpoint...")
    resp = requests.get(f"{BASE_URL}/dataset")
    if resp.ok:
        print("Dataset Status:", resp.json())
    else:
        print("Dataset Status Failed:", resp.text)

    # 4. Train
    print("\n4. Training model...")
    payload = {
        "data": [{"raw_time_series": generate_weak_chunk()}],
        "label": "Pranshul",
        "train_model": True
    }
    resp = requests.post(f"{BASE_URL}/train_data", json=payload)
    if resp.ok:
        print("Train Result:", resp.json().get('metrics'))
    else:
        print("Train Failed:", resp.text)

    # 5. Predict
    print("\n5. Predicting...")
    chunk = generate_weak_chunk()
    resp = requests.post(f"{BASE_URL}/predictfootsteps", json={"data": chunk})
    if resp.ok:
        print("Prediction:", resp.json())
    else:
        print("Prediction Failed:", resp.text)

    # 6. Status
    print("\n6. Final Status...")
    resp = requests.get(f"{BASE_URL}/status")
    print("Status:", resp.json())

if __name__ == "__main__":
    test_full_pipeline()
