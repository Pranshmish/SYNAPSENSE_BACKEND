import requests
import numpy as np
import random
import time

BASE_URL = "http://localhost:8000"
PERSONS = ['Pranshul', 'Aditi', 'Apurv', 'Samir', 'Intruder']

def generate_chunk():
    # Generate a synthetic chunk that passes validation (energy > 2200, etc.)
    # A simple sine wave should work
    t = np.linspace(0, 1, 200)
    freq = random.uniform(10, 50)
    amp = random.uniform(100, 500)
    noise = np.random.normal(0, 10, 200)
    chunk = amp * np.sin(2 * np.pi * freq * t) + noise
    return chunk.tolist()

def test_intruder_flow():
    print("1. Resetting model...")
    resp = requests.post(f"{BASE_URL}/reset_model")
    if not resp.ok:
        print(f"Error: {resp.status_code} - {resp.text}")
    try:
        print(resp.json())
    except:
        print("JSON Decode Error. Raw text:", resp.text)

    print("\n2. Adding samples...")
    for person in PERSONS:
        print(f"  Adding data for {person}...")
        chunks = [{"raw_time_series": generate_chunk()} for _ in range(10)]
        payload = {
            "data": chunks,
            "label": person,
            "train_model": False
        }
        resp = requests.post(f"{BASE_URL}/train_data", json=payload)
        if not resp.ok:
            print(f"Failed to add data for {person}: {resp.status_code} - {resp.text}")
            continue
        try:
            # print(resp.json()) # Optional: print success
            pass
        except:
            print(f"JSON Error for {person}: {resp.text}")

    print("\n3. Training model...")
    # Send one more chunk and trigger train
    payload = {
        "data": [{"raw_time_series": generate_chunk()}],
        "label": "Pranshul",
        "train_model": True
    }
    resp = requests.post(f"{BASE_URL}/train_data", json=payload)
    if not resp.ok:
        print(f"Train Error: {resp.status_code} - {resp.text}")
    try:
        print("Train response:", resp.json())
    except:
        print("Train JSON Error:", resp.text)

    print("\n4. Testing Prediction...")
    test_chunk = generate_chunk()
    resp = requests.post(f"{BASE_URL}/predictfootsteps", json={"data": test_chunk})
    print("Predict response:", resp.json())
    
    print("\n5. Checking Status...")
    resp = requests.get(f"{BASE_URL}/status")
    print("Status response:", resp.json())

if __name__ == "__main__":
    test_intruder_flow()
