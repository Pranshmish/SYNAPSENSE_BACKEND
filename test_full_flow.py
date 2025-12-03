import requests
import numpy as np
import random
import time
import json

BASE_URL = "http://localhost:8000"
SAMPLE_RATE = 200
DURATION = 1.0 # 200 samples

def generate_synthetic_footstep(amplitude=2000, freq=15, noise_level=5, baseline=2048):
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))
    # Damped sine wave
    envelope = np.exp(-5 * t)
    signal = amplitude * np.sin(2 * np.pi * freq * t) * envelope
    # Add noise
    noise = np.random.normal(0, noise_level, len(t))
    return (signal + noise + baseline).tolist()

def run_test():
    print("🚀 STARTING FULL SYSTEM TEST 🚀")
    print("=================================")

    # 1. Reset Model
    print("\n[1] Resetting System...")
    try:
        resp = requests.post(f"{BASE_URL}/reset_model")
        if resp.status_code == 200:
            print("✅ Reset successful")
        else:
            print(f"❌ Reset failed: {resp.text}")
            return
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # 2. Generate and Upload Training Data
    print("\n[2] Uploading Training Data...")
    
    # Pranshul: 14Hz (was 12)
    # Aditi: 16Hz (was 25) - Closer frequencies to make classification harder
    persons = {
        'Pranshul': {'amp': 2500, 'freq': 14},
        'Aditi': {'amp': 2200, 'freq': 16}
    }
    
    samples_per_person = 30 # Increase samples for better CV
    
    for name, params in persons.items():
        print(f"   -> Generating {samples_per_person} samples for {name}...")
        chunks = []
        for _ in range(samples_per_person):
            # Add significant variance and noise
            amp = params['amp'] + random.uniform(-300, 300)
            freq = params['freq'] + random.uniform(-2, 2) # Overlapping frequencies
            
            # High noise level to prevent 100% accuracy
            chunk = generate_synthetic_footstep(amplitude=amp, freq=freq, noise_level=50)
            chunks.append({"raw_time_series": chunk})
            
        payload = {
            "data": chunks,
            "label": name,
            "train_model": False
        }
        
        resp = requests.post(f"{BASE_URL}/train_data", json=payload)
        if resp.status_code == 200:
            print(f"   ✅ Uploaded {name} data")
        else:
            print(f"   ❌ Failed to upload {name}: {resp.text}")

    # 3. Train Model
    print("\n[3] Training Model...")
    # Send one dummy sample to trigger training or use explicit train endpoint if available
    # We'll use the explicit train endpoint if it exists, otherwise train_data with train_model=True
    
    # Let's try explicit /train endpoint first (from main.py code)
    resp = requests.post(f"{BASE_URL}/train")
    if resp.status_code == 200:
        result = resp.json()
        acc = result.get('metrics', {}).get('training_accuracy', 0)
        print(f"✅ Training successful! Accuracy: {acc}%")
        print(f"   Classes: {result.get('metrics', {}).get('classes')}")
    else:
        print(f"❌ Training failed: {resp.text}")
        return

    # 4. Test Predictions
    print("\n[4] Testing Predictions...")
    
    # 4a. Test Known (Pranshul)
    print("   [4a] Testing Known Person (Pranshul)...")
    pranshul_signal = generate_synthetic_footstep(amplitude=2500, freq=12)
    resp = requests.post(f"{BASE_URL}/predictfootsteps", json={"data": pranshul_signal})
    if resp.status_code == 200:
        res = resp.json()
        print(f"      Result: {res['prediction']} (Conf: {res['confidence']:.2f})")
        if res['prediction'] == 'Pranshul':
            print("      ✅ Correctly identified Pranshul")
        else:
            print("      ❌ Incorrect identification")
    else:
        print(f"      ❌ Request failed: {resp.text}")

    # 4b. Test Known (Aditi)
    print("   [4b] Testing Known Person (Aditi)...")
    aditi_signal = generate_synthetic_footstep(amplitude=2200, freq=25)
    resp = requests.post(f"{BASE_URL}/predictfootsteps", json={"data": aditi_signal})
    if resp.status_code == 200:
        res = resp.json()
        print(f"      Result: {res['prediction']} (Conf: {res['confidence']:.2f})")
        if res['prediction'] == 'Aditi':
            print("      ✅ Correctly identified Aditi")
        else:
            print("      ❌ Incorrect identification")
    else:
        print(f"      ❌ Request failed: {resp.text}")

    # 4c. Test Intruder (Random Noise / Very Different Signal)
    print("   [4c] Testing Intruder (High Freq Noise)...")
    # Generate signal with very high frequency (50Hz) which is outside training distribution
    intruder_signal = generate_synthetic_footstep(amplitude=2000, freq=60, baseline=2048)
    resp = requests.post(f"{BASE_URL}/predictfootsteps", json={"data": intruder_signal})
    if resp.status_code == 200:
        res = resp.json()
        print(f"      Result: {res['prediction']}")
        print(f"      Is Intruder: {res['is_intruder']}")
        print(f"      Reason: {res.get('intruder_reason')}")
        print(f"      Probabilities: {res['probabilities']}")
        
        if res['is_intruder']:
            print("      ✅ Correctly flagged as Intruder")
        else:
            print("      ❌ Failed to detect intruder")
    else:
        print(f"      ❌ Request failed: {resp.text}")

    # 5. Check Status
    print("\n[5] Final System Status...")
    resp = requests.get(f"{BASE_URL}/status")
    print(json.dumps(resp.json(), indent=2))

if __name__ == "__main__":
    run_test()
