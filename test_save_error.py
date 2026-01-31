import requests
import json
import numpy as np

url = "http://localhost:8000/train_data"

# Create a realistic "footstep" signal
# 200 samples, baseline 2048, with a bell-shaped envelope and 100Hz oscillation
t = np.linspace(0, 1, 200)
signal = 2048 + 500 * np.exp(-((t-0.5)**2)/0.01) * np.sin(2*np.pi*100*t)
signal += np.random.normal(0, 10, 200)  # Add some noise
raw_data = signal.astype(int).tolist()

payload = {
    "data": [
        {
            "raw_time_series": raw_data,
            "channel": 1,
            "filtered_waveform": [x - 2048 for x in raw_data],
            "fft_data": {"frequencies": [10, 20, 30], "magnitudes": [0.1, 0.5, 0.2]}
        }
    ],
    "label": "HOME_DebugUser",
    "train_model": False
}

try:
    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
