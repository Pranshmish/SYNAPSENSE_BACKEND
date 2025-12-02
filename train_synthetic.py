import numpy as np
from storage import StorageManager
from ml import MLManager
from features import FootstepFeatureExtractor
import os
import shutil

# Instantiate Managers
ml_manager = MLManager()
storage_manager = StorageManager()
extractor = FootstepFeatureExtractor()

# Config
SAMPLE_RATE = 200
DURATION = 1.0 # 200 samples
SAMPLES_PER_PERSON = 20

def generate_synthetic_footstep(amplitude=2000, freq=15, noise_level=5):
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))
    # Damped sine wave
    envelope = np.exp(-5 * t)
    signal = amplitude * np.sin(2 * np.pi * freq * t) * envelope
    # Add noise
    noise = np.random.normal(0, noise_level, len(t))
    # Add baseline
    baseline = 2048
    return signal + noise + baseline

def train_synthetic():
    print("🚀 Starting synthetic data generation and training...")
    
    # 1. Reset existing data
    print("1. Resetting dataset...")
    if os.path.exists('dataset'):
        shutil.rmtree('dataset')
    if os.path.exists('models'):
        shutil.rmtree('models')
    if os.path.exists('db/samples.db'):
        os.remove('db/samples.db')
    
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('db', exist_ok=True)
    storage_manager._init_db()

    # 2. Generate data for each person
    persons = {
        'Pranshul': {'amp': 2500, 'freq': 12},
        'Aditi': {'amp': 2200, 'freq': 18},
        'Apurv': {'amp': 2800, 'freq': 10},
        'Samir': {'amp': 2400, 'freq': 15}
    }

    for name, params in persons.items():
        print(f"2. Generating {SAMPLES_PER_PERSON} samples for {name}...")
        count = 0
        for i in range(SAMPLES_PER_PERSON):
            # Generate raw signal
            raw_signal = generate_synthetic_footstep(
                amplitude=params['amp'] + np.random.randint(-200, 200),
                freq=params['freq'] + np.random.uniform(-1, 1),
                noise_level=10
            )
            
            # Extract features
            features = extractor.process_chunk(raw_signal)
            
            if features:
                storage_manager.save_sample(name, features)
                count += 1
            else:
                print(f"   ⚠️ Sample {i} rejected by validation")
        print(f"   -> Saved {count} samples for {name}")

    # 3. Train the model
    print("3. Training model...")
    # We need to fetch data from storage first to pass to train method
    data, labels = storage_manager.get_all_samples()
    metrics = ml_manager.train(data, labels)
    
    if metrics and "error" not in metrics:
        print("✅ Training complete!")
        print(f"Metrics: {metrics}")
    else:
        print(f"❌ Training failed! {metrics}")

if __name__ == "__main__":
    train_synthetic()
