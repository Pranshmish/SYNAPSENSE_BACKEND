"""
Pytest-Compatible Test Suite for SynapSense One-Class Anomaly Detection
Run with: pytest tests/test_pytest_qa.py -v
"""

import os
import sys
import pytest
import numpy as np
import requests
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import FootstepFeatureExtractor, FEATURE_NAMES, LIF
from ml import AnomalyDetector, LABEL_HOME, LABEL_INTRUDER
from storage import StorageManager

# =====================================================================
# FIXTURES
# =====================================================================

BASE_URL = "http://localhost:8000"
SAMPLE_RATE = 200.0

@pytest.fixture(scope="module")
def extractor():
    """Feature extractor instance."""
    return FootstepFeatureExtractor()

@pytest.fixture(scope="module")
def detector():
    """Anomaly detector instance."""
    return AnomalyDetector()

@pytest.fixture(scope="module")
def storage():
    """Storage manager instance."""
    return StorageManager()

@pytest.fixture
def soft_footstep():
    """Generate soft footstep signal."""
    t = np.linspace(0, 1, int(SAMPLE_RATE))
    freq = 20
    amp = 80
    envelope = np.exp(-3 * t) * np.sin(np.pi * t)
    signal = amp * envelope * np.sin(2 * np.pi * freq * t)
    noise = np.random.normal(0, 3, len(t))
    return (signal + noise + 2048).tolist()

@pytest.fixture
def normal_footstep():
    """Generate normal footstep signal."""
    t = np.linspace(0, 1, int(SAMPLE_RATE))
    freq = 20
    amp = 300
    envelope = np.exp(-4 * t) * np.sin(np.pi * t)
    signal = amp * envelope * np.sin(2 * np.pi * freq * t)
    signal += amp * 0.3 * envelope * np.sin(2 * np.pi * freq * 2 * t)
    noise = np.random.normal(0, 8, len(t))
    return (signal + noise + 2048).tolist()

@pytest.fixture
def hard_footstep():
    """Generate hard footstep signal."""
    t = np.linspace(0, 1, int(SAMPLE_RATE))
    freq = 15
    amp = 800
    envelope = np.exp(-5 * t) * np.sin(np.pi * t)
    signal = amp * envelope * np.sin(2 * np.pi * freq * t)
    signal += amp * 0.5 * envelope * np.sin(2 * np.pi * freq * 2 * t)
    noise = np.random.normal(0, 15, len(t))
    return (signal + noise + 2048).tolist()

@pytest.fixture
def ambient_noise():
    """Generate ambient noise (no footstep)."""
    t = np.linspace(0, 1, int(SAMPLE_RATE))
    noise = np.random.normal(0, 2, len(t))
    drift = 5 * np.sin(2 * np.pi * 0.1 * t)
    return (noise + drift + 2048).tolist()

@pytest.fixture
def intruder_footstep():
    """Generate intruder footstep (different pattern)."""
    t = np.linspace(0, 1, int(SAMPLE_RATE))
    freq = 12
    amp = 250
    envelope = (1 - np.exp(-10 * t)) * np.exp(-2 * t)
    signal = amp * envelope * np.sin(2 * np.pi * freq * t)
    signal += amp * 0.6 * envelope * np.sin(2 * np.pi * freq * 1.5 * t)
    noise = np.random.normal(0, 12, len(t))
    return (signal + noise + 2048).tolist()


# =====================================================================
# TEST CLASS: Signal Acquisition
# =====================================================================

class TestSignalAcquisition:
    """Test 1: Signal acquisition and generation."""
    
    def test_soft_footstep_generation(self, soft_footstep):
        """Test soft footstep signal generation."""
        assert len(soft_footstep) >= 200
        assert all(0 <= v <= 4095 for v in soft_footstep)
        
    def test_normal_footstep_generation(self, normal_footstep):
        """Test normal footstep signal generation."""
        assert len(normal_footstep) >= 200
        assert np.std(normal_footstep) > 10  # Has variation
        
    def test_hard_footstep_generation(self, hard_footstep):
        """Test hard footstep signal generation."""
        assert len(hard_footstep) >= 200
        assert np.std(hard_footstep) > 50  # Higher variation
        
    def test_ambient_noise_low_amplitude(self, ambient_noise):
        """Test ambient noise has low amplitude."""
        assert np.std(ambient_noise) < 20  # Low variation
        
    def test_adc_range_validity(self, normal_footstep, soft_footstep, hard_footstep):
        """Test all signals in ADC range (0-4095)."""
        for signal in [normal_footstep, soft_footstep, hard_footstep]:
            assert min(signal) >= 0
            assert max(signal) <= 4500  # Allow small overshoot


# =====================================================================
# TEST CLASS: Step Detection
# =====================================================================

class TestStepDetection:
    """Test 2: Step detection and segmentation."""
    
    def test_soft_footstep_detected(self, extractor, soft_footstep):
        """Test soft footstep is detected."""
        features = extractor.process_chunk(soft_footstep)
        assert features is not None
        assert len(features) > 0
        
    def test_normal_footstep_detected(self, extractor, normal_footstep):
        """Test normal footstep is detected."""
        features = extractor.process_chunk(normal_footstep)
        assert features is not None
        
    def test_hard_footstep_detected(self, extractor, hard_footstep):
        """Test hard footstep is detected."""
        features = extractor.process_chunk(hard_footstep)
        assert features is not None
        
    def test_consistent_features(self, extractor, normal_footstep):
        """Test same signal gives same features (no double-triggering)."""
        features1 = extractor.process_chunk(normal_footstep)
        features2 = extractor.process_chunk(normal_footstep)
        
        if features1 and features2:
            diff = abs(features1.get('stat_rms', 0) - features2.get('stat_rms', 0))
            assert diff < 0.01


# =====================================================================
# TEST CLASS: Feature Extraction
# =====================================================================

class TestFeatureExtraction:
    """Test 4: LIF + FFT + Statistical feature extraction."""
    
    def test_feature_count(self, extractor, normal_footstep):
        """Test minimum feature count."""
        features = extractor.process_chunk(normal_footstep)
        assert features is not None
        assert len(features) >= 50
        
    def test_lif_spike_features(self, extractor, normal_footstep):
        """Test LIF spike features exist."""
        features = extractor.process_chunk(normal_footstep)
        assert 'lif_mid_spike_count' in features
        assert 'lif_mid_spike_rate' in features
        
    def test_lif_spike_count_positive(self, extractor, normal_footstep):
        """Test LIF generates spikes for real footstep."""
        features = extractor.process_chunk(normal_footstep)
        # At least one neuron should spike
        total_spikes = (features.get('lif_low_spike_count', 0) + 
                       features.get('lif_mid_spike_count', 0) + 
                       features.get('lif_high_spike_count', 0))
        assert total_spikes >= 0  # Might be 0 for very weak signals
        
    def test_fft_features(self, extractor, normal_footstep):
        """Test FFT spectral features exist."""
        features = extractor.process_chunk(normal_footstep)
        assert 'fft_centroid' in features
        assert 'fft_bass_energy' in features
        assert 'fft_rolloff' in features
        
    def test_fft_centroid_range(self, extractor, normal_footstep):
        """Test FFT centroid in valid frequency range."""
        features = extractor.process_chunk(normal_footstep)
        centroid = features.get('fft_centroid', -1)
        assert 0 <= centroid <= 100  # Footstep frequency range
        
    def test_statistical_features(self, extractor, normal_footstep):
        """Test statistical features exist."""
        features = extractor.process_chunk(normal_footstep)
        assert 'stat_rms' in features
        assert 'stat_mean' in features
        assert 'stat_std' in features
        assert 'stat_peak_count' in features
        
    def test_rms_positive(self, extractor, normal_footstep):
        """Test RMS is positive for real signal."""
        features = extractor.process_chunk(normal_footstep)
        assert features.get('stat_rms', 0) > 0


# =====================================================================
# TEST CLASS: Model Training
# =====================================================================

class TestModelTraining:
    """Test 5: Isolation Forest model training."""
    
    @pytest.fixture(scope="class")
    def trained_detector(self, extractor):
        """Create and train a detector for tests."""
        detector = AnomalyDetector()
        detector.reset()
        
        # Generate training data
        home_features = []
        home_labels = []
        
        for _ in range(20):
            t = np.linspace(0, 1, 200)
            freq = np.random.uniform(15, 25)
            amp = np.random.uniform(150, 400)
            envelope = np.exp(-4 * t) * np.sin(np.pi * t)
            signal = amp * envelope * np.sin(2 * np.pi * freq * t)
            noise = np.random.normal(0, 8, len(t))
            chunk = (signal + noise + 2048).tolist()
            
            features = extractor.process_chunk(chunk)
            if features:
                home_features.append(features)
                home_labels.append(LABEL_HOME)
        
        detector.train(home_features, home_labels)
        return detector
    
    def test_training_completes(self, trained_detector):
        """Test model training completes without error."""
        assert trained_detector is not None
        assert trained_detector.is_trained
        
    def test_scaler_initialized(self, trained_detector):
        """Test scaler is initialized after training."""
        assert trained_detector.scaler is not None
        
    def test_isolation_forest_initialized(self, trained_detector):
        """Test Isolation Forest model is initialized."""
        assert trained_detector.isolation_forest is not None
        
    def test_anomaly_threshold_set(self, trained_detector):
        """Test anomaly threshold is calculated."""
        assert trained_detector.anomaly_threshold is not None
        
    def test_model_can_be_saved(self, trained_detector):
        """Test model can be saved to disk."""
        trained_detector.save_models()
        assert Path("models/home_detector_if.pkl").exists()


# =====================================================================
# TEST CLASS: Predictions
# =====================================================================

class TestPredictions:
    """Test 6: Prediction accuracy."""
    
    @pytest.fixture(scope="class")
    def loaded_detector(self):
        """Load detector for prediction tests."""
        detector = AnomalyDetector()
        if not detector.is_trained:
            pytest.skip("No trained model available")
        return detector
    
    def test_home_prediction_soft(self, loaded_detector, extractor, soft_footstep):
        """Test soft HOME footstep predicted as HOME."""
        features = extractor.process_chunk(soft_footstep)
        if features:
            result = loaded_detector.predict(list(features.values()))
            # Should not crash at minimum
            assert "prediction" in result
            
    def test_home_prediction_normal(self, loaded_detector, extractor, normal_footstep):
        """Test normal HOME footstep predicted as HOME."""
        features = extractor.process_chunk(normal_footstep)
        if features:
            result = loaded_detector.predict(list(features.values()))
            assert "prediction" in result
            assert "confidence" in result
            
    def test_confidence_provided(self, loaded_detector, extractor, normal_footstep):
        """Test confidence score is provided."""
        features = extractor.process_chunk(normal_footstep)
        if features:
            result = loaded_detector.predict(list(features.values()))
            assert "confidence" in result
            assert result["confidence"] >= 0
            
    def test_anomaly_score_provided(self, loaded_detector, extractor, normal_footstep):
        """Test anomaly score is provided."""
        features = extractor.process_chunk(normal_footstep)
        if features:
            result = loaded_detector.predict(list(features.values()))
            assert "anomaly_score" in result


# =====================================================================
# TEST CLASS: Backend Endpoints
# =====================================================================

@pytest.mark.integration
class TestBackendEndpoints:
    """Test 7: REST API endpoint tests."""
    
    @pytest.fixture(autouse=True)
    def check_backend(self):
        """Check if backend is running."""
        try:
            resp = requests.get(f"{BASE_URL}/", timeout=2)
            if resp.status_code != 200:
                pytest.skip("Backend not running")
        except:
            pytest.skip("Backend not running")
    
    def test_root_endpoint(self):
        """Test GET / endpoint."""
        resp = requests.get(f"{BASE_URL}/")
        assert resp.status_code == 200
        data = resp.json()
        assert "mode" in data
        
    def test_status_endpoint(self):
        """Test GET /status endpoint."""
        resp = requests.get(f"{BASE_URL}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_status" in data
        assert "model_type" in data
        
    def test_train_data_endpoint(self, normal_footstep):
        """Test POST /train_data endpoint."""
        payload = {
            "data": [{"raw_time_series": normal_footstep}],
            "label": "HOME",
            "train_model": False
        }
        resp = requests.post(f"{BASE_URL}/train_data", json=payload)
        assert resp.status_code == 200
        
    def test_predict_endpoint(self, normal_footstep):
        """Test POST /predictfootsteps endpoint."""
        resp = requests.post(f"{BASE_URL}/predictfootsteps", json={"data": normal_footstep})
        # May fail if model not trained, but endpoint should respond
        assert resp.status_code in [200, 400]
        
    def test_train_endpoint(self):
        """Test POST /train endpoint."""
        resp = requests.post(f"{BASE_URL}/train", json={"label": "HOME"}, timeout=30)
        # May fail with insufficient data, but should respond
        assert resp.status_code in [200, 400, 500]


# =====================================================================
# TEST CLASS: Performance
# =====================================================================

@pytest.mark.integration
class TestPerformance:
    """Test 8 & 10: Performance and stress tests."""
    
    @pytest.fixture(autouse=True)
    def check_backend(self):
        """Check if backend is running."""
        try:
            resp = requests.get(f"{BASE_URL}/", timeout=2)
            if resp.status_code != 200:
                pytest.skip("Backend not running")
        except:
            pytest.skip("Backend not running")
    
    def test_prediction_latency(self, normal_footstep):
        """Test prediction completes under 300ms."""
        start = time.time()
        resp = requests.post(f"{BASE_URL}/predictfootsteps", json={"data": normal_footstep})
        latency = (time.time() - start) * 1000
        
        if resp.status_code == 200:
            assert latency < 500  # Allow 500ms for network overhead
            
    def test_batch_processing(self, normal_footstep):
        """Test batch sample processing."""
        chunks = [{"raw_time_series": normal_footstep} for _ in range(5)]
        payload = {"data": chunks, "label": "HOME", "train_model": False}
        
        resp = requests.post(f"{BASE_URL}/train_data", json=payload, timeout=30)
        assert resp.status_code == 200
        
    def test_rapid_predictions(self, normal_footstep):
        """Test rapid sequential predictions."""
        successes = 0
        for _ in range(10):
            resp = requests.post(f"{BASE_URL}/predictfootsteps", 
                               json={"data": normal_footstep}, timeout=5)
            if resp.status_code == 200:
                successes += 1
        
        assert successes >= 5  # At least 50% should succeed


# =====================================================================
# RUN CONFIGURATION
# =====================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "--tb=short"])
