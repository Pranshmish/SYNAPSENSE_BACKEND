import os
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Any

MODELS_DIR = "models"
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Feature order must match features.py output list
FEATURE_NAMES = [
    "peak_amplitude", "energy", "step_duration", "rise_time", "fall_time", 
    "zero_crossings", "std", "rms", "skewness", "kurtosis", "dominant_freq",
    "spectral_centroid", "spectral_entropy", "spectral_bandwidth", 
    "low_freq_ratio", "spike_count", "firing_rate", "isi_mean", 
    "isi_std", "isi_cv", "first_spike_latency"
]

class MLManager:
    def __init__(self):
        self._init_dirs()
        self.svm = None
        self.rf = None
        self.scaler = None
        self.load_models()

    def _init_dirs(self):
        os.makedirs(MODELS_DIR, exist_ok=True)

    def load_models(self):
        if os.path.exists(SVM_MODEL_PATH) and os.path.exists(RF_MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                self.svm = joblib.load(SVM_MODEL_PATH)
                self.rf = joblib.load(RF_MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                print("Models loaded successfully.")
            except Exception as e:
                print(f"Error loading models: {e}")
        else:
            print("Models not found. Need training.")

    def prepare_data(self, data: List[Dict[str, float]]) -> np.ndarray:
        # Convert list of dicts to numpy array with consistent order
        X = []
        for item in data:
            row = [item.get(k, 0.0) for k in FEATURE_NAMES]
            X.append(row)
        return np.array(X)

    def train(self, data: List[Dict[str, float]], labels: List[str]) -> Dict[str, float]:
        if len(data) < 5: # Minimal data check
            return {"error": "Not enough data to train"}

        X = self.prepare_data(data)
        y = np.array(labels)

        # Scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split
        # Stratified split if possible
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, stratify=y, random_state=42
            )
        except ValueError:
            # Fallback if stratify fails (e.g. single sample class)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

        # Train SVM
        self.svm = SVC(kernel='rbf', probability=True, random_state=42)
        self.svm.fit(X_train, y_train)

        # Train RF
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf.fit(X_train, y_train)

        # Evaluate (using RF as primary or average?)
        # User wants "Return metrics". I'll return metrics for RF as it's usually more robust out of box, 
        # or maybe both? User example: "metrics": {"accuracy": 0.92...}
        # I'll use RF predictions for the returned metrics or maybe the best one.
        # Let's use RF for metrics display.
        
        y_pred = self.rf.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='macro', zero_division=0)
        }

        # Save
        joblib.dump(self.svm, SVM_MODEL_PATH)
        joblib.dump(self.rf, RF_MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)

        return metrics

    def predict(self, features: List[float]) -> Dict[str, Any]:
        if not self.svm or not self.rf or not self.scaler:
            return {"error": "Models not trained"}

        # Reshape and scale
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Predict (using RF for now, or ensemble?)
        # User example: "probabilities": {"person1": 0.12...}
        # RF gives probabilities.
        
        probs = self.rf.predict_proba(X_scaled)[0]
        classes = self.rf.classes_
        
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
        prediction = self.rf.predict(X_scaled)[0]
        confidence = float(prob_dict[prediction])

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": prob_dict
        }
