"""
Multi-Model Manager for SynapSense
Supports switching between RF, MLP, and Hybrid LSTM+SNN models.

Models:
1. RandomForestEnsemble - RF + Isolation Forest (Hackathon Winner ~95% CV)
2. MLPClassifier - Neural Network (Current ~94% CV)
3. HybridLSTMSNN - LSTM + Spiking Neural Network (Future ~96%+)
"""

import os
import json
import numpy as np
import joblib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from features import FEATURE_NAMES

# ============== PATHS ==============
MODELS_DIR = "models"
MODEL_REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")

# RF Model paths
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_ensemble.pkl")
RF_ISO_PATH = os.path.join(MODELS_DIR, "rf_isolation.pkl")
RF_SCALER_PATH = os.path.join(MODELS_DIR, "rf_scaler.pkl")
RF_ENCODER_PATH = os.path.join(MODELS_DIR, "rf_encoder.pkl")
RF_METADATA_PATH = os.path.join(MODELS_DIR, "rf_metadata.pkl")

# MLP Model paths (same as existing mlp_model.py)
MLP_MODEL_PATH = os.path.join(MODELS_DIR, "simple_mlp.pkl")
MLP_SCALER_PATH = os.path.join(MODELS_DIR, "mlp_scaler.pkl")
MLP_ENCODER_PATH = os.path.join(MODELS_DIR, "mlp_encoder.pkl")
MLP_METADATA_PATH = os.path.join(MODELS_DIR, "mlp_metadata.pkl")

# Hybrid Model paths
HYBRID_MODEL_PATH = os.path.join(MODELS_DIR, "hybrid_snn.pt")
HYBRID_SCALER_PATH = os.path.join(MODELS_DIR, "hybrid_scaler.pkl")
HYBRID_METADATA_PATH = os.path.join(MODELS_DIR, "hybrid_metadata.pkl")

# ============== MODEL TYPES ==============
MODEL_TYPES = {
    "RandomForestEnsemble": {
        "display_name": "Random Forest + Isolation Forest",
        "short_name": "RF",
        "description": "Ensemble classifier with anomaly detection",
        "ready": True
    },
    "MLPClassifier": {
        "display_name": "MLP Neural Network",
        "short_name": "MLP",
        "description": "Multi-layer perceptron classifier",
        "ready": True
    },
    "HybridLSTMSNN": {
        "display_name": "Hybrid LSTM + SNN",
        "short_name": "Hybrid",
        "description": "LSTM + Spiking Neural Network (Coming Soon)",
        "ready": False  # Placeholder for future
    }
}

# Top 20 robust features (shared across all models)
ROBUST_FEATURES = [
    'stat_zcr', 'stat_rms', 'stat_std', 'stat_peak_count', 'stat_peak_spacing',
    'fft_centroid', 'fft_bandwidth', 'fft_rolloff', 'fft_flux', 'fft_flatness',
    'fft_energy_low', 'fft_energy_mid', 'fft_energy_high', 'fft_peak_freq',
    'lif_spike_rate', 'lif_isi_mean', 'lif_isi_cv', 'lif_burst_ratio',
    'lif_potential_mean', 'stat_entropy'
]


class ModelRegistry:
    """Track all trained models and their performance"""
    
    def __init__(self):
        self._init_dirs()
        self.registry = self._load_registry()
    
    def _init_dirs(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
    
    def _load_registry(self) -> Dict:
        if os.path.exists(MODEL_REGISTRY_PATH):
            try:
                with open(MODEL_REGISTRY_PATH, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "models": {},
            "active_model": "MLPClassifier",  # Default
            "last_updated": None
        }
    
    def _save_registry(self):
        self.registry["last_updated"] = datetime.now().isoformat()
        with open(MODEL_REGISTRY_PATH, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_name: str, metrics: Dict):
        """Register a trained model with its metrics"""
        self.registry["models"][model_name] = {
            "trained": True,
            "cv_accuracy": metrics.get("cv_accuracy", 0),
            "training_accuracy": metrics.get("training_accuracy", 0),
            "samples_used": metrics.get("total_samples", 0),
            "trained_at": datetime.now().isoformat(),
            **MODEL_TYPES.get(model_name, {})
        }
        self._save_registry()
    
    def set_active_model(self, model_name: str):
        """Set the active model for predictions"""
        if model_name in MODEL_TYPES:
            self.registry["active_model"] = model_name
            self._save_registry()
    
    def get_active_model(self) -> str:
        return self.registry.get("active_model", "MLPClassifier")
    
    def get_all_models(self) -> Dict:
        """Get all available models with their status"""
        result = {}
        for name, info in MODEL_TYPES.items():
            model_data = self.registry["models"].get(name, {})
            result[name] = {
                **info,
                "trained": model_data.get("trained", False),
                "cv_accuracy": model_data.get("cv_accuracy", 0),
                "samples_used": model_data.get("samples_used", 0),
                "trained_at": model_data.get("trained_at"),
                "is_active": self.registry["active_model"] == name
            }
        return result


class RandomForestEnsembleModel:
    """
    Random Forest + Isolation Forest Ensemble
    - RF for classification
    - Isolation Forest for anomaly detection (INTRUDER)
    """
    
    def __init__(self):
        self.rf_model: Optional[RandomForestClassifier] = None
        self.iso_model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.encoder: Optional[LabelEncoder] = None
        self.metadata: Dict = {}
        self.is_trained = False
        self.feature_indices: List[int] = []
        self._load_models()
    
    def _get_feature_indices(self) -> List[int]:
        indices = []
        for feat in ROBUST_FEATURES:
            if feat in FEATURE_NAMES:
                indices.append(FEATURE_NAMES.index(feat))
        return indices
    
    def _select_features(self, X: np.ndarray) -> np.ndarray:
        if not self.feature_indices:
            self.feature_indices = self._get_feature_indices()
        
        if len(self.feature_indices) == 0:
            return X[:, :20] if X.shape[1] > 20 else X
        
        selected = []
        for idx in self.feature_indices:
            if idx < X.shape[1]:
                selected.append(X[:, idx])
        
        if len(selected) == 0:
            return X[:, :20] if X.shape[1] > 20 else X
            
        return np.column_stack(selected)
    
    def _load_models(self) -> bool:
        try:
            if all(os.path.exists(p) for p in [RF_MODEL_PATH, RF_ISO_PATH, RF_SCALER_PATH, RF_ENCODER_PATH]):
                self.rf_model = joblib.load(RF_MODEL_PATH)
                self.iso_model = joblib.load(RF_ISO_PATH)
                self.scaler = joblib.load(RF_SCALER_PATH)
                self.encoder = joblib.load(RF_ENCODER_PATH)
                if os.path.exists(RF_METADATA_PATH):
                    self.metadata = joblib.load(RF_METADATA_PATH)
                self.is_trained = True
                print("[RF] RandomForest Ensemble loaded successfully")
                return True
        except Exception as e:
            print(f"[RF] Error loading models: {e}")
        return False
    
    def _save_models(self):
        joblib.dump(self.rf_model, RF_MODEL_PATH)
        joblib.dump(self.iso_model, RF_ISO_PATH)
        joblib.dump(self.scaler, RF_SCALER_PATH)
        joblib.dump(self.encoder, RF_ENCODER_PATH)
        joblib.dump(self.metadata, RF_METADATA_PATH)
        print("[RF] Models saved successfully")
    
    def prepare_features(self, data: List[Dict[str, float]]) -> np.ndarray:
        X = []
        for item in data:
            if isinstance(item, dict):
                row = [float(item.get(name, 0.0)) for name in FEATURE_NAMES]
            else:
                row = [float(v) for v in item]
            X.append(row)
        return np.array(X, dtype=np.float64)
    
    def generate_synthetic_intruder(self, home_data: np.ndarray, n_samples: int) -> np.ndarray:
        """Generate synthetic INTRUDER samples"""
        synthetic = []
        
        # Method 1: Strong noise (40%)
        n_noise = int(n_samples * 0.4)
        for _ in range(n_noise):
            idx = np.random.randint(0, len(home_data))
            sample = home_data[idx].copy()
            noise_level = np.random.uniform(2, 5)
            noise = np.random.randn(len(sample)) * np.std(home_data, axis=0) * noise_level
            synthetic.append(sample + noise)
        
        # Method 2: Random scaling (30%)
        n_scale = int(n_samples * 0.3)
        for _ in range(n_scale):
            idx = np.random.randint(0, len(home_data))
            sample = home_data[idx].copy()
            scales = np.random.uniform(0.3, 3.0, len(sample))
            synthetic.append(sample * scales)
        
        # Method 3: Mix + offset (30%)
        n_mix = n_samples - n_noise - n_scale
        for _ in range(n_mix):
            idx1, idx2 = np.random.choice(len(home_data), 2, replace=True)
            sample1, sample2 = home_data[idx1], home_data[idx2]
            alpha = np.random.uniform(0.3, 0.7)
            offset = np.random.randn(len(sample1)) * np.std(home_data, axis=0)
            synthetic.append(alpha * sample1 + (1 - alpha) * sample2 + offset)
        
        return np.array(synthetic)
    
    def train(self, data: List[Dict[str, float]], labels: List[str]) -> Dict[str, Any]:
        """Train RF + Isolation Forest ensemble with K-Fold CV"""
        print(f"[RF] Training on {len(data)} samples...")
        
        # Prepare features
        X_full = self.prepare_features(data)
        X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Select robust features
        self.feature_indices = self._get_feature_indices()
        X = self._select_features(X_full)
        
        # Normalize labels
        normalized_labels = []
        for label in labels:
            if label.upper().startswith('HOME') or label.upper() == 'HOME':
                normalized_labels.append('HOME')
            else:
                normalized_labels.append('HOME')  # Treat all as HOME for training
        
        # Get HOME samples
        home_indices = [i for i, l in enumerate(normalized_labels) if l == 'HOME']
        X_home = X[home_indices]
        
        if len(X_home) < 5:
            return {"success": False, "error": f"Need at least 5 HOME samples, got {len(X_home)}"}
        
        # Generate synthetic INTRUDER samples
        n_intruder = len(X_home)
        X_intruder = self.generate_synthetic_intruder(X_home, n_intruder)
        
        # Combine data
        X_combined = np.vstack([X_home, X_intruder])
        y_labels = ['HOME'] * len(X_home) + ['INTRUDER'] * len(X_intruder)
        
        # Encode labels
        self.encoder = LabelEncoder()
        y_combined = self.encoder.fit_transform(y_labels)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_combined)
        
        # ============== K-FOLD CROSS-VALIDATION ==============
        n_folds = min(5, len(X_home))
        print(f"[RF] Running {n_folds}-Fold Cross-Validation...")
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_combined)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_combined[train_idx], y_combined[val_idx]
            
            fold_rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            fold_rf.fit(X_train, y_train)
            
            fold_acc = accuracy_score(y_val, fold_rf.predict(X_val)) * 100
            cv_scores.append(fold_acc)
            print(f"[RF] Fold {fold+1}/{n_folds}: {fold_acc:.2f}%")
        
        mean_cv = np.mean(cv_scores)
        std_cv = np.std(cv_scores)
        print(f"[RF] CV Accuracy: {mean_cv:.2f}% (+/- {std_cv:.2f}%)")
        
        # ============== TRAIN FINAL MODELS ==============
        print("[RF] Training final RF model...")
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_scaled, y_combined)
        
        # Train Isolation Forest on HOME samples only
        print("[RF] Training Isolation Forest for anomaly detection...")
        X_home_scaled = self.scaler.transform(X_home)
        self.iso_model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.iso_model.fit(X_home_scaled)
        
        training_acc = self.rf_model.score(X_scaled, y_combined) * 100
        
        # Feature importance
        importances = self.rf_model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:10]
        feature_names = [ROBUST_FEATURES[i] if i < len(ROBUST_FEATURES) else f"feat_{i}" for i in top_indices]
        top_features = [(feature_names[i], float(importances[top_indices[i]])) for i in range(len(top_indices))]
        
        self.metadata = {
            "cv_scores": cv_scores,
            "cv_accuracy": round(mean_cv, 2),
            "cv_std": round(std_cv, 2),
            "training_accuracy": round(training_acc, 2),
            "n_folds": n_folds,
            "home_samples": len(X_home),
            "intruder_samples": len(X_intruder),
            "total_samples": len(X_combined),
            "classes": list(self.encoder.classes_),
            "top_features": top_features
        }
        
        self.is_trained = True
        self._save_models()
        
        return {
            "success": True,
            "metrics": {
                "cv_accuracy": round(mean_cv, 2),
                "cv_std": round(std_cv, 2),
                "cv_scores": [round(s, 2) for s in cv_scores],
                "training_accuracy": round(training_acc, 2),
                "n_folds": n_folds,
                "home_samples": len(X_home),
                "intruder_samples": len(X_intruder),
                "total_samples": len(X_combined)
            },
            "top_features": [{"name": n, "importance": v} for n, v in top_features[:5]],
            "model_name": "RandomForestEnsemble"
        }
    
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """Predict with RF + Isolation Forest ensemble"""
        if not self.is_trained:
            return {"success": False, "error": "Model not trained"}
        
        try:
            # Prepare features
            X = np.array(features, dtype=np.float64).reshape(1, -1)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = self._select_features(X)
            X_scaled = self.scaler.transform(X)
            
            # RF prediction
            rf_proba = self.rf_model.predict_proba(X_scaled)[0]
            rf_pred = self.rf_model.predict(X_scaled)[0]
            
            # Isolation Forest anomaly score
            anomaly_score = self.iso_model.decision_function(X_scaled)[0]
            is_anomaly = self.iso_model.predict(X_scaled)[0] == -1
            
            # Get class names and probabilities
            classes = self.encoder.classes_
            prob_dict = {cls: float(prob) for cls, prob in zip(classes, rf_proba)}
            
            # Determine prediction
            rf_label = classes[rf_pred]
            confidence = float(max(rf_proba))
            
            # Apply INTRUDER detection rules
            is_intruder = False
            rule_applied = None
            
            if is_anomaly:
                is_intruder = True
                rule_applied = "Isolation Forest Anomaly"
            elif confidence < 0.85:
                is_intruder = True
                rule_applied = "Low Confidence (<85%)"
            elif anomaly_score < -0.2:
                is_intruder = True
                rule_applied = "Anomaly Score < -0.2"
            
            prediction = "INTRUDER" if is_intruder else rf_label
            
            # Confidence band
            if confidence >= 0.9:
                confidence_band = "high"
                color_code = "#22c55e" if not is_intruder else "#ef4444"
            elif confidence >= 0.75:
                confidence_band = "medium"
                color_code = "#eab308"
            else:
                confidence_band = "low"
                color_code = "#ef4444"
            
            return {
                "success": True,
                "prediction": prediction,
                "confidence": round(confidence, 3),
                "is_intruder": is_intruder,
                "alert": "🚨 INTRUDER DETECTED!" if is_intruder else f"✅ HOME: {rf_label}",
                "probabilities": prob_dict,
                "anomaly_score": round(float(anomaly_score), 3),
                "is_anomaly": is_anomaly,
                "confidence_band": confidence_band,
                "color_code": color_code,
                "rule_applied": rule_applied,
                "model_used": "RandomForestEnsemble"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_status(self) -> Dict:
        if not self.is_trained:
            return {"trained": False}
        return {
            "trained": True,
            "accuracy": self.metadata.get("cv_accuracy", 0),
            "cv_accuracy": self.metadata.get("cv_accuracy", 0),
            "cv_std": self.metadata.get("cv_std", 0),
            "home_samples": self.metadata.get("home_samples", 0),
            "intruder_samples": self.metadata.get("intruder_samples", 0),
            "classes": self.metadata.get("classes", [])
        }


class HybridLSTMSNNModel:
    """
    Placeholder for Hybrid LSTM + SNN Model
    To be implemented with PyTorch + SpikingJelly
    """
    
    def __init__(self):
        self.is_trained = False
        self.metadata = {}
    
    def train(self, data: List[Dict], labels: List[str]) -> Dict[str, Any]:
        return {
            "success": False,
            "error": "Hybrid LSTM+SNN model is coming soon! Use RF or MLP for now."
        }
    
    def predict(self, features: List[float]) -> Dict[str, Any]:
        return {
            "success": False,
            "error": "Hybrid model not available yet"
        }
    
    def get_status(self) -> Dict:
        return {
            "trained": False,
            "message": "Coming Soon - Hybrid LSTM+SNN"
        }


class MultiModelManager:
    """
    Central manager for all models with switching capability
    """
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.rf_model = RandomForestEnsembleModel()
        self.hybrid_model = HybridLSTMSNNModel()
        
        # MLP model is imported from mlp_model.py to avoid duplication
        from mlp_model import mlp_classifier
        self.mlp_model = mlp_classifier
        
        print(f"[ModelManager] Initialized. Active model: {self.registry.get_active_model()}")
    
    def get_model(self, model_name: str):
        """Get model instance by name"""
        if model_name == "RandomForestEnsemble":
            return self.rf_model
        elif model_name == "MLPClassifier":
            return self.mlp_model
        elif model_name == "HybridLSTMSNN":
            return self.hybrid_model
        else:
            return None
    
    def train_model(self, model_name: str, data: List[Dict], labels: List[str]) -> Dict[str, Any]:
        """Train a specific model"""
        model = self.get_model(model_name)
        if not model:
            return {"success": False, "error": f"Unknown model: {model_name}"}
        
        result = model.train(data, labels)
        
        if result.get("success", False):
            self.registry.register_model(model_name, result.get("metrics", {}))
            result["model_name"] = model_name
        
        return result
    
    def predict_with_model(self, model_name: str, features: List[float]) -> Dict[str, Any]:
        """Predict using a specific model"""
        model = self.get_model(model_name)
        if not model:
            return {"success": False, "error": f"Unknown model: {model_name}"}
        
        if not model.is_trained:
            return {"success": False, "error": f"{model_name} is not trained yet"}
        
        result = model.predict(features)
        result["model_used"] = model_name
        return result
    
    def predict_with_active_model(self, features: List[float]) -> Dict[str, Any]:
        """Predict using the currently active model"""
        active = self.registry.get_active_model()
        return self.predict_with_model(active, features)
    
    def set_active_model(self, model_name: str) -> Dict[str, Any]:
        """Switch the active model"""
        if model_name not in MODEL_TYPES:
            return {"success": False, "error": f"Unknown model: {model_name}"}
        
        model = self.get_model(model_name)
        if not model.is_trained:
            return {"success": False, "error": f"{model_name} is not trained. Train it first!"}
        
        self.registry.set_active_model(model_name)
        return {
            "success": True,
            "message": f"Active model set to {model_name}",
            "active_model": model_name
        }
    
    def get_all_models_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        return {
            "models": self.registry.get_all_models(),
            "active_model": self.registry.get_active_model()
        }
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models for dropdown"""
        models = []
        all_status = self.registry.get_all_models()
        
        for name, info in all_status.items():
            models.append({
                "name": name,
                "display_name": info["display_name"],
                "short_name": info["short_name"],
                "description": info["description"],
                "trained": info["trained"],
                "cv_accuracy": info["cv_accuracy"],
                "is_active": info["is_active"],
                "ready": info["ready"]
            })
        
        return models


# Global instance
model_manager = MultiModelManager()
