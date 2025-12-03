"""
Multi-Class Footstep Classifier with Intruder Detection
Trains on specific family members (Pranshul, Aditi, etc.)
Detects intruders based on low confidence or ambiguous probabilities.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from features import FEATURE_NAMES

MODELS_DIR = "models"
RF_MODEL_PATH = os.path.join(MODELS_DIR, "footstep_classifier_rf.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.pkl")

class AnomalyDetector:
    """
    Multi-Class Classifier (Random Forest) that supports Intruder Detection
    via probability thresholding.
    
    Replaces the previous One-Class Anomaly Detector.
    """
    
    def __init__(self):
        self._init_dirs()
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.metadata: Dict[str, Any] = {}
        self.is_trained = False
        self.load_models()
        
    def _init_dirs(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        
    def load_models(self) -> bool:
        if (os.path.exists(RF_MODEL_PATH) and 
            os.path.exists(SCALER_PATH) and 
            os.path.exists(METADATA_PATH)):
            try:
                self.model = joblib.load(RF_MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.metadata = joblib.load(METADATA_PATH)
                self.is_trained = True
                print(f"[ML] Multi-class classifier loaded. Classes: {self.model.classes_}")
                return True
            except Exception as e:
                print(f"[ML] Error loading models: {e}")
                self.is_trained = False
                return False
        else:
            print("[ML] No trained models found.")
            return False
            
    def save_models(self):
        try:
            joblib.dump(self.model, RF_MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            joblib.dump(self.metadata, METADATA_PATH)
            print("[ML] Models saved successfully.")
        except Exception as e:
            print(f"[ML] Error saving models: {e}")
            
    def reset(self):
        self.model = None
        self.scaler = None
        self.metadata = {}
        self.is_trained = False
        
        for path in [RF_MODEL_PATH, SCALER_PATH, METADATA_PATH]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"[ML] Error removing {path}: {e}")
        print("[ML] Models reset.")
        
    def prepare_features(self, data: List[Dict[str, float]]) -> np.ndarray:
        X = []
        for item in data:
            if isinstance(item, dict):
                row = [float(item.get(name, 0.0)) for name in FEATURE_NAMES]
            else:
                row = [float(v) for v in item]
            X.append(row)
        return np.array(X, dtype=np.float64)
    
    def train(self, data: List[Dict[str, float]], labels: List[str]) -> Dict[str, Any]:
        """
        Train Random Forest on all provided labels.
        """
        if len(set(labels)) < 2:
            return {
                "success": False,
                "error": f"Need at least 2 different classes to train. Got: {list(set(labels))}"
            }
            
        X = self.prepare_features(data)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest with Regularization to prevent overfitting
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,           # Limit depth to prevent memorization
            min_samples_split=5,    # Require more samples to split a node
            min_samples_leaf=2,     # Require at least 2 samples in a leaf
            max_features='sqrt',
            oob_score=True,         # Use Out-of-Bag score for validation
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, labels)
        
        # Calculate Training Accuracy
        training_accuracy = self.model.score(X_scaled, labels) * 100
        
        # Get OOB Score (Out-of-Bag) - good proxy for validation accuracy
        oob_accuracy = 0.0
        if hasattr(self.model, 'oob_score_'):
            oob_accuracy = self.model.oob_score_ * 100
        
        # Attempt Cross-Validation
        cv_accuracy = 0.0
        try:
            class_counts = {label: labels.count(label) for label in set(labels)}
            min_samples = min(class_counts.values())
            
            if min_samples >= 2:
                n_splits = min(5, min_samples)
                if n_splits >= 2:
                    cv_scores = cross_val_score(self.model, X_scaled, labels, cv=n_splits)
                    cv_accuracy = np.mean(cv_scores) * 100
        except Exception as e:
            print(f"[ML] CV failed: {e}")
        
        # Determine Final Accuracy Metric (Prioritize CV > OOB > Training)
        # If CV/OOB are too low (e.g. < 50% on very small data), they might be misleading, 
        # but generally they are better indicators of generalization than training acc.
        
        final_accuracy = training_accuracy
        if cv_accuracy > 0:
            final_accuracy = cv_accuracy
        elif oob_accuracy > 0:
            final_accuracy = oob_accuracy
            
        # Feature Importance
        importances = self.model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:10]
        top_features = [(FEATURE_NAMES[i], float(importances[i])) for i in top_indices]
        
        self.metadata = {
            "classes": list(self.model.classes_),
            "training_accuracy": final_accuracy,
            "raw_training_accuracy": training_accuracy,
            "cv_accuracy": cv_accuracy,
            "oob_accuracy": oob_accuracy,
            "top_features": top_features,
            "sample_count": len(data)
        }
        
        self.is_trained = True
        self.save_models()
        
        return {
            "success": True,
            "metrics": {
                "training_accuracy": round(final_accuracy, 2),
                "classes": list(self.model.classes_)
            },
            "top_features": [{"name": n, "importance": v} for n, v in top_features]
        }
        
    def predict_proba(self, features: List[float]) -> Dict[str, Any]:
        """
        Get probabilities for all classes.
        """
        if not self.is_trained or self.model is None:
            return {"success": False, "error": "Model not trained"}
            
        try:
            X = np.array(features, dtype=np.float64).reshape(1, -1)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = self.scaler.transform(X)
            
            probs = self.model.predict_proba(X_scaled)[0]
            classes = self.model.classes_
            
            prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
            
            return {
                "success": True,
                "probabilities": prob_dict
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        if not self.is_trained:
            return {"trained": False, "message": "Model not trained"}
            
        return {
            "trained": True,
            "classes": self.metadata.get("classes", []),
            "accuracy": self.metadata.get("training_accuracy", 0),
            "sample_count": self.metadata.get("sample_count", 0)
        }

# Global instance
ml_manager = AnomalyDetector()

# Legacy wrappers
def train_model(data, labels): return ml_manager.train(data, labels)
def predict_footstep(features): 
    # This is legacy, but main.py will use predict_proba now
    return ml_manager.predict_proba(features)
def get_model_status(): return ml_manager.get_status()
def reset_model(): ml_manager.reset()
