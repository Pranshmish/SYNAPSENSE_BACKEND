"""
One-Class Anomaly Detection ML Module for HOME Footstep Recognition
Uses Isolation Forest for anomaly detection - trains ONLY on HOME samples.
Intruders/unknowns are detected as anomalies (outliers).
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from features import FEATURE_NAMES, get_feature_vector

MODELS_DIR = "models"
IF_MODEL_PATH = os.path.join(MODELS_DIR, "home_detector_if.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.pkl")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "home_detector_svm.pkl")

# Labels
LABEL_HOME = "HOME"
LABEL_INTRUDER = "INTRUDER"  # Detected as anomaly, not trained


class AnomalyDetector:
    """
    One-Class Anomaly Detection for HOME footstep recognition.
    Trains ONLY on HOME footsteps - intruders are detected as anomalies.
    
    Primary: IsolationForest (fast, robust)
    Secondary: OneClassSVM (stricter, optional)
    """
    
    def __init__(self):
        self._init_dirs()
        self.isolation_forest: Optional[IsolationForest] = None
        self.one_class_svm: Optional[OneClassSVM] = None
        self.scaler: Optional[StandardScaler] = None
        self.metadata: Dict[str, Any] = {}
        self.is_trained = False
        self.anomaly_threshold = -0.1  # Default threshold for decision function
        self.load_models()
        
    def _init_dirs(self):
        """Ensure model directory exists."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        
    def load_models(self) -> bool:
        """Load trained models from disk if available."""
        if (os.path.exists(IF_MODEL_PATH) and 
            os.path.exists(SCALER_PATH) and 
            os.path.exists(METADATA_PATH)):
            try:
                self.isolation_forest = joblib.load(IF_MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.metadata = joblib.load(METADATA_PATH)
                self.anomaly_threshold = self.metadata.get('anomaly_threshold', -0.1)
                
                # Load optional SVM if exists
                if os.path.exists(SVM_MODEL_PATH):
                    self.one_class_svm = joblib.load(SVM_MODEL_PATH)
                    
                self.is_trained = True
                print(f"[ML] Anomaly detector loaded. HOME samples: {self.metadata.get('home_samples', 'N/A')}")
                return True
            except Exception as e:
                print(f"[ML] Error loading models: {e}")
                self.is_trained = False
                return False
        else:
            print("[ML] No trained models found. Training required.")
            return False
            
    def save_models(self):
        """Save trained models to disk."""
        try:
            joblib.dump(self.isolation_forest, IF_MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            joblib.dump(self.metadata, METADATA_PATH)
            
            if self.one_class_svm is not None:
                joblib.dump(self.one_class_svm, SVM_MODEL_PATH)
                
            print("[ML] Models saved successfully.")
        except Exception as e:
            print(f"[ML] Error saving models: {e}")
            
    def reset(self):
        """Reset/delete all trained models."""
        self.isolation_forest = None
        self.one_class_svm = None
        self.scaler = None
        self.metadata = {}
        self.is_trained = False
        self.anomaly_threshold = -0.1
        
        for path in [IF_MODEL_PATH, SCALER_PATH, METADATA_PATH, SVM_MODEL_PATH]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"[ML] Error removing {path}: {e}")
                    
        print("[ML] Models reset.")
        
    def prepare_features(self, data: List[Dict[str, float]]) -> np.ndarray:
        """
        Convert list of feature dictionaries to numpy array.
        Ensures consistent feature ordering matching FEATURE_NAMES.
        """
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
        Train the anomaly detector on HOME samples ONLY.
        
        Args:
            data: List of feature dictionaries
            labels: List of labels (only HOME samples are used)
            
        Returns:
            Dictionary with training metrics and status
        """
        # Filter to HOME samples only
        home_data = []
        home_count = 0
        other_count = 0
        
        for item, label in zip(data, labels):
            label_upper = label.upper()
            if label_upper == LABEL_HOME or label_upper == "HOME_SAMPLE":
                home_data.append(item)
                home_count += 1
            else:
                other_count += 1
                
        # Validate - need enough HOME samples
        if home_count < 5:
            return {
                "success": False,
                "error": f"Not enough HOME samples. Need at least 5, got {home_count}.",
                "samples_provided": len(data),
                "home_samples": home_count
            }
            
        print(f"[ML] Training on {home_count} HOME samples (ignoring {other_count} other samples)")
            
        # Prepare feature matrix (HOME only)
        X = self.prepare_features(home_data)
        
        # Handle NaN/Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Initialize and fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest (primary model)
        # contamination=0.02 assumes ~2% of HOME samples might be outliers
        self.isolation_forest = IsolationForest(
            n_estimators=200,
            max_samples='auto',
            contamination=0.02,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        self.isolation_forest.fit(X_scaled)
        
        # Calculate anomaly scores for threshold calibration
        anomaly_scores = -self.isolation_forest.decision_function(X_scaled)
        
        # Auto-calibrate threshold: mean + 2*std of HOME scores
        # This ensures most HOME samples are classified correctly
        score_mean = float(np.mean(anomaly_scores))
        score_std = float(np.std(anomaly_scores))
        self.anomaly_threshold = score_mean + 2 * score_std  # Higher score = more anomalous
        
        # Predictions on training data (for validation)
        train_predictions = self.isolation_forest.predict(X_scaled)
        home_correctly_identified = int(np.sum(train_predictions == 1))  # 1 = inlier (HOME)
        home_misclassified = int(np.sum(train_predictions == -1))  # -1 = outlier
        
        training_accuracy = home_correctly_identified / len(train_predictions) * 100
        
        # K-Fold cross-validation for robustness estimate
        kf = KFold(n_splits=min(5, home_count), shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X_scaled):
            X_train_cv = X_scaled[train_idx]
            X_val_cv = X_scaled[val_idx]
            
            if_cv = IsolationForest(
                n_estimators=100,
                contamination=0.02,
                random_state=42,
                n_jobs=-1
            )
            if_cv.fit(X_train_cv)
            
            val_pred = if_cv.predict(X_val_cv)
            cv_accuracy = np.mean(val_pred == 1) * 100
            cv_scores.append(cv_accuracy)
            
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        
        # Optional: Train OneClassSVM as secondary model
        if home_count >= 10:
            try:
                self.one_class_svm = OneClassSVM(
                    kernel='rbf',
                    gamma='scale',
                    nu=0.05
                )
                self.one_class_svm.fit(X_scaled)
                print("[ML] OneClassSVM trained as secondary detector.")
            except Exception as e:
                print(f"[ML] OneClassSVM training skipped: {e}")
                self.one_class_svm = None
        
        # Feature importance (based on variance contribution)
        feature_variance = np.var(X_scaled, axis=0)
        feature_importance = feature_variance / (np.sum(feature_variance) + 1e-10)
        top_features = sorted(
            zip(FEATURE_NAMES, feature_importance), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Store metadata
        self.metadata = {
            "home_samples": home_count,
            "training_accuracy": training_accuracy,
            "cv_accuracy_mean": cv_mean,
            "cv_accuracy_std": cv_std,
            "anomaly_threshold": self.anomaly_threshold,
            "score_mean": score_mean,
            "score_std": score_std,
            "home_correctly_identified": home_correctly_identified,
            "home_misclassified": home_misclassified,
            "top_features": top_features,
            "model_type": "IsolationForest",
            "has_svm_backup": self.one_class_svm is not None
        }
        
        # Save models
        self.is_trained = True
        self.save_models()
        
        return {
            "success": True,
            "model_type": "One-Class Anomaly Detection (Isolation Forest)",
            "metrics": {
                "training_accuracy": round(training_accuracy, 2),
                "cv_accuracy": f"{cv_mean:.1f}% ± {cv_std:.1f}%",
                "home_recognition_rate": f"{home_correctly_identified}/{home_count}",
                "anomaly_threshold": round(self.anomaly_threshold, 4)
            },
            "dataset": {
                "home_samples_used": home_count,
                "other_samples_ignored": other_count,
                "total_provided": len(data)
            },
            "calibration": {
                "score_mean": round(score_mean, 4),
                "score_std": round(score_std, 4),
                "threshold": round(self.anomaly_threshold, 4)
            },
            "top_features": [{"name": name, "importance": float(imp)} for name, imp in top_features[:5]],
            "note": "Model trained on HOME samples only. Unknown patterns will be flagged as INTRUDER."
        }
        
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """
        Predict if footstep is HOME (normal) or INTRUDER (anomaly).
        
        Args:
            features: List of features matching FEATURE_NAMES order
            
        Returns:
            Dictionary with prediction, confidence, and anomaly scores
        """
        if not self.is_trained or self.isolation_forest is None or self.scaler is None:
            return {
                "success": False,
                "error": "Model not trained. Please collect HOME samples and train.",
                "prediction": None
            }
            
        try:
            # Reshape to 2D array
            X = np.array(features, dtype=np.float64).reshape(1, -1)
            
            # Handle NaN/Inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get Isolation Forest prediction
            # predict returns: 1 = inlier (HOME), -1 = outlier (INTRUDER)
            if_prediction = self.isolation_forest.predict(X_scaled)[0]
            
            # Get anomaly score (higher = more anomalous)
            # decision_function returns negative for outliers, positive for inliers
            raw_score = self.isolation_forest.decision_function(X_scaled)[0]
            anomaly_score = -raw_score  # Flip so higher = more anomalous
            
            # Calculate confidence based on distance from threshold
            score_mean = self.metadata.get('score_mean', 0)
            score_std = self.metadata.get('score_std', 1)
            
            # Normalize anomaly score to 0-1 range for confidence
            if if_prediction == 1:  # HOME
                # How confident we are this is HOME (lower anomaly score = higher confidence)
                z_score = (anomaly_score - score_mean) / (score_std + 1e-6)
                confidence = float(1 / (1 + np.exp(z_score)))  # Sigmoid
                confidence = max(0.5, min(0.99, confidence))  # Clamp
            else:  # INTRUDER (anomaly)
                # Higher anomaly score = higher confidence it's an intruder
                z_score = (anomaly_score - self.anomaly_threshold) / (score_std + 1e-6)
                confidence = float(1 / (1 + np.exp(-z_score)))  # Sigmoid
                confidence = max(0.5, min(0.99, confidence))
            
            # Map prediction to label
            prediction = LABEL_HOME if if_prediction == 1 else LABEL_INTRUDER
            is_intruder = prediction == LABEL_INTRUDER
            
            # Secondary SVM check (if available)
            svm_agrees = None
            if self.one_class_svm is not None:
                svm_prediction = self.one_class_svm.predict(X_scaled)[0]
                svm_agrees = (svm_prediction == 1 and prediction == LABEL_HOME) or \
                            (svm_prediction == -1 and prediction == LABEL_INTRUDER)
            
            return {
                "success": True,
                "prediction": prediction,
                "is_intruder": is_intruder,
                "confidence": round(confidence, 3),
                "anomaly_score": round(float(anomaly_score), 4),
                "threshold": round(float(self.anomaly_threshold), 4),
                "raw_score": round(float(raw_score), 4),
                "svm_agrees": svm_agrees,
                "probabilities": {
                    LABEL_HOME: round(1 - confidence if is_intruder else confidence, 3),
                    LABEL_INTRUDER: round(confidence if is_intruder else 1 - confidence, 3)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "prediction": None
            }
            
    def get_status(self) -> Dict[str, Any]:
        """Get current model status and metadata."""
        if not self.is_trained:
            return {
                "trained": False,
                "message": "No model trained. Collect HOME samples and train."
            }
            
        return {
            "trained": True,
            "model_type": "One-Class Anomaly Detection",
            "home_samples": self.metadata.get("home_samples", 0),
            "training_accuracy": self.metadata.get("training_accuracy", 0),
            "cv_accuracy": self.metadata.get("cv_accuracy_mean", 0),
            "anomaly_threshold": self.metadata.get("anomaly_threshold", 0),
            "has_svm_backup": self.metadata.get("has_svm_backup", False),
            "note": "Detects intruders as anomalies - no intruder training data needed"
        }


# Backward compatibility alias
class BinaryClassifier(AnomalyDetector):
    """Backward compatibility wrapper - redirects to AnomalyDetector."""
    pass


# Global instance for use by API
ml_manager = AnomalyDetector()


# Legacy compatibility functions
def train_model(data: List[Dict[str, float]], labels: List[str]) -> Dict[str, Any]:
    """Train the model (legacy wrapper)."""
    return ml_manager.train(data, labels)


def predict_footstep(features: List[float]) -> Dict[str, Any]:
    """Predict footstep (legacy wrapper)."""
    return ml_manager.predict(features)


def get_model_status() -> Dict[str, Any]:
    """Get model status (legacy wrapper)."""
    return ml_manager.get_status()


def reset_model():
    """Reset model (legacy wrapper)."""
    ml_manager.reset()
