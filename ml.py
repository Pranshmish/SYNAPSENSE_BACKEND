"""
Binary Classification ML Module for HOME vs INTRUDER Detection
Uses RandomForest as primary classifier with comprehensive feature set.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from features import FEATURE_NAMES, get_feature_vector

MODELS_DIR = "models"
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.pkl")

# Binary classification labels
LABEL_HOME = "HOME"
LABEL_INTRUDER = "INTRUDER"


class BinaryClassifier:
    """
    Binary classifier for HOME vs INTRUDER footstep detection.
    Uses RandomForest with StandardScaler preprocessing.
    """
    
    def __init__(self):
        self._init_dirs()
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.metadata: Dict[str, Any] = {}
        self.is_trained = False
        self.load_models()
        
    def _init_dirs(self):
        """Ensure model directory exists."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        
    def load_models(self) -> bool:
        """Load trained models from disk if available."""
        if (os.path.exists(RF_MODEL_PATH) and 
            os.path.exists(SCALER_PATH) and 
            os.path.exists(METADATA_PATH)):
            try:
                self.model = joblib.load(RF_MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.metadata = joblib.load(METADATA_PATH)
                self.is_trained = True
                print(f"[ML] Models loaded. Accuracy: {self.metadata.get('accuracy', 'N/A')}")
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
            joblib.dump(self.model, RF_MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            joblib.dump(self.metadata, METADATA_PATH)
            print("[ML] Models saved successfully.")
        except Exception as e:
            print(f"[ML] Error saving models: {e}")
            
    def reset(self):
        """Reset/delete all trained models."""
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
        """
        Convert list of feature dictionaries to numpy array.
        Ensures consistent feature ordering matching FEATURE_NAMES.
        """
        X = []
        for item in data:
            # Handle both dict and list inputs
            if isinstance(item, dict):
                row = [item.get(name, 0.0) for name in FEATURE_NAMES]
            else:
                row = list(item)  # Assume it's already a list
            X.append(row)
        return np.array(X, dtype=np.float64)
        
    def train(self, data: List[Dict[str, float]], labels: List[str]) -> Dict[str, Any]:
        """
        Train the binary classifier on provided data.
        
        Args:
            data: List of feature dictionaries
            labels: List of labels (HOME or INTRUDER)
            
        Returns:
            Dictionary with training metrics and status
        """
        # Validate inputs
        if len(data) < 10:
            return {
                "success": False,
                "error": "Not enough data. Need at least 10 samples.",
                "samples_provided": len(data)
            }
            
        # Convert labels to binary
        binary_labels = []
        home_count = 0
        intruder_count = 0
        
        for label in labels:
            label_upper = label.upper()
            if label_upper == LABEL_HOME or label_upper == "HOME_SAMPLE":
                binary_labels.append(LABEL_HOME)
                home_count += 1
            else:
                binary_labels.append(LABEL_INTRUDER)
                intruder_count += 1
                
        # Check class balance
        if home_count < 3 or intruder_count < 3:
            return {
                "success": False,
                "error": f"Need at least 3 samples per class. HOME: {home_count}, INTRUDER: {intruder_count}"
            }
            
        # Prepare feature matrix
        X = self.prepare_features(data)
        y = np.array(binary_labels)
        
        # Handle NaN/Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Initialize and fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data (stratified to maintain class balance)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, stratify=y, random_state=42
            )
        except ValueError:
            # Fallback if stratification fails
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
        # Train RandomForest
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        
        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, pos_label=LABEL_HOME, zero_division=0))
        recall = float(recall_score(y_test, y_pred, pos_label=LABEL_HOME, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, pos_label=LABEL_HOME, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[LABEL_HOME, LABEL_INTRUDER])
        
        # Cross-validation for more robust estimate
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=min(5, len(y) // 2), scoring='accuracy')
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        
        # Feature importance
        feature_importance = dict(zip(FEATURE_NAMES, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Store metadata
        self.metadata = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "cv_accuracy_mean": cv_mean,
            "cv_accuracy_std": cv_std,
            "confusion_matrix": cm.tolist(),
            "home_samples": home_count,
            "intruder_samples": intruder_count,
            "total_samples": len(data),
            "top_features": top_features,
            "classes": [LABEL_HOME, LABEL_INTRUDER]
        }
        
        # Save models
        self.is_trained = True
        self.save_models()
        
        return {
            "success": True,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "cv_accuracy": f"{cv_mean:.3f} ± {cv_std:.3f}"
            },
            "dataset": {
                "total": len(data),
                "home": home_count,
                "intruder": intruder_count
            },
            "confusion_matrix": {
                "true_home_pred_home": int(cm[0, 0]),
                "true_home_pred_intruder": int(cm[0, 1]),
                "true_intruder_pred_home": int(cm[1, 0]),
                "true_intruder_pred_intruder": int(cm[1, 1])
            },
            "top_features": [{"name": name, "importance": float(imp)} for name, imp in top_features[:5]],
            "classes": [LABEL_HOME, LABEL_INTRUDER]
        }
        
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """
        Predict HOME or INTRUDER from feature vector.
        
        Args:
            features: List of features matching FEATURE_NAMES order
            
        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        if not self.is_trained or self.model is None or self.scaler is None:
            return {
                "success": False,
                "error": "Model not trained. Please train first.",
                "prediction": None
            }
            
        try:
            # Reshape to 2D array
            X = np.array(features, dtype=np.float64).reshape(1, -1)
            
            # Handle NaN/Inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get prediction and probabilities
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Map probabilities to labels
            prob_dict = {
                str(cls): float(prob) 
                for cls, prob in zip(self.model.classes_, probabilities)
            }
            
            confidence = float(prob_dict[prediction])
            is_intruder = prediction == LABEL_INTRUDER
            
            return {
                "success": True,
                "prediction": prediction,
                "is_intruder": is_intruder,
                "confidence": confidence,
                "probabilities": prob_dict
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
                "message": "No model trained. Please collect data and train."
            }
            
        return {
            "trained": True,
            "accuracy": self.metadata.get("accuracy", 0),
            "precision": self.metadata.get("precision", 0),
            "recall": self.metadata.get("recall", 0),
            "f1": self.metadata.get("f1", 0),
            "total_samples": self.metadata.get("total_samples", 0),
            "home_samples": self.metadata.get("home_samples", 0),
            "intruder_samples": self.metadata.get("intruder_samples", 0),
            "classes": self.metadata.get("classes", [LABEL_HOME, LABEL_INTRUDER])
        }


# Global instance for use by API
ml_manager = BinaryClassifier()


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
