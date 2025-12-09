"""
Robust Multi-Model Manager for HOME vs INTRUDER Classification
==============================================================
Uses person classifier + anomaly detection + decision layer.
All HOME datasets (Sameer, Dixit, Pandey) are real footsteps.
INTRUDER = any unrecognized person (anomaly detection).

Strategy:
1. Train person classifier on HOME data (Sameer vs Dixit vs Pandey)
2. Train IsolationForest anomaly detector on ALL merged HOME data
3. Decision layer combines classifier confidence + anomaly score
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ROBUST FEATURE SELECTION (use ALL features for 315+ samples)
# ============================================================================
ROBUST_FEATURES = None  # None = use all features (best for 300+ samples)

# Optional: Uncomment to use only select features for smaller datasets
# ROBUST_FEATURES = [
#     # Statistical (most stable across sessions)
#     'stat_mean', 'stat_std', 'stat_var', 'stat_max', 'stat_min', 'stat_rms',
#     'stat_skewness', 'stat_kurtosis', 'stat_energy', 'stat_power', 'stat_zcr',
#     'stat_peak_count', 'stat_peak_mean', 'stat_signal_entropy',
#     # FFT (frequency signatures)
#     'fft_total_energy', 'fft_centroid', 'fft_spread', 'fft_rolloff', 'fft_dominant_freq',
#     'fft_bass_energy', 'fft_low_mid_energy', 'fft_high_mid_energy', 'fft_high_energy',
#     # LIF neuron (biologically-inspired)
#     'lif_low_spike_count', 'lif_mid_spike_count', 'lif_high_spike_count',
#     'lif_low_spike_rate', 'lif_mid_spike_rate', 'lif_high_spike_rate',
#     'lif_total_spikes', 'lif_low_high_ratio'
# ]

# Thresholds for decision layer
HOME_CONFIDENT_THRESHOLD = 0.70  # Person classifier confidence for HOME
ANOMALY_SAFE_THRESHOLD = -0.10   # Above this = likely HOME
ANOMALY_STRONG_THRESHOLD = -0.30  # Below this = definitely INTRUDER

# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class TrainingResult:
    success: bool
    model_name: str
    accuracy: float = 0.0
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    classes: List[str] = field(default_factory=list)
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictionResult:
    label: str
    confidence: float
    person_label: str = "UNKNOWN"
    person_confidence: float = 0.0
    anomaly_score: float = 0.0
    decision_reason: str = ""
    all_probabilities: Dict[str, float] = field(default_factory=dict)

# ============================================================================
# FEATURE SELECTOR
# ============================================================================
class RobustFeatureSelector:
    """Selects robust features that generalize well across sessions."""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.selected_indices = []
        self.selected_names = []
        self._compute_indices()
    
    def _compute_indices(self):
        """Find indices of robust features in the full feature list."""
        if ROBUST_FEATURES is None:
            # Use all features
            self.selected_indices = list(range(len(self.feature_names)))
            self.selected_names = self.feature_names.copy()
            print(f"[RobustFeatureSelector] Using ALL {len(self.feature_names)} features")
        else:
            # Select specific features
            self.selected_indices = []
            self.selected_names = []
            
            for robust_feat in ROBUST_FEATURES:
                for idx, name in enumerate(self.feature_names):
                    if name == robust_feat:
                        self.selected_indices.append(idx)
                        self.selected_names.append(name)
                        break
            
            print(f"[RobustFeatureSelector] Selected {len(self.selected_indices)} features out of {len(self.feature_names)}")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select only robust features."""
        if len(self.selected_indices) == 0:
            print("[WARN] No features selected, using all features")
            return X
        if len(self.selected_indices) == X.shape[1]:
            return X  # All features selected
        return X[:, self.selected_indices]
    
    def get_feature_names(self) -> List[str]:
        return self.selected_names if self.selected_names else self.feature_names

# ============================================================================
# BASE MODEL CLASS
# ============================================================================
class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, model_id: str, display_name: str, description: str):
        self.model_id = model_id
        self.display_name = display_name
        self.description = description
        self.is_trained = False
        self.classes: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.feature_selector: Optional[RobustFeatureSelector] = None
        self.cv_accuracy: float = 0.0  # Store CV accuracy after training
        self.cv_std: float = 0.0  # Store CV standard deviation
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> TrainingResult:
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> PredictionResult:
        pass
    
    @abstractmethod
    def save(self, path: str) -> bool:
        pass
    
    @abstractmethod
    def load(self, path: str) -> bool:
        pass
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "is_trained": self.is_trained,
            "classes": self.classes,
            "cv_accuracy": self.cv_accuracy,
            "cv_std": self.cv_std
        }

# ============================================================================
# RANDOM FOREST ENSEMBLE MODEL (ONE-MODEL-PER-PERSON VERSION)
# ============================================================================
class RandomForestEnsembleModel(BaseModel):
    """
    One-Model-Per-Person RF classifier.
    
    Instead of a single multi-class classifier, trains separate binary
    classifiers for each person (Sameer, Pandey, Dixit).
    
    At prediction:
    - Runs all person models
    - If ANY model is confident → HOME (that person)
    - If NONE are confident → INTRUDER
    """
    
    def __init__(self):
        super().__init__(
            model_id="rf_ensemble",
            display_name="Random Forest Ensemble",
            description="One-model-per-person binary classifiers for HOME vs INTRUDER"
        )
        self.person_ensemble = None  # Will be imported when needed
        self.person_classes: List[str] = []
    
    def _get_ensemble(self):
        """Lazy import and get person ensemble manager"""
        if self.person_ensemble is None:
            from person_model_manager import get_person_ensemble
            self.person_ensemble = get_person_ensemble()
        return self.person_ensemble
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> TrainingResult:
        """
        Train one binary model per person.
        """
        try:
            print(f"\n{'='*60}")
            print("TRAINING ONE-MODEL-PER-PERSON RF ENSEMBLE")
            print(f"{'='*60}")
            
            from person_model_manager import train_person_models, get_person_ensemble
            
            # Train using person ensemble manager
            result = train_person_models(model_family="RF")
            
            if not result['success']:
                return TrainingResult(
                    success=False,
                    model_name=self.display_name,
                    message=result.get('message', 'Training failed')
                )
            
            # Update local state
            self.person_ensemble = get_person_ensemble()
            self.is_trained = True
            self.cv_accuracy = result.get('overall_accuracy', 0.0)
            self.person_classes = self.person_ensemble.person_names
            self.classes = ["HOME", "INTRUDER"]
            
            # Calculate cv_std from person results
            cv_accs = [r['cv_accuracy'] for r in result.get('person_results', []) if r['success']]
            self.cv_std = float(np.std(cv_accs)) if cv_accs else 0.0
            
            return TrainingResult(
                success=True,
                model_name=self.display_name,
                accuracy=self.cv_accuracy,
                cv_scores=cv_accs,
                cv_mean=self.cv_accuracy,
                cv_std=self.cv_std,
                classes=self.person_classes,
                message=f"Trained {len(self.person_classes)} person models with avg CV {self.cv_accuracy:.1%}",
                details={
                    "person_classes": self.person_classes,
                    "person_results": result.get('person_results', []),
                    "model_family": "RF",
                    "threshold": 0.80
                }
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return TrainingResult(
                success=False,
                model_name=self.display_name,
                message=f"Training failed: {str(e)}"
            )
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Predict HOME vs INTRUDER using person ensemble.
        """
        if not self.is_trained:
            return PredictionResult(label="UNKNOWN", confidence=0.0, decision_reason="Model not trained")
        
        try:
            from person_model_manager import predict_person
            
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            result = predict_person(X, model_family="RF")
            
            return PredictionResult(
                label=result['final_label'],
                confidence=result['confidence'],
                person_label=result.get('matched_person') or "UNKNOWN",
                person_confidence=result['confidence'] if result['matched_person'] else 0.0,
                anomaly_score=0.0,  # Not used in person ensemble
                decision_reason=result['decision_reason'],
                all_probabilities=result.get('person_probs', {})
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return PredictionResult(label="UNKNOWN", confidence=0.0, decision_reason=f"Error: {str(e)}")
    
    def save(self, path: str) -> bool:
        """Save model status to disk (person models are saved separately)."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model_data = {
                'is_trained': self.is_trained,
                'cv_accuracy': self.cv_accuracy,
                'cv_std': self.cv_std,
                'person_classes': self.person_classes,
                'classes': self.classes
            }
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"[SAVE] RF Ensemble status saved to {path} (accuracy={self.cv_accuracy:.1%})")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """Load model from disk and restore person ensemble."""
        try:
            # First try to load person ensemble models
            from person_model_manager import get_person_ensemble
            self.person_ensemble = get_person_ensemble()
            
            if self.person_ensemble.is_trained:
                self.is_trained = True
                self.cv_accuracy = self.person_ensemble.cv_accuracy
                self.person_classes = self.person_ensemble.person_names
                self.classes = ["HOME", "INTRUDER"]
                print(f"[LOAD] RF Person Ensemble loaded (accuracy={self.cv_accuracy:.1%})")
                print(f"  Person classes: {self.person_classes}")
                return True
            
            # Fallback: try to load from pickle file
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    model_data = pickle.load(f)
                self.is_trained = model_data.get('is_trained', False)
                self.cv_accuracy = model_data.get('cv_accuracy', 0.0)
                self.cv_std = model_data.get('cv_std', 0.0)
                self.person_classes = model_data.get('person_classes', [])
                self.classes = model_data.get('classes', ["HOME", "INTRUDER"])
                return self.is_trained
            
            return False
        except Exception as e:
            print(f"[ERROR] Failed to load: {e}")
            return False

# ============================================================================
# MLP CLASSIFIER MODEL (One-Model-Per-Person)
# ============================================================================
class MLPClassifierModel(BaseModel):
    """
    MLP model using one-model-per-person approach.
    
    Uses sklearn MLPClassifier for each person's binary model.
    """
    
    def __init__(self):
        super().__init__(
            model_id="mlp_classifier",
            display_name="MLP Neural Network",
            description="One-model-per-person MLP classifiers for HOME vs INTRUDER"
        )
        self.person_ensemble = None
        self.person_classes: List[str] = []
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> TrainingResult:
        """Train one MLP binary model per person."""
        try:
            print(f"\n{'='*60}")
            print("TRAINING ONE-MODEL-PER-PERSON MLP ENSEMBLE")
            print(f"{'='*60}")
            
            from person_model_manager import train_person_models, get_person_ensemble
            
            # Train using person ensemble manager with MLP models
            result = train_person_models(model_family="MLP")
            
            if not result['success']:
                return TrainingResult(
                    success=False,
                    model_name=self.display_name,
                    message=result.get('message', 'Training failed')
                )
            
            # Update local state
            self.person_ensemble = get_person_ensemble()
            self.is_trained = True
            self.cv_accuracy = result.get('overall_accuracy', 0.0)
            self.person_classes = self.person_ensemble.person_names
            self.classes = ["HOME", "INTRUDER"]
            
            # Calculate cv_std from person results
            cv_accs = [r['cv_accuracy'] for r in result.get('person_results', []) if r['success']]
            self.cv_std = float(np.std(cv_accs)) if cv_accs else 0.0
            
            return TrainingResult(
                success=True,
                model_name=self.display_name,
                accuracy=self.cv_accuracy,
                cv_scores=cv_accs,
                cv_mean=self.cv_accuracy,
                cv_std=self.cv_std,
                classes=self.person_classes,
                message=f"Trained {len(self.person_classes)} MLP person models with avg CV {self.cv_accuracy:.1%}",
                details={
                    "person_classes": self.person_classes,
                    "person_results": result.get('person_results', []),
                    "model_family": "MLP",
                    "threshold": 0.80
                }
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return TrainingResult(
                success=False,
                model_name=self.display_name,
                message=f"MLP training failed: {str(e)}"
            )
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Predict using MLP person ensemble."""
        if not self.is_trained:
            return PredictionResult(label="UNKNOWN", confidence=0.0, decision_reason="MLP not trained")
        
        try:
            from person_model_manager import predict_person
            
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            result = predict_person(X, model_family="MLP")
            
            return PredictionResult(
                label=result['final_label'],
                confidence=result['confidence'],
                person_label=result.get('matched_person') or "UNKNOWN",
                person_confidence=result['confidence'] if result['matched_person'] else 0.0,
                anomaly_score=0.0,
                decision_reason=result['decision_reason'],
                all_probabilities=result.get('person_probs', {})
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return PredictionResult(label="UNKNOWN", confidence=0.0, decision_reason=f"Error: {str(e)}")
    
    def save(self, path: str) -> bool:
        """Save model status (person models saved separately)."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model_data = {
                'is_trained': self.is_trained,
                'cv_accuracy': self.cv_accuracy,
                'cv_std': self.cv_std,
                'person_classes': self.person_classes,
                'classes': self.classes
            }
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            return True
        except:
            return False
    
    def load(self, path: str) -> bool:
        """Load model from disk and restore person ensemble."""
        try:
            from person_model_manager import get_person_ensemble
            self.person_ensemble = get_person_ensemble()
            
            # Check if MLP models are trained
            mlp_models = [k for k in self.person_ensemble.person_models.keys() if k.startswith("MLP_")]
            
            if len(mlp_models) >= 2:
                self.is_trained = True
                self.person_classes = self.person_ensemble.person_names
                self.classes = ["HOME", "INTRUDER"]
                
                # Calculate average accuracy from loaded models
                cv_accs = []
                for key in mlp_models:
                    model = self.person_ensemble.person_models.get(key)
                    if model and model.is_trained:
                        cv_accs.append(model.cv_accuracy)
                
                self.cv_accuracy = float(np.mean(cv_accs)) if cv_accs else 0.0
                print(f"[LOAD] MLP Person Ensemble loaded (accuracy={self.cv_accuracy:.1%})")
                return True
            
            return False
        except Exception as e:
            print(f"[ERROR] Failed to load MLP: {e}")
            return False

# ============================================================================
# HYBRID LSTM-SNN MODEL (Placeholder)
# ============================================================================
class HybridLSTMSNNModel(BaseModel):
    """Hybrid LSTM + Spiking Neural Network - placeholder for future implementation."""
    
    def __init__(self):
        super().__init__(
            model_id="hybrid_lstm_snn",
            display_name="Hybrid LSTM-SNN",
            description="Advanced temporal + spiking model (coming soon)"
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> TrainingResult:
        return TrainingResult(
            success=False,
            model_name=self.display_name,
            message="Hybrid LSTM-SNN not yet implemented. Use RF Ensemble or MLP."
        )
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        return PredictionResult(
            label="UNKNOWN",
            confidence=0.0,
            decision_reason="Hybrid LSTM-SNN not yet implemented"
        )
    
    def save(self, path: str) -> bool:
        return False
    
    def load(self, path: str) -> bool:
        return False

# ============================================================================
# MODEL REGISTRY
# ============================================================================
class ModelRegistry:
    """Registry of all available models."""
    
    _models: Dict[str, BaseModel] = {}
    
    @classmethod
    def register(cls, model: BaseModel):
        cls._models[model.model_id] = model
        print(f"[Registry] Registered model: {model.model_id}")
    
    @classmethod
    def get(cls, model_id: str) -> Optional[BaseModel]:
        return cls._models.get(model_id)
    
    @classmethod
    def get_all(cls) -> Dict[str, BaseModel]:
        return cls._models
    
    @classmethod
    def list_models(cls) -> List[Dict[str, Any]]:
        return [
            {
                "id": m.model_id,
                "name": m.display_name,
                "description": m.description,
                "is_trained": m.is_trained,
                "classes": m.classes,
                "cv_accuracy": m.cv_accuracy,
                "cv_std": m.cv_std
            }
            for m in cls._models.values()
        ]

# Register all models
ModelRegistry.register(RandomForestEnsembleModel())
ModelRegistry.register(MLPClassifierModel())
ModelRegistry.register(HybridLSTMSNNModel())

# ============================================================================
# MULTI-MODEL MANAGER
# ============================================================================
class MultiModelManager:
    """
    Manages multiple models and provides unified interface.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.active_model_id: str = "rf_ensemble"
        os.makedirs(models_dir, exist_ok=True)
        
        # Try to load saved models
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all saved models from disk."""
        for model_id, model in ModelRegistry.get_all().items():
            model_path = os.path.join(self.models_dir, f"{model_id}.pkl")
            if model.load(model_path):
                print(f"[Manager] Loaded {model_id}")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with their status."""
        models = ModelRegistry.list_models()
        for m in models:
            m['is_active'] = (m['id'] == self.active_model_id)
        return models
    
    def set_active_model(self, model_id: str) -> bool:
        """Set the active model for predictions."""
        if ModelRegistry.get(model_id):
            self.active_model_id = model_id
            print(f"[Manager] Active model set to: {model_id}")
            return True
        return False
    
    def train_model(self, model_id: str, datasets: List[str], dataset_dir: str = "dataset") -> TrainingResult:
        """
        Train a specific model on selected datasets.
        
        For RF Ensemble: expects HOME_* datasets, trains person classifier + anomaly detector
        """
        model = ModelRegistry.get(model_id)
        if not model:
            return TrainingResult(success=False, model_name=model_id, message=f"Model {model_id} not found")
        
        # Load and merge datasets
        X_all, y_all, feature_names = self._load_datasets(datasets, dataset_dir)
        
        if X_all is None or len(X_all) == 0:
            return TrainingResult(success=False, model_name=model.display_name, message="No data loaded from datasets")
        
        print(f"\n[Manager] Training {model_id} on {len(X_all)} samples from {len(datasets)} datasets")
        
        # Train
        result = model.train(X_all, y_all, feature_names)
        
        # Save if successful
        if result.success:
            model_path = os.path.join(self.models_dir, f"{model_id}.pkl")
            model.save(model_path)
        
        return result
    
    def _load_datasets(self, datasets: List[str], dataset_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """
        Load and merge multiple datasets.
        
        For HOME_* datasets, extracts person name as label (e.g., HOME_Sameer → Sameer)
        Handles directory structure: HOME_Dixit/features_HOME_Dixit.csv
        """
        all_X = []
        all_y = []
        feature_names = []
        
        for dataset_name in datasets:
            # Find the CSV file
            csv_path = None
            
            # Try direct path first
            if os.path.exists(dataset_name):
                csv_path = dataset_name
            else:
                # Check for HOME_* format (e.g., HOME_Dixit → HOME_Dixit/features_HOME_Dixit.csv)
                if dataset_name.startswith("HOME_"):
                    potential_path = os.path.join(dataset_dir, dataset_name, f"features_{dataset_name}.csv")
                    if os.path.exists(potential_path):
                        csv_path = potential_path
                
                # Also try without HOME_ prefix
                if csv_path is None:
                    for subdir in os.listdir(dataset_dir) if os.path.exists(dataset_dir) else []:
                        subdir_path = os.path.join(dataset_dir, subdir)
                        if os.path.isdir(subdir_path):
                            # Match HOME_Dixit to HOME_Dixit/features_HOME_Dixit.csv
                            potential_path = os.path.join(subdir_path, f"features_{subdir}.csv")
                            if os.path.exists(potential_path) and subdir == dataset_name:
                                csv_path = potential_path
                                break
            
            if csv_path is None:
                print(f"[WARN] Dataset not found: {dataset_name}")
                continue
            
            try:
                df = pd.read_csv(csv_path)
                print(f"[LOAD] {dataset_name}: {len(df)} samples from {csv_path}")
                
                if feature_names == []:
                    # Get feature columns (exclude label columns and metadata)
                    exclude_cols = ['label', '_label', '_class', '_timestamp']
                    feature_names = [c for c in df.columns if c not in exclude_cols]
                
                X = df[feature_names].values
                
                # Extract person name from dataset name (HOME_Sameer → Sameer)
                if dataset_name.startswith("HOME_"):
                    person_name = dataset_name.replace("HOME_", "")
                else:
                    person_name = dataset_name
                
                y = np.array([person_name] * len(X))
                
                all_X.append(X)
                all_y.append(y)
                
            except Exception as e:
                print(f"[ERROR] Failed to load {dataset_name}: {e}")
                continue
        
        if len(all_X) == 0:
            return None, None, []
        
        X_merged = np.vstack(all_X)
        y_merged = np.concatenate(all_y)
        
        print(f"[MERGED] Total: {len(X_merged)} samples, {len(np.unique(y_merged))} classes")
        print(f"  Classes: {np.unique(y_merged)}")
        
        return X_merged, y_merged, feature_names
    
    def predict(self, X: np.ndarray, model_id: Optional[str] = None) -> PredictionResult:
        """
        Make prediction using specified model or active model.
        """
        model_id = model_id or self.active_model_id
        model = ModelRegistry.get(model_id)
        
        if not model:
            return PredictionResult(label="UNKNOWN", confidence=0.0, decision_reason=f"Model {model_id} not found")
        
        if not model.is_trained:
            return PredictionResult(label="UNKNOWN", confidence=0.0, decision_reason=f"Model {model_id} not trained")
        
        return model.predict(X)
    
    def get_model_status(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific model."""
        model = ModelRegistry.get(model_id)
        if model:
            status = model.get_status()
            status['is_active'] = (model_id == self.active_model_id)
            return status
        return None

# ============================================================================
# GLOBAL INSTANCE
# ============================================================================
_manager_instance: Optional[MultiModelManager] = None

def get_model_manager() -> MultiModelManager:
    """Get or create the global model manager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = MultiModelManager()
    return _manager_instance

# ============================================================================
# CONVENIENCE FUNCTIONS FOR API
# ============================================================================
def get_available_models() -> List[Dict[str, Any]]:
    """Get list of available models."""
    return get_model_manager().get_available_models()

def train_selected_model(model_id: str, datasets: List[str]) -> Dict[str, Any]:
    """Train a model on selected datasets."""
    manager = get_model_manager()
    result = manager.train_model(model_id, datasets)
    return {
        "success": result.success,
        "model_name": result.model_name,
        "accuracy": result.accuracy,
        "cv_mean": result.cv_mean,
        "cv_std": result.cv_std,
        "cv_scores": result.cv_scores,
        "classes": result.classes,
        "message": result.message,
        "details": result.details
    }

def predict_with_model(features: np.ndarray, model_id: Optional[str] = None) -> Dict[str, Any]:
    """Make prediction with specified or active model."""
    manager = get_model_manager()
    result = manager.predict(features, model_id)
    return {
        "label": result.label,
        "confidence": result.confidence,
        "person_label": result.person_label,
        "person_confidence": result.person_confidence,
        "anomaly_score": result.anomaly_score,
        "decision_reason": result.decision_reason,
        "all_probabilities": result.all_probabilities
    }

def set_active_model(model_id: str) -> bool:
    """Set the active model."""
    return get_model_manager().set_active_model(model_id)
