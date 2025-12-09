"""
One-Model-Per-Person HOME/INTRUDER Classification
=================================================
Trains a separate binary model for each home person (Sameer, Pandey, Dixit).
At prediction time, runs ALL person-models and decides HOME vs INTRUDER
based on which models match.

Strategy:
- Binary classifiers: Sameer vs Not-Sameer, Pandey vs Not-Pandey, Dixit vs Not-Dixit
- For a new footstep: compute each model's probability
- If ANY model is confidently positive → HOME (that person)
- If NONE are confident → INTRUDER
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
PERSON_MATCH_THRESHOLD = 0.80  # If best_prob >= this → HOME

# Use absolute paths based on script location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PERSON_MODELS_DIR = os.path.join(_SCRIPT_DIR, "models", "person")
DATASET_DIR = os.path.join(_SCRIPT_DIR, "dataset")

# Robust features for person classification
PERSON_MODEL_FEATURES = [
    # Statistical (most stable across sessions)
    'stat_std', 'stat_rms', 'stat_kurtosis', 'stat_zero_crossings', 'stat_peak_count',
    'stat_signal_entropy', 
    # FFT (frequency signatures)
    'fft_total_energy', 'fft_centroid', 'fft_spread', 'fft_rolloff', 'fft_flatness',
    # MFCC (temporal envelope)
    'mfcc_0', 'mfcc_1', 'mfcc_2', 'mfcc_3',
    # LIF neuron (biologically-inspired)
    'lif_low_spike_rate', 'lif_mid_spike_rate', 'lif_high_spike_rate', 'lif_total_spikes'
]

# Model configurations
RF_CONFIG = {
    'n_estimators': 150,
    'max_depth': 10,
    'min_samples_leaf': 2,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

MLP_CONFIG = {
    'hidden_layer_sizes': (64, 32),
    'alpha': 0.001,
    'max_iter': 500,
    'early_stopping': True,
    'validation_fraction': 0.2,
    'random_state': 42
}

# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class PersonModelResult:
    """Result from a single person model prediction"""
    person_name: str
    probability: float
    is_match: bool

@dataclass
class PersonEnsemblePrediction:
    """Combined prediction from all person models"""
    final_label: str  # HOME or INTRUDER
    matched_person: Optional[str]  # Sameer, Pandey, Dixit, or None
    confidence: float
    person_probs: Dict[str, float]
    person_match_threshold: float
    model_family: str  # RF or MLP
    decision_reason: str

@dataclass
class PersonModelTrainResult:
    """Training result for a single person model"""
    person_name: str
    success: bool
    accuracy: float = 0.0
    cv_accuracy: float = 0.0
    cv_std: float = 0.0
    roc_auc: float = 0.0
    n_positive: int = 0
    n_negative: int = 0
    message: str = ""

@dataclass
class EnsembleTrainResult:
    """Training result for all person models"""
    success: bool
    model_family: str
    person_results: List[PersonModelTrainResult] = field(default_factory=list)
    overall_accuracy: float = 0.0
    message: str = ""

# ============================================================================
# FEATURE SELECTOR FOR PERSON MODELS
# ============================================================================
class PersonFeatureSelector:
    """Selects features for person classification"""
    
    def __init__(self, all_feature_names: List[str]):
        self.all_feature_names = all_feature_names
        self.selected_indices = []
        self.selected_names = []
        self._compute_indices()
    
    def _compute_indices(self):
        """Find indices of selected features"""
        self.selected_indices = []
        self.selected_names = []
        
        for feat in PERSON_MODEL_FEATURES:
            for idx, name in enumerate(self.all_feature_names):
                if name == feat:
                    self.selected_indices.append(idx)
                    self.selected_names.append(name)
                    break
        
        # If no specific features found, use all
        if len(self.selected_indices) == 0:
            self.selected_indices = list(range(len(self.all_feature_names)))
            self.selected_names = self.all_feature_names.copy()
            print(f"[PersonFeatureSelector] Using ALL {len(self.all_feature_names)} features")
        else:
            print(f"[PersonFeatureSelector] Selected {len(self.selected_indices)} features")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select only the chosen features"""
        if len(self.selected_indices) == 0:
            return X
        if len(self.selected_indices) == X.shape[1]:
            return X
        return X[:, self.selected_indices]

# ============================================================================
# SINGLE PERSON MODEL (Binary Classifier)
# ============================================================================
class SinglePersonModel:
    """
    Binary classifier for one person: Person vs Not-Person
    Can use either RandomForest or MLP as the base model.
    """
    
    def __init__(self, person_name: str, model_type: str = "RF"):
        self.person_name = person_name
        self.model_type = model_type.upper()  # "RF" or "MLP"
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector: Optional[PersonFeatureSelector] = None
        self.is_trained = False
        self.cv_accuracy = 0.0
        self.cv_std = 0.0
        self.roc_auc = 0.0
    
    def _create_model(self):
        """Create the underlying classifier"""
        if self.model_type == "RF":
            return RandomForestClassifier(**RF_CONFIG)
        else:  # MLP
            return MLPClassifier(**MLP_CONFIG)
    
    def train(self, X_pos: np.ndarray, X_neg: np.ndarray, feature_names: List[str]) -> PersonModelTrainResult:
        """
        Train binary classifier.
        
        Args:
            X_pos: Positive samples (this person's footsteps)
            X_neg: Negative samples (other persons' footsteps)
            feature_names: List of feature column names
        
        Returns:
            PersonModelTrainResult with metrics
        """
        try:
            print(f"\n[{self.person_name}Model] Training {self.model_type} classifier...")
            print(f"  Positive samples: {len(X_pos)}, Negative samples: {len(X_neg)}")
            
            # Feature selection
            self.feature_selector = PersonFeatureSelector(feature_names)
            X_pos_sel = self.feature_selector.transform(X_pos)
            X_neg_sel = self.feature_selector.transform(X_neg)
            
            # Combine and create labels
            X = np.vstack([X_pos_sel, X_neg_sel])
            y = np.array([1] * len(X_pos) + [0] * len(X_neg))
            
            # Scale
            X_scaled = self.scaler.fit_transform(X)
            
            # Create and train model
            self.model = self._create_model()
            
            # Cross-validation
            n_splits = min(5, min(len(X_pos), len(X_neg)))
            if n_splits >= 2:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
                self.cv_accuracy = float(cv_scores.mean())
                self.cv_std = float(cv_scores.std())
                print(f"  CV Accuracy: {self.cv_accuracy:.1%} (+/- {self.cv_std*2:.1%})")
            else:
                cv_scores = []
                self.cv_accuracy = 0.0
                self.cv_std = 0.0
            
            # Train on full data
            self.model.fit(X_scaled, y)
            
            # Training accuracy
            train_pred = self.model.predict(X_scaled)
            train_acc = accuracy_score(y, train_pred)
            
            # ROC-AUC if possible
            try:
                train_proba = self.model.predict_proba(X_scaled)[:, 1]
                self.roc_auc = roc_auc_score(y, train_proba)
            except:
                self.roc_auc = 0.0
            
            self.is_trained = True
            
            return PersonModelTrainResult(
                person_name=self.person_name,
                success=True,
                accuracy=train_acc,
                cv_accuracy=self.cv_accuracy,
                cv_std=self.cv_std,
                roc_auc=self.roc_auc,
                n_positive=len(X_pos),
                n_negative=len(X_neg),
                message=f"{self.person_name} model trained: CV={self.cv_accuracy:.1%}"
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return PersonModelTrainResult(
                person_name=self.person_name,
                success=False,
                message=f"Training failed: {str(e)}"
            )
    
    def predict_proba(self, X: np.ndarray) -> float:
        """
        Get probability that this sample is from this person.
        
        Returns:
            Probability (0-1) that this is the person
        """
        if not self.is_trained:
            return 0.0
        
        try:
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            X_sel = self.feature_selector.transform(X)
            X_scaled = self.scaler.transform(X_sel)
            
            proba = self.model.predict_proba(X_scaled)[0][1]  # Probability of class 1 (this person)
            return float(proba)
        except Exception as e:
            print(f"[{self.person_name}Model] Predict error: {e}")
            return 0.0
    
    def save(self, path: str) -> bool:
        """Save model to disk"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {
                'person_name': self.person_name,
                'model_type': self.model_type,
                'model': self.model,
                'scaler': self.scaler,
                'feature_selector_names': self.feature_selector.selected_names if self.feature_selector else [],
                'feature_selector_indices': self.feature_selector.selected_indices if self.feature_selector else [],
                'is_trained': self.is_trained,
                'cv_accuracy': self.cv_accuracy,
                'cv_std': self.cv_std,
                'roc_auc': self.roc_auc
            }
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            print(f"[{self.person_name}Model] Saved to {path}")
            return True
        except Exception as e:
            print(f"[{self.person_name}Model] Save error: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """Load model from disk"""
        try:
            if not os.path.exists(path):
                return False
            
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.person_name = data['person_name']
            self.model_type = data['model_type']
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = data['is_trained']
            self.cv_accuracy = data.get('cv_accuracy', 0.0)
            self.cv_std = data.get('cv_std', 0.0)
            self.roc_auc = data.get('roc_auc', 0.0)
            
            # Reconstruct feature selector
            if data.get('feature_selector_names'):
                self.feature_selector = PersonFeatureSelector([])
                self.feature_selector.selected_names = data['feature_selector_names']
                self.feature_selector.selected_indices = data['feature_selector_indices']
            
            print(f"[{self.person_name}Model] Loaded (CV={self.cv_accuracy:.1%})")
            return True
        except Exception as e:
            print(f"[{self.person_name}Model] Load error: {e}")
            return False

# ============================================================================
# PERSON ENSEMBLE MANAGER
# ============================================================================
class PersonEnsembleManager:
    """
    Manages multiple person-specific binary models.
    Coordinates training and prediction across all person models.
    """
    
    def __init__(self, models_dir: str = PERSON_MODELS_DIR):
        self.models_dir = models_dir
        self.person_models: Dict[str, SinglePersonModel] = {}
        self.model_family = "RF"  # Current model family (RF or MLP)
        self.is_trained = False
        self.person_names: List[str] = []
        self.cv_accuracy = 0.0
        os.makedirs(models_dir, exist_ok=True)
        
        # Try to load existing models
        self._load_models()
    
    def _load_models(self):
        """Load any existing person models"""
        for model_type in ["RF", "MLP"]:
            subdir = os.path.join(self.models_dir, model_type.lower())
            if os.path.exists(subdir):
                for filename in os.listdir(subdir):
                    if filename.endswith("Model.pkl"):
                        person_name = filename.replace("Model.pkl", "")
                        model = SinglePersonModel(person_name, model_type)
                        if model.load(os.path.join(subdir, filename)):
                            key = f"{model_type}_{person_name}"
                            self.person_models[key] = model
                            if person_name not in self.person_names:
                                self.person_names.append(person_name)
        
        # Check if we have a trained set
        rf_count = sum(1 for k in self.person_models if k.startswith("RF_"))
        mlp_count = sum(1 for k in self.person_models if k.startswith("MLP_"))
        
        if rf_count >= 2:
            self.model_family = "RF"
            self.is_trained = True
        elif mlp_count >= 2:
            self.model_family = "MLP"
            self.is_trained = True
        
        print(f"[PersonEnsemble] Loaded {len(self.person_models)} person models (RF:{rf_count}, MLP:{mlp_count})")
    
    def train(self, model_family: str = "RF", dataset_dir: str = DATASET_DIR) -> EnsembleTrainResult:
        """
        Train all person models.
        
        For each person:
        1. Load that person's CSV as positive samples
        2. Load other persons' CSVs as negative samples
        3. Train binary classifier
        """
        self.model_family = model_family.upper()
        print(f"\n{'='*60}")
        print(f"TRAINING PERSON ENSEMBLE ({self.model_family})")
        print(f"Dataset directory: {dataset_dir}")
        print(f"{'='*60}")
        
        # Find all HOME_* datasets
        person_datasets = {}
        
        if os.path.exists(dataset_dir):
            subdirs = os.listdir(dataset_dir)
            print(f"[DEBUG] Found subdirectories: {subdirs}")
            for subdir in subdirs:
                if subdir.startswith("HOME_"):
                    person_name = subdir.replace("HOME_", "")
                    csv_path = os.path.join(dataset_dir, subdir, f"features_{subdir}.csv")
                    if os.path.exists(csv_path):
                        try:
                            # Get file modification time for debugging
                            file_stat = os.stat(csv_path)
                            mod_time = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                            
                            df = pd.read_csv(csv_path)
                            # Get feature columns
                            exclude = ['label', '_label', '_class', '_timestamp', '_person']
                            features = [c for c in df.columns if c not in exclude]
                            X = df[features].values
                            
                            # Get timestamp range from data
                            if '_timestamp' in df.columns:
                                timestamps = df['_timestamp'].dropna()
                                if len(timestamps) > 0:
                                    latest_ts = timestamps.iloc[-1] if len(timestamps) > 0 else "N/A"
                                    print(f"[LOAD] {person_name}: {len(X)} samples | File modified: {mod_time} | Latest sample: {latest_ts}")
                                else:
                                    print(f"[LOAD] {person_name}: {len(X)} samples | File modified: {mod_time}")
                            else:
                                print(f"[LOAD] {person_name}: {len(X)} samples | File modified: {mod_time}")
                            
                            person_datasets[person_name] = {
                                'X': X,
                                'feature_names': features,
                                'csv_path': csv_path
                            }
                        except Exception as e:
                            print(f"[ERROR] Failed to load {subdir}: {e}")
                    else:
                        print(f"[WARN] CSV not found: {csv_path}")
        else:
            print(f"[ERROR] Dataset directory does not exist: {dataset_dir}")
        
        if len(person_datasets) < 2:
            return EnsembleTrainResult(
                success=False,
                model_family=self.model_family,
                message=f"Need at least 2 persons, found {len(person_datasets)}"
            )
        
        self.person_names = list(person_datasets.keys())
        print(f"\nPersons: {self.person_names}")
        
        # Train a binary model for each person
        results = []
        accuracies = []
        
        for person_name in self.person_names:
            print(f"\n--- Training {person_name}Model ---")
            
            # Positive samples: this person
            X_pos = person_datasets[person_name]['X']
            feature_names = person_datasets[person_name]['feature_names']
            
            # Negative samples: all other persons
            X_neg_list = []
            for other_name, other_data in person_datasets.items():
                if other_name != person_name:
                    X_neg_list.append(other_data['X'])
            
            X_neg = np.vstack(X_neg_list) if X_neg_list else np.empty((0, X_pos.shape[1]))
            
            # Balance negative samples (sample same amount as positive if more)
            if len(X_neg) > len(X_pos) * 2:
                indices = np.random.choice(len(X_neg), size=len(X_pos) * 2, replace=False)
                X_neg = X_neg[indices]
            
            # Create and train model
            model = SinglePersonModel(person_name, self.model_family)
            result = model.train(X_pos, X_neg, feature_names)
            results.append(result)
            
            if result.success:
                accuracies.append(result.cv_accuracy)
                
                # Save model
                save_dir = os.path.join(self.models_dir, self.model_family.lower())
                os.makedirs(save_dir, exist_ok=True)
                model.save(os.path.join(save_dir, f"{person_name}Model.pkl"))
                
                # Store in memory
                key = f"{self.model_family}_{person_name}"
                self.person_models[key] = model
        
        # Calculate overall metrics
        successful = sum(1 for r in results if r.success)
        self.is_trained = successful >= 2
        self.cv_accuracy = np.mean(accuracies) if accuracies else 0.0
        
        print(f"\n{'='*60}")
        print(f"ENSEMBLE TRAINING COMPLETE")
        print(f"  Models trained: {successful}/{len(results)}")
        print(f"  Average CV Accuracy: {self.cv_accuracy:.1%}")
        print(f"{'='*60}")
        
        return EnsembleTrainResult(
            success=self.is_trained,
            model_family=self.model_family,
            person_results=results,
            overall_accuracy=self.cv_accuracy,
            message=f"Trained {successful} person models with avg CV {self.cv_accuracy:.1%}"
        )
    
    def predict(self, X: np.ndarray, model_family: Optional[str] = None) -> PersonEnsemblePrediction:
        """
        Predict HOME or INTRUDER using all person models.
        
        1. Run all person models
        2. Get probability from each
        3. If best probability >= threshold → HOME (that person)
        4. Otherwise → INTRUDER
        """
        model_family = (model_family or self.model_family).upper()
        
        if not self.is_trained:
            return PersonEnsemblePrediction(
                final_label="UNKNOWN",
                matched_person=None,
                confidence=0.0,
                person_probs={},
                person_match_threshold=PERSON_MATCH_THRESHOLD,
                model_family=model_family,
                decision_reason="Person models not trained"
            )
        
        # Get probabilities from each person model
        person_probs = {}
        
        for person_name in self.person_names:
            key = f"{model_family}_{person_name}"
            model = self.person_models.get(key)
            
            if model and model.is_trained:
                prob = model.predict_proba(X)
                person_probs[person_name] = prob
            else:
                person_probs[person_name] = 0.0
        
        if not person_probs:
            return PersonEnsemblePrediction(
                final_label="UNKNOWN",
                matched_person=None,
                confidence=0.0,
                person_probs={},
                person_match_threshold=PERSON_MATCH_THRESHOLD,
                model_family=model_family,
                decision_reason="No trained models found"
            )
        
        # Find best match
        best_person = max(person_probs, key=person_probs.get)
        best_prob = person_probs[best_person]
        
        # Decision
        if best_prob >= PERSON_MATCH_THRESHOLD:
            return PersonEnsemblePrediction(
                final_label="HOME",
                matched_person=best_person,
                confidence=best_prob,
                person_probs=person_probs,
                person_match_threshold=PERSON_MATCH_THRESHOLD,
                model_family=model_family,
                decision_reason=f"Matched {best_person} ({best_prob:.1%} >= {PERSON_MATCH_THRESHOLD:.0%})"
            )
        else:
            return PersonEnsemblePrediction(
                final_label="INTRUDER",
                matched_person=None,
                confidence=1.0 - best_prob,  # Confidence in INTRUDER classification
                person_probs=person_probs,
                person_match_threshold=PERSON_MATCH_THRESHOLD,
                model_family=model_family,
                decision_reason=f"No match (best: {best_person} at {best_prob:.1%} < {PERSON_MATCH_THRESHOLD:.0%})"
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all person models"""
        status = {
            'is_trained': self.is_trained,
            'model_family': self.model_family,
            'person_names': self.person_names,
            'cv_accuracy': self.cv_accuracy,
            'threshold': PERSON_MATCH_THRESHOLD,
            'person_models': {}
        }
        
        for key, model in self.person_models.items():
            status['person_models'][key] = {
                'person_name': model.person_name,
                'model_type': model.model_type,
                'is_trained': model.is_trained,
                'cv_accuracy': model.cv_accuracy,
                'cv_std': model.cv_std,
                'roc_auc': model.roc_auc
            }
        
        return status

# ============================================================================
# GLOBAL INSTANCE
# ============================================================================
_person_ensemble: Optional[PersonEnsembleManager] = None

def get_person_ensemble() -> PersonEnsembleManager:
    """Get or create the global person ensemble manager"""
    global _person_ensemble
    if _person_ensemble is None:
        _person_ensemble = PersonEnsembleManager()
    return _person_ensemble

def reset_person_ensemble():
    """Reset the global person ensemble manager to force fresh data loading"""
    global _person_ensemble
    print("[PersonEnsemble] Resetting global instance for fresh data loading")
    _person_ensemble = None

def get_dataset_sample_counts() -> Dict[str, Any]:
    """Get current sample counts from CSV files (fresh read from disk)"""
    counts = {}
    total = 0
    
    if os.path.exists(DATASET_DIR):
        for subdir in os.listdir(DATASET_DIR):
            if subdir.startswith("HOME_"):
                person_name = subdir.replace("HOME_", "")
                csv_path = os.path.join(DATASET_DIR, subdir, f"features_{subdir}.csv")
                if os.path.exists(csv_path):
                    try:
                        file_stat = os.stat(csv_path)
                        mod_time = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                        
                        df = pd.read_csv(csv_path)
                        count = len(df)
                        
                        latest_timestamp = None
                        if '_timestamp' in df.columns and len(df) > 0:
                            latest_timestamp = df['_timestamp'].iloc[-1]
                        
                        counts[person_name] = {
                            'samples': count,
                            'file_modified': mod_time,
                            'latest_sample': latest_timestamp,
                            'csv_path': csv_path
                        }
                        total += count
                    except Exception as e:
                        counts[person_name] = {'error': str(e)}
    
    return {
        'persons': counts,
        'total_samples': total,
        'dataset_dir': DATASET_DIR
    }

# ============================================================================
# CONVENIENCE FUNCTIONS FOR API
# ============================================================================
def train_person_models(model_family: str = "RF", force_reload: bool = True) -> Dict[str, Any]:
    """
    Train all person models.
    
    Args:
        model_family: "RF" or "MLP"
        force_reload: If True, reset singleton to force fresh data loading from disk
    """
    # Show what data we'll be training on
    print("\n[train_person_models] Checking dataset before training...")
    sample_counts = get_dataset_sample_counts()
    print(f"[train_person_models] Dataset: {sample_counts}")
    
    # Force reload ensures we get fresh data from disk
    if force_reload:
        reset_person_ensemble()
    
    manager = get_person_ensemble()
    result = manager.train(model_family)
    
    return {
        "success": result.success,
        "model_family": result.model_family,
        "overall_accuracy": result.overall_accuracy,
        "person_results": [
            {
                "person_name": r.person_name,
                "success": r.success,
                "cv_accuracy": r.cv_accuracy,
                "cv_std": r.cv_std,
                "roc_auc": r.roc_auc,
                "n_positive": r.n_positive,
                "n_negative": r.n_negative,
                "message": r.message
            }
            for r in result.person_results
        ],
        "message": result.message
    }

def predict_person(features: np.ndarray, model_family: str = "RF") -> Dict[str, Any]:
    """Predict HOME/INTRUDER using person ensemble"""
    manager = get_person_ensemble()
    result = manager.predict(features, model_family)
    
    return {
        "final_label": result.final_label,
        "matched_person": result.matched_person,
        "confidence": result.confidence,
        "person_probs": result.person_probs,
        "person_match_threshold": result.person_match_threshold,
        "model_family": result.model_family,
        "decision_reason": result.decision_reason,
        # Compatibility with existing API
        "label": result.final_label,
        "person_label": result.matched_person or "UNKNOWN"
    }

def get_person_ensemble_status() -> Dict[str, Any]:
    """Get status of person ensemble"""
    return get_person_ensemble().get_status()
