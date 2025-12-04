"""
Pydantic models for Hybrid Anomaly Detection API
Primary: One-class (Isolation Forest) trains ONLY on HOME samples
Secondary: Binary fallback for testing with INTRUDER samples
"""

from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Literal


class TrainDataRequest(BaseModel):
    """Request for submitting training data."""
    data: List[Dict[str, List[float]]]  # [{"raw_time_series": [...200 samples...]}, ...]
    label: str  # "HOME" for training (INTRUDER optional for binary fallback)
    train_model: bool = False  # Whether to trigger training after saving


class TrainResponse(BaseModel):
    """Response after processing training data."""
    success: bool
    samples_per_person: Dict[str, int]  # {"HOME": 45, "INTRUDER": 10}
    features_extracted: Optional[List[List[float]]] = None
    metrics: Optional[Dict[str, Any]] = None  # Training metrics if train_model=True
    model_classes: List[str] = ["HOME"]  # Only HOME class for training
    valid_samples: Optional[int] = None  # Number of valid samples processed
    label_used: Optional[str] = None  # Normalized label that was used
    model_type: Optional[str] = "Hybrid Anomaly Detection"


class PredictRequest(BaseModel):
    """Request for footstep prediction."""
    data: List[float]  # Single chunk of ~200 ADC samples


class PredictResponse(BaseModel):
    """
    Enhanced response with comprehensive prediction details.
    
    Includes:
    - anomaly_score: Raw score from Isolation Forest (higher = more anomalous)
    - threshold: Calibrated threshold for HOME vs INTRUDER decision
    - confidence_band: 'high'/'medium'/'low' for UI color coding
    - color_code: Direct color suggestion for UI display
    """
    prediction: str  # "HOME" or "INTRUDER" (anomaly)
    confidence: float  # 0.0 to 1.0
    is_intruder: bool  # True if detected as anomaly
    alert: str  # Human-readable alert message
    probabilities: Dict[str, float]  # {"HOME": 0.85, "INTRUDER": 0.15}
    
    # Enhanced scoring details (now required)
    anomaly_score: float  # Raw anomaly score from model
    threshold: float  # Threshold used for decision
    confidence_band: Literal["high", "medium", "low"]  # Confidence level band
    color_code: str  # "green", "red", or "yellow" for UI display
    
    # Model agreement (optional)
    svm_agrees: Optional[bool] = None  # Whether SVM backup agrees
    binary_prediction: Optional[str] = None  # Binary model prediction if available
    z_score: Optional[float] = None  # Normalized anomaly score
    intruder_reason: Optional[str] = None  # Reason for intruder classification


class StatusResponse(BaseModel):
    """Response with system status."""
    samples_per_person: Dict[str, int]  # Sample counts per label
    model_status: str  # "Ready" or "Not Trained"
    classes: List[str]  # ["HOME"] - only HOME needed for training
    intruder_threshold: float  # Anomaly threshold for intruder detection
    accuracy: Optional[float] = None  # Training accuracy if trained
    home_samples: Optional[int] = None  # Total HOME samples
    intruder_samples: Optional[int] = None  # Not used for training, display only
    model_type: Optional[str] = "Hybrid Anomaly Detection"
    has_binary_fallback: Optional[bool] = None  # Whether binary model is available


class DatasetStatusResponse(BaseModel):
    """Response with dataset overview."""
    persons: List[Dict[str, Any]]  # [{"name": "HOME", "samples": 45}, ...]
    total_samples: int
    model_status: str  # "trained" or "needs_training"
    model_accuracy: Optional[float] = None
    classes: List[str] = ["HOME"]  # Only HOME for training
    model_type: Optional[str] = "Hybrid Anomaly Detection"


class UploadResponse(BaseModel):
    """Response after dataset upload."""
    success: bool
    imported_samples: int
    samples_per_person: Dict[str, int]
    message: str


class TrainMLPRequest(BaseModel):
    """Request for MLP training with optional dataset selection."""
    selected_datasets: Optional[List[str]] = None  # List of dataset names to train on (None = all)


class TrainSelectedModelRequest(BaseModel):
    """Request for training a specific model type."""
    model_name: str  # "RandomForestEnsemble", "MLPClassifier", or "HybridLSTMSNN"
    selected_datasets: Optional[List[str]] = None  # List of dataset names to train on (None = all)


class PredictSelectedModelRequest(BaseModel):
    """Request for prediction with a specific model."""
    data: List[float]  # Raw ADC samples
    model_name: Optional[str] = None  # If None, uses active model


class SetActiveModelRequest(BaseModel):
    """Request to set the active model for predictions."""
    model_name: str  # Model to set as active


class ModelStatusResponse(BaseModel):
    """Response with model status and available models."""
    models: Dict[str, Any]  # All models with their status
    active_model: str  # Currently active model name
