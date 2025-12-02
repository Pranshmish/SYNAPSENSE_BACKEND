"""
Pydantic models for One-Class Anomaly Detection API
Trains ONLY on HOME samples - detects intruders as anomalies.
"""

from pydantic import BaseModel
from typing import List, Dict, Optional, Any


class TrainDataRequest(BaseModel):
    """Request for submitting training data."""
    data: List[Dict[str, List[float]]]  # [{"raw_time_series": [...200 samples...]}, ...]
    label: str  # "HOME" for training (INTRUDER not needed for anomaly detection)
    train_model: bool = False  # Whether to trigger training after saving


class TrainResponse(BaseModel):
    """Response after processing training data."""
    success: bool
    samples_per_person: Dict[str, int]  # {"HOME": 45}
    features_extracted: Optional[List[List[float]]] = None
    metrics: Optional[Dict[str, Any]] = None  # Training metrics if train_model=True
    model_classes: List[str] = ["HOME"]  # Only HOME class for training
    valid_samples: Optional[int] = None  # Number of valid samples processed
    label_used: Optional[str] = None  # Normalized label that was used
    model_type: Optional[str] = "One-Class Anomaly Detection"


class PredictRequest(BaseModel):
    """Request for footstep prediction."""
    data: List[float]  # Single chunk of ~200 ADC samples


class PredictResponse(BaseModel):
    """Response with prediction result."""
    prediction: str  # "HOME" or "INTRUDER" (anomaly)
    confidence: float  # 0.0 to 1.0
    is_intruder: bool  # True if detected as anomaly
    alert: str  # Human-readable alert message
    probabilities: Dict[str, float]  # {"HOME": 0.85, "INTRUDER": 0.15}
    anomaly_score: Optional[float] = None  # Raw anomaly score from model
    threshold: Optional[float] = None  # Threshold used for decision


class StatusResponse(BaseModel):
    """Response with system status."""
    samples_per_person: Dict[str, int]  # Sample counts per label
    model_status: str  # "Ready" or "Not Trained"
    classes: List[str]  # ["HOME"] - only HOME needed for training
    intruder_threshold: float  # Anomaly threshold for intruder detection
    accuracy: Optional[float] = None  # Training accuracy if trained
    home_samples: Optional[int] = None  # Total HOME samples
    intruder_samples: Optional[int] = None  # Not used for training, display only
    model_type: Optional[str] = "One-Class Anomaly Detection"


class DatasetStatusResponse(BaseModel):
    """Response with dataset overview."""
    persons: List[Dict[str, Any]]  # [{"name": "HOME", "samples": 45}, ...]
    total_samples: int
    model_status: str  # "trained" or "needs_training"
    model_accuracy: Optional[float] = None
    classes: List[str] = ["HOME"]  # Only HOME for training
    model_type: Optional[str] = "One-Class Anomaly Detection"


class UploadResponse(BaseModel):
    """Response after dataset upload."""
    success: bool
    imported_samples: int
    samples_per_person: Dict[str, int]
    message: str
