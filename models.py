"""
Pydantic models for Binary HOME vs INTRUDER Classification API
"""

from pydantic import BaseModel
from typing import List, Dict, Optional, Any


class TrainDataRequest(BaseModel):
    """Request for submitting training data."""
    data: List[Dict[str, List[float]]]  # [{"raw_time_series": [...200 samples...]}, ...]
    label: str  # "HOME", "HOME_SAMPLE", "INTRUDER", or "INTRUDER_SAMPLE"
    train_model: bool = False  # Whether to trigger training after saving


class TrainResponse(BaseModel):
    """Response after processing training data."""
    success: bool
    samples_per_person: Dict[str, int]  # {"HOME": 45, "INTRUDER": 32}
    features_extracted: Optional[List[List[float]]] = None
    metrics: Optional[Dict[str, Any]] = None  # Training metrics if train_model=True
    model_classes: List[str] = ["HOME", "INTRUDER"]
    valid_samples: Optional[int] = None  # Number of valid samples processed
    label_used: Optional[str] = None  # Normalized label that was used


class PredictRequest(BaseModel):
    """Request for footstep prediction."""
    data: List[float]  # Single chunk of ~200 ADC samples


class PredictResponse(BaseModel):
    """Response with prediction result."""
    prediction: str  # "HOME" or "INTRUDER"
    confidence: float  # 0.0 to 1.0
    is_intruder: bool  # True if prediction is INTRUDER
    alert: str  # Human-readable alert message
    probabilities: Dict[str, float]  # {"HOME": 0.85, "INTRUDER": 0.15}


class StatusResponse(BaseModel):
    """Response with system status."""
    samples_per_person: Dict[str, int]  # Sample counts per label
    model_status: str  # "Ready" or "Not Trained"
    classes: List[str]  # ["HOME", "INTRUDER"]
    intruder_threshold: float  # Threshold for intruder classification
    accuracy: Optional[float] = None  # Model accuracy if trained
    home_samples: Optional[int] = None  # Total HOME samples
    intruder_samples: Optional[int] = None  # Total INTRUDER samples


class DatasetStatusResponse(BaseModel):
    """Response with dataset overview."""
    persons: List[Dict[str, Any]]  # [{"name": "HOME", "samples": 45}, ...]
    total_samples: int
    model_status: str  # "trained" or "needs_training"
    model_accuracy: Optional[float] = None
    classes: List[str] = ["HOME", "INTRUDER"]


class UploadResponse(BaseModel):
    """Response after dataset upload."""
    success: bool
    imported_samples: int
    samples_per_person: Dict[str, int]
    message: str
