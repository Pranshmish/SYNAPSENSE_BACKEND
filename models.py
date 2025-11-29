from pydantic import BaseModel
from typing import List, Dict, Optional

class TrainDataRequest(BaseModel):
    data: List[Dict[str, List[float]]]  # "data": [{"raw_time_series": [...]}, ...]
    label: str
    train_model: bool = False

class TrainResponse(BaseModel):
    success: bool
    samples_per_person: Dict[str, int]
    features_extracted: Optional[List[List[float]]] = None
    metrics: Optional[Dict[str, float]] = None
    model_classes: List[str] = ["Pranshul", "Aditi", "Apurv", "Samir", "Intruder"]

class PredictRequest(BaseModel):
    data: List[float]  # Single 200-sample chunk

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    is_intruder: bool
    alert: str
    probabilities: Dict[str, float]

class StatusResponse(BaseModel):
    samples_per_person: Dict[str, int]
    model_status: str
    classes: List[str]
    intruder_threshold: float
