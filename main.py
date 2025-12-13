"""
FastAPI Backend for HOME/INTRUDER Footstep Classification
Features:
- Dual dataset saving (HOME.csv + individual files)
- Simple MLP model (92% accuracy on 150 samples)
- Prediction rules for robust INTRUDER detection
"""

from fastapi import FastAPI, HTTPException, Path, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import os
import shutil
import glob
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import zipfile
import tempfile
from io import BytesIO

from models import (
    TrainDataRequest, TrainResponse, PredictRequest, PredictResponse, StatusResponse, 
    TrainMLPRequest, TrainSelectedModelRequest, PredictSelectedModelRequest, 
    SetActiveModelRequest, ModelStatusResponse
)
from features import FootstepFeatureExtractor, FEATURE_NAMES, extract_features
from storage import StorageManager
from ml import AnomalyDetector
from mlp_model import mlp_classifier, MLPClassifierWrapper
from model_manager import (
    get_model_manager, get_available_models as get_available_models_func,
    train_selected_model as train_selected_model_func, 
    predict_with_model, set_active_model as set_active_model_func,
    ModelRegistry
)
from tracking_router import router as tracking_router

# Model types for backward compatibility
MODEL_TYPES = {
    "RandomForestEnsemble": {"ready": True, "name": "Random Forest Ensemble"},
    "MLPClassifier": {"ready": True, "name": "MLP Neural Network"},
    "HybridLSTMSNN": {"ready": False, "name": "Hybrid LSTM-SNN"}
}

INTRUDER_CONF_THRESH = 0.85  # Updated: Higher threshold for MLP
INTRUDER_MARGIN_THRESH = 0.15

app = FastAPI(title="SynapSense - HOME/INTRUDER Footstep Classifier")
app.include_router(tracking_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
extractor = FootstepFeatureExtractor()
storage = StorageManager()
classifier = AnomalyDetector()  # Legacy RF classifier

# Static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "SynapSense HOME/INTRUDER Classifier",
        "version": "4.0",
        "mode": "Dual Dataset + MLP Model",
        "features": [
            "Dual CSV saving (HOME.csv + individual files)",
            "Simple MLP (150 samples → 92% accuracy)",
            "Prediction rules for robust detection"
        ]
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current system status including sample counts and model state."""
    counts = storage.get_sample_counts()
    model_status = classifier.get_status()
    mlp_status = mlp_classifier.get_status()
    dual_status = storage.get_dual_dataset_status()
    
    # Prefer MLP if trained, else fall back to RF
    active_model = "MLP" if mlp_status.get("trained", False) else "RF"
    is_trained = mlp_status.get("trained", False) or model_status.get("trained", False)
    accuracy = mlp_status.get("accuracy", model_status.get("accuracy", 0))
    
    return StatusResponse(
        samples_per_person=counts,
        model_status="Ready" if is_trained else "Not Trained",
        classes=mlp_status.get("classes", model_status.get("classes", [])),
        intruder_threshold=INTRUDER_CONF_THRESH,
        accuracy=accuracy,
        home_samples=dual_status["home_csv"]["samples"],
        intruder_samples=dual_status["intruder_csv"]["samples"],
        model_type=f"Simple {active_model}" if is_trained else "Not Trained"
    )


@app.post("/train_data", response_model=TrainResponse)
async def train_data(request: TrainDataRequest):
    """
    Process training data with DUAL DATASET SAVING.
    
    Labels should be:
    - "HOME" or "HOME_*" for home user footsteps (e.g., HOME_Apurv)
    - "INTRUDER" or "INTRUDER_*" for intruder footsteps
    
    Data is saved to:
    1. HOME.csv or INTRUDER.csv (aggregated samples)
    2. HOME_{person} or INTRUDER_{person}/features_{person}.csv (individual files)
    3. Analysis plots (FFT, LIF, Combined) in plots/ directory
    """
    try:
        # Use raw label directly for multi-class training
        label = request.label
        print(f"[TRAIN] Received label='{label}'")
        
        # Check if this should be saved as INTRUDER (label starts with INTRUDER)
        save_as_intruder = label.upper().startswith('INTRUDER')
            
        extracted_features_list = []
        valid_samples = 0
        rejected_chunks = 0
        
        # Process each chunk
        for item in request.data:
            raw_chunk = item.get("raw_time_series")
            if not raw_chunk or len(raw_chunk) < 20:  # Lowered from 50 to 20
                rejected_chunks += 1
                print(f"[TRAIN] Rejected chunk: len={len(raw_chunk) if raw_chunk else 0}")
                continue
            
            # Extract optional analysis data from request
            fft_data = item.get("fft_data")  # {frequencies: [...], magnitudes: [...]}
            lif_data = item.get("lif_data")  # {membrane: [...], spikes: [...], time: [...]}
            filtered_waveform = item.get("filtered_waveform")  # Filtered signal
                
            # Extract features with validation
            features = extractor.process_chunk(raw_chunk)
            
            if features:
                # Add waveform and analysis data to features for storage
                features['_raw_waveform'] = raw_chunk
                if filtered_waveform:
                    features['_filtered_waveform'] = filtered_waveform
                if fft_data:
                    features['_fft_data'] = fft_data
                if lif_data:
                    features['_lif_data'] = lif_data
                
                # DUAL SAVE: Save to both main CSV and individual file with analysis plots
                save_result = storage.save_sample_dual(label, features, save_as_intruder=save_as_intruder)
                extracted_features_list.append(list(features.values()))
                valid_samples += 1
                print(f"[TRAIN] ✓ Saved sample #{valid_samples} for {label} (dual: {save_result}, intruder: {save_as_intruder})")
            else:
                rejected_chunks += 1
                import numpy as np
                data = np.array(raw_chunk)
                print(f"[TRAIN] ✗ Feature extraction failed: len={len(raw_chunk)}, std={np.std(data):.6f}, mean={np.mean(data):.2f}")
        
        print(f"[TRAIN] Result: {valid_samples} saved, {rejected_chunks} rejected for label={label}")
        
        # Get updated counts
        counts = storage.get_sample_counts()
        dual_status = storage.get_dual_dataset_status()
        
        metrics = None
        model_classes = []
        
        if request.train_model:
            # Load all data for training
            all_features, all_labels = storage.get_all_samples()
            
            if len(all_features) >= 10:
                train_result = classifier.train(all_features, all_labels)
                
                if train_result.get("success", False):
                    metrics = train_result.get("metrics", {})
                    metrics["confusion_matrix"] = train_result.get("confusion_matrix")
                    metrics["top_features"] = train_result.get("top_features")
                else:
                    metrics = {"error": train_result.get("error", "Training failed")}
            else:
                metrics = {"error": f"Need at least 10 samples. Current: {len(all_features)}"}
        
        # Add dual dataset status to response
        response_metrics = metrics or {}
        response_metrics["dual_dataset"] = {
            "home_samples": dual_status["home_csv"]["samples"],
            "progress_percent": dual_status["progress_percent"],
            "target": dual_status["target_samples"]
        }
                
        return TrainResponse(
            success=valid_samples > 0,
            samples_per_person=counts,
            features_extracted=extracted_features_list if extracted_features_list else None,
            metrics=response_metrics,
            model_classes=model_classes,
            valid_samples=valid_samples,
            label_used=label
        )
    except Exception as e:
        import traceback
        print(f"[TRAIN] ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predictfootsteps", response_model=PredictResponse)
async def predict_footsteps(request: PredictRequest):
    """
    Predict whether footstep is HOME or INTRUDER.
    Returns enhanced prediction with anomaly score, confidence band, and color coding.
    """
    # Extract features from chunk
    features_dict = extractor.process_chunk(request.data)
    
    if not features_dict:
        raise HTTPException(
            status_code=400, 
            detail="Invalid signal - could not extract features. May be noise or flatline."
        )
    
    # Convert to feature vector (ordered)
    features_list = [features_dict.get(k, 0.0) for k in FEATURE_NAMES]
    
    # Get probabilities
    result = classifier.predict_proba(features_list)
    
    if not result.get("success", False):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Prediction failed. Model may not be trained.")
        )
    
    probs = result["probabilities"]
    
    # Sort probabilities to find max and second max
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    if not sorted_probs:
        raise HTTPException(status_code=500, detail="No probabilities returned")
        
    max_label, max_prob = sorted_probs[0]
    second_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
    
    # Default values
    is_intruder = False
    prediction = max_label
    intruder_reason = None
    confidence = max_prob
    
    # Intruder Detection Logic
    if max_prob < INTRUDER_CONF_THRESH:
        is_intruder = True
        prediction = "Intruder/Unknown"
        intruder_reason = "low_confidence"
    elif (max_prob - second_prob) < INTRUDER_MARGIN_THRESH:
        is_intruder = True
        prediction = "Intruder/Unknown"
        intruder_reason = "ambiguous_probs"
        
    # Determine UI color code and alert
    if is_intruder:
        color_code = "red"
        alert = f"🚨 Intruder / Unknown Footstep Detected ({intruder_reason})"
        confidence_band = "low"
    else:
        color_code = "green"
        alert = f"✅ {prediction} ({confidence:.1%})"
        confidence_band = "high" if confidence > 0.8 else "medium"

    return PredictResponse(
        prediction=prediction,
        confidence=confidence,
        is_intruder=is_intruder,
        alert=alert,
        probabilities=probs,
        anomaly_score=0.0, # Not used in this logic but required by schema
        threshold=INTRUDER_CONF_THRESH,
        confidence_band=confidence_band,
        color_code=color_code,
        intruder_reason=intruder_reason
    )


# ============== NEW MLP ENDPOINTS ==============

@app.post("/train_mlp")
async def train_mlp_model(request: TrainMLPRequest = None):
    """
    Train MLP classifier using one-model-per-person approach.
    
    Features:
    - Trains separate MLP binary classifier for each person
    - Uses robust features for best generalization
    - K-Fold Cross-Validation for robust accuracy estimation
    - Optional: Train on selected datasets only
    - Returns detailed dataset information
    """
    from person_model_manager import train_person_models, get_person_ensemble
    
    # Train using person ensemble with MLP models
    result = train_person_models(model_family="MLP")
    
    if not result.get("success", False):
        raise HTTPException(
            status_code=400,
            detail=result.get("message", "MLP training failed")
        )
    
    # Format response for frontend
    ensemble = get_person_ensemble()
    cv_accuracy = result.get('overall_accuracy', 0.0)
    
    # Add dataset details
    dual_status = storage.get_dual_dataset_status()
    
    return {
        "success": True,
        "metrics": {
            "cv_accuracy": round(cv_accuracy * 100, 1),
            "cv_std": round(result.get('cv_std', 0.0) * 100, 1) if result.get('cv_std') else 0,
            "training_accuracy": round(cv_accuracy * 100, 1),
            "total_samples": dual_status.get('total_samples', 0)
        },
        "classes": ensemble.person_names,
        "person_results": result.get('person_results', []),
        "model_family": "MLP",
        "dual_dataset": dual_status,
        "dataset_details": {
            "dataset_names": [f"HOME_{p}" for p in ensemble.person_names],
            "total_samples": dual_status.get('total_samples', 0)
        }
    }


@app.post("/predict_mlp")
async def predict_with_mlp(request: PredictRequest):
    """
    Predict using MLP person ensemble.
    
    Returns enhanced prediction with:
    - Per-person probability matching
    - Confidence bands (high/medium/low)
    - Color coding for UI
    """
    from person_model_manager import get_person_ensemble, predict_person
    
    ensemble = get_person_ensemble()
    
    # Check if MLP models are trained
    mlp_models = [k for k in ensemble.person_models.keys() if k.startswith("MLP_")]
    if len(mlp_models) < 2:
        raise HTTPException(
            status_code=400,
            detail="MLP not trained. Call /train_mlp first."
        )
    
    # Validate input data
    if not request.data or len(request.data) < 20:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: need at least 20 samples, got {len(request.data) if request.data else 0}"
        )
    
    print(f"[PREDICT_MLP] Received {len(request.data)} samples, range: [{min(request.data):.1f}, {max(request.data):.1f}]")
    
    # Extract features
    features_dict = extractor.process_chunk(request.data)
    
    if not features_dict:
        # Get more info about why feature extraction failed
        data = np.array(request.data, dtype=np.float64)
        peak = np.max(np.abs(data))
        raise HTTPException(
            status_code=400,
            detail=f"Feature extraction failed. Samples: {len(request.data)}, Peak: {peak:.1f}. Signal may be too weak (need peak > 50)."
        )
    
    features_list = [features_dict.get(k, 0.0) for k in FEATURE_NAMES]
    features_array = np.array(features_list).reshape(1, -1)
    
    # Get MLP prediction using person ensemble
    result = predict_person(features_array, model_family="MLP")
    
    # Determine confidence band and color
    conf = result["confidence"]
    is_intruder = result["final_label"] == "INTRUDER"
    
    if conf >= 0.85:
        confidence_band = "high"
        color_code = "#22c55e" if not is_intruder else "#ef4444"
    elif conf >= 0.65:
        confidence_band = "medium"
        color_code = "#eab308" if is_intruder else "#22c55e"
    else:
        confidence_band = "low"
        color_code = "#94a3b8"
    
    # Build alert message
    if is_intruder:
        alert = f"⚠️ INTRUDER DETECTED! {result.get('decision_reason', '')}"
    else:
        person = result.get('matched_person', 'HOME')
        alert = f"✓ HOME - {person} ({conf:.0%})"
    
    return PredictResponse(
        prediction=result["final_label"],
        confidence=conf,
        is_intruder=is_intruder,
        alert=alert,
        probabilities=result.get("person_probs", {}),
        anomaly_score=0.0,
        threshold=result.get("person_match_threshold", 0.80),
        confidence_band=confidence_band,
        color_code=color_code,
        intruder_reason=result.get("decision_reason")
    )


@app.get("/dataset_status")
async def get_dataset_status_detailed():
    """
    Get detailed dual dataset status.
    
    Returns:
    - HOME.csv status (samples, persons)
    - Individual file status
    - Progress toward 150 sample target
    """
    from person_model_manager import get_person_ensemble_status
    
    dual_status = storage.get_dual_dataset_status()
    counts = storage.get_sample_counts()
    ensemble_status = get_person_ensemble_status()
    
    return {
        "dual_dataset": dual_status,
        "sample_counts": counts,
        "mlp_model": {
            "trained": ensemble_status.get("is_trained", False),
            "accuracy": ensemble_status.get("cv_accuracy", 0) * 100,
            "model_family": ensemble_status.get("model_family", "RF"),
            "person_names": ensemble_status.get("person_names", [])
        },
        "target_samples": 150,
        "ready_to_train": dual_status.get("total_samples", 0) >= 10,
        "recommended_action": _get_recommended_action(dual_status, ensemble_status)
    }


def _get_recommended_action(dual_status: Dict, ensemble_status: Dict) -> str:
    """Get recommended next action based on current state"""
    total_samples = dual_status.get("total_samples", 0)
    is_trained = ensemble_status.get("is_trained", False)
    
    if total_samples < 10:
        return f"Collect more HOME samples ({total_samples}/10 minimum)"
    elif total_samples < 150 and not is_trained:
        return f"Collect more samples ({total_samples}/150) or train model now"
    elif not is_trained:
        return "Ready to train! Call /train_mlp or /train_selected_model"
    else:
        accuracy = ensemble_status.get("cv_accuracy", 0) * 100
        if accuracy < 90:
            return f"Model accuracy {accuracy:.1f}%. Collect more samples to improve."
        return f"Model ready! Accuracy: {accuracy:.1f}%"


# ============== MULTI-MODEL ENDPOINTS ==============

@app.get("/available_models")
async def get_available_models():
    """
    Get list of all available models with their status.
    Used for populating model selector dropdown.
    """
    manager = get_model_manager()
    raw_models = manager.get_available_models()
    
    # Map model IDs to frontend expected names
    id_to_name = {
        "rf_ensemble": "RandomForestEnsemble",
        "mlp_classifier": "MLPClassifier",
        "hybrid_lstm_snn": "HybridLSTMSNN"
    }
    
    id_to_display = {
        "rf_ensemble": "Random Forest + Isolation Forest",
        "mlp_classifier": "MLP Neural Network",
        "hybrid_lstm_snn": "Hybrid LSTM + SNN"
    }
    
    id_to_short = {
        "rf_ensemble": "RF",
        "mlp_classifier": "MLP",
        "hybrid_lstm_snn": "Hybrid"
    }
    
    # Transform to frontend format
    models = []
    for m in raw_models:
        model_id = m.get("id", "")
        cv_accuracy = m.get("cv_accuracy", 0)
        # Convert to percentage if it's a decimal (0-1 range)
        if cv_accuracy > 0 and cv_accuracy <= 1:
            cv_accuracy = cv_accuracy * 100
        models.append({
            "name": id_to_name.get(model_id, model_id),
            "display_name": id_to_display.get(model_id, m.get("name", "")),
            "short_name": id_to_short.get(model_id, model_id[:3].upper()),
            "description": m.get("description", ""),
            "ready": model_id != "hybrid_lstm_snn",  # Hybrid not ready yet
            "trained": m.get("is_trained", False),
            "cv_accuracy": round(cv_accuracy, 1),  # Use actual accuracy from model
            "is_active": m.get("is_active", False),
            "classes": m.get("classes", [])
        })
    
    # Map active model ID to frontend name
    active_id = manager.active_model_id
    active_name = id_to_name.get(active_id, active_id)
    
    return {
        "models": models,
        "active_model": active_name
    }


@app.post("/train_selected_model")
async def train_selected_model(request: TrainSelectedModelRequest):
    """
    Train a specific model type.
    
    Supports:
    - rf_ensemble: RF + Isolation Forest (~95% CV)
    - mlp_classifier: Neural Network (~94% CV)
    - hybrid_lstm_snn: LSTM + SNN (Coming Soon)
    """
    # Map frontend model names to model_manager IDs
    model_name_map = {
        "RandomForestEnsemble": "rf_ensemble",
        "MLPClassifier": "mlp_classifier",
        "HybridLSTMSNN": "hybrid_lstm_snn"
    }
    
    model_id = model_name_map.get(request.model_name, request.model_name)
    model = ModelRegistry.get(model_id)
    
    if not model:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {request.model_name}. Available: rf_ensemble, mlp_classifier, hybrid_lstm_snn"
        )
    
    if model_id == "hybrid_lstm_snn":
        raise HTTPException(
            status_code=400,
            detail="Hybrid LSTM-SNN is not available yet. Coming soon!"
        )
    
    # Get datasets to train on
    datasets_to_use = request.selected_datasets if request.selected_datasets else []
    
    if not datasets_to_use:
        # Use all available HOME datasets
        counts = storage.get_sample_counts()
        datasets_to_use = list(counts.keys())
    
    print(f"[TRAIN] Training {model_id} on datasets: {datasets_to_use}")
    
    # Train using new model manager
    result = train_selected_model_func(model_id, datasets_to_use)
    
    if not result.get("success", False):
        raise HTTPException(
            status_code=400,
            detail=result.get("message", "Training failed")
        )
    
    # Format response to match frontend expectations
    cv_mean = result.get("cv_mean", 0)
    cv_std = result.get("cv_std", 0)
    cv_scores = result.get("cv_scores", [])
    
    response = {
        "success": True,
        "model_name": result.get("model_name", request.model_name),
        "metrics": {
            "training_accuracy": round(cv_mean * 100, 1),
            "cv_accuracy": round(cv_mean * 100, 1),
            "cv_std": round(cv_std * 100, 1),
            "cv_scores": [round(s * 100, 1) for s in cv_scores],
            "n_folds": len(cv_scores) if cv_scores else result.get("details", {}).get("n_folds", 5),
            "home_samples": result.get("details", {}).get("total_samples", 0) or sum(result.get("details", {}).get("samples_per_class", {}).values()) if result.get("details") else 0,
            "intruder_samples": 0,
            "total_samples": result.get("details", {}).get("total_samples", 0) or sum(result.get("details", {}).get("samples_per_class", {}).values()) if result.get("details") else 0,
        },
        "classes": result.get("classes", []),
        "dataset_details": {
            "dataset_names": datasets_to_use,
            "selected_datasets": datasets_to_use,
            "datasets": datasets_to_use
        },
        "dual_dataset": storage.get_dual_dataset_status(),
        "top_features": []
    }
    
    return response


@app.post("/predict_selected_model")
async def predict_with_selected_model(request: PredictSelectedModelRequest):
    """
    Predict using a specific model or the active model.
    
    If model_name is not provided, uses the currently active model.
    Returns enhanced prediction with anomaly detection.
    """
    # Validate input data
    if not request.data or len(request.data) < 20:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: need at least 20 samples, got {len(request.data) if request.data else 0}"
        )
    
    print(f"[PREDICT_MODEL] Received {len(request.data)} samples, model: {request.model_name}")
    
    # Extract features
    features_dict = extractor.process_chunk(request.data)
    
    if not features_dict:
        # Get more info about why feature extraction failed
        data = np.array(request.data, dtype=np.float64)
        peak = np.max(np.abs(data))
        raise HTTPException(
            status_code=400,
            detail=f"Feature extraction failed. Samples: {len(request.data)}, Peak: {peak:.1f}. Signal may be too weak."
        )
    
    features_list = [features_dict.get(k, 0.0) for k in FEATURE_NAMES]
    features_array = np.array(features_list).reshape(1, -1)
    
    # Map frontend model names to model_manager IDs
    model_name_map = {
        "RandomForestEnsemble": "rf_ensemble",
        "MLPClassifier": "mlp_classifier",
        "HybridLSTMSNN": "hybrid_lstm_snn"
    }
    
    # Determine which model to use
    manager = get_model_manager()
    model_id = None
    if request.model_name:
        model_id = model_name_map.get(request.model_name, request.model_name)
    
    result = predict_with_model(features_array, model_id)
    
    # Determine if intruder
    is_intruder = result["label"] == "INTRUDER"
    
    # Determine confidence band and color
    conf = result["confidence"]
    if conf >= 0.85:
        confidence_band = "high"
        color_code = "#22c55e" if not is_intruder else "#ef4444"  # green or red
    elif conf >= 0.65:
        confidence_band = "medium"
        color_code = "#eab308" if is_intruder else "#22c55e"  # yellow or green
    else:
        confidence_band = "low"
        color_code = "#94a3b8"  # gray
    
    # Build alert message
    if is_intruder:
        alert = f"⚠️ INTRUDER DETECTED! {result.get('decision_reason', '')}"
    else:
        person = result.get('person_label', 'HOME')
        alert = f"✓ HOME - {person} ({conf:.0%})"
    
    return PredictResponse(
        prediction=result["label"],
        confidence=conf,
        is_intruder=is_intruder,
        alert=alert,
        probabilities=result.get("all_probabilities", {}),
        anomaly_score=result.get("anomaly_score", 0.0),
        threshold=INTRUDER_CONF_THRESH,
        confidence_band=confidence_band,
        color_code=color_code,
        intruder_reason=result.get("decision_reason")
    )


@app.post("/set_active_model")
async def set_active_model_endpoint(request: SetActiveModelRequest):
    """
    Switch the active model for predictions.
    
    The model must be trained before it can be set as active.
    """
    # Map frontend model names to model_manager IDs
    model_name_map = {
        "RandomForestEnsemble": "rf_ensemble",
        "MLPClassifier": "mlp_classifier",
        "HybridLSTMSNN": "hybrid_lstm_snn"
    }
    
    # Reverse map for response
    id_to_name = {v: k for k, v in model_name_map.items()}
    
    model_id = model_name_map.get(request.model_name, request.model_name)
    success = set_active_model_func(model_id)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to set active model: {request.model_name}"
        )
    
    return {"success": True, "active_model": id_to_name.get(model_id, model_id)}


@app.get("/model_status")
async def get_model_status():
    """
    Get comprehensive status of all models.
    
    Returns status for RF, MLP, and Hybrid models with:
    - Training status
    - CV accuracy
    - Samples used
    - Whether it's currently active
    """
    manager = get_model_manager()
    models = manager.get_available_models()
    
    return {
        "models": models,
        "active_model": manager.active_model_id
    }


@app.post("/reset_model")
async def reset_model():
    """Reset all data: models, dataset, and database."""
    deleted_samples = 0
    deleted_persons = 0
    
    # Count what will be deleted
    if os.path.exists('dataset'):
        for person_dir in os.listdir('dataset'):
            person_path = f'dataset/{person_dir}'
            if os.path.isdir(person_path):
                csv_files = glob.glob(f'{person_path}/*.csv')
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        deleted_samples += len(df)
                    except:
                        pass
                deleted_persons += 1
    
    # Clear everything
    shutil.rmtree('dataset', ignore_errors=True)
    shutil.rmtree('models', ignore_errors=True)
    
    if os.path.exists('db/samples.db'):
        try:
            os.remove('db/samples.db')
        except:
            pass
    
    # Re-initialize
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('db', exist_ok=True)
    
    storage._init_db()
    classifier.reset()
    mlp_classifier.reset()  # Also reset MLP
    
    return {
        'success': True,
        'reset_time': datetime.now().isoformat(),
        'deleted': {
            'samples': deleted_samples,
            'persons': deleted_persons
        },
        'message': 'All data and models cleared successfully'
    }


@app.get("/dataset")
async def get_dataset_status():
    """Get overview of stored dataset."""
    persons = []
    total_samples = 0
    
    if os.path.exists('dataset'):
        for person_dir in os.listdir('dataset'):
            person_path = f'dataset/{person_dir}'
            if os.path.isdir(person_path):
                csv_files = glob.glob(f'{person_path}/*.csv')
                sample_count = 0
                for f in csv_files:
                    try:
                        df = pd.read_csv(f)
                        sample_count += len(df)
                    except:
                        pass
                # Use directory name as label type
                label_type = person_dir
                    
                persons.append({
                    'name': person_dir,
                    'samples': sample_count,
                    'type': label_type
                })
                total_samples += sample_count
    
    model_trained = classifier.is_trained
    model_status = classifier.get_status()
    
    return {
        'persons': persons,
        'total_samples': total_samples,
        'model_status': 'trained' if model_trained else 'needs_training',
        'model_accuracy': model_status.get('accuracy') if model_trained else None,
        'classes': model_status.get("classes", [])
    }


@app.delete("/dataset/{person}")
async def delete_person_dataset(person: str = Path(...)):
    """Delete dataset for a specific person/label."""
    person_path = f'dataset/{person}'
    deleted_samples = 0
    
    if os.path.exists(person_path):
        csv_files = glob.glob(f'{person_path}/*.csv')
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                deleted_samples += len(df)
            except:
                pass
        shutil.rmtree(person_path, ignore_errors=True)
    
    # Also delete from database
    try:
        conn = sqlite3.connect('db/samples.db')
        conn.execute('DELETE FROM samples WHERE person=?', (person,))
        conn.commit()
        conn.close()
    except:
        pass
    
    return {
        'success': True,
        'message': f'Deleted {person} ({deleted_samples} samples)'
    }


@app.get("/dataset/download")
async def download_dataset():
    """Download complete dataset as ZIP file."""
    if not os.path.exists('dataset'):
        raise HTTPException(status_code=404, detail="No dataset found")
    
    # Check if there's any data
    has_data = False
    for person_dir in os.listdir('dataset'):
        if os.path.isdir(f'dataset/{person_dir}'):
            if glob.glob(f'dataset/{person_dir}/*.csv'):
                has_data = True
                break
    
    if not has_data:
        raise HTTPException(status_code=404, detail="Dataset is empty")
    
    # Create ZIP file
    os.makedirs('static', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"synapsense_dataset_{timestamp}.zip"
    zip_path = f"static/{zip_filename}"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for person_dir in os.listdir('dataset'):
            person_path = f'dataset/{person_dir}'
            if os.path.isdir(person_path):
                # Add all files recursively
                for root, dirs, files in os.walk(person_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, 'dataset')
                        # Ensure forward slashes for ZIP compatibility
                        arcname = arcname.replace(os.path.sep, '/')
                        zipf.write(file_path, arcname)
    
    return FileResponse(
        zip_path,
        media_type='application/zip',
        filename=zip_filename,
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
    )


# ============== INDIVIDUAL DATASET DOWNLOAD ==============
@app.get("/dataset/download/{person}")
async def download_individual_dataset(person: str = Path(...)):
    """
    Download individual person's dataset as ZIP file.
    Includes: features CSV, raw signals, and waveform plots (if available).
    """
    person_path = f'dataset/{person}'
    
    if not os.path.exists(person_path):
        raise HTTPException(status_code=404, detail=f"Dataset for '{person}' not found")
    
    # Check if there's any data
    csv_files = glob.glob(f'{person_path}/*.csv')
    if not csv_files:
        raise HTTPException(status_code=404, detail=f"No data found for '{person}'")
    
    # Create ZIP file
    os.makedirs('static', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"{person}_dataset_{timestamp}.zip"
    zip_path = f"static/{zip_filename}"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all files recursively
        for root, dirs, files in os.walk(person_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, person_path)
                # Ensure forward slashes for ZIP compatibility
                arcname = arcname.replace(os.path.sep, '/')
                zipf.write(file_path, arcname)
    
    return FileResponse(
        zip_path,
        media_type='application/zip',
        filename=zip_filename,
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
    )


# ============== DATASET PREVIEW ENDPOINTS ==============
@app.get("/dataset/preview/{person}")
async def preview_person_dataset(person: str = Path(...), limit: int = 20):
    """
    Get preview of a person's dataset including:
    - Sample count and file info
    - List of recent samples with timestamps
    - Feature statistics summary
    - Available waveforms count
    """
    person_path = f'dataset/{person}'
    
    if not os.path.exists(person_path):
        raise HTTPException(status_code=404, detail=f"Dataset for '{person}' not found")
    
    result = {
        "person": person,
        "samples": [],
        "total_samples": 0,
        "waveform_count": 0,
        "plot_count": 0,
        "feature_stats": {},
        "file_info": {}
    }
    
    # Read features CSV
    csv_path = f'{person_path}/features_{person}.csv'
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            result["total_samples"] = len(df)
            result["file_info"]["features_csv"] = {
                "path": csv_path,
                "size_kb": round(os.path.getsize(csv_path) / 1024, 2),
                "columns": list(df.columns)
            }
            
            # Get recent samples (last N rows with timestamp)
            if '_timestamp' in df.columns:
                recent = df.tail(limit)[['_timestamp', '_label', '_class'] if '_class' in df.columns else ['_timestamp']].copy()
                recent['sample_index'] = range(len(df) - len(recent), len(df))
                result["samples"] = recent.to_dict(orient='records')
            
            # Feature statistics for numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude private columns
            feature_cols = [c for c in numeric_cols if not c.startswith('_')][:10]
            if feature_cols:
                result["feature_stats"] = df[feature_cols].describe().to_dict()
                
        except Exception as e:
            result["file_info"]["error"] = str(e)
    
    # Count waveforms
    waveforms_dir = f'{person_path}/waveforms'
    if os.path.exists(waveforms_dir):
        result["waveform_count"] = len([f for f in os.listdir(waveforms_dir) if f.endswith('.csv')])
    
    # Count plots
    plots_dir = f'{person_path}/plots'
    if os.path.exists(plots_dir):
        result["plot_count"] = len([f for f in os.listdir(plots_dir) if f.endswith('.png')])
    
    return result


@app.get("/dataset/preview/{person}/samples")
async def get_person_samples_list(person: str = Path(...), offset: int = 0, limit: int = 50):
    """
    Get paginated list of samples for a person with their waveform availability.
    """
    person_path = f'dataset/{person}'
    
    if not os.path.exists(person_path):
        raise HTTPException(status_code=404, detail=f"Dataset for '{person}' not found")
    
    samples = []
    csv_path = f'{person_path}/features_{person}.csv'
    waveforms_dir = f'{person_path}/waveforms'
    
    # Get list of available waveforms
    available_waveforms = set()
    if os.path.exists(waveforms_dir):
        for f in os.listdir(waveforms_dir):
            if f.startswith('wave_') and f.endswith('.csv'):
                # Extract timestamp: wave_YYYYMMDD_HHMMSS_ffffff.csv
                ts = f.replace('wave_', '').replace('.csv', '')
                available_waveforms.add(ts)
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            total = len(df)
            
            # Get requested slice
            slice_df = df.iloc[offset:offset+limit]
            
            for idx, row in slice_df.iterrows():
                sample = {
                    "index": idx,
                    "has_waveform": False,
                    "waveform_id": None
                }
                
                # Add timestamp if available
                if '_timestamp' in row:
                    sample["timestamp"] = row['_timestamp']
                    # Check if waveform exists for this timestamp
                    ts_clean = str(row['_timestamp']).replace('-', '').replace(':', '').replace(' ', '_')[:15]
                    for wf_ts in available_waveforms:
                        if wf_ts.startswith(ts_clean):
                            sample["has_waveform"] = True
                            sample["waveform_id"] = wf_ts
                            break
                
                if '_label' in row:
                    sample["label"] = row['_label']
                
                # Include a few key features for preview
                for feat in ['stat_rms', 'stat_energy', 'fft_dominant_freq', 'spike_count']:
                    if feat in row and not pd.isna(row[feat]):
                        sample[feat] = round(float(row[feat]), 4)
                
                samples.append(sample)
            
            return {
                "person": person,
                "total": total,
                "offset": offset,
                "limit": limit,
                "samples": samples
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")
    
    return {"person": person, "total": 0, "offset": offset, "limit": limit, "samples": []}


@app.get("/dataset/preview/{person}/waveform/{waveform_id}")
async def get_sample_waveform_data(person: str = Path(...), waveform_id: str = Path(...)):
    """
    Get raw waveform data for a specific sample.
    Returns the waveform amplitude data for visualization.
    """
    waveform_path = f'dataset/{person}/waveforms/wave_{waveform_id}.csv'
    
    if not os.path.exists(waveform_path):
        raise HTTPException(status_code=404, detail=f"Waveform not found: {waveform_id}")
    
    try:
        df = pd.read_csv(waveform_path)
        amplitudes = df['amplitude'].tolist()
        
        return {
            "waveform_id": waveform_id,
            "person": person,
            "samples": len(amplitudes),
            "duration_ms": len(amplitudes) * 5,  # 200Hz = 5ms per sample
            "amplitude": amplitudes,
            "stats": {
                "min": float(min(amplitudes)),
                "max": float(max(amplitudes)),
                "mean": float(np.mean(amplitudes)),
                "std": float(np.std(amplitudes))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading waveform: {str(e)}")


@app.get("/dataset/list")
async def list_all_datasets():
    """
    List all available datasets with their sample counts and download links.
    Useful for the frontend dataset manager.
    """
    datasets = []
    
    if not os.path.exists('dataset'):
        return {"datasets": [], "total_samples": 0}
    
    total_samples = 0
    
    for person_dir in sorted(os.listdir('dataset')):
        person_path = f'dataset/{person_dir}'
        if os.path.isdir(person_path):
            csv_path = f'{person_path}/features_{person_dir}.csv'
            
            dataset_info = {
                "name": person_dir,
                "sample_count": 0,
                "waveform_count": 0,
                "size_kb": 0,
                "last_modified": None,
                "download_url": f"/dataset/download/{person_dir}",
                "preview_url": f"/dataset/preview/{person_dir}"
            }
            
            # Get sample count from CSV
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    dataset_info["sample_count"] = len(df)
                    dataset_info["size_kb"] = round(os.path.getsize(csv_path) / 1024, 2)
                    dataset_info["last_modified"] = datetime.fromtimestamp(
                        os.path.getmtime(csv_path)
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    total_samples += len(df)
                except:
                    pass
            
            # Get waveform count
            waveforms_dir = f'{person_path}/waveforms'
            if os.path.exists(waveforms_dir):
                dataset_info["waveform_count"] = len([
                    f for f in os.listdir(waveforms_dir) if f.endswith('.csv')
                ])
            
            datasets.append(dataset_info)
    
    return {
        "datasets": datasets,
        "total_samples": total_samples,
        "total_datasets": len(datasets)
    }

@app.post("/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a dataset ZIP file to restore/merge data.
    ZIP should contain folders (HOME/, INTRUDER/) with CSV files.
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported")
    
    # Save uploaded file
    content = await file.read()
    
    try:
        with zipfile.ZipFile(BytesIO(content), 'r') as zipf:
            # Validate structure
            file_list = zipf.namelist()
            csv_files = [f for f in file_list if f.endswith('.csv')]
            
            if not csv_files:
                raise HTTPException(status_code=400, detail="No CSV files found in ZIP")
            
            imported_count = 0
            
            for csv_path in csv_files:
                # Extract label from path (e.g., Pranshul/features.csv -> Pranshul)
                parts = csv_path.split('/')
                if len(parts) >= 2:
                    label = parts[0]
                else:
                    # Try to infer from filename or default to Unknown
                    label = "Unknown"
                
                # Read CSV
                with zipf.open(csv_path) as csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                        
                        # Save each row as a sample
                        for _, row in df.iterrows():
                            features = row.to_dict()
                            storage.save_sample(label, features)
                            imported_count += 1
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")
                        continue
            
            counts = storage.get_sample_counts()
            
            return {
                'success': True,
                'imported_samples': imported_count,
                'samples_per_person': counts,
                'message': f'Successfully imported {imported_count} samples'
            }
            
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")


@app.post("/train")
async def train_model():
    """
    Explicitly train the model using all stored data.
    Separate endpoint from train_data for manual control.
    """
    all_features, all_labels = storage.get_all_samples()
    
    if len(all_features) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough data. Need at least 10 samples, have {len(all_features)}"
        )
    
    result = classifier.train(all_features, all_labels)
    
    if not result.get("success", False):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Training failed")
        )
    
    return result


@app.get("/model/features")
async def get_feature_names():
    """Get list of features used by the model."""
    return {
        "feature_count": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "categories": {
            "statistical": [f for f in FEATURE_NAMES if f.startswith("stat_")],
            "fft": [f for f in FEATURE_NAMES if f.startswith("fft_")],
            "lif": [f for f in FEATURE_NAMES if f.startswith("lif_")]
        }
    }


@app.get("/person_ensemble_status")
async def get_person_ensemble_status_endpoint():
    """
    Get status of person ensemble models.
    
    Returns information about all trained per-person binary classifiers
    for both RF and MLP model families.
    """
    try:
        from person_model_manager import get_person_ensemble_status
        status = get_person_ensemble_status()
        return {
            "success": True,
            "is_trained": status.get('is_trained', False),
            "model_family": status.get('model_family', 'RF'),
            "person_names": status.get('person_names', []),
            "cv_accuracy": status.get('cv_accuracy', 0.0),
            "threshold": status.get('threshold', 0.80),
            "person_models": status.get('person_models', {})
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "is_trained": False
        }


@app.get("/dataset_sample_counts")
async def get_dataset_sample_counts_endpoint():
    """
    Get fresh sample counts directly from CSV files on disk.
    
    This reads the actual CSV files to show current sample counts,
    including file modification times and latest sample timestamps.
    Useful for debugging data staleness issues.
    """
    try:
        from person_model_manager import get_dataset_sample_counts
        counts = get_dataset_sample_counts()
        return {
            "success": True,
            **counts
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================================
# SIGNAL & WAVELET VISUALIZATION ENDPOINTS
# ============================================================================
from pydantic import BaseModel

class WaveletConfigRequest(BaseModel):
    type: str = "cwt"
    family: str = "morl"
    num_scales: int = 32

class SignalVisualizationRequest(BaseModel):
    sample_id: str
    source: str
    max_points: int = 2048
    fft_n_points: int = 1024
    wavelet: Optional[WaveletConfigRequest] = None


@app.get("/api/visualization/samples/{source}")
async def list_visualization_samples(source: str):
    """
    List all available waveform samples for a dataset source.
    
    Args:
        source: Dataset source name (e.g., "HOME_Dixit")
    
    Returns:
        List of samples with filename, timestamp, and path
    """
    try:
        from visualization import list_samples
        samples = list_samples(source)
        return {
            "success": True,
            "source": source,
            "count": len(samples),
            "samples": samples
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/api/visualization/signal_wavelet")
async def get_signal_wavelet_visualization(request: SignalVisualizationRequest):
    """
    Compute and return time series, FFT spectrum, and wavelet scalogram
    for a selected sample.
    
    This is for VISUALIZATION ONLY - does not modify any datasets or models.
    
    Request body:
        - sample_id: Timestamp or filename of the sample
        - source: Dataset source (e.g., "HOME_Dixit")
        - max_points: Maximum points for time series (default 2048)
        - fft_n_points: FFT size (default 1024)
        - wavelet: Optional wavelet config {type, family, num_scales}
    
    Response:
        - time_series: {t: [...], x: [...]}
        - fft: {freq: [...], magnitude: [...]}
        - wavelet: {scales, frequencies, power (2D), time}
        - sample_info: metadata
    """
    try:
        from visualization import get_signal_visualization
        
        wavelet_type = "cwt"
        wavelet_family = "morl"
        num_scales = 32
        
        if request.wavelet:
            wavelet_type = request.wavelet.type
            wavelet_family = request.wavelet.family
            num_scales = request.wavelet.num_scales
        
        result = get_signal_visualization(
            source=request.source,
            sample_id=request.sample_id,
            max_points=request.max_points,
            fft_n_points=request.fft_n_points,
            wavelet_type=wavelet_type,
            wavelet_family=wavelet_family,
            num_scales=num_scales
        )
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/visualization/sources")
async def list_visualization_sources():
    """
    List all available dataset sources that have waveform data.
    
    Returns list of sources (e.g., ["HOME_Dixit", "HOME_Pandey", "HOME_Sameer"])
    """
    try:
        import os
        from visualization import DATASET_DIR
        
        sources = []
        if os.path.exists(DATASET_DIR):
            for name in os.listdir(DATASET_DIR):
                source_path = os.path.join(DATASET_DIR, name)
                waveforms_path = os.path.join(source_path, "waveforms")
                if os.path.isdir(source_path) and os.path.exists(waveforms_path):
                    # Count waveforms
                    waveform_count = len([f for f in os.listdir(waveforms_path) if f.endswith('.csv')])
                    if waveform_count > 0:
                        sources.append({
                            "name": name,
                            "waveform_count": waveform_count
                        })
        
        return {
            "success": True,
            "sources": sources
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
