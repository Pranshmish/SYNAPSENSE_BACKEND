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

from models import TrainDataRequest, TrainResponse, PredictRequest, PredictResponse, StatusResponse, TrainMLPRequest
from features import FootstepFeatureExtractor, FEATURE_NAMES, extract_features
from storage import StorageManager
from ml import AnomalyDetector
from mlp_model import mlp_classifier, MLPClassifierWrapper

INTRUDER_CONF_THRESH = 0.85  # Updated: Higher threshold for MLP
INTRUDER_MARGIN_THRESH = 0.15

app = FastAPI(title="SynapSense - HOME/INTRUDER Footstep Classifier")

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
    1. HOME.csv (aggregated HOME samples)
    2. HOME_{person}/features_HOME_{person}.csv (individual files)
    """
    try:
        # Use raw label directly for multi-class training
        label = request.label
        print(f"[TRAIN] Received label='{label}'")
            
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
                
            # Extract features with validation
            features = extractor.process_chunk(raw_chunk)
            
            if features:
                # DUAL SAVE: Save to both main CSV and individual file
                save_result = storage.save_sample_dual(label, features)
                extracted_features_list.append(list(features.values()))
                valid_samples += 1
                print(f"[TRAIN] ✓ Saved sample #{valid_samples} for {label} (dual: {save_result})")
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
    Train Simple MLP classifier on HOME data.
    
    Features:
    - Uses 20 robust features for best generalization
    - Generates synthetic INTRUDER samples automatically
    - Dropout + L2 regularization to prevent overfitting
    - K-Fold Cross-Validation for robust accuracy estimation
    - Optional: Train on selected datasets only
    - Returns detailed dataset information
    """
    # Parse request body (handle case where body is empty or None)
    selected_datasets = None
    if request and request.selected_datasets:
        selected_datasets = request.selected_datasets
        print(f"[TRAIN_MLP] Training on selected datasets: {selected_datasets}")
    
    # Get samples based on selection
    if selected_datasets and len(selected_datasets) > 0:
        all_features, all_labels, dataset_details = storage.get_samples_by_datasets(selected_datasets)
    else:
        all_features, all_labels, dataset_details = storage.get_all_samples_with_details()
        print(f"[TRAIN_MLP] Training on ALL datasets: {dataset_details.get('dataset_names', [])}")
    
    if len(all_features) < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 5 HOME samples, have {len(all_features)}"
        )
    
    result = mlp_classifier.train(all_features, all_labels)
    
    if not result.get("success", False):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "MLP training failed")
        )
    
    # Add dataset details to result
    result["dataset_details"] = dataset_details
    
    # Add dual dataset status
    dual_status = storage.get_dual_dataset_status()
    result["dual_dataset"] = dual_status
    
    return result


@app.post("/predict_mlp")
async def predict_with_mlp(request: PredictRequest):
    """
    Predict using MLP with prediction rules.
    
    Returns enhanced prediction with:
    - Rule-based INTRUDER detection
    - Confidence bands (high/medium/low)
    - Color coding for UI
    """
    if not mlp_classifier.is_trained:
        raise HTTPException(
            status_code=400,
            detail="MLP not trained. Call /train_mlp first."
        )
    
    # Extract features
    features_dict = extractor.process_chunk(request.data)
    
    if not features_dict:
        raise HTTPException(
            status_code=400,
            detail="Invalid signal - could not extract features."
        )
    
    features_list = [features_dict.get(k, 0.0) for k in FEATURE_NAMES]
    
    # Get MLP prediction with rules
    result = mlp_classifier.predict(features_list)
    
    if not result.get("success", False):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Prediction failed")
        )
    
    return PredictResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        is_intruder=result["is_intruder"],
        alert=result["alert"],
        probabilities=result["probabilities"],
        anomaly_score=0.0,
        threshold=INTRUDER_CONF_THRESH,
        confidence_band=result["confidence_band"],
        color_code=result["color_code"],
        intruder_reason=result.get("rule_applied")
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
    dual_status = storage.get_dual_dataset_status()
    counts = storage.get_sample_counts()
    mlp_status = mlp_classifier.get_status()
    
    return {
        "dual_dataset": dual_status,
        "sample_counts": counts,
        "mlp_model": mlp_status,
        "target_samples": 150,
        "ready_to_train": dual_status["home_csv"]["samples"] >= 10,
        "recommended_action": _get_recommended_action(dual_status, mlp_status)
    }


def _get_recommended_action(dual_status: Dict, mlp_status: Dict) -> str:
    """Get recommended next action based on current state"""
    home_samples = dual_status["home_csv"]["samples"]
    
    if home_samples < 10:
        return f"Collect more HOME samples ({home_samples}/10 minimum)"
    elif home_samples < 150 and not mlp_status.get("trained", False):
        return f"Collect more samples ({home_samples}/150) or train MLP now"
    elif not mlp_status.get("trained", False):
        return "Ready to train MLP! Call /train_mlp"
    else:
        accuracy = mlp_status.get("accuracy", 0)
        if accuracy < 90:
            return f"Model accuracy {accuracy}%. Collect more samples to improve."
        return f"Model ready! Accuracy: {accuracy}%"


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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"synapsense_dataset_{timestamp}.zip"
    zip_path = f"static/{zip_filename}"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for person_dir in os.listdir('dataset'):
            person_path = f'dataset/{person_dir}'
            if os.path.isdir(person_path):
                csv_files = glob.glob(f'{person_path}/*.csv')
                for csv_file in csv_files:
                    arcname = csv_file.replace('dataset/', '')
                    zipf.write(csv_file, arcname)
    
    return FileResponse(
        zip_path,
        media_type='application/zip',
        filename=zip_filename,
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
    )


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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
