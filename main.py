"""
FastAPI Backend for One-Class Anomaly Detection Footstep Recognition
Trains ONLY on HOME samples - detects intruders as anomalies.
No intruder training data required!
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

from models import TrainDataRequest, TrainResponse, PredictRequest, PredictResponse, StatusResponse
from features import FootstepFeatureExtractor, FEATURE_NAMES, extract_features
from storage import StorageManager
from ml import AnomalyDetector, LABEL_HOME, LABEL_INTRUDER

app = FastAPI(title="SynapSense - Anomaly Detection Footstep Recognizer")

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
classifier = AnomalyDetector()  # One-class anomaly detection

# Static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "SynapSense Anomaly Detection Backend",
        "version": "3.0",
        "mode": "One-Class (HOME only training)",
        "note": "Intruders detected as anomalies - no intruder training needed"
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current system status including sample counts and model state."""
    counts = storage.get_sample_counts()
    model_status = classifier.get_status()
    
    return StatusResponse(
        samples_per_person=counts,
        model_status="Ready" if model_status.get("trained", False) else "Not Trained",
        classes=[LABEL_HOME],  # Only HOME class for training
        intruder_threshold=model_status.get("anomaly_threshold", 0.5),
        accuracy=model_status.get("training_accuracy"),
        home_samples=model_status.get("home_samples", counts.get("HOME", 0)),
        intruder_samples=counts.get("INTRUDER", 0),  # For display only, not used in training
        model_type=model_status.get("model_type", "One-Class Anomaly Detection")
    )


@app.post("/train_data", response_model=TrainResponse)
async def train_data(request: TrainDataRequest):
    """
    Process training data and optionally train the model.
    
    Labels should be:
    - "HOME" or "HOME_SAMPLE" for home user footsteps
    - "INTRUDER" or "INTRUDER_SAMPLE" for intruder footsteps
    """
    try:
        # Normalize label to binary
        label = request.label.upper()
        if label in ["HOME", "HOME_SAMPLE"]:
            binary_label = LABEL_HOME
        else:
            binary_label = LABEL_INTRUDER
            
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
                # Save valid sample
                storage.save_sample(binary_label, features)
                extracted_features_list.append(list(features.values()))
                valid_samples += 1
                print(f"[TRAIN] ✓ Saved sample #{valid_samples} for {binary_label}")
            else:
                rejected_chunks += 1
                import numpy as np
                data = np.array(raw_chunk)
                print(f"[TRAIN] ✗ Feature extraction failed: len={len(raw_chunk)}, std={np.std(data):.6f}, mean={np.mean(data):.2f}")
        
        print(f"[TRAIN] Result: {valid_samples} saved, {rejected_chunks} rejected for label={binary_label}")
        
        # Get updated counts
        counts = storage.get_sample_counts()
        
        metrics = None
        model_classes = [LABEL_HOME, LABEL_INTRUDER]
        
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
                
        return TrainResponse(
            success=valid_samples > 0,
            samples_per_person=counts,
            features_extracted=extracted_features_list if extracted_features_list else None,
            metrics=metrics,
            model_classes=model_classes,
            valid_samples=valid_samples,
            label_used=binary_label
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
    Returns prediction with confidence and probability distribution.
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
    
    # Get prediction
    result = classifier.predict(features_list)
    
    if not result.get("success", False):
        # Model not trained - return error response
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Prediction failed. Model may not be trained.")
        )
    
    prediction = result["prediction"]
    confidence = result["confidence"]
    is_intruder = result["is_intruder"]
    probabilities = result["probabilities"]
    
    # Generate alert message
    if is_intruder:
        if confidence > 0.8:
            alert = f"🚨 INTRUDER DETECTED (High Confidence: {confidence:.1%})"
        else:
            alert = f"⚠️ Possible Intruder ({confidence:.1%} confidence)"
    else:
        if confidence > 0.8:
            alert = f"✅ HOME User Verified ({confidence:.1%})"
        else:
            alert = f"🏠 Likely Home User ({confidence:.1%} confidence)"
    
    return PredictResponse(
        prediction=prediction,
        confidence=confidence,
        is_intruder=is_intruder,
        alert=alert,
        probabilities=probabilities
    )


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
                persons.append({
                    'name': person_dir,
                    'samples': sample_count,
                    'type': 'HOME' if person_dir == 'HOME' else 'INTRUDER'
                })
                total_samples += sample_count
    
    model_trained = classifier.is_trained
    model_status = classifier.get_status()
    
    return {
        'persons': persons,
        'total_samples': total_samples,
        'model_status': 'trained' if model_trained else 'needs_training',
        'model_accuracy': model_status.get('accuracy') if model_trained else None,
        'classes': [LABEL_HOME, LABEL_INTRUDER]
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
                # Extract label from path (e.g., HOME/features_HOME.csv -> HOME)
                parts = csv_path.split('/')
                if len(parts) >= 2:
                    label = parts[0].upper()
                else:
                    # Try to infer from filename
                    if 'home' in csv_path.lower():
                        label = LABEL_HOME
                    else:
                        label = LABEL_INTRUDER
                
                # Normalize to binary labels
                if label not in [LABEL_HOME, LABEL_INTRUDER]:
                    if label in ['HOME_SAMPLE', 'HOME']:
                        label = LABEL_HOME
                    else:
                        label = LABEL_INTRUDER
                
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
