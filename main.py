from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import shutil
import glob
import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

from models import TrainDataRequest, TrainResponse, PredictRequest, PredictResponse, StatusResponse
from features import FootstepFeatureExtractor
from storage import StorageManager
from ml import MLManager

app = FastAPI(title="Footstep ML Pipeline Backend")

PERSONS = ['Pranshul', 'Aditi', 'Apurv', 'Samir', 'Intruder']

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Managers
extractor = FootstepFeatureExtractor()
storage = StorageManager()
ml_manager = MLManager()

# Static mount for health check or simple UI if needed
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {"message": "Footstep ML Pipeline Backend is running"}

@app.get("/status", response_model=StatusResponse)
async def get_status():
    counts = storage.get_sample_counts()
    model_status = "Ready" if ml_manager.rf else "Not Trained"
    classes = list(ml_manager.rf.classes_) if ml_manager.rf else []
    return StatusResponse(
        samples_per_person=counts, 
        model_status=model_status,
        classes=classes,
        intruder_threshold=0.75
    )

@app.post("/train_data", response_model=TrainResponse)
async def train_data(request: TrainDataRequest):
    extracted_features_list = []
    
    # Process chunks
    for item in request.data:
        raw_chunk = item.get("raw_time_series")
        if not raw_chunk:
            continue
            
        # Extract features (includes validation)
        features = extractor.process_chunk(raw_chunk)
        
        if features:
            # Save if valid
            storage.save_sample(request.label, features)
            extracted_features_list.append(list(features.values()))
    
    # Get updated counts
    counts = storage.get_sample_counts()
    
    metrics = None
    if request.train_model:
        # Load all data for training
        all_features, all_labels = storage.get_all_samples()
        if len(all_features) > 0:
            metrics = ml_manager.train(all_features, all_labels)
        else:
            metrics = {"error": "No data available for training"}
            
    return TrainResponse(
        success=True,
        samples_per_person=counts,
        features_extracted=extracted_features_list if extracted_features_list else None,
        metrics=metrics
    )

@app.post("/predictfootsteps", response_model=PredictResponse)
async def predict_footsteps(request: PredictRequest):
    # Extract features from single chunk
    features_dict = extractor.process_chunk(request.data)
    
    if not features_dict:
        raise HTTPException(status_code=400, detail="Invalid chunk or noise detected")
        
    # Convert dict values to list (ensure order matches ML expectation)
    # features.py returns dict. ml.py expects list of floats for predict.
    # We need to ensure order. 
    # Ideally features.py should return both or we use the keys from ml.py
    
    # Let's use the keys defined in ml.py to be safe
    from ml import FEATURE_NAMES
    features_list = [features_dict.get(k, 0.0) for k in FEATURE_NAMES]
    
    result = ml_manager.predict(features_list)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    prediction = result["prediction"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]
    
    # INTRUDER LOGIC
    is_intruder = False
    alert_msg = ''
    
    if prediction == 'Intruder' or confidence < 0.75:
        is_intruder = True
        alert_msg = '🚨 INTRUDER DETECTED!'
        # If low confidence, force prediction to show as Intruder logic might imply
        # But user wants "prediction" to be the person OR Intruder.
        # If confidence < 0.75, should we change prediction to "Intruder"?
        # User example: "low_confidence_intruder": {"prediction": "Pranshul", "confidence": 0.72, "is_intruder": true...}
        # So we keep the original prediction but flag it.
    else:
        is_intruder = False
        alert_msg = f'✅ Family: {prediction}'
        
    return PredictResponse(
        prediction=prediction,
        confidence=confidence,
        is_intruder=is_intruder,
        alert=alert_msg,
        probabilities=probabilities
    )

@app.post('/reset_model')
async def reset_model():
    deleted_samples = 0
    deleted_persons = 0
    if os.path.exists('dataset'):
        for person_dir in os.listdir('dataset'):
            if os.path.isdir(f'dataset/{person_dir}'):
                csv_files = glob.glob(f'dataset/{person_dir}/*.csv')
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        deleted_samples += len(df)
                    except: pass
                deleted_persons += 1
    shutil.rmtree('dataset', ignore_errors=True)
    shutil.rmtree('models', ignore_errors=True)
    if os.path.exists('db/samples.db'):
        try:
            os.remove('db/samples.db')
        except: pass
    
    # Re-init directories
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('db', exist_ok=True)
    
    # Re-init DB
    storage._init_db()
    
    return {'success': True, 'reset_time': datetime.now().isoformat(), 'deleted': {'samples': deleted_samples, 'persons': deleted_persons}}

@app.get('/dataset')
async def get_dataset_status():
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
                        sample_count += len(pd.read_csv(f).values)
                    except: pass
                persons.append({'name': person_dir, 'samples': sample_count})
                total_samples += sample_count
    model_trained = os.path.exists('models/svm_model.pkl')
    return {'persons': persons, 'total_samples': total_samples, 'model_status': 'trained' if model_trained else 'needs_training'}

@app.delete('/dataset/{person}')
async def delete_person_dataset(person: str = Path(...)):
    person_path = f'dataset/{person}'
    deleted_samples = 0
    if os.path.exists(person_path):
        csv_files = glob.glob(f'{person_path}/*.csv')
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                deleted_samples += len(df)
            except: pass
        shutil.rmtree(person_path, ignore_errors=True)
    
    conn = sqlite3.connect('db/samples.db')
    conn.execute('DELETE FROM samples WHERE person=?', (person,))
    conn.commit()
    conn.close()
    
    return {'success': True, 'message': f'Deleted {person} ({deleted_samples} samples)'}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
