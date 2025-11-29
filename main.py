from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from typing import List, Dict, Any

from models import TrainDataRequest, TrainResponse, PredictRequest, PredictResponse, StatusResponse
from features import FootstepFeatureExtractor
from storage import StorageManager
from ml import MLManager

app = FastAPI(title="Footstep ML Pipeline Backend")

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
    return StatusResponse(samples_per_person=counts, model_status=model_status)

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
        
    return PredictResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"]
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
