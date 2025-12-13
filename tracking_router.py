from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import numpy as np
from tracking_engine import engine

router = APIRouter(prefix="/tracking", tags=["tracking"])

class ManualInput(BaseModel):
    Piezo1: float
    Piezo2: float
    Piezo3: float
    Piezo4: float
    Piezo5: float
    Piezo6: float

class ESP32Data(BaseModel):
    timestamp_ms: Optional[int]
    fs: int = 2000
    samples: int
    channels: Dict[str, List[float]]

@router.post("/process_manual")
async def process_manual(data: ManualInput):
    """
    Process single manual reading (snapshot).
    """
    try:
        # Generate synthetic 0.5s window with these peaks
        fs = 2000
        n_samples = 1000
        
        channels = {}
        amplitudes = [data.Piezo1, data.Piezo2, data.Piezo3, data.Piezo4, data.Piezo5, data.Piezo6]
        
        for i, amp in enumerate(amplitudes):
            # Create a pulse in the middle
            sig = np.random.normal(0, 0.01, n_samples) # Noise
            # Ricker wavelet pulse (Mexican hat)
            width = 50
            center = 500
            t_pulse = np.arange(n_samples)
            pulse = amp * (1 - 2 * (np.pi * (t_pulse - center) / width)**2) * np.exp(-(np.pi * (t_pulse - center) / width)**2)
            channels[f"c{i+1}"] = (sig + pulse).tolist()
            
        result = engine.process_signals(channels, fs=fs)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process_json")
async def process_json(payload: ESP32Data):
    try:
        result = engine.process_signals(payload.channels, fs=payload.fs)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/generate_test_data")
async def generate_two_persons():
    """Generates a 2-person walking scenario for testing."""
    try:
        data = engine.generate_two_person_data()
        return data
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/generate_real_test_data")
async def generate_real_test_data():
    """Generates a walking scenario using REAL footstep signatures."""
    try:
        data = engine.generate_real_footstep_test()
        return data
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/generate_single_person")
async def generate_single_person():
    try:
        data = engine.generate_single_person_data()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/generate_overlap")
async def generate_overlap():
    try:
        data = engine.generate_overlapped_data()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/generate_stop_go")
async def generate_stop_go():
    try:
        data = engine.generate_stop_go_data()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/generate_circle")
async def generate_circle():
    try:
        data = engine.generate_circle_data()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/generate_three_persons")
async def generate_three_persons():
    try:
        data = engine.generate_three_person_data()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
