"""
Signal & Wavelet Visualization Utilities
========================================
Provides on-the-fly computation of:
- Time-domain signal display
- FFT spectrum
- Wavelet (CWT) scalogram

This module is for VISUALIZATION ONLY - does NOT modify any existing
dataset format, feature extraction, or model training logic.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Use absolute paths based on script location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(_SCRIPT_DIR, "dataset")

# Try to import PyWavelets, fallback gracefully if not available
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("[Visualization] PyWavelets not installed. Wavelet features disabled.")
    print("  Install with: pip install PyWavelets")


@dataclass
class WaveletConfig:
    """Configuration for wavelet computation"""
    wavelet_type: str = "cwt"      # 'cwt' for continuous wavelet transform
    wavelet_family: str = "morl"   # Morlet wavelet (good for frequency analysis)
    num_scales: int = 32           # Number of scales for CWT
    sampling_rate: float = 200.0   # Hz (default for our sensor)


@dataclass 
class VisualizationResult:
    """Result container for signal visualization data"""
    success: bool
    error: Optional[str] = None
    
    # Time series data
    time_series: Optional[Dict[str, List[float]]] = None
    
    # FFT data
    fft: Optional[Dict[str, List[float]]] = None
    
    # Wavelet data
    wavelet: Optional[Dict[str, Any]] = None
    
    # MFCC data
    mfcc: Optional[Dict[str, Any]] = None
    
    # LIF neuron data
    lif: Optional[Dict[str, Any]] = None
    
    # Metadata
    sample_info: Optional[Dict[str, Any]] = None


class SignalVisualizer:
    """
    Handles on-the-fly computation of signal visualizations.
    Does NOT save any outputs to datasets.
    """
    
    def __init__(self, dataset_dir: str = DATASET_DIR):
        self.dataset_dir = dataset_dir
    
    def list_available_samples(self, source: str) -> List[Dict[str, str]]:
        """
        List all available waveform samples for a given source (e.g., HOME_Dixit).
        
        Returns list of dicts with:
        - filename: wave file name
        - timestamp: extracted timestamp
        - path: full path to file
        """
        samples = []
        
        # Build path to waveforms directory
        waveforms_dir = os.path.join(self.dataset_dir, source, "waveforms")
        
        if not os.path.exists(waveforms_dir):
            return samples
        
        for filename in sorted(os.listdir(waveforms_dir)):
            if filename.startswith("wave_") and filename.endswith(".csv"):
                # Extract timestamp from filename: wave_YYYYMMDD_HHMMSS_ffffff.csv
                parts = filename.replace("wave_", "").replace(".csv", "").split("_")
                if len(parts) >= 3:
                    date_str = parts[0]
                    time_str = parts[1]
                    micro_str = parts[2] if len(parts) > 2 else "000000"
                    
                    # Format as readable timestamp
                    try:
                        timestamp = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                    except:
                        timestamp = filename
                    
                    samples.append({
                        "filename": filename,
                        "timestamp": timestamp,
                        "path": os.path.join(waveforms_dir, filename)
                    })
        
        return samples
    
    def load_waveform(self, source: str, sample_id: str, max_points: int = 2048) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Load a waveform from disk.
        
        Args:
            source: Dataset source (e.g., "HOME_Dixit")
            sample_id: Either a timestamp like "2025-12-04 10:34:31" or a filename
            max_points: Maximum points to return (truncates/downsamples if needed)
        
        Returns:
            Tuple of (waveform_array, error_message)
        """
        waveforms_dir = os.path.join(self.dataset_dir, source, "waveforms")
        
        if not os.path.exists(waveforms_dir):
            return None, f"Waveforms directory not found: {waveforms_dir}"
        
        # Try to find the matching file
        target_file = None
        
        # If sample_id looks like a filename
        if sample_id.startswith("wave_") and sample_id.endswith(".csv"):
            target_file = os.path.join(waveforms_dir, sample_id)
        
        # If sample_id is a timestamp, search for matching file
        elif " " in sample_id or "-" in sample_id:
            # Convert timestamp to filename format: "2025-12-04 10:34:31" -> "20251204_103431"
            try:
                clean_ts = sample_id.replace("-", "").replace(":", "").replace(" ", "_")
                # Look for files starting with wave_ and containing this timestamp prefix
                for filename in os.listdir(waveforms_dir):
                    if filename.startswith(f"wave_{clean_ts[:15]}"):  # Match date_time
                        target_file = os.path.join(waveforms_dir, filename)
                        break
            except:
                pass
        
        # Try direct match as filename
        if target_file is None:
            potential_path = os.path.join(waveforms_dir, sample_id)
            if os.path.exists(potential_path):
                target_file = potential_path
        
        # Try with wave_ prefix
        if target_file is None:
            potential_path = os.path.join(waveforms_dir, f"wave_{sample_id}.csv")
            if os.path.exists(potential_path):
                target_file = potential_path
        
        if target_file is None or not os.path.exists(target_file):
            return None, f"Sample not found: {sample_id} in {source}"
        
        # Load the waveform
        try:
            df = pd.read_csv(target_file)
            waveform = df['amplitude'].values
            
            # Truncate or downsample if needed
            if len(waveform) > max_points:
                # Downsample by taking every nth point
                step = len(waveform) // max_points
                waveform = waveform[::step][:max_points]
            
            return waveform.astype(float), None
            
        except Exception as e:
            return None, f"Error loading waveform: {str(e)}"
    
    def compute_fft(self, waveform: np.ndarray, sampling_rate: float = 200.0, n_points: int = 1024) -> Dict[str, List[float]]:
        """
        Compute FFT spectrum of the waveform.
        
        Args:
            waveform: Time series signal
            sampling_rate: Sampling frequency in Hz
            n_points: FFT size (zero-pads if needed)
        
        Returns:
            Dict with 'freq' (Hz) and 'magnitude' arrays
        """
        # Pad or truncate to n_points
        if len(waveform) < n_points:
            padded = np.zeros(n_points)
            padded[:len(waveform)] = waveform
            waveform = padded
        else:
            waveform = waveform[:n_points]
        
        # Apply window function to reduce spectral leakage
        window = np.hanning(len(waveform))
        windowed = waveform * window
        
        # Compute FFT (real signal, so use rfft)
        fft_result = np.fft.rfft(windowed)
        magnitude = np.abs(fft_result)
        
        # Normalize
        magnitude = magnitude / len(waveform)
        
        # Compute frequency bins
        freq = np.fft.rfftfreq(len(waveform), 1.0 / sampling_rate)
        
        return {
            "freq": freq.tolist(),
            "magnitude": magnitude.tolist()
        }
    
    def compute_wavelet(self, waveform: np.ndarray, config: WaveletConfig) -> Optional[Dict[str, Any]]:
        """
        Compute continuous wavelet transform (CWT) of the waveform.
        
        Args:
            waveform: Time series signal
            config: Wavelet computation configuration
        
        Returns:
            Dict with 'scales', 'frequencies', 'power' (2D), 'time' arrays
            or None if PyWavelets not available
        """
        if not PYWT_AVAILABLE:
            return None
        
        # Define scales (logarithmically spaced for better frequency coverage)
        # Scale relates to frequency: f ≈ center_frequency * sampling_rate / scale
        min_scale = 1
        max_scale = min(len(waveform) // 2, 128)
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), config.num_scales)
        
        try:
            # Compute CWT
            coefficients, frequencies = pywt.cwt(
                waveform, 
                scales, 
                config.wavelet_family,
                sampling_period=1.0/config.sampling_rate
            )
            
            # Compute power (|coefficients|^2)
            power = np.abs(coefficients) ** 2
            
            # Normalize power for visualization (log scale often helps)
            power = np.log10(power + 1e-10)  # Add small epsilon to avoid log(0)
            
            # Time vector
            time = np.arange(len(waveform)) / config.sampling_rate
            
            return {
                "scales": scales.tolist(),
                "frequencies": frequencies.tolist(),
                "power": power.tolist(),  # 2D array: [num_scales x signal_length]
                "time": time.tolist(),
                "wavelet_family": config.wavelet_family,
                "num_scales": config.num_scales
            }
            
        except Exception as e:
            print(f"[Visualization] Wavelet computation error: {e}")
            return None
    
    def visualize_sample(
        self,
        source: str,
        sample_id: str,
        max_points: int = 2048,
        fft_n_points: int = 1024,
        wavelet_config: Optional[WaveletConfig] = None
    ) -> VisualizationResult:
        """
        Main method to compute all visualizations for a sample.
        
        Args:
            source: Dataset source (e.g., "HOME_Dixit")
            sample_id: Sample identifier (timestamp or filename)
            max_points: Max points for time series
            fft_n_points: FFT size
            wavelet_config: Wavelet parameters (uses defaults if None)
        
        Returns:
            VisualizationResult with all computed data
        """
        # Default wavelet config
        if wavelet_config is None:
            wavelet_config = WaveletConfig()
        
        # Load waveform
        waveform, error = self.load_waveform(source, sample_id, max_points)
        
        if waveform is None:
            return VisualizationResult(
                success=False,
                error=error
            )
        
        # Create time vector
        time = (np.arange(len(waveform)) / wavelet_config.sampling_rate).tolist()
        
        # Compute FFT
        fft_result = self.compute_fft(waveform, wavelet_config.sampling_rate, fft_n_points)
        
        # Compute Wavelet
        wavelet_result = self.compute_wavelet(waveform, wavelet_config)
        
        # Compute MFCC
        mfcc_result = self.compute_mfcc(waveform, wavelet_config.sampling_rate)
        
        # Compute LIF
        lif_result = self.compute_lif(waveform, wavelet_config.sampling_rate)
        
        return VisualizationResult(
            success=True,
            time_series={
                "t": time,
                "x": waveform.tolist()
            },
            fft=fft_result,
            wavelet=wavelet_result,
            mfcc=mfcc_result,
            lif=lif_result,
            sample_info={
                "source": source,
                "sample_id": sample_id,
                "num_points": len(waveform),
                "duration_sec": len(waveform) / wavelet_config.sampling_rate,
                "sampling_rate": wavelet_config.sampling_rate,
                "wavelet_available": PYWT_AVAILABLE
            }
        )
    
    def compute_mfcc(self, waveform: np.ndarray, sampling_rate: float = 200.0) -> Optional[Dict[str, Any]]:
        """
        Compute Mel-Frequency Cepstral Coefficients (MFCC).
        
        Returns:
            Dict with 'coefficients' (2D array), 'time_frames', 'n_mfcc'
        """
        try:
            from scipy.fft import fft, dct
            
            # MFCC Parameters
            n_mfcc = 13
            n_mels = 26
            frame_size = min(64, len(waveform))
            hop_size = frame_size // 2
            
            # Normalize waveform
            waveform = waveform - np.mean(waveform)
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
            
            # Calculate number of frames
            n_frames = max(1, (len(waveform) - frame_size) // hop_size + 1)
            
            mfccs = []
            for i in range(n_frames):
                start = i * hop_size
                end = min(start + frame_size, len(waveform))
                frame = waveform[start:end]
                
                # Pad if necessary
                if len(frame) < frame_size:
                    frame = np.pad(frame, (0, frame_size - len(frame)))
                
                # Apply Hanning window
                frame = frame * np.hanning(len(frame))
                
                # FFT
                spectrum = np.abs(np.fft.rfft(frame, n=frame_size))
                
                # Log compression
                log_spectrum = np.log(spectrum + 1e-10)
                
                # DCT to get cepstral coefficients
                cepstral = dct(log_spectrum, type=2, norm='ortho')[:n_mfcc]
                mfccs.append(cepstral.tolist())
            
            mfcc_array = np.array(mfccs).T  # [n_mfcc x n_frames]
            time_frames = [i * hop_size / sampling_rate for i in range(n_frames)]
            
            return {
                "coefficients": mfcc_array.tolist(),
                "time_frames": time_frames,
                "n_mfcc": n_mfcc,
                "n_frames": n_frames
            }
        except Exception as e:
            print(f"[Visualization] MFCC computation error: {e}")
            return None
    
    def compute_lif(self, waveform: np.ndarray, sampling_rate: float = 200.0) -> Optional[Dict[str, Any]]:
        """
        Compute Leaky Integrate-and-Fire (LIF) neuron response.
        
        Returns:
            Dict with 'membrane', 'spikes', 'time', 'spike_count', 'spike_rate'
        """
        try:
            # LIF Parameters
            tau = 0.020  # Time constant (20ms)
            threshold = 0.025
            refractory_period = 0.010  # 10ms
            dt = 1.0 / sampling_rate
            
            # Normalize waveform
            signal = waveform - np.mean(waveform)
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                signal = signal / max_val * 0.1  # Scale to reasonable input current
            
            # LIF simulation
            membrane = np.zeros(len(signal))
            spikes = np.zeros(len(signal))
            potential = 0.0
            refractory_count = 0
            refractory_samples = int(refractory_period * sampling_rate)
            
            alpha = 0.22  # Smoothing factor
            smoothed_input = 0.0
            
            for i in range(len(signal)):
                # Smooth input
                smoothed_input = alpha * np.abs(signal[i]) + (1 - alpha) * smoothed_input
                
                if refractory_count > 0:
                    refractory_count -= 1
                    potential *= 0.5
                else:
                    # Leaky integration
                    leak = np.exp(-dt / tau)
                    potential = potential * leak + smoothed_input * (1 - leak)
                    
                    # Spike check
                    if potential >= threshold:
                        spikes[i] = 1
                        potential = 0
                        refractory_count = refractory_samples
                
                membrane[i] = potential
            
            time = np.arange(len(signal)) / sampling_rate
            spike_count = int(np.sum(spikes))
            duration = len(signal) / sampling_rate
            
            return {
                "membrane": membrane.tolist(),
                "spikes": spikes.tolist(),
                "time": time.tolist(),
                "spike_count": spike_count,
                "spike_rate": spike_count / duration if duration > 0 else 0,
                "threshold": threshold
            }
        except Exception as e:
            print(f"[Visualization] LIF computation error: {e}")
            return None


# ============================================================================
# CONVENIENCE FUNCTIONS FOR API
# ============================================================================
_visualizer_instance: Optional[SignalVisualizer] = None

def get_visualizer() -> SignalVisualizer:
    """Get or create the global visualizer instance."""
    global _visualizer_instance
    if _visualizer_instance is None:
        _visualizer_instance = SignalVisualizer()
    return _visualizer_instance


def list_samples(source: str) -> List[Dict[str, str]]:
    """List available samples for a dataset source."""
    return get_visualizer().list_available_samples(source)


def get_signal_visualization(
    source: str,
    sample_id: str,
    max_points: int = 2048,
    fft_n_points: int = 1024,
    wavelet_type: str = "cwt",
    wavelet_family: str = "morl",
    num_scales: int = 32
) -> Dict[str, Any]:
    """
    Get visualization data for a sample.
    
    Returns dict suitable for JSON response.
    """
    config = WaveletConfig(
        wavelet_type=wavelet_type,
        wavelet_family=wavelet_family,
        num_scales=num_scales
    )
    
    result = get_visualizer().visualize_sample(
        source=source,
        sample_id=sample_id,
        max_points=max_points,
        fft_n_points=fft_n_points,
        wavelet_config=config
    )
    
    if not result.success:
        return {
            "success": False,
            "error": result.error
        }
    
    return {
        "success": True,
        "time_series": result.time_series,
        "fft": result.fft,
        "wavelet": result.wavelet,
        "mfcc": result.mfcc,
        "lif": result.lif,
        "sample_info": result.sample_info
    }
