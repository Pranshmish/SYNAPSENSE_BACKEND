"""
Enhanced Feature Extraction for HOME Footstep Anomaly Detection
Uses LIF spike features + FFT spectral features + MFCC + statistical features (~85 total)

Improvements:
- Wider bandpass filter (8-200Hz) to capture full footstep spectrum
- MFCC features for temporal envelope analysis
- Signal entropy for complexity measure
- Smoothed RMS for amplitude envelope
- Adaptive gain normalization
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from scipy.fft import fft, fftfreq, dct
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class LIF:
    """
    Leaky Integrate-and-Fire Neuron with enhanced spike feature extraction.
    Used for encoding footstep vibration signals into spike patterns.
    """
    
    def __init__(self, threshold: float = 0.5, decay: float = 0.95, 
                 resting_potential: float = 0.0, refractory_period: int = 5):
        self.threshold = threshold
        self.decay = decay
        self.resting_potential = resting_potential
        self.refractory_period = refractory_period
        self.potential = resting_potential
        self.spike_times = []
        self.refractory_counter = 0
        self.potential_history = []
        
    def reset(self):
        """Reset neuron state for new signal processing."""
        self.potential = self.resting_potential
        self.spike_times = []
        self.refractory_counter = 0
        self.potential_history = []
        
    def step(self, input_current: float, time_idx: int) -> bool:
        """
        Process one time step of input current.
        Returns True if spike occurred.
        """
        spike = False
        
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            self.potential = self.resting_potential
        else:
            # Leaky integration
            self.potential = self.potential * self.decay + input_current
            
            # Check for spike
            if self.potential >= self.threshold:
                spike = True
                self.spike_times.append(time_idx)
                self.potential = self.resting_potential
                self.refractory_counter = self.refractory_period
                
        self.potential_history.append(self.potential)
        return spike
    
    def process_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Process entire signal and return binary spike train.
        """
        self.reset()
        spikes = np.zeros(len(signal), dtype=np.int8)
        
        for i, val in enumerate(signal):
            if self.step(val, i):
                spikes[i] = 1
                
        return spikes
    
    def get_spike_features(self, signal_length: int, sample_rate: float = 200.0) -> Dict[str, float]:
        """
        Extract comprehensive spike features for classification.
        Returns dictionary of spike-based features.
        """
        spike_times = np.array(self.spike_times)
        n_spikes = len(spike_times)
        duration = signal_length / sample_rate
        
        features = {}
        
        # Basic spike statistics
        features['spike_count'] = n_spikes
        features['spike_rate'] = n_spikes / duration if duration > 0 else 0
        
        if n_spikes < 2:
            # Not enough spikes for ISI analysis
            features['isi_mean'] = 0
            features['isi_std'] = 0
            features['isi_cv'] = 0
            features['isi_min'] = 0
            features['isi_max'] = 0
            features['burst_count'] = 0
            features['burst_ratio'] = 0
        else:
            # Inter-spike interval (ISI) features
            isi = np.diff(spike_times) / sample_rate * 1000  # Convert to ms
            features['isi_mean'] = np.mean(isi)
            features['isi_std'] = np.std(isi)
            features['isi_cv'] = features['isi_std'] / features['isi_mean'] if features['isi_mean'] > 0 else 0
            features['isi_min'] = np.min(isi)
            features['isi_max'] = np.max(isi)
            
            # Burst detection (ISI < 20ms indicates burst)
            burst_threshold = 20  # ms
            bursts = isi < burst_threshold
            features['burst_count'] = np.sum(bursts)
            features['burst_ratio'] = features['burst_count'] / len(isi) if len(isi) > 0 else 0
        
        # Spike timing features
        if n_spikes > 0:
            features['first_spike_time'] = spike_times[0] / sample_rate
            features['last_spike_time'] = spike_times[-1] / sample_rate
            features['spike_duration'] = features['last_spike_time'] - features['first_spike_time']
        else:
            features['first_spike_time'] = 0
            features['last_spike_time'] = 0
            features['spike_duration'] = 0
            
        # Potential history features
        if len(self.potential_history) > 0:
            pot = np.array(self.potential_history)
            features['potential_mean'] = np.mean(pot)
            features['potential_max'] = np.max(pot)
            features['potential_std'] = np.std(pot)
        else:
            features['potential_mean'] = 0
            features['potential_max'] = 0
            features['potential_std'] = 0
            
        return features


class FootstepFeatureExtractor:
    """
    Complete feature extraction pipeline for footstep vibration signals.
    Extracts ~85 features combining LIF, FFT, MFCC, and statistical approaches.
    
    Improvements:
    - Wider bandpass (8-200Hz) for full footstep spectrum
    - MFCC for temporal envelope
    - Signal entropy for complexity
    - Adaptive gain normalization
    """
    
    def __init__(self, sample_rate: float = 200.0):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2
        
        # Multiple LIF neurons with different sensitivities
        self.lif_neurons = {
            'low': LIF(threshold=0.3, decay=0.90, refractory_period=3),
            'mid': LIF(threshold=0.5, decay=0.95, refractory_period=5),
            'high': LIF(threshold=0.7, decay=0.97, refractory_period=7)
        }
        
        # Frequency bands for spectral analysis (Hz)
        self.freq_bands = {
            'sub_bass': (1, 10),      # Very low frequency rumble
            'bass': (10, 25),          # Main footstep frequency
            'low_mid': (25, 50),       # Secondary harmonics
            'high_mid': (50, 80),      # High frequency content
            'high': (80, 150),         # Extended high frequency
        }
        
        # MFCC configuration
        self.n_mfcc = 13  # Number of MFCC coefficients
        self.n_mels = 26  # Number of mel filter banks
        
    def butterworth_filter(self, data: np.ndarray, lowcut: float = 8.0, 
                          highcut: float = 95.0, order: int = 4) -> np.ndarray:
        """
        Apply Butterworth bandpass filter to isolate footstep frequencies.
        Range: 12-180Hz (Matches frontend graph logic).
        """
        low = lowcut / self.nyquist
        high = min(highcut / self.nyquist, 0.99)
        
        if low >= high or low <= 0:
            return data
            
        try:
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, data)
        except Exception:
            return data

    def apply_frontend_processing(self, data: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Apply exact processing chain used by frontend graph:
        1. Spike Multiplier (0.8x)
        2. Bandpass Filter (12-180 Hz)
        3. Amplitude Gate (>= 60 ADC)
        
        Returns:
            Tuple[filtered_data, is_valid]
        """
        # 1. Apply Spike Multiplier (0.8x)
        data = data * 0.8
        
        # 2. Apply Bandpass Filter (12-180 Hz)
        filtered = self.butterworth_filter(data, lowcut=12.0, highcut=180.0)
        
        # 3. Apply Amplitude Gate (>= 50 ADC)
        # Check if peak amplitude exceeds gate
        peak_amplitude = np.max(np.abs(filtered))
        if peak_amplitude < 50:
            return filtered, False
            
        return filtered, True
            
    def normalize_signal(self, data: np.ndarray) -> np.ndarray:
        """Normalize signal to [-1, 1] range for consistent processing."""
        data = np.array(data, dtype=np.float64)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            return data / max_val
        return data
    
    def adaptive_gain_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Apply adaptive gain normalization to handle variable signal amplitudes.
        Helps with low/medium vibration signals in synthetic tests.
        """
        data = np.array(data, dtype=np.float64)
        
        # Calculate windowed RMS for adaptive gain
        window_size = max(20, len(data) // 10)
        
        # Pad data for windowed processing
        padded = np.pad(data, (window_size // 2, window_size // 2), mode='reflect')
        
        # Calculate local RMS envelope
        local_rms = np.zeros(len(data))
        for i in range(len(data)):
            window = padded[i:i + window_size]
            local_rms[i] = np.sqrt(np.mean(window ** 2)) + 1e-8
        
        # Smooth the envelope
        local_rms = gaussian_filter1d(local_rms, sigma=window_size // 4)
        
        # Target RMS (median of the local RMS values)
        target_rms = np.median(local_rms)
        if target_rms < 1e-6:
            target_rms = np.std(data)
        
        # Apply adaptive gain
        gain = np.clip(target_rms / local_rms, 0.1, 10)  # Limit gain range
        normalized = data * gain
        
        # Final normalization to [-1, 1]
        max_val = np.max(np.abs(normalized))
        if max_val > 0:
            normalized = normalized / max_val
            
        return normalized
        
    def validate_chunk(self, data: np.ndarray, min_std: float = 0.0001) -> bool:
        """
        Validate that the chunk contains meaningful footstep signal.
        VERY lenient - only rejects completely flat signals.
        DC ratio check REMOVED as ADC data is naturally positive (0-4095).
        """
        if len(data) < 20:  # Reduced minimum length
            return False
            
        data_std = np.std(data)
        
        # Only reject completely flat signals (no variation at all)
        # No DC ratio check - ADC values are naturally positive!
        return data_std > min_std
        
    def extract_statistical_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain statistical features from the signal.
        """
        features = {}
        
        # Basic statistics
        features['stat_mean'] = np.mean(data)
        features['stat_std'] = np.std(data)
        features['stat_var'] = np.var(data)
        features['stat_max'] = np.max(data)
        features['stat_min'] = np.min(data)
        features['stat_range'] = features['stat_max'] - features['stat_min']
        features['stat_rms'] = np.sqrt(np.mean(data ** 2))
        
        # Shape features
        features['stat_skewness'] = self._skewness(data)
        features['stat_kurtosis'] = self._kurtosis(data)
        
        # Zero crossings (indicative of oscillation frequency)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(data - np.mean(data)))) > 0)
        features['stat_zero_crossings'] = zero_crossings
        features['stat_zcr'] = zero_crossings / len(data)  # Zero crossing rate
        
        # Peak features
        abs_data = np.abs(data)
        peaks, properties = find_peaks(abs_data, height=np.std(abs_data) * 0.5)
        features['stat_peak_count'] = len(peaks)
        features['stat_peak_mean'] = np.mean(properties['peak_heights']) if len(peaks) > 0 else 0
        
        # Energy features
        features['stat_energy'] = np.sum(data ** 2)
        features['stat_power'] = features['stat_energy'] / len(data)
        
        # Smoothed RMS envelope features
        smoothed_rms = self._smoothed_rms_envelope(data)
        features['stat_smoothed_rms_mean'] = np.mean(smoothed_rms)
        features['stat_smoothed_rms_std'] = np.std(smoothed_rms)
        features['stat_smoothed_rms_max'] = np.max(smoothed_rms)
        
        # Signal entropy (complexity measure)
        features['stat_signal_entropy'] = self._signal_entropy(data)
        
        return features
    
    def _smoothed_rms_envelope(self, data: np.ndarray, window_size: int = 20) -> np.ndarray:
        """Calculate smoothed RMS envelope of the signal."""
        # Calculate squared values
        squared = data ** 2
        
        # Apply convolution for windowed mean
        window = np.ones(window_size) / window_size
        if len(data) >= window_size:
            smoothed = np.convolve(squared, window, mode='same')
            return np.sqrt(smoothed)
        else:
            return np.sqrt(np.mean(squared)) * np.ones(len(data))
    
    def _signal_entropy(self, data: np.ndarray, n_bins: int = 32) -> float:
        """
        Calculate Shannon entropy of the signal's amplitude distribution.
        Higher entropy = more random/noisy, Lower entropy = more structured.
        """
        # Create histogram of signal values
        hist, _ = np.histogram(data, bins=n_bins, density=True)
        
        # Add small epsilon to avoid log(0)
        hist = hist + 1e-10
        hist = hist / np.sum(hist)  # Normalize
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        return entropy
        
    def extract_fft_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency-domain features using FFT.
        """
        features = {}
        
        # Compute FFT
        n = len(data)
        fft_vals = fft(data)
        fft_magnitude = np.abs(fft_vals[:n // 2])
        freqs = fftfreq(n, 1 / self.sample_rate)[:n // 2]
        
        # Normalize magnitude
        fft_magnitude = fft_magnitude / (n / 2)
        
        # Total spectral energy
        total_energy = np.sum(fft_magnitude ** 2)
        features['fft_total_energy'] = total_energy
        
        # Band energies (relative)
        for band_name, (low, high) in self.freq_bands.items():
            mask = (freqs >= low) & (freqs < high)
            band_energy = np.sum(fft_magnitude[mask] ** 2)
            features[f'fft_{band_name}_energy'] = band_energy
            features[f'fft_{band_name}_ratio'] = band_energy / (total_energy + 1e-10)
            
        # Spectral centroid (center of mass of spectrum)
        features['fft_centroid'] = np.sum(freqs * fft_magnitude) / (np.sum(fft_magnitude) + 1e-10)
        
        # Spectral spread (bandwidth)
        centroid = features['fft_centroid']
        features['fft_spread'] = np.sqrt(
            np.sum(((freqs - centroid) ** 2) * fft_magnitude) / (np.sum(fft_magnitude) + 1e-10)
        )
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumsum = np.cumsum(fft_magnitude ** 2)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * total_energy)
        features['fft_rolloff'] = freqs[min(rolloff_idx, len(freqs) - 1)]
        
        # Dominant frequency
        dom_idx = np.argmax(fft_magnitude)
        features['fft_dominant_freq'] = freqs[dom_idx]
        features['fft_dominant_magnitude'] = fft_magnitude[dom_idx]
        
        # Spectral flatness (how noise-like vs tonal)
        geometric_mean = np.exp(np.mean(np.log(fft_magnitude + 1e-10)))
        arithmetic_mean = np.mean(fft_magnitude)
        features['fft_flatness'] = geometric_mean / (arithmetic_mean + 1e-10)
        
        return features
    
    def extract_mfcc_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract MFCC (Mel-Frequency Cepstral Coefficients) features.
        Excellent for capturing temporal envelope characteristics.
        """
        features = {}
        
        # Compute FFT
        n = len(data)
        fft_vals = fft(data)
        power_spectrum = np.abs(fft_vals[:n // 2]) ** 2
        freqs = fftfreq(n, 1 / self.sample_rate)[:n // 2]
        
        # Create mel filter banks
        mel_filters = self._create_mel_filterbank(n // 2, self.n_mels)
        
        # Apply mel filters to power spectrum
        mel_energies = np.dot(mel_filters, power_spectrum)
        mel_energies = np.maximum(mel_energies, 1e-10)  # Avoid log(0)
        
        # Apply log compression
        log_mel_energies = np.log(mel_energies)
        
        # Apply DCT to get MFCCs
        mfccs = dct(log_mel_energies, type=2, norm='ortho')[:self.n_mfcc]
        
        # Store MFCC coefficients as features
        for i, mfcc_val in enumerate(mfccs):
            features[f'mfcc_{i}'] = float(mfcc_val)
        
        # MFCC statistics
        features['mfcc_mean'] = float(np.mean(mfccs))
        features['mfcc_std'] = float(np.std(mfccs))
        features['mfcc_max'] = float(np.max(mfccs))
        features['mfcc_min'] = float(np.min(mfccs))
        
        return features
    
    def _create_mel_filterbank(self, n_fft_bins: int, n_mels: int) -> np.ndarray:
        """Create mel-scale triangular filter bank."""
        # Mel scale conversion functions
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)
        
        # Frequency range
        low_freq = 1
        high_freq = self.sample_rate / 2
        
        # Mel points
        low_mel = hz_to_mel(low_freq)
        high_mel = hz_to_mel(high_freq)
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert to FFT bin indices
        bin_points = np.floor((n_fft_bins * 2) * hz_points / self.sample_rate).astype(int)
        bin_points = np.clip(bin_points, 0, n_fft_bins - 1)
        
        # Create filterbank
        filterbank = np.zeros((n_mels, n_fft_bins))
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            # Rising edge
            for j in range(left, center):
                if center != left:
                    filterbank[i, j] = (j - left) / (center - left)
            
            # Falling edge
            for j in range(center, right):
                if right != center:
                    filterbank[i, j] = (right - j) / (right - center)
        
        return filterbank
        
    def extract_lif_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract LIF neuron-based features from multiple neurons.
        """
        features = {}
        
        # Process with each neuron sensitivity
        for name, neuron in self.lif_neurons.items():
            neuron.reset()
            neuron.process_signal(data)
            
            # Get comprehensive spike features
            spike_features = neuron.get_spike_features(len(data), self.sample_rate)
            
            # Add prefix for this neuron
            for feat_name, value in spike_features.items():
                features[f'lif_{name}_{feat_name}'] = value
                
        # Cross-neuron features
        spike_counts = [self.lif_neurons[n].spike_times for n in ['low', 'mid', 'high']]
        features['lif_total_spikes'] = sum(len(sc) for sc in spike_counts)
        features['lif_low_high_ratio'] = (
            len(spike_counts[0]) / (len(spike_counts[2]) + 1) 
            if len(spike_counts[2]) > 0 else len(spike_counts[0])
        )
        
        return features
        
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of distribution."""
        n = len(data)
        if n < 3:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
        
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of distribution."""
        n = len(data)
        if n < 4:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
        
    def process_chunk(self, raw_data: List[float]) -> Optional[Dict[str, Any]]:
        """
        Process a chunk of raw ADC samples and extract all features.
        Returns None if chunk is invalid.
        
        Pipeline (Matches Frontend):
        1. Apply Spike Multiplier (0.8x)
        2. Apply Bandpass Filter (12-180 Hz)
        3. Apply Amplitude Gate (>= 60 ADC)
        4. Extract features from FILTERED waveform
        """
        # Convert to numpy array
        data = np.array(raw_data, dtype=np.float64)
        
        # Apply frontend-matched processing
        filtered, is_valid = self.apply_frontend_processing(data)
        
        if not is_valid:
            print(f"[FEATURES] Rejected by Amplitude Gate (< 60 ADC): peak={np.max(np.abs(filtered)):.2f}")
            return None
            
        # Validate chunk length
        if len(filtered) < 20:
            return None
        
        print(f"[FEATURES] Validation OK: len={len(filtered)}, peak={np.max(np.abs(filtered)):.2f}")
        
        # Apply adaptive gain normalization (helps with low/medium vibration)
        # Note: We use the filtered data for this
        normalized = self.adaptive_gain_normalize(filtered)
        
        # Extract all feature sets
        try:
            features = {}
            features.update(self.extract_statistical_features(normalized))
            features.update(self.extract_fft_features(normalized))
            features.update(self.extract_mfcc_features(normalized))
            features.update(self.extract_lif_features(normalized))
            
            # Convert all numpy types to Python native types for JSON serialization
            features = {k: float(v) if hasattr(v, 'item') else float(v) for k, v in features.items()}
            
            # Add the filtered waveform to the result so it can be saved
            features['_filtered_waveform'] = filtered.tolist()
            
            print(f"[FEATURES] ✓ Extracted {len(features)} features")
            return features
        except Exception as e:
            print(f"[FEATURES] ✗ Feature extraction error: {e}")
            return None
        
    def extract_features_batch(self, samples_list: List[List[float]]) -> List[Dict[str, float]]:
        """
        Process multiple chunks and extract features for each.
        """
        results = []
        for samples in samples_list:
            features = self.process_chunk(samples)
            if features is not None:
                results.append(features)
        return results


# Feature names for ML model (must match extraction order)
FEATURE_NAMES = [
    # Statistical features (19) - includes new smoothed RMS and entropy
    'stat_mean', 'stat_std', 'stat_var', 'stat_max', 'stat_min', 'stat_range',
    'stat_rms', 'stat_skewness', 'stat_kurtosis', 'stat_zero_crossings', 'stat_zcr',
    'stat_peak_count', 'stat_peak_mean', 'stat_energy', 'stat_power',
    'stat_smoothed_rms_mean', 'stat_smoothed_rms_std', 'stat_smoothed_rms_max',
    'stat_signal_entropy',
    
    # FFT features (19) - includes new high band
    'fft_total_energy', 
    'fft_sub_bass_energy', 'fft_sub_bass_ratio',
    'fft_bass_energy', 'fft_bass_ratio',
    'fft_low_mid_energy', 'fft_low_mid_ratio',
    'fft_high_mid_energy', 'fft_high_mid_ratio',
    'fft_high_energy', 'fft_high_ratio',
    'fft_centroid', 'fft_spread', 'fft_rolloff',
    'fft_dominant_freq', 'fft_dominant_magnitude', 'fft_flatness',
    
    # MFCC features (17) - NEW
    'mfcc_0', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6',
    'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12',
    'mfcc_mean', 'mfcc_std', 'mfcc_max', 'mfcc_min',
    
    # LIF features - low threshold neuron (15)
    'lif_low_spike_count', 'lif_low_spike_rate', 'lif_low_isi_mean', 'lif_low_isi_std',
    'lif_low_isi_cv', 'lif_low_isi_min', 'lif_low_isi_max', 'lif_low_burst_count',
    'lif_low_burst_ratio', 'lif_low_first_spike_time', 'lif_low_last_spike_time',
    'lif_low_spike_duration', 'lif_low_potential_mean', 'lif_low_potential_max', 'lif_low_potential_std',
    
    # LIF features - mid threshold neuron (15)
    'lif_mid_spike_count', 'lif_mid_spike_rate', 'lif_mid_isi_mean', 'lif_mid_isi_std',
    'lif_mid_isi_cv', 'lif_mid_isi_min', 'lif_mid_isi_max', 'lif_mid_burst_count',
    'lif_mid_burst_ratio', 'lif_mid_first_spike_time', 'lif_mid_last_spike_time',
    'lif_mid_spike_duration', 'lif_mid_potential_mean', 'lif_mid_potential_max', 'lif_mid_potential_std',
    
    # LIF features - high threshold neuron (15)
    'lif_high_spike_count', 'lif_high_spike_rate', 'lif_high_isi_mean', 'lif_high_isi_std',
    'lif_high_isi_cv', 'lif_high_isi_min', 'lif_high_isi_max', 'lif_high_burst_count',
    'lif_high_burst_ratio', 'lif_high_first_spike_time', 'lif_high_last_spike_time',
    'lif_high_spike_duration', 'lif_high_potential_mean', 'lif_high_potential_max', 'lif_high_potential_std',
    
    # Cross-neuron LIF features (2)
    'lif_total_spikes', 'lif_low_high_ratio'
]


def get_feature_vector(features: Dict[str, float]) -> List[float]:
    """Convert feature dictionary to ordered vector matching FEATURE_NAMES."""
    return [features.get(name, 0.0) for name in FEATURE_NAMES]


# Convenience function for backward compatibility
def extract_features(raw_data: List[float]) -> Optional[List[float]]:
    """
    Extract features from raw ADC samples.
    Returns feature vector or None if invalid.
    """
    extractor = FootstepFeatureExtractor()
    features = extractor.process_chunk(raw_data)
    if features is None:
        return None
    return get_feature_vector(features)
