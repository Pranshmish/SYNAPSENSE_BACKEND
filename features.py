import numpy as np
from scipy import signal as sp_signal
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis, entropy
from datetime import datetime

# Config from reference
SAMPLE_RATE = 200
BP_LOW = 8
BP_HIGH = 90  # Reference line 35 says BP_HIGH=90.
FILTER_ORDER = 4
LIF_TAU = 0.020
LIF_THRESH = 0.025
LIF_REFRAC = 0.010

# Validation Thresholds
MIN_ENERGY = 2200.0  # Reference
MIN_SIGNAL_STD = 6.0
MIN_PEAK_DEVIATION = 35.0 # Reference
MIN_SNR = 1.5
MIN_PEAK_RATIO = 1.3
MIN_DOMINANT_FREQ = 10.0
MAX_DOMINANT_FREQ = 80.0

class LIF:
    def __init__(self, tau=LIF_TAU, thresh=LIF_THRESH, refrac=LIF_REFRAC, fs=SAMPLE_RATE):
        self.tau = tau
        self.thresh = thresh
        self.refrac = refrac
        self.dt = 1.0 / fs

    def encode(self, sig):
        n = len(sig)
        spikes = np.zeros(n)
        membrane = np.zeros(n)
        v = 0.0
        last_t = -1.0
        
        for i in range(n):
            t = i * self.dt
            if t - last_t < self.refrac:
                v = 0.0
            else:
                v += (-v / self.tau + abs(sig[i])) * self.dt
                if v >= self.thresh:
                    spikes[i] = 1
                    v = 0.0
                    last_t = t
            membrane[i] = v
        return spikes, membrane

class FootstepFeatureExtractor:
    def __init__(self):
        self.lif = LIF()
        self.fs = SAMPLE_RATE

    def butterworth_filter(self, data):
        nyq = self.fs / 2
        lo = BP_LOW / nyq
        hi = BP_HIGH / nyq
        try:
            b, a = sp_signal.butter(FILTER_ORDER, [lo, hi], btype='band')
            return sp_signal.filtfilt(b, a, data)
        except Exception as e:
            print(f"Filter error: {e}")
            return data

    def validate_chunk(self, data):
        # NaN check
        if np.isnan(data).any() or np.isinf(data).any():
            return False, "Contains NaN/Inf"
        
        # Centered data for energy/stats
        centered = data - np.mean(data)
        
        # Energy check
        energy = np.sum(centered**2)
        if energy < MIN_ENERGY:
            return False, f"Low Energy: {energy:.1f} < {MIN_ENERGY}"
            
        # Std check
        sig_std = np.std(data)
        if sig_std < MIN_SIGNAL_STD:
            return False, f"Low Std: {sig_std:.1f}"

        return True, "Valid"

    def process_chunk(self, raw_samples):
        # 1. Convert to numpy
        raw = np.array(raw_samples, dtype=float)
        
        # 2. Validation
        is_valid, msg = self.validate_chunk(raw)
        if not is_valid:
            return None
            
        # 3. Preprocessing
        # Center the signal
        sig = raw - np.mean(raw)
        # Filter
        sig = self.butterworth_filter(sig)
        
        # 4. Feature Extraction
        try:
            # Time domain
            peak = float(np.max(np.abs(sig)))
            energy = float(np.sum(sig**2))
            dur = len(sig) / self.fs
            
            pk_idx = np.argmax(np.abs(sig))
            pk_val = np.abs(sig[pk_idx])
            
            # Rise/Fall time
            rs = np.where(np.abs(sig[:pk_idx]) > 0.1*pk_val)[0]
            re = np.where(np.abs(sig[:pk_idx]) > 0.9*pk_val)[0]
            rise = float((re[0]-rs[0])/self.fs) if len(rs) and len(re) else 0.0
            
            fs_indices = np.where(np.abs(sig[pk_idx:]) > 0.9*pk_val)[0]
            fe_indices = np.where(np.abs(sig[pk_idx:]) < 0.1*pk_val)[0]
            fall = float((fe_indices[0]-fs_indices[-1])/self.fs) if len(fs_indices) and len(fe_indices) else 0.0
            
            zc = int(np.sum(np.diff(np.sign(sig)) != 0))
            std = float(np.std(sig))
            rms = float(np.sqrt(np.mean(sig**2)))
            sk = float(skew(sig))
            kt = float(kurtosis(sig))
            
            # Freq domain
            win = sig * np.hanning(len(sig))
            fft_m = np.abs(rfft(win)) / len(sig)
            freqs = rfftfreq(len(sig), 1/self.fs)
            
            if len(fft_m) > 1:
                dom = float(freqs[np.argmax(fft_m[1:])+1])
            else:
                dom = 0.0
                
            tot = np.sum(fft_m)
            if tot > 0:
                cent = float(np.sum(freqs*fft_m)/tot)
                bw = float(np.sqrt(np.sum((freqs-cent)**2*fft_m)/tot))
            else:
                cent = 0.0
                bw = 0.0
                
            ent = float(entropy(fft_m + 1e-10))
            
            # Low freq ratio (5-30Hz)
            lo_idx = (freqs >= 5) & (freqs <= 30)
            lo_rat = float(np.sum(fft_m[lo_idx]*2)/tot) if tot > 0 else 0.0
            
            # LIF features
            spk, mem = self.lif.encode(sig)
            n_spk = int(np.sum(spk))
            fr = float(n_spk/dur) if dur > 0 else 0.0
            
            st = np.where(spk == 1)[0] / self.fs
            if len(st) > 1:
                isis = np.diff(st)
                isi_m = float(np.mean(isis))
                isi_s = float(np.std(isis))
                isi_cv = float(isi_s/isi_m) if isi_m > 0 else 0.0
            else:
                isi_m, isi_s, isi_cv = 0.0, 0.0, 0.0
                
            lat = float(st[0]) if len(st) > 0 else 0.0
            
            # Feature vector (ordered list)
            features = [
                peak, energy, dur, rise, fall, 
                zc, std, rms, sk, kt, dom,
                cent, ent, bw, 
                lo_rat, n_spk, fr, isi_m, 
                isi_s, isi_cv, lat
            ]
            
            # Also return as dict for easier debugging/CSV
            features_dict = {
                "peak_amplitude": peak, "energy": energy, "step_duration": dur, "rise_time": rise, "fall_time": fall, 
                "zero_crossings": zc, "std": std, "rms": rms, "skewness": sk, "kurtosis": kt, "dominant_freq": dom,
                "spectral_centroid": cent, "spectral_entropy": ent, "spectral_bandwidth": bw, 
                "low_freq_ratio": lo_rat, "spike_count": n_spk, "firing_rate": fr, "isi_mean": isi_m, 
                "isi_std": isi_s, "isi_cv": isi_cv, "first_spike_latency": lat
            }
            
            return features_dict
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
