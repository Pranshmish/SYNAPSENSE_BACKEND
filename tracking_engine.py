import numpy as np
import scipy.signal as signal
import pywt
from sklearn.decomposition import FastICA
from scipy.optimize import least_squares
from typing import List, Dict, Tuple, Any

# Constants
FS = 2000  # Sampling frequency
ROOM_DIM = (10, 10)  # 10x10 meters
SENSOR_POSITIONS = np.array([
    [0, 0], [10, 0], [10, 10], [0, 10], [5, 0], [5, 10]
])
WAVELET_NAME = 'db4'
VELOCITY = 300.0

class KalmanFilter:
    def __init__(self, initial_state=None, dt=0.5):
        self.dt = dt
        self.x = np.zeros(4) if initial_state is None else np.array(initial_state)
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.P = np.eye(4) * 10
        self.R = np.eye(2) * 1.0
        self.Q = np.eye(4) * 0.1
        
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]
    
    def update(self, z):
        z = np.array(z)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2]

class TrackingEngine:
    def __init__(self):
        self.kalman_filters = {}
        self.next_person_id = 1
        self.tracks = {}

    def process_signals(self, channel_data: Dict[str, List[float]], fs: int = FS) -> Dict:
        self.kalman_filters = {}
        self.next_person_id = 1
        self.tracks = {}

        keys = sorted(channel_data.keys())
        raw_signals = np.array([channel_data[k] for k in keys])
        
        filtered_signals = self._preprocess_signals(raw_signals, fs)
        denoised_signals = self._apply_dwt_denoising(filtered_signals)
        events = self._detect_events_and_toas(denoised_signals, fs)
        
        footsteps_result = []
        for event in events:
            pos = self._multilaterate(event['toas'])
            if pos is not None:
                person_id, smooth_pos = self._update_tracks(pos, event['timestamp'])
                footsteps_result.append({
                    "id": len(footsteps_result) + 1,
                    "timestamp": round(event['timestamp'], 2),
                    "raw_TOA": [round(t, 4) for t in event['toas']],
                    "position": [round(float(pos[0]), 2), round(float(pos[1]), 2)],
                    "smoothed": [round(float(smooth_pos[0]), 2), round(float(smooth_pos[1]), 2)],
                    "person_id": person_id
                })
        
        persons_output = []
        colors = ["#007bff", "#ff3b30", "#28a745", "#ffc107"]
        for pid, points in self.tracks.items():
            persons_output.append({
                "id": pid,
                "color": colors[(pid - 1) % len(colors)],
                "footsteps_detected": len(points),
                "trajectory": [p['raw'] for p in points],
                "smoothed": [p['smooth'] for p in points],
                "times": [p['t'] for p in points] # Add timestamps to output
            })
            
        return {
            "persons": persons_output,
            "footsteps": footsteps_result,
            "debug_signals": {
                "separated": denoised_signals.tolist()[0][:200] if len(denoised_signals) > 0 else [] 
            }
        }

    def generate_single_person_data(self) -> Dict[str, Any]:
        """Simulates one person walking straight."""
        duration = 10.0; fs = 2000; n_samples = int(duration * fs)
        channels = {f"c{i+1}": np.random.normal(0, 0.005, n_samples) for i in range(6)}
        steps = 10
        for i in range(steps):
             # Walk (2,1) -> (8,8)
             x = 2.0 + (i/steps)*6.0
             y = 1.0 + (i/steps)*7.0
             t = 1.0 + i * (0.8 + np.random.normal(0, 0.1)) # Variable speed
             self._add_footstep_pulse(channels, x, y, t, fs)
        return {"timestamp_ms": 1000, "fs": fs, "samples": n_samples, "channels": {k: v.tolist() for k, v in channels.items()}}

    def generate_overlapped_data(self) -> Dict[str, Any]:
        """Simulates high overlap scenario."""
        duration = 10.0; fs = 2000; n_samples = int(duration * fs)
        channels = {f"c{i+1}": np.random.normal(0, 0.005, n_samples) for i in range(6)}
        
        # Two fast walkers crossing in middle
        steps = 10
        for i in range(steps):
             # A: (0,5) -> (10,5)
             self._add_footstep_pulse(channels, i*1.0, 5.0, 1.0+i*0.5, fs)
             # B: (10,5) -> (0,5) overlaps A
             self._add_footstep_pulse(channels, 10.0-i*1.0, 5.0, 1.1+i*0.5, fs) # 0.1s offset, heavy overlap
             
        return {"timestamp_ms": 1000, "fs": fs, "samples": n_samples, "channels": {k: v.tolist() for k, v in channels.items()}}

    def generate_real_footstep_test(self) -> Dict[str, Any]:
        # Using the real template
        footstep_template = [
             1395, 382, 825, 990, 945, 352, 1200, 457, 247, 832, 810, 322, 
             937, 1027, 870, 1170, 750, 772, 16410, 16410, 16410, 16410, 
             16410, 16410, 16410, 16410
        ]
        template = np.array(footstep_template, dtype=float)
        template = template - np.min(template)
        template = template / np.max(template)
        
        duration = 10.0; fs = 2000; n_samples = int(duration * fs)
        channels = {f"c{i+1}": np.random.normal(0, 0.005, n_samples) for i in range(6)}
        
        steps = 15
        # Varying speed simulation
        # Starts slow, speeds up, slows down
        current_time = 1.0
        for i in range(steps):
            x = 2.0 + (i / steps) * 6.0
            y = 2.0 + (i / steps) * 6.0
            
            # Simulate variable speed (interval between steps changes)
            speed_factor = 0.5 + 0.5 * np.sin(i / 3.0) # Varies between 0.5s and 1.0s interval
            current_time += speed_factor
            
            self._inject_real_footstep(channels, x, y, current_time, fs, template)
            
        return {"timestamp_ms": 1700000000, "fs": fs, "samples": n_samples, "channels": {k: v.tolist() for k, v in channels.items()}}

    def generate_stop_go_data(self) -> Dict[str, Any]:
        """Simulates a person walking, stopping, then walking again."""
        duration = 15.0; fs = 2000; n_samples = int(duration * fs)
        channels = {f"c{i+1}": np.random.normal(0, 0.005, n_samples) for i in range(6)}
        
        # Part 1: Walk (1,1) -> (4,4)
        for i in range(5):
             self._add_footstep_pulse(channels, 1.0+i*0.6, 1.0+i*0.6, 1.0+i*0.8, fs)
        
        # Pause: No steps for 3 seconds
        
        # Part 2: Continue (4,4) -> (8,8)
        start_t = 1.0 + 4*0.8 + 3.0 # Start after 3s pause
        for i in range(5):
             self._add_footstep_pulse(channels, 4.0+i*0.8, 4.0+i*0.8, start_t+i*0.8, fs)
             
        return {"timestamp_ms": 1000, "fs": fs, "samples": n_samples, "channels": {k: v.tolist() for k, v in channels.items()}}

    def generate_circle_data(self) -> Dict[str, Any]:
        """Simulates a person walking in a circle."""
        duration = 10.0; fs = 2000; n_samples = int(duration * fs)
        channels = {f"c{i+1}": np.random.normal(0, 0.005, n_samples) for i in range(6)}
        
        steps = 16
        center = (5, 5); radius = 3.0
        for i in range(steps):
             theta = (i / steps) * 2 * np.pi
             x = center[0] + radius * np.cos(theta)
             y = center[1] + radius * np.sin(theta)
             self._add_footstep_pulse(channels, x, y, 1.0+i*0.5, fs)
             
        return {"timestamp_ms": 1000, "fs": fs, "samples": n_samples, "channels": {k: v.tolist() for k, v in channels.items()}}

    def generate_three_person_data(self) -> Dict[str, Any]:
        """Simulates three random walkers."""
        duration = 20.0; fs = 2000; n_samples = int(duration * fs)
        channels = {f"c{i+1}": np.random.normal(0, 0.005, n_samples) for i in range(6)}
        
        # Person A: Linear
        for i in range(15):
             self._add_footstep_pulse(channels, 1.0+(i/15)*8, 1.0+(i/15)*8, 1.0+i*1.0, fs)
        # Person B: Vertical
        for i in range(12):
             self._add_footstep_pulse(channels, 8.0, 1.0+(i/12)*8, 1.5+i*1.2, fs)
        # Person C: Horizontal
        for i in range(12):
             self._add_footstep_pulse(channels, 1.0+(i/12)*8, 8.0, 2.0+i*1.1, fs)

        return {"timestamp_ms": 1000, "fs": fs, "samples": n_samples, "channels": {k: v.tolist() for k, v in channels.items()}}

    def _inject_real_footstep(self, channels, x, y, t0, fs, template):
        pos = np.array([x, y])
        for i, s_pos in enumerate(SENSOR_POSITIONS):
            dist = np.linalg.norm(s_pos - pos)
            delay = dist / VELOCITY
            t_arrival = t0 + delay
            start_idx = int(t_arrival * fs)
            amp = 1.0 / (dist + 0.1)
            L = len(template); end_idx = start_idx + L
            s_start = max(0, start_idx); s_end = min(len(channels['c1']), end_idx)
            if s_end > s_start:
                t_start = s_start - start_idx
                t_end = t_start + (s_end - s_start)
                channels[f"c{i+1}"][s_start:s_end] += template[t_start:t_end] * amp * 5.0

    def generate_two_person_data(self) -> Dict[str, Any]:
        duration = 20.0; fs = 2000; n_samples = int(duration * fs)
        channels = {f"c{i+1}": np.random.normal(0, 0.005, n_samples) for i in range(6)}
        steps_A = 10; steps_B = 10
        for i in range(steps_A):
            self._add_footstep_pulse(channels, 1.0+(i/steps_A)*4.0, 5.0+np.random.normal(0,0.2), 1.0+i*1.5, fs)
        for i in range(steps_B):
            self._add_footstep_pulse(channels, 9.0-(i/steps_B)*4.0, 5.0+np.random.normal(0,0.2), 1.5+i*1.5, fs)
        return {"timestamp_ms": 1000, "fs": fs, "samples": n_samples, "channels": {k: v.tolist() for k, v in channels.items()}}

    def _add_footstep_pulse(self, channels, x, y, t0, fs):
        pos = np.array([x, y])
        for i, s_pos in enumerate(SENSOR_POSITIONS):
            dist = np.linalg.norm(s_pos - pos)
            delay = dist / VELOCITY
            t_arrival = t0 + delay
            center_idx = int(t_arrival * fs)
            if 0 <= center_idx < len(channels['c1']):
                width = 50
                window_x = np.arange(-width*2, width*2)
                pulse = (1 - 2 * (np.pi * window_x / width)**2) * np.exp(-(np.pi * window_x / width)**2)
                amp = 1.0 / (dist + 0.5) 
                start = center_idx - width*2
                end = center_idx + width*2
                p_start = max(0, -start)
                s_start = max(0, start)
                s_end = min(len(channels['c1']), end)
                if s_end > s_start:
                    channels[f"c{i+1}"][s_start:s_end] += pulse[p_start:p_start + (s_end-s_start)] * amp
    
    # ... (Rest of the helper methods remain unchanged)
    def _preprocess_signals(self, signals, fs):
        if signals.shape[1] < 30: return signals
        nyquist = 0.5 * fs
        try:
            b, a = signal.butter(4, [2/nyquist, 120/nyquist], btype='band')
            filtered = signal.filtfilt(b, a, signals, axis=1)
            norm = np.max(np.abs(filtered), axis=1, keepdims=True)
            norm[norm == 0] = 1
            return filtered / norm
        except: return signals

    def _apply_dwt_denoising(self, signals):
        denoised = []
        for sig in signals:
            try:
                coeffs = pywt.wavedec(sig, WAVELET_NAME, level=4)
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(len(sig) if len(sig)>0 else 1))
                new_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
                rec = pywt.waverec(new_coeffs, WAVELET_NAME)
                if len(rec) > len(sig): rec = rec[:len(sig)]
                denoised.append(rec)
            except: denoised.append(sig)
        return np.array(denoised)

    def _detect_events_and_toas(self, signals, fs):
        energy = np.sum(signals**2, axis=0)
        peaks, _ = signal.find_peaks(energy, height=1.0, distance=fs//4)
        events = []
        for p in peaks:
            window = 100
            start = max(0, p - window)
            end = min(signals.shape[1], p + window)
            toas = []
            for ch_idx in range(len(signals)):
                segment = signals[ch_idx, start:end]
                if len(segment) == 0:
                    toas.append(0); continue
                local_idx = np.argmax(np.abs(segment))
                toas.append((start + local_idx) / fs)
            events.append({"timestamp": p/fs, "toas": toas})
        return events

    def _multilaterate(self, toas):
        toas = np.array(toas)
        if np.all(toas == 0): return None
        def residuals(vars):
            x, y, t0 = vars
            dists_est = np.sqrt(np.sum((SENSOR_POSITIONS[:len(toas)] - np.array([x, y]))**2, axis=1))
            dists_meas = VELOCITY * (toas - t0)
            return dists_est - dists_meas
        x0 = [5, 5, np.min(toas)]
        try:
            res = least_squares(residuals, x0, bounds=([0, 0, -np.inf], [10, 10, np.inf]))
            if res.success: return res.x[:2]
        except: pass
        return None

    def _update_tracks(self, pos, t):
        min_dist = 1000
        best_id = -1
        for pid, kf in self.kalman_filters.items():
            kf.predict()
        for pid, kf in self.kalman_filters.items():
            pred = kf.x[:2]
            dist = np.linalg.norm(pred - pos)
            if dist < min_dist:
                min_dist = dist; best_id = pid
        if min_dist < 3.0:
            kf = self.kalman_filters[best_id]
            smooth_pos = kf.update(pos)
            self.tracks[best_id].append({"raw": pos.tolist(), "smooth": smooth_pos.tolist(), "t": t})
            return best_id, smooth_pos
        else:
            new_id = self.next_person_id
            self.next_person_id += 1
            kf = KalmanFilter(initial_state=[pos[0], pos[1], 0, 0])
            self.kalman_filters[new_id] = kf
            kf.update(pos)
            self.tracks[new_id] = [{"raw": pos.tolist(), "smooth": pos.tolist(), "t": t}]
            return new_id, pos

engine = TrackingEngine()
