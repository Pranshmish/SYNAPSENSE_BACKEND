import sqlite3
import csv
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Use absolute paths based on script location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_SCRIPT_DIR, "db", "samples.db")
DATASET_DIR = os.path.join(_SCRIPT_DIR, "dataset")

# Main aggregated files for HOME/INTRUDER classification
HOME_CSV_PATH = os.path.join(DATASET_DIR, "HOME.csv")
INTRUDER_CSV_PATH = os.path.join(DATASET_DIR, "INTRUDER.csv")

class StorageManager:
    def __init__(self):
        self._init_db()
        self._init_dirs()

    def _init_dirs(self):
        os.makedirs("db", exist_ok=True)
        os.makedirs(DATASET_DIR, exist_ok=True)

    def _init_db(self):
        os.makedirs("db", exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person TEXT NOT NULL,
                features_json TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def save_sample_dual(self, person: str, features: Dict[str, Any], save_as_intruder: bool = False) -> Dict[str, bool]:
        """
        DUAL DATASET SAVING:
        1. Saves to main HOME.csv or INTRUDER.csv (for HOME/INTRUDER classification)
        2. Saves to individual HOME_{person}.csv or INTRUDER_{person}.csv (for person identification)
        
        Args:
            person: Person name (e.g., "Apurv", "HOME_Apurv", "INTRUDER_Test")
            features: Feature dictionary including optional waveforms and analysis data
            save_as_intruder: If True, save as INTRUDER class instead of HOME
        
        Example: "Apurv" with save_as_intruder=False → saves to:
          - HOME.csv (label='HOME')
          - HOME_Apurv/features_HOME_Apurv.csv (label='Apurv', class='HOME')
        
        Example: "Test" with save_as_intruder=True → saves to:
          - INTRUDER.csv (label='INTRUDER')
          - INTRUDER_Test/features_INTRUDER_Test.csv (label='Test', class='INTRUDER')
        
        Returns dict with save status for each file.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_safe = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        result = {"main_csv": False, "individual_csv": False, "sqlite": False, "analysis_plots": False}
        
        # Extract waveforms and analysis data if present (don't save to feature CSV)
        filtered_waveform = features.pop('_filtered_waveform', None)
        raw_waveform = features.pop('_raw_waveform', None)
        fft_data = features.pop('_fft_data', None)
        lif_data = features.pop('_lif_data', None)
        
        # Determine main class based on save_as_intruder flag
        main_class = 'INTRUDER' if save_as_intruder else 'HOME'
        
        # Normalize person label - extract name from any format
        # "INTRUDER_Apurv" -> "Apurv", "HOME_Apurv" -> "Apurv", "Apurv" -> "Apurv"
        individual_name = person
        if '_' in person:
            individual_name = person.split('_', 1)[1]
        
        # Final person label with appropriate prefix
        if individual_name and individual_name.upper() not in ['HOME', 'INTRUDER']:
            normalized_person = f"{main_class}_{individual_name}"
        else:
            normalized_person = main_class
        
        # 1. Save to main aggregated CSV (HOME.csv or INTRUDER.csv)
        main_csv_path = INTRUDER_CSV_PATH if save_as_intruder else HOME_CSV_PATH
        try:
            features_with_meta = features.copy()
            features_with_meta['_label'] = main_class
            features_with_meta['_person'] = individual_name
            features_with_meta['_timestamp'] = timestamp
            
            file_exists = os.path.exists(main_csv_path)
            fieldnames = list(features_with_meta.keys())
            
            with open(main_csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(features_with_meta)
            result["main_csv"] = True
        except Exception as e:
            print(f"[STORAGE] Error saving to main CSV: {e}")
        
        # 2. Save to individual person CSV
        try:
            person_dir = os.path.join(DATASET_DIR, normalized_person)
            os.makedirs(person_dir, exist_ok=True)
            csv_path = os.path.join(person_dir, f"features_{normalized_person}.csv")
            
            features_individual = features.copy()
            features_individual['_label'] = individual_name
            features_individual['_class'] = main_class
            features_individual['_timestamp'] = timestamp
            
            file_exists = os.path.exists(csv_path)
            fieldnames = list(features_individual.keys())
            
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(features_individual)
            result["individual_csv"] = True
        except Exception as e:
            print(f"[STORAGE] Error saving individual CSV: {e}")
        
        # 3. Save to SQLite
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO samples (person, features_json, timestamp) VALUES (?, ?, ?)",
                      (normalized_person, json.dumps(features), timestamp))
            conn.commit()
            conn.close()
            result["sqlite"] = True
        except Exception as e:
            print(f"[STORAGE] Error saving to SQLite: {e}")
        
        # 4. Save waveforms and analysis data
        person_dir = os.path.join(DATASET_DIR, normalized_person)
        
        # Save raw signal (before any processing)
        if raw_waveform:
            self._save_raw_signal(person_dir, raw_waveform, timestamp_safe, normalized_person)
        
        # Save filtered waveform with plot
        if filtered_waveform:
            self._save_waveform(normalized_person, filtered_waveform, timestamp)
        
        # 5. Save comprehensive analysis plots (FFT, LIF, Combined)
        if raw_waveform or filtered_waveform or fft_data or lif_data:
            try:
                self._save_analysis_plots(
                    person_dir=person_dir,
                    timestamp_safe=timestamp_safe,
                    person=normalized_person,
                    raw_waveform=raw_waveform,
                    filtered_waveform=filtered_waveform,
                    fft_data=fft_data,
                    lif_data=lif_data
                )
                result["analysis_plots"] = True
            except Exception as e:
                print(f"[STORAGE] Error in analysis plots: {e}")
        
        return result

    def save_sample(self, person: str, features: Dict[str, Any]):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_safe = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Extract filtered waveform if present
        filtered_waveform = features.pop('_filtered_waveform', None)
        
        # 1. Save to SQLite
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO samples (person, features_json, timestamp) VALUES (?, ?, ?)",
                  (person, json.dumps(features), timestamp))
        conn.commit()
        conn.close()
        
        # 2. Save to CSV (Features)
        person_dir = os.path.join(DATASET_DIR, person)
        os.makedirs(person_dir, exist_ok=True)
        csv_path = os.path.join(person_dir, f"features_{person}.csv")
        
        file_exists = os.path.exists(csv_path)
        
        # Ensure consistent field order
        fieldnames = list(features.keys())
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(features)
            
        # 3. Save Waveform (CSV + PNG)
        if filtered_waveform:
            # Save Waveform CSV
            wave_dir = os.path.join(person_dir, "waveforms")
            os.makedirs(wave_dir, exist_ok=True)
            wave_csv_path = os.path.join(wave_dir, f"wave_{timestamp_safe}.csv")
            
            try:
                import pandas as pd
                df_wave = pd.DataFrame({'amplitude': filtered_waveform})
                df_wave.to_csv(wave_csv_path, index=False)
            except Exception as e:
                print(f"[STORAGE] Error saving waveform CSV: {e}")
                
            # Save Waveform PNG
            plot_dir = os.path.join(person_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"plot_{timestamp_safe}.png")
            
            try:
                import matplotlib
                matplotlib.use('Agg') # Non-interactive backend
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(10, 4))
                plt.plot(filtered_waveform, color='green', linewidth=1.5)
                plt.title(f"Filtered Footstep Event - {person} ({timestamp})")
                plt.xlabel("Sample")
                plt.ylabel("Amplitude (Filtered)")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
            except Exception as e:
                print(f"[STORAGE] Error saving waveform PNG: {e}")

    def _save_waveform(self, person: str, filtered_waveform: List, timestamp: str):
        """Helper to save waveform data"""
        timestamp_safe = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        person_dir = os.path.join(DATASET_DIR, person)
        os.makedirs(person_dir, exist_ok=True)
        
        # Save Waveform CSV
        wave_dir = os.path.join(person_dir, "waveforms")
        os.makedirs(wave_dir, exist_ok=True)
        wave_csv_path = os.path.join(wave_dir, f"wave_{timestamp_safe}.csv")
        
        try:
            import pandas as pd
            df_wave = pd.DataFrame({'amplitude': filtered_waveform})
            df_wave.to_csv(wave_csv_path, index=False)
        except Exception as e:
            print(f"[STORAGE] Error saving waveform CSV: {e}")
        
        # Save Waveform PNG
        plot_dir = os.path.join(person_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"plot_{timestamp_safe}.png")
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 4))
            plt.plot(filtered_waveform, color='green', linewidth=1.5)
            plt.title(f"Filtered Footstep Event - {person} ({timestamp})")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude (Filtered)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
        except Exception as e:
            print(f"[STORAGE] Error saving waveform PNG: {e}")

    def _save_raw_signal(self, person_dir: str, raw_waveform: List, timestamp_safe: str, person: str):
        """Save raw signal (before any filtering/processing) to a separate directory."""
        try:
            raw_dir = os.path.join(person_dir, "raw_signals")
            os.makedirs(raw_dir, exist_ok=True)
            raw_csv_path = os.path.join(raw_dir, f"raw_{timestamp_safe}.csv")
            
            import pandas as pd
            df_raw = pd.DataFrame({'raw_adc': raw_waveform})
            df_raw.to_csv(raw_csv_path, index=False)
            print(f"[STORAGE] ✓ Saved raw signal: {raw_csv_path}")
        except Exception as e:
            print(f"[STORAGE] Error saving raw signal CSV: {e}")

    def _save_analysis_plots(self, person_dir: str, timestamp_safe: str, person: str, 
                             raw_waveform: List = None, filtered_waveform: List = None,
                             fft_data: Dict = None, lif_data: Dict = None, mfcc_data: Dict = None):
        """
        Save comprehensive analysis plots for each footstep event:
        - Raw Waveform
        - Filtered Waveform  
        - FFT Spectrum
        - MFCC (Mel-Frequency Cepstral Coefficients)
        - LIF Neuron Response
        - Combined Analysis (all in one figure)
        
        Also saves the data as CSVs for later visualization.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            
            plots_dir = os.path.join(person_dir, "plots")
            analysis_dir = os.path.join(person_dir, "analysis_data")
            os.makedirs(plots_dir, exist_ok=True)
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Sample rate assumption
            sample_rate = 200  # Hz
            
            # ===== 1. Save FFT Data & Plot =====
            if fft_data and 'frequencies' in fft_data and 'magnitudes' in fft_data:
                # Save FFT CSV
                import pandas as pd
                fft_df = pd.DataFrame({
                    'frequency': fft_data['frequencies'],
                    'magnitude': fft_data['magnitudes']
                })
                fft_csv_path = os.path.join(analysis_dir, f"fft_{timestamp_safe}.csv")
                fft_df.to_csv(fft_csv_path, index=False)
                
                # Save FFT Plot
                plt.figure(figsize=(10, 4))
                plt.bar(fft_data['frequencies'][:50], fft_data['magnitudes'][:50], 
                        color='#8b5cf6', alpha=0.8, width=0.8)
                plt.title(f"FFT Spectrum - {person}", fontsize=12, fontweight='bold')
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Magnitude")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"fft_{timestamp_safe}.png"), dpi=100)
                plt.close()
                print(f"[STORAGE] ✓ Saved FFT: {fft_csv_path}")
            
            # ===== 2. Compute and Save MFCC =====
            computed_mfcc = None
            if raw_waveform or filtered_waveform:
                signal = np.array(filtered_waveform if filtered_waveform else raw_waveform, dtype=np.float64)
                signal = signal - np.mean(signal)  # Remove DC offset
                
                # Try to use librosa if available
                try:
                    import librosa
                    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_fft=64, hop_length=32)
                    computed_mfcc = mfccs
                except ImportError:
                    # Manual MFCC computation without librosa
                    # Use simple DCT-based approach
                    from scipy.fftpack import dct
                    
                    # Frame the signal
                    frame_size = min(64, len(signal))
                    hop_size = frame_size // 2
                    n_frames = max(1, (len(signal) - frame_size) // hop_size + 1)
                    
                    # Create mel filterbank (simplified)
                    n_mels = 26
                    n_mfcc = 13
                    
                    mfccs = []
                    for i in range(n_frames):
                        start = i * hop_size
                        end = min(start + frame_size, len(signal))
                        frame = signal[start:end]
                        
                        # Apply window
                        if len(frame) < frame_size:
                            frame = np.pad(frame, (0, frame_size - len(frame)))
                        frame = frame * np.hanning(len(frame))
                        
                        # FFT
                        spectrum = np.abs(np.fft.rfft(frame, n=frame_size))
                        
                        # Simple log compression
                        log_spectrum = np.log(spectrum + 1e-10)
                        
                        # DCT to get cepstral coefficients
                        cepstral = dct(log_spectrum, type=2, norm='ortho')[:n_mfcc]
                        mfccs.append(cepstral)
                    
                    computed_mfcc = np.array(mfccs).T
                
                # Use provided MFCC data if available, otherwise use computed
                if mfcc_data and 'mfcc' in mfcc_data:
                    computed_mfcc = np.array(mfcc_data['mfcc'])
                
                if computed_mfcc is not None and computed_mfcc.size > 0:
                    # Save MFCC CSV
                    import pandas as pd
                    mfcc_df = pd.DataFrame(computed_mfcc, 
                                          index=[f"MFCC_{i}" for i in range(computed_mfcc.shape[0])])
                    mfcc_csv_path = os.path.join(analysis_dir, f"mfcc_{timestamp_safe}.csv")
                    mfcc_df.to_csv(mfcc_csv_path)
                    
                    # Save MFCC Plot (heatmap)
                    plt.figure(figsize=(10, 4))
                    plt.imshow(computed_mfcc, aspect='auto', origin='lower', cmap='viridis')
                    plt.colorbar(label='Coefficient Value')
                    plt.title(f"MFCC - {person}", fontsize=12, fontweight='bold')
                    plt.xlabel("Time Frame")
                    plt.ylabel("MFCC Coefficient")
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f"mfcc_{timestamp_safe}.png"), dpi=100)
                    plt.close()
                    print(f"[STORAGE] ✓ Saved MFCC: {mfcc_csv_path}")
            
            # ===== 3. Save LIF Data & Plot =====
            if lif_data and 'membrane' in lif_data:
                # Save LIF CSV
                import pandas as pd
                lif_df = pd.DataFrame({
                    'time': lif_data.get('time', list(range(len(lif_data['membrane'])))),
                    'membrane': lif_data['membrane'],
                    'spikes': lif_data.get('spikes', [0] * len(lif_data['membrane']))
                })
                lif_csv_path = os.path.join(analysis_dir, f"lif_{timestamp_safe}.csv")
                lif_df.to_csv(lif_csv_path, index=False)
                
                # Save LIF Plot
                plt.figure(figsize=(10, 4))
                membrane = np.array(lif_data['membrane'])
                time_axis = np.array(lif_data.get('time', range(len(membrane))))
                plt.plot(time_axis, membrane, color='#10b981', linewidth=1.5, label='Membrane Potential')
                plt.axhline(y=0.025, color='#ef4444', linestyle='--', label='Threshold')
                
                # Mark spikes
                if 'spikes' in lif_data:
                    spike_indices = np.where(np.array(lif_data['spikes']) > 0)[0]
                    if len(spike_indices) > 0 and len(spike_indices) < len(time_axis):
                        plt.scatter(time_axis[spike_indices], membrane[spike_indices], 
                                   color='#ef4444', s=50, zorder=5, label='Spikes')
                
                plt.title(f"LIF Neuron Response - {person}", fontsize=12, fontweight='bold')
                plt.xlabel("Time")
                plt.ylabel("Membrane Potential")
                plt.legend(loc='upper right')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"lif_{timestamp_safe}.png"), dpi=100)
                plt.close()
                print(f"[STORAGE] ✓ Saved LIF: {lif_csv_path}")
            
            # ===== 4. Create Combined Analysis Plot (3x2 grid) =====
            fig, axes = plt.subplots(3, 2, figsize=(14, 12))
            fig.suptitle(f"Footstep Analysis - {person} ({timestamp_safe})", fontsize=14, fontweight='bold')
            
            # Row 1, Col 1: Raw Waveform
            ax1 = axes[0, 0]
            if raw_waveform:
                ax1.plot(raw_waveform, color='#00eaff', linewidth=1)
                ax1.set_title("Raw ADC Signal", fontsize=11)
                ax1.set_xlabel("Sample")
                ax1.set_ylabel("ADC Value")
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, "No raw data", ha='center', va='center', transform=ax1.transAxes)
            
            # Row 1, Col 2: Filtered Waveform
            ax2 = axes[0, 1]
            if filtered_waveform:
                ax2.plot(filtered_waveform, color='#22c55e', linewidth=1.5)
                ax2.set_title("Filtered Signal", fontsize=11)
                ax2.set_xlabel("Sample")
                ax2.set_ylabel("Amplitude")
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, "No filtered data", ha='center', va='center', transform=ax2.transAxes)
            
            # Row 2, Col 1: FFT Spectrum
            ax3 = axes[1, 0]
            if fft_data and 'frequencies' in fft_data:
                ax3.bar(fft_data['frequencies'][:50], fft_data['magnitudes'][:50], 
                       color='#8b5cf6', alpha=0.8, width=0.8)
                ax3.set_title("FFT Spectrum", fontsize=11)
                ax3.set_xlabel("Frequency (Hz)")
                ax3.set_ylabel("Magnitude")
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, "No FFT data", ha='center', va='center', transform=ax3.transAxes)
            
            # Row 2, Col 2: MFCC
            ax4 = axes[1, 1]
            if computed_mfcc is not None and computed_mfcc.size > 0:
                im = ax4.imshow(computed_mfcc, aspect='auto', origin='lower', cmap='viridis')
                ax4.set_title("MFCC Coefficients", fontsize=11)
                ax4.set_xlabel("Time Frame")
                ax4.set_ylabel("MFCC Index")
                plt.colorbar(im, ax=ax4, label='Value')
            else:
                ax4.text(0.5, 0.5, "No MFCC data", ha='center', va='center', transform=ax4.transAxes)
            
            # Row 3, Col 1: LIF Response
            ax5 = axes[2, 0]
            if lif_data and 'membrane' in lif_data:
                membrane = np.array(lif_data['membrane'])
                ax5.plot(membrane, color='#10b981', linewidth=1.5)
                ax5.axhline(y=0.025, color='#ef4444', linestyle='--', alpha=0.7)
                ax5.set_title("LIF Neuron Response", fontsize=11)
                ax5.set_xlabel("Time")
                ax5.set_ylabel("Membrane Potential")
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, "No LIF data", ha='center', va='center', transform=ax5.transAxes)
            
            # Row 3, Col 2: Spike Train or additional info
            ax6 = axes[2, 1]
            if lif_data and 'spikes' in lif_data:
                spikes = np.array(lif_data['spikes'])
                spike_times = np.where(spikes > 0)[0]
                ax6.eventplot([spike_times], colors=['#ef4444'], lineoffsets=0.5, linelengths=0.8)
                ax6.set_title(f"Spike Train ({len(spike_times)} spikes)", fontsize=11)
                ax6.set_xlabel("Sample")
                ax6.set_ylabel("Neuron")
                ax6.set_ylim(0, 1)
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, "No spike data", ha='center', va='center', transform=ax6.transAxes)
            
            plt.tight_layout()
            combined_path = os.path.join(plots_dir, f"combined_{timestamp_safe}.png")
            plt.savefig(combined_path, dpi=120)
            plt.close()
            print(f"[STORAGE] ✓ Saved combined analysis: {combined_path}")
            
        except Exception as e:
            print(f"[STORAGE] Error saving analysis plots: {e}")
            import traceback
            traceback.print_exc()

    def get_dual_dataset_status(self) -> Dict[str, Any]:
        """Get status of dual dataset (HOME.csv and individual files)"""
        status = {
            "home_csv": {"exists": False, "samples": 0, "persons": []},
            "intruder_csv": {"exists": False, "samples": 0},
            "individual_files": [],
            "total_samples": 0,
            "target_samples": 150,
            "progress_percent": 0
        }
        
        # Check HOME.csv
        if os.path.exists(HOME_CSV_PATH):
            try:
                import pandas as pd
                df = pd.read_csv(HOME_CSV_PATH)
                status["home_csv"]["exists"] = True
                status["home_csv"]["samples"] = len(df)
                if '_person' in df.columns:
                    status["home_csv"]["persons"] = df['_person'].unique().tolist()
            except Exception as e:
                print(f"[STORAGE] Error reading HOME.csv: {e}")
        
        # Check INTRUDER.csv
        if os.path.exists(INTRUDER_CSV_PATH):
            try:
                import pandas as pd
                df = pd.read_csv(INTRUDER_CSV_PATH)
                status["intruder_csv"]["exists"] = True
                status["intruder_csv"]["samples"] = len(df)
            except Exception as e:
                print(f"[STORAGE] Error reading INTRUDER.csv: {e}")
        
        # Check individual files
        if os.path.exists(DATASET_DIR):
            for person_dir in os.listdir(DATASET_DIR):
                person_path = os.path.join(DATASET_DIR, person_dir)
                if os.path.isdir(person_path):
                    csv_path = os.path.join(person_path, f"features_{person_dir}.csv")
                    if os.path.exists(csv_path):
                        try:
                            import pandas as pd
                            df = pd.read_csv(csv_path)
                            status["individual_files"].append({
                                "name": person_dir,
                                "samples": len(df)
                            })
                        except:
                            pass
        
        # ALWAYS use individual file counts - they are the actual training data
        # HOME.csv is just an aggregate that may be out of sync
        individual_total = sum(f["samples"] for f in status["individual_files"])
        status["total_samples"] = individual_total + status["intruder_csv"]["samples"]
        
        if status["target_samples"] > 0:
            status["progress_percent"] = min(100, int(status["total_samples"] / status["target_samples"] * 100))
        
        return status

    def get_sample_counts(self) -> Dict[str, int]:
        """Get sample counts from individual CSV files (not SQLite)"""
        counts = {}
        if os.path.exists(DATASET_DIR):
            for person_dir in os.listdir(DATASET_DIR):
                person_path = os.path.join(DATASET_DIR, person_dir)
                if os.path.isdir(person_path):
                    csv_path = os.path.join(person_path, f"features_{person_dir}.csv")
                    if os.path.exists(csv_path):
                        try:
                            import pandas as pd
                            df = pd.read_csv(csv_path)
                            counts[person_dir] = len(df)
                        except Exception as e:
                            print(f"[STORAGE] Error reading {csv_path}: {e}")
        return counts

    def get_all_samples(self):
        """Get all samples with data and labels"""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT person, features_json FROM samples")
        rows = c.fetchall()
        conn.close()
        
        data = []
        labels = []
        for person, features_json in rows:
            features = json.loads(features_json)
            data.append(features)
            labels.append(person)
            
        return data, labels

    def get_all_samples_with_details(self):
        """
        Get all samples with detailed dataset information.
        Returns: (data, labels, dataset_details)
        """
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get samples
        c.execute("SELECT person, features_json FROM samples")
        rows = c.fetchall()
        
        # Get per-person counts
        c.execute("SELECT person, COUNT(*) FROM samples GROUP BY person ORDER BY person")
        person_counts = c.fetchall()
        
        conn.close()
        
        data = []
        labels = []
        for person, features_json in rows:
            features = json.loads(features_json)
            data.append(features)
            labels.append(person)
        
        # Build detailed dataset info
        dataset_details = {
            "total_samples": len(data),
            "datasets": [
                {"name": person, "samples": count} 
                for person, count in person_counts
            ],
            "dataset_names": [person for person, _ in person_counts]
        }
            
        return data, labels, dataset_details

    def get_samples_by_datasets(self, selected_datasets):
        """
        Get samples filtered by selected dataset names.
        
        Args:
            selected_datasets: List of dataset names to include (e.g., ['HOME_Apurv', 'HOME_Samir'])
        
        Returns: (data, labels, dataset_details)
        """
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Build SQL query with placeholders
        placeholders = ','.join('?' * len(selected_datasets))
        
        # Get samples for selected datasets only
        c.execute(f"SELECT person, features_json FROM samples WHERE person IN ({placeholders})", selected_datasets)
        rows = c.fetchall()
        
        # Get per-person counts for selected datasets
        c.execute(f"SELECT person, COUNT(*) FROM samples WHERE person IN ({placeholders}) GROUP BY person ORDER BY person", selected_datasets)
        person_counts = c.fetchall()
        
        conn.close()
        
        data = []
        labels = []
        for person, features_json in rows:
            features = json.loads(features_json)
            data.append(features)
            labels.append(person)
        
        # Build detailed dataset info
        dataset_details = {
            "total_samples": len(data),
            "datasets": [
                {"name": person, "samples": count} 
                for person, count in person_counts
            ],
            "dataset_names": [person for person, _ in person_counts],
            "selected_datasets": selected_datasets
        }
            
        return data, labels, dataset_details
