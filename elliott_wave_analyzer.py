
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from ai_models import AlphaGoModel, SelfPlayModel
from deep_learning_elliott_wave import DeepLearningElliottWave

class ElliottWaveAnalyzer:
    def __init__(self, ohlc_data):
        self.data = ohlc_data
        self.waves = []
        self.alphago_model = AlphaGoModel()
        self.self_play_model = SelfPlayModel()
        self.dl_ew_model = DeepLearningElliottWave(sequence_length=60, n_features=5) # Initialize DL model

    def _find_swing_points(self, data_series, order=5):
        """Finds swing highs and swing lows using peak detection."""
        # Find peaks (swing highs)
        peaks, _ = find_peaks(data_series["high"], distance=order)
        # Find troughs (swing lows) by inverting the series
        troughs, _ = find_peaks(-data_series["low"], distance=order)
        
        swing_points = []
        for p in peaks:
            swing_points.append({"index": p, "price": data_series["high"].iloc[p], "type": "high"})
        for t in troughs:
            swing_points.append({"index": t, "price": data_series["low"].iloc[t], "type": "low"})
            
        # Sort swing points by index (time)
        swing_points.sort(key=lambda x: x["index"])
        
        return swing_points

    def _get_wave_points(self, start_idx, end_idx):
        segment = self.data.iloc[start_idx:end_idx+1]
        if segment.empty:
            return None

        # Use the actual open and close prices for start and end of the wave
        start_price = segment["open"].iloc[0]
        end_price = segment["close"].iloc[-1]
        
        high = segment["high"].max()
        low = segment["low"].min()
        
        high_idx_global = segment["high"].idxmax()
        low_idx_global = segment["low"].idxmin()

        return {"start_idx": start_idx, "end_idx": end_idx, 
                "start_price": start_price, 
                "end_price": end_price,
                "high": high, "low": low, 
                "high_idx": high_idx_global, "low_idx": low_idx_global}

    def _check_impulsive_rules(self, wave1, wave2, wave3, wave4, wave5):
        if not all([wave1, wave2, wave3, wave4, wave5]):
            return False

        # Rule #1: Wave 2 never falls below the starting point of Wave 1.
        if wave2["low"] < wave1["start_price"]:
            return False

        # Rule #2: Wave 3 is often the longest wave, but never the shortest of the waves 1-3-5.
        len1 = abs(wave1["end_price"] - wave1["start_price"])
        len3 = abs(wave3["end_price"] - wave3["start_price"])
        len5 = abs(wave5["end_price"] - wave5["start_price"])
        
        if len3 < len1 or len3 < len5:
            return False

        # Rule #3: Wave 4 can’t enter Wave 2 territory.
        # Assuming Wave 2 territory is between its high and low
        if (wave4["low"] < wave2["high"] and wave4["high"] > wave2["low"]):
            return False

        # Fibonacci relationships (guidelines, not strict rules for validity)
        # Wave 2 often retraces 50%, 61.8%, or 78.6% of Wave 1
        # Wave 3 is often 1.618 or 2.618 times Wave 1
        # Wave 4 often retraces 38.2% or 50% of Wave 3
        # Wave 5 is often equal to Wave 1, or 0.618 of Wave 1, or 0.618 of Wave 1-3 combined

        return True

    def _check_extended_rules(self, wave1, wave2, wave3, wave4, wave5, extended_wave_num):
        # Extended waves follow impulsive rules, but one wave extends into 5 sub-waves.
        # This function will check the overall 9-wave structure and the rules for the extended wave.
        # For simplicity, we\"ll assume the extended wave is already identified and passed in as sub-waves.

        # Check overall impulsive rules for the main 5 waves
        if not self._check_impulsive_rules(wave1, wave2, wave3, wave4, wave5):
            return False

        # Additional check for extended wave (assuming it\"s a 5-wave structure within one of the main waves)
        # This part would require more complex logic to verify the sub-waves of the extended wave.
        # For now, we\"ll just return True if the main impulsive rules are met.
        return True

    def _check_leading_diagonal_rules(self, wave1, wave2, wave3, wave4, wave5):
        if not all([wave1, wave2, wave3, wave4, wave5]):
            return False

        # Rule #1: Wave 2 never falls below the starting point of Wave 1.
        if wave2["low"] < wave1["start_price"]:
            return False

        # Rule #2: Wave 3 is often the longest wave, but never the shortest of the waves 1-3-5.
        len1 = abs(wave1["end_price"] - wave1["start_price"])
        len3 = abs(wave3["end_price"] - wave3["start_price"])
        len5 = abs(wave5["end_price"] - wave5["start_price"])
        
        if len3 < len1 or len3 < len5:
            return False

        # Rule #3: Wave 4 must hold above the start of Wave 2 (i.e., it overlaps with Wave 1).
        # This is the key difference from impulsive waves.
        # For a bullish LD, wave 4 low should be below wave 1 high, and wave 4 high should be above wave 2 low.
        # For a bearish LD, wave 4 high should be above wave 1 low, and wave 4 low should be below wave 2 high.
        # Simplified check for overlap: if wave4 low is within wave1 range, or wave4 high is within wave1 range
        # More precisely, wave 4 enters the territory of wave 1.
        # This implies that wave 4 low is less than wave 1 high (for bullish) or wave 4 high is greater than wave 1 low (for bearish)
        # And wave 4 must hold above the start of wave 2 (i.e., wave 4 low > wave 2 start_price for bullish)

        # Assuming bullish trend for now, as per example in PDF
        if wave4["low"] > wave2["start_price"]:
            return False # This is the opposite of overlap, so it should be False if it doesn\"t overlap

        # Check for overlap with Wave 1 (simplified)
        if not (wave4["low"] < wave1["high"] and wave4["high"] > wave1["low"]):
            return False

        return True

    def _check_ending_diagonal_rules(self, wave1, wave2, wave3, wave4, wave5):
        if not all([wave1, wave2, wave3, wave4, wave5]):
            return False

        # Rule #1: Wave 2 never falls below the starting point of Wave 1.
        if wave2["low"] < wave1["start_price"]:
            return False

        # Rule #2: Wave 3 is often the longest wave, but never the shortest of the waves 1-3-5.
        len1 = abs(wave1["end_price"] - wave1["start_price"])
        len3 = abs(wave3["end_price"] - wave3["start_price"])
        len5 = abs(wave5["end_price"] - wave5["start_price"])
        
        if len3 < len1 or len3 < len5:
            return False

        # Rule #3: Wave 4 must end above the starting point of Wave 2.
        # This implies overlap between Wave 4 and Wave 2.
        if not (wave4["low"] < wave2["high"] and wave4["high"] > wave2["low"]):
            return False

        # Additionally, all sub-waves in an Ending Diagonal are 3-wave structures (3-3-3-3-3).
        # This would require a more detailed sub-wave analysis, which is not implemented yet.

        return True

    def _check_zig_zag_rules(self, waveA, waveB, waveC):
        if not all([waveA, waveB, waveC]):
            return False

        # Rule #1: Wave B must end below the start of wave A (for a downward Zig-Zag) or above the start of wave A (for an upward Zig-Zag).
        # Assuming bullish Zig-Zag (A-B-C moves higher) as per Image 17 example
        if waveA["start_price"] < waveA["end_price"]: # Bullish A wave
            if waveB["end_price"] < waveA["start_price"]:
                return False
        else: # Bearish A wave
            if waveB["end_price"] > waveA["start_price"]:
                return False

        # Rule #2: Wave C must break below the end of wave A (for a downward Zig-Zag) or above the end of wave A (for an upward Zig-Zag).
        if waveA["start_price"] < waveA["end_price"]: # Bullish A wave
            if waveC["end_price"] < waveA["end_price"]:
                return False
        else: # Bearish A wave
            if waveC["end_price"] > waveA["end_price"]:
                return False

        # Additional check: Sub-wave structure 5-3-5 (A-B-C)
        # This would require recursive wave identification, which is not implemented yet.

        return True

    def _check_flat_rules(self, waveA, waveB, waveC, flat_type):
        if not all([waveA, waveB, waveC]):
            return False

        # Rule #1: All waves in a flat correction are 3-wave structures (3-3-5).
        # This would require recursive wave identification, not implemented yet.

        if flat_type == "regular":
            # Rule #1 - Wave B must retrace to 90% of wave A (78.6% allowed in Forex).
            retrace_ratio = abs(waveB["end_price"] - waveA["start_price"]) / abs(waveA["end_price"] - waveA["start_price"])
            if not (0.786 <= retrace_ratio <= 0.9):
                return False
            # Rule #2 - Wave C must end below the ending of wave A.
            if waveC["end_price"] > waveA["end_price"]:
                return False
        elif flat_type == "expanding":
            # Rule #1 - Wave B needs to break through the start level of wave A, but can’t end above an inverse of 161.8% of wave A.
            if not (abs(waveB["end_price"] - waveA["start_price"]) > abs(waveA["end_price"] - waveA["start_price"])):
                return False
            # Rule #2 - Wave C needs to end below the end level of wave A.
            if waveC["end_price"] > waveA["end_price"]:
                return False
        elif flat_type == "running":
            # Rule #1 - Wave B needs to break the start level of wave A, but can’t end above an inverse of 161.8% of wave A.
            if not (abs(waveB["end_price"] - waveA["start_price"]) > abs(waveA["end_price"] - waveA["start_price"])):
                return False
            # Rule #2 - Wave C needs to end below the end level of wave A.
            if waveC["end_price"] > waveA["end_price"]:
                return False
        else:
            return False # Invalid flat type

        return True

    def _check_triangle_rules(self, waveA, waveB, waveC, waveD, waveE, triangle_type):
        if not all([waveA, waveB, waveC, waveD, waveE]):
            return False

        # All waves in a triangle are 3-wave structures (3-3-3-3-3).
        # This would require recursive wave identification, not implemented yet.

        if triangle_type == "contracting":
            # Rule #1 - Wave B must be smaller than wave A
            if abs(waveB["end_price"] - waveB["start_price"]) >= abs(waveA["end_price"] - waveA["start_price"]):
                return False
            # Rule #2 - Wave C must be smaller than wave B
            if abs(waveC["end_price"] - waveC["start_price"]) >= abs(waveB["end_price"] - waveB["start_price"]):
                return False
            # Rule #3 - Wave D must be smaller than wave C
            if abs(waveD["end_price"] - waveD["start_price"]) >= abs(waveC["end_price"] - waveC["start_price"]):
                return False
            # Rule #4 - Wave E must be smaller than wave D
            if abs(waveE["end_price"] - waveE["start_price"]) >= abs(waveD["end_price"] - waveD["start_price"]):
                return False
            # Converging trendlines
            # This would require checking the slopes of trendlines connecting wave ends.
        elif triangle_type == "barrier":
            # One side is flat, other is converging.
            pass
        elif triangle_type == "running":
            # Wave B is longer than A, and C, D, E are smaller.
            pass
        elif triangle_type == "expanding":
            # Diverging trendlines.
            pass
        else:
            return False # Invalid triangle type

        return True

    def _check_double_three_rules(self, waveW, waveX, waveY):
        if not all([waveW, waveX, waveY]):
            return False

        # Rule #1 - Wave X must end above the starting level of wave W
        if waveX["end_price"] < waveW["start_price"]:
            return False
        # Rule #2 - Wave Y must end above the ending level of wave W
        if waveY["end_price"] < waveW["end_price"]:
            return False
        
        # Sub-waves of W, X, Y are typically corrective patterns (e.g., Zig-Zag, Flat, Triangle).
        # This would require recursive wave identification.

        return True

    def _check_triple_three_rules(self, waveW, waveX, waveY, waveX2, waveZ):
        if not all([waveW, waveX, waveY, waveX2, waveZ]):
            return False

        # Rule #1 - Wave X must end above the starting level of wave W
        if waveX["end_price"] < waveW["start_price"]:
            return False
        # Rule #2 - Wave Y must ends above the ending level of wave W
        if waveY["end_price"] < waveW["end_price"]:
            return False
        # Rule #3 - Wave X2 must ends above the starting level of wave Y
        if waveX2["end_price"] < waveY["start_price"]:
            return False
        # Rule #4 - Wave Z must end above the starting level of wave Y
        if waveZ["end_price"] < waveY["end_price"]:
            return False

        # Sub-waves of W, X, Y, X2, Z are typically corrective patterns.
        # This would require recursive wave identification.

        return True

    def _segment_waves(self):
        """Segments the OHLC data into potential waves using swing points."""
        swing_points = self._find_swing_points(self.data)
        
        segments = []
        if len(swing_points) < 2:
            return segments

        # Iterate through swing points to identify potential wave segments
        # This is a basic segmentation. More advanced methods would consider
        # different wave degrees and fractal nature.
        for i in range(len(swing_points) - 1):
            start_idx = swing_points[i]["index"]
            end_idx = swing_points[i+1]["index"]
            segment = self._get_wave_points(start_idx, end_idx)
            if segment:
                segments.append(segment)
        return segments

    def run_analysis(self):
        print("Starting Elliott Wave analysis...")
        
        # Step 1: Segment the data into potential waves
        potential_waves = self._segment_waves()
        print(f"Found {len(potential_waves)} potential wave segments.")

        # Step 2: Use AI to identify and validate patterns
        # Prepare data for deep learning model
        ohlc_for_dl = self.data[["open", "high", "low", "close", "volume"]].copy()
        X_dl, y_dl_dummy = self.dl_ew_model.prepare_data(ohlc_for_dl)

        if X_dl.shape[0] > 0:
            # Train the deep learning model (can be pre-trained and loaded)
            # For demonstration, we'll train a small model here
            self.dl_ew_model.train_model(X_dl, y_dl_dummy, model_type='cnn_lstm', epochs=5, batch_size=32)
            print("Deep Learning model trained for pattern recognition.")

            # Predict patterns using the trained DL model
            dl_predictions = self.dl_ew_model.predict_patterns(X_dl)
            # You would then use these predictions to refine or validate patterns
            # For example, if dl_predictions[i] indicates a strong bullish impulsive wave,
            # you can prioritize or confirm the rule-based identification.
            print("Deep Learning model made predictions.")

        identified_patterns = []
        for i in range(len(potential_waves) - 4): # Need at least 5 waves for impulsive/diagonal
            wave1 = potential_waves[i]
            wave2 = potential_waves[i+1]
            wave3 = potential_waves[i+2]
            wave4 = potential_waves[i+3]
            wave5 = potential_waves[i+4]

            if self._check_impulsive_rules(wave1, wave2, wave3, wave4, wave5):
                identified_patterns.append({"type": "impulsive", "waves": [wave1, wave2, wave3, wave4, wave5]})
            # Add checks for other patterns (extended, diagonal, corrective, complex)
            # This would involve more complex logic to combine waves into patterns

        # Store the identified patterns for further use by the trading strategy
        self.waves = identified_patterns

        print("Elliott Wave analysis complete.")

if __name__ == "__main__":
    # Load preprocessed data (assuming it\\\\\"s available from data_preprocessing.py)
    # Note: In a real scenario, you would load data from the database here.
    # For standalone testing, we use dummy data or load from CSV if available.
    try:
        from database import OHLCDatabase
        db = OHLCDatabase()
        data = db.get_ohlc_data("XAUUSD")
    except Exception as e:
        print(f"Could not load data from database: {e}. Loading from CSV for testing.")
        # Fallback for testing if database is not populated or accessible
        data = pd.read_csv("../upload/XAUUSD_M1_Backtest.csv", sep=\t, header=0)
        data.columns = [col.replace("<", "").replace(">", "") for col in data.columns]
        data["timestamp"] = pd.to_datetime(data["DATE"] + " " + data["TIME"])
        data = data.set_index("timestamp")
        data = data[["OPEN", "HIGH", "LOW", "CLOSE", "TICKVOL", "VOL", "SPREAD"]]
        data.columns = data.columns.str.lower() # Convert column names to lowercase

    analyzer = ElliottWaveAnalyzer(data)
    analyzer.run_analysis()

    print("Elliott Wave analysis complete.")


