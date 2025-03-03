import sys
sys.path.append("../")
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Tuple, Dict

from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter

def logit(x, lower=0.0, upper=10.0, eps=1e-8):
    """
    Convert x in [lower, upper] to an unbounded real z in (-inf, +inf).
    We clamp x slightly to avoid taking log(0).
    """
    x_clamped = np.clip(x, lower + eps, upper - eps)
    scale = upper - lower
    ratio = (x_clamped - lower) / scale
    return np.log(ratio / (1 - ratio))

def inv_logit(z, lower=0.0, upper=10.0):
    """
    Convert an unbounded real z in (-inf, +inf) back to [lower, upper].
    """
    scale = upper - lower
    return lower + scale / (1.0 + np.exp(-z))

class OdorStateManager:
    def __init__(self, config, whiff_intermittency):
        # Start odor at base level, but store internally in z-space
        z_init = logit(config.base_odor_level, 0, 10)
        self.z_current = z_init
        self.z_prev = z_init

        # For reference/tracking we also keep actual odor in [0..10]
        self.current_concentration = config.base_odor_level
        self.prev_concentration = config.base_odor_level

        # For other state logic
        self.recent_history = [0] * 1000
        self.recent_concentrations = [config.base_odor_level] * 10
        self.recent_intermittencies = list(np.random.choice(whiff_intermittency, 5))
        self.in_whiff_state = False
        self.state_duration = 0

@dataclass
class OdorConfig:
    rows_per_second: int = 200
    base_odor_level: float = 0.6
    distance_threshold: float = 3
    # AR(2) base coefficients
    ar1: float = 0.98
    ar2: float = -0.02
    warmup_steps: int = 200
    low_threshold: float = 0.05
    history_length: int = 7
    # Transition matrix for whiff states
    transition_matrix: np.ndarray = np.array([[0.15, 0.85],
                                            [0.15, 0.85]])

class ParallelOdorPredictor:
    def __init__(self, fitted_p_heatmap, xedges, yedges, fdf, fdf_nowhiff, test_locations):
        self.config = OdorConfig()
        self.fdf = fdf
        self.fdf_nowhiff = fdf_nowhiff
        self.test_locations = test_locations
        self.fitted_p_heatmap = fitted_p_heatmap
        self.xedges = xedges
        self.yedges = yedges
        self.setup_data()

    def setup_data(self):
        distance_bins = np.arange(0, 41, 1)
        nearest_bins = np.arange(0, 9, 1)
        
        self.bin_data_dict = {}
        
        for i in range(len(distance_bins)-1):
            for j in range(len(nearest_bins)-1):
                start_dist, end_dist = distance_bins[i], distance_bins[i+1]
                start_near, end_near = nearest_bins[j], nearest_bins[j+1]
                
                bin_data = self.fdf[
                    (self.fdf['avg_distance_along_streakline'] >= start_dist) & 
                    (self.fdf['avg_distance_along_streakline'] < end_dist) &
                    (self.fdf['avg_nearest_from_streakline'] >= start_near) &
                    (self.fdf['avg_nearest_from_streakline'] < end_near)
                ]['odor_intermittency'].dropna().values
                
                self.bin_data_dict[(start_dist, end_dist, start_near, end_near)] = bin_data

    def get_spatial_prob(self, x, y):
        """Get probability from heatmap for given coordinates."""
        x_idx = np.digitize(x, self.xedges) - 1
        y_idx = np.digitize(y, self.yedges) - 1
        
        # Boundary checking
        x_idx = np.clip(x_idx, 0, len(self.xedges)-2)
        y_idx = np.clip(y_idx, 0, len(self.yedges)-2)
        
        return self.fitted_p_heatmap[x_idx, y_idx]

    @staticmethod
    def calculate_distance_from_source(x: float, y: float) -> float:
        """Euclidean distance from source (assume (0,0)) to point (x,y)."""
        return np.sqrt(x**2 + y**2)

    def generate_intermittency(self, distance_along: float, distance_from: float,
                             state: OdorStateManager, default: float = 0.05) -> float:
        last_values = np.array(state.recent_intermittencies[-self.config.history_length:])
        low_frequency = np.mean(last_values < self.config.low_threshold)
        
        for (sd, ed, sn, en), values in self.bin_data_dict.items():
            if (sd <= distance_along < ed) and (sn <= distance_from < en):
                if len(values) > 0:
                    if low_frequency > 0.5:
                        median_val = np.median(values)
                        subset = values[values < median_val]
                        if len(subset) > 0:
                            intermittency = np.random.choice(subset)
                        else:
                            intermittency = np.random.choice(values)
                    else:
                        intermittency = np.random.choice(values)
                    return np.clip(intermittency, np.min(values), np.max(values))
        return default

    def update_whiff_posterior(self, prior_prob: float, state: OdorStateManager) -> float:
        """
        Example function to incorporate a Markov chain / state transition
        plus some heuristics about recent whiffs, etc.
        """
        whiff_state = 1 if state.in_whiff_state else 0
        num_recent_whiffs = sum(state.recent_history[-20:])
        
        # Time since last whiff
        time_since_whiff = 0
        for i in range(len(state.recent_history)-1, -1, -1):
            if state.recent_history[i]:
                break
            time_since_whiff += 1
        scaler=0.25
        time_since_last_whiff = min(1.5, time_since_whiff) if time_since_whiff > 50 else 1.0
        recent_whiff_memory = (1 + (num_recent_whiffs) * scaler) * time_since_last_whiff
                
        posterior = ((prior_prob * scaler)
                    * self.config.transition_matrix[whiff_state][1]
                    * recent_whiff_memory)
        
        return posterior

    def update_ar2_concentration(self, current: float, prev: float, target: float, 
                               noise_scale: float) -> float:
        noise = noise_scale * (np.random.randn() - 0.5) * 0.5
        x_next = (0.85 * (self.config.ar1 * (current - target) + 
                 self.config.ar2 * (prev - target)) + target + noise)
        return x_next

    def update_ar2_in_zspace(self, z_current: float, z_prev: float,
                            z_target: float, distance: float,
                            base_noise_scale: float = 0.1,
                            jump_prob: float = 0.05) -> float:
        distance_factor = np.exp(-distance / 50.0)  # decays with distance

        # Possibly adjust AR(1), AR(2), etc. coefficients
        ar1_local = self.config.ar1 * (1 + 0.1 * distance_factor)
        ar2_local = self.config.ar2 * (1 - 0.1 * distance_factor)
        
        # Base random noise
        noise = base_noise_scale * (1 + 2 * distance_factor) * np.random.randn()
        
        # Optional "jumps"
        if np.random.rand() < jump_prob:
            jump_size = np.random.uniform(-1, 1) * base_noise_scale * 3
            noise += jump_size

        # AR(2) update in unbounded space
        z_next = 0.85 * (ar1_local * (z_current - z_target)
                         + ar2_local * (z_prev - z_target)) \
                 + z_target + noise

        return z_next

    def process_segment(self, start_idx: int, end_idx: int,
                       state: OdorStateManager) -> Tuple[np.ndarray, np.ndarray]:
        locations_segment = self.test_locations[start_idx:end_idx]
        concentrations = np.full(len(locations_segment), self.config.base_odor_level)
        predictions = np.zeros(len(locations_segment), dtype=int)

        test_locations = locations_segment
        whiff_locations = self.fdf[['avg_distance_along_streakline','avg_nearest_from_streakline']].values
        nowhiff_locations = self.fdf_nowhiff[['avg_distance_along_streakline','avg_nearest_from_streakline']].values

        distances_from_source = np.array([
            self.calculate_distance_from_source(x, y) 
            for x, y in test_locations
        ])
        distances = cdist(test_locations, whiff_locations)
        distances_nowhiff = cdist(test_locations, nowhiff_locations)

        i = 0
        while i < len(locations_segment):
            if start_idx + i < self.config.warmup_steps:
                i += 1
                continue

            x, y = locations_segment[i]
            prior_prob = self.get_spatial_prob(x, y)
            posterior = self.update_whiff_posterior(prior_prob, state)

            if state.in_whiff_state:
                state.state_duration += 1
                min_duration = 0.1 * self.config.rows_per_second
                if state.state_duration > min_duration:
                    continue_prob = 0.5 * prior_prob
                    state.in_whiff_state = (np.random.rand() < continue_prob)
            else:
                transition_prob = posterior
                state.in_whiff_state = (np.random.rand() < transition_prob * 0.5)
                if state.in_whiff_state:
                    state.state_duration = 0

            if state.in_whiff_state and (np.min(distances[i]) <= self.config.distance_threshold):
                nearest_idx = np.argmin(distances[i])
                mean_concentration = self.fdf.mean_concentration.values[nearest_idx]
                std_dev_whiff = self.fdf.std_whiff.values[nearest_idx]
                duration = int(self.fdf.length_of_encounter.values[nearest_idx] *
                             self.config.rows_per_second)
                rows_to_fill = min(duration, len(locations_segment) - i)

                z_target = logit(mean_concentration, 0, 10)

                for j in range(rows_to_fill):
                    dist_here = distances_from_source[i + j] if (i+j) < len(distances_from_source) else 0
                    z_next = self.update_ar2_in_zspace(
                        state.z_current, state.z_prev, z_target,
                        distance=dist_here,
                        base_noise_scale=0.15 * std_dev_whiff,
                        jump_prob=0.05
                    )
                    odor_next = inv_logit(z_next, 0, 10)

                    state.z_prev = state.z_current
                    state.z_current = z_next
                    state.prev_concentration = state.current_concentration
                    state.current_concentration = odor_next

                    concentrations[i+j] = odor_next
                    predictions[i+j] = 1

                    state.recent_concentrations.append(odor_next)
                    state.recent_concentrations.pop(0)
                    state.recent_history.append(1)
                    state.recent_history.pop(0)

                i += rows_to_fill

                dist_along = locations_segment[i-1][0]
                dist_from = locations_segment[i-1][1]
                intermittency = self.generate_intermittency(dist_along, dist_from, state)
                state.recent_intermittencies.append(intermittency)
                state.recent_intermittencies.pop(0)

                intermittency_duration = int(intermittency * self.config.rows_per_second * 0.9)
                i += intermittency_duration

            else:
                nearest_idx = np.argmin(distances_nowhiff[i])
                no_whiff_mean = self.fdf_nowhiff.wc_nowhiff.values[nearest_idx]
                no_whiff_std = self.fdf_nowhiff.wsd_nowhiff.values[nearest_idx]
                
                new_concentration = self.update_ar2_concentration(
                    state.current_concentration,
                    state.prev_concentration,
                    no_whiff_mean,
                    0.05 * no_whiff_std
                )
                new_concentration = np.clip(new_concentration, 0.6, 1.0)
                
                if i >= 10:
                    window_data = concentrations[i-10:i]
                    window_data = np.append(window_data, new_concentration)
                    window = np.ones(10)/10.0
                    new_concentration = np.convolve(window_data, window, mode='valid')[-1]
                
                state.prev_concentration = state.current_concentration
                state.current_concentration = new_concentration
                
                concentrations[i] = new_concentration
                state.recent_concentrations.append(new_concentration)
                state.recent_concentrations.pop(0)
                state.recent_history.append(0)
                state.recent_history.pop(0)
                
                i += 1

        return concentrations, predictions

    def predict(self) -> Dict[str, np.ndarray]:
            segment_size = 2000
            total_segments = (
                len(self.test_locations) // segment_size +
                (1 if len(self.test_locations) % segment_size else 0)
            )

            all_concentrations = []
            all_predictions = []

            # Create the state manager
            state = OdorStateManager(self.config, self.fdf.odor_intermittency.values)

            for seg_idx in range(total_segments):
                start_idx = seg_idx * segment_size
                end_idx = min((seg_idx + 1) * segment_size, len(self.test_locations))

                concentrations, predictions = self.process_segment(start_idx, end_idx, state)
                all_concentrations.append(concentrations)
                all_predictions.append(predictions)

            # Concatenate results
            final_concentrations = np.concatenate(all_concentrations)
            final_predictions = np.concatenate(all_predictions)
            final_concentrations = gaussian_filter(final_concentrations, sigma=0.8)

            return {
                'concentrations': final_concentrations,
                'predictions': final_predictions
            }

def main(fitted_p_heatmap,xedges,yedges,fdf,fdf_nowhiff, test_locations):
    predictor = ParallelOdorPredictor(
        fitted_p_heatmap=fitted_p_heatmap,
        xedges=xedges,
        yedges=yedges,
        fdf=fdf,
        fdf_nowhiff=fdf_nowhiff,
        test_locations=test_locations
    )
    return predictor.predict()

# if __name__ == "__main__":
#     dirname = '../data/simulator/rigolli/'
#     hmap_data = np.load(str(dirname) + "hmap.npz")
#     # fdf = pd.read_hdf(str(dirname) + 'whiff.h5')
#     # fdf_nowhiff = pd.read_hdf(str(dirname) + 'nowhiff.h5')
    
#     # Example test locations
#     test_locations = df_test[['distance_along_streakline', 'nearest_from_streakline']].values
#     fitted_p_heatmap = hmap_data['fitted_p_heatmap']
#     xedges = hmap_data['xedges']
#     yedges = hmap_data['yedges']
    
#     results = main(fitted_p_heatmap,xedges,yedges,fdf,fdf_nowhiff, test_locations)
#     df_test.loc[:, 'predicted_odor'] = results['concentrations']
#     df_test.loc[:, 'whiff_predicted'] = results['predictions']