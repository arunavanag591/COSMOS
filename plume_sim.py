import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional


class OdorPredictor:
    def __init__(self, fitted_p_heatmap, xedges, yedges, fdf, fdf_nowhiff, distance_threshold=4, rows_per_second=200, base_odor_level=0.58):
        self.fitted_p_heatmap = fitted_p_heatmap
        self.xedges = xedges
        self.yedges = yedges
        self.distance_threshold = distance_threshold
        self.rows_per_second = rows_per_second
        self.base_odor_level = base_odor_level
        self.initialize_data(fdf, fdf_nowhiff)

    def initialize_data(self, fdf, fdf_nowhiff):
        # Extract necessary data as arrays for faster access
        self.whiff_locations = fdf[['avg_distance_along_streakline', 'avg_nearest_from_streakline']].values
        self.nowhiff_locations = fdf_nowhiff[['avg_distance_along_streakline', 'avg_nearest_from_streakline']].values
        self.whiff_means = fdf.mean_concentration.values
        self.whiff_stds = fdf.std_whiff.values
        self.whiff_duration = fdf.length_of_encounter.values
        self.nowhiff_means = fdf_nowhiff.wc_nowhiff.values
        self.nowhiff_wsd = fdf_nowhiff.wsd_nowhiff.values

    def predict_whiff_from_probability(self, x, y):
        x_bin = np.digitize(x, self.xedges) - 1
        y_bin = np.digitize(y, self.yedges) - 1
        if x_bin < 0 or x_bin >= self.fitted_p_heatmap.shape[0] or y_bin < 0 or y_bin >= self.fitted_p_heatmap.shape[1]:
            return False, 0
        whiff_prob = self.fitted_p_heatmap[x_bin, y_bin]
        return np.random.rand() < whiff_prob, whiff_prob

    @staticmethod
    def moving_average(data, window_size=5):
        # Adjust moving average to keep the same length as input data
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')

    def predict_odor_concentration(self, x, y):
        # Convert DataFrame columns to NumPy arrays
        df_test = pd.DataFrame({'distance_along_streakline': x, 'nearest_from_streakline': y})
        odor_concentration_samples = np.full(len(df_test), self.base_odor_level)
        whiff_prediction_samples = np.zeros(len(df_test), dtype=int)

        test_locations = df_test[['distance_along_streakline', 'nearest_from_streakline']].values
        distances = cdist(test_locations, self.whiff_locations)
        distances_nowhiff = cdist(test_locations, self.nowhiff_locations)

        # Process whiff and no-whiff regions
        odor_concentration_samples, whiff_prediction_samples = self.process_whiff_regions(
            distances, odor_concentration_samples, whiff_prediction_samples, test_locations
        )
        odor_concentration_samples = self.process_no_whiff_regions(
            distances_nowhiff, odor_concentration_samples
        )

        # Return results as DataFrame
        df_test['predicted_odor'] = odor_concentration_samples
        df_test['whiff_predicted'] = whiff_prediction_samples
        return df_test

    def process_whiff_regions(self, distances, odor_concentration_samples, whiff_prediction_samples, test_locations):
        for i, (dist_along, nearest_from) in enumerate(test_locations):
            whiff_predicted, _ = self.predict_whiff_from_probability(dist_along, nearest_from)
            if whiff_predicted and np.min(distances[i]) <= self.distance_threshold:
                nearest_whiff_idx = np.argmin(distances[i])
                mean_concentration = self.whiff_means[nearest_whiff_idx]
                std_dev_whiff = self.whiff_stds[nearest_whiff_idx]
                duration = int(self.whiff_duration[nearest_whiff_idx] * self.rows_per_second)
                rows_to_fill = min(duration, len(odor_concentration_samples) - i)

                generated_concentrations = np.random.normal(mean_concentration, std_dev_whiff, rows_to_fill)
                generated_concentrations = np.clip(generated_concentrations, 4, 10.2)
                odor_concentration_samples[i:i + rows_to_fill] = generated_concentrations
                whiff_prediction_samples[i:i + rows_to_fill] = 1
        return odor_concentration_samples, whiff_prediction_samples

    def process_no_whiff_regions(self, distances_nowhiff, odor_concentration_samples):
        i = 0
        while i < len(odor_concentration_samples):
            if odor_concentration_samples[i] == self.base_odor_level:
                nearest_no_whiff_idx = np.argmin(distances_nowhiff[i])
                no_whiff_mean = self.nowhiff_means[nearest_no_whiff_idx]
                no_whiff_std = self.nowhiff_wsd[nearest_no_whiff_idx]

                start = i
                while i < len(odor_concentration_samples) and odor_concentration_samples[i] == self.base_odor_level:
                    i += 1
                end = i

                no_whiff_concentrations = np.random.normal(no_whiff_mean, no_whiff_std, end - start)
                no_whiff_concentrations = np.clip(no_whiff_concentrations, 0.58, 1)
                
                # Check if moving average should be applied
                if end - start >= 5:  # Apply only if length >= window size
                    smoothed_concentrations = self.moving_average(no_whiff_concentrations, window_size=5)
                    odor_concentration_samples[start:end] = smoothed_concentrations[:end - start]
                else:
                    odor_concentration_samples[start:end] = no_whiff_concentrations
            else:
                i += 1
        return odor_concentration_samples
    


@dataclass
class OdorConfig:
    rows_per_second: int = 200
    base_odor_level: float = 0.70
    distance_threshold: float = 3
    ar1: float = 0.98
    ar2: float = -0.02
    warmup_steps: int = 1000
    low_threshold: float = 0.05
    history_length: int = 7
    transition_matrix: np.ndarray = np.array([[0.15, 0.85], [0.15, 0.85]])

class OdorStateManager:
    def __init__(self, config: OdorConfig, whiff_intermittency: np.ndarray):
        self.recent_history = [0] * 1000
        self.recent_concentrations = [config.base_odor_level] * 10
        self.recent_intermittencies = list(np.random.choice(whiff_intermittency, 5))
        self.current_concentration = config.base_odor_level
        self.prev_concentration = config.base_odor_level
        self.in_whiff_state = False
        self.state_duration = 0

class SimpleOdorPredictor:
    def __init__(self, fitted_p_heatmap: np.ndarray, xedges: np.ndarray, 
                 yedges: np.ndarray, fdf: pd.DataFrame, fdf_nowhiff: pd.DataFrame):
        """
        Initialize the predictor with heatmap data and whiff/no-whiff information.
        """
        self.config = OdorConfig()
        self.fitted_p_heatmap = fitted_p_heatmap
        self.xedges = xedges
        self.yedges = yedges
        self.setup_data(fdf, fdf_nowhiff)

    def setup_data(self, fdf: pd.DataFrame, fdf_nowhiff: pd.DataFrame):
        """Initialize data arrays and distance bins."""
        self.whiff_locations = fdf[['avg_distance_along_streakline', 'avg_nearest_from_streakline']].values
        self.nowhiff_locations = fdf_nowhiff[['avg_distance_along_streakline', 'avg_nearest_from_streakline']].values
        self.whiff_means = fdf.mean_concentration.values
        self.whiff_stds = fdf.std_whiff.values
        self.whiff_duration = fdf.length_of_encounter.values
        self.nowhiff_means = fdf_nowhiff.wc_nowhiff.values
        self.nowhiff_wsd = fdf_nowhiff.wsd_nowhiff.values
        self.whiff_intermittency = fdf.odor_intermittency.values

        # Setup distance bins for intermittency
        distance_bins = np.arange(0, 60, 15)
        self.bin_data_dict = {}
        for i in range(len(distance_bins)-1):
            start_dist, end_dist = distance_bins[i], distance_bins[i+1]
            bin_data = fdf[
                (fdf['avg_distance_along_streakline'] >= start_dist) &
                (fdf['avg_distance_along_streakline'] < end_dist)
            ]['odor_intermittency'].dropna().values
            self.bin_data_dict[(start_dist, end_dist)] = bin_data

    def get_spatial_probability(self, x: float, y: float) -> float:
        """Calculate spatial probability from heatmap for given coordinates."""
        x_bin = np.digitize(x, self.xedges) - 1
        y_bin = np.digitize(y, self.yedges) - 1
        
        if (x_bin < 0 or x_bin >= self.fitted_p_heatmap.shape[0] or 
            y_bin < 0 or y_bin >= self.fitted_p_heatmap.shape[1]):
            return 0.0
            
        return self.fitted_p_heatmap[x_bin, y_bin]

    def generate_intermittency(self, distance: float, state: OdorStateManager, default: float = 0.05) -> float:
        """Generate intermittency value based on distance and state."""
        last_values = np.array(state.recent_intermittencies[-self.config.history_length:])
        low_frequency = np.mean(last_values < self.config.low_threshold)
        
        for (sd, ed), values in self.bin_data_dict.items():
            if sd <= distance < ed and len(values) > 0:
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
        """Update whiff state posterior probability."""
        whiff_state = 1 if state.in_whiff_state else 0
        num_recent_whiffs = sum(state.recent_history[-20:])
        
        time_since_whiff = 0
        for i in range(len(state.recent_history)-1, -1, -1):
            if state.recent_history[i]:
                break
            time_since_whiff += 1
        
        time_factor = min(1.5, time_since_whiff / 35) if time_since_whiff > 50 else 1.0
        boost_factor = (1 + (num_recent_whiffs / 30) * 0.3) * time_factor
        
        recent_avg_concentration = np.mean(state.recent_concentrations[-10:])
        ar_factor = max(0, (recent_avg_concentration - self.config.base_odor_level) / 
                       (4.5 - self.config.base_odor_level))
        concentration_factor = np.clip(1 + ar_factor * 0.2, 0.6, 1.4)
        
        posterior = ((prior_prob * 0.2) * self.config.transition_matrix[whiff_state][1] * 
                    boost_factor * concentration_factor)
        return posterior / (posterior + (1 - posterior))


    def update_ar2_concentration(self, current: float, prev: float, target: float, 
                               noise_scale: float) -> float:
        """Update concentration using AR(2) model."""
        noise = noise_scale * (np.random.randn() - 0.5) * 0.5
        x_next = (0.85 * (self.config.ar1 * (current - target) + 
                 self.config.ar2 * (prev - target)) + target + noise)
        return x_next

    def predict_odor_concentration(self, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Predict odor concentrations for given coordinates.
        
        Args:
            x: Array of x coordinates
            y: Array of y coordinates
            
        Returns:
            DataFrame with predicted odor concentrations and whiff predictions
        """
        # Convert to numpy arrays if needed
        if isinstance(x, pd.Series):
            x = x.values
        if isinstance(y, pd.Series):
            y = y.values

        # Create DataFrame
        df_result = pd.DataFrame({
            'distance_along_streakline': x,
            'nearest_from_streakline': y
        })
        
        # Process data in sequential segments
        segment_size = 2000
        total_segments = len(x) // segment_size + (1 if len(x) % segment_size else 0)
        
        all_concentrations = []
        all_predictions = []
        
        # Initialize state
        state = OdorStateManager(self.config, self.whiff_intermittency)
        
        for seg_idx in range(total_segments):
            start_idx = seg_idx * segment_size
            end_idx = min((seg_idx + 1) * segment_size, len(x))
            
            # Get segment data
            segment_x = x[start_idx:end_idx]
            segment_y = y[start_idx:end_idx]
            segment_locations = np.column_stack((segment_x, segment_y))
            
            # Calculate distances for this segment
            distances = cdist(segment_locations, self.whiff_locations)
            distances_nowhiff = cdist(segment_locations, self.nowhiff_locations)
            
            # Initialize segment arrays
            concentrations = np.full(len(segment_x), self.config.base_odor_level)
            predictions = np.zeros(len(segment_x), dtype=int)
            
            i = 0
            while i < len(segment_x):
                if start_idx + i < self.config.warmup_steps:
                    i += 1
                    continue

                spatial_prob = self.get_spatial_probability(segment_x[i], segment_y[i])
                posterior = self.update_whiff_posterior(spatial_prob, state)

                # Update whiff state
                if state.in_whiff_state:
                    state.state_duration += 1
                    min_duration = 0.1 * self.config.rows_per_second
                    if state.state_duration > min_duration:
                        continue_prob = 0.5 * spatial_prob
                        state.in_whiff_state = np.random.rand() < continue_prob
                else:
                    concentration_factor = ((state.current_concentration - self.config.base_odor_level) / 
                                         (4.5 - self.config.base_odor_level))
                    transition_prob = posterior * (1 + max(0, concentration_factor) * 0.3)
                    state.in_whiff_state = np.random.rand() < transition_prob * 0.5
                    if state.in_whiff_state:
                        state.state_duration = 0

                if state.in_whiff_state and np.min(distances[i]) <= self.config.distance_threshold:
                    # Process whiff state
                    nearest_idx = np.argmin(distances[i])
                    mean_concentration = self.whiff_means[nearest_idx]
                    std_dev_whiff = self.whiff_stds[nearest_idx]
                    duration = int(self.whiff_duration[nearest_idx] * self.config.rows_per_second)
                    rows_to_fill = min(duration, len(segment_x) - i)

                    for j in range(rows_to_fill):
                        new_concentration = self.update_ar2_concentration(
                            state.current_concentration,
                            state.prev_concentration,
                            mean_concentration,
                            0.15 * std_dev_whiff
                        )
                        new_concentration = np.clip(new_concentration, 4.5, 10.2)
                        
                        state.prev_concentration = state.current_concentration
                        state.current_concentration = new_concentration
                        
                        concentrations[i + j] = new_concentration
                        predictions[i + j] = 1
                        
                        state.recent_concentrations.append(new_concentration)
                        state.recent_concentrations.pop(0)
                        state.recent_history.append(1)
                        state.recent_history.pop(0)

                    i += rows_to_fill
                    
                    # Handle intermittency
                    if i < len(segment_x):
                        intermittency = self.generate_intermittency(segment_x[i-1], state)
                        state.recent_intermittencies.append(intermittency)
                        state.recent_intermittencies.pop(0)
                        
                        intermittency_duration = int(intermittency * self.config.rows_per_second * 0.9)
                        i += intermittency_duration
                else:
                    # Process no-whiff state
                    nearest_idx = np.argmin(distances_nowhiff[i])
                    no_whiff_mean = self.nowhiff_means[nearest_idx]
                    no_whiff_std = self.nowhiff_wsd[nearest_idx]
                    
                    new_concentration = self.update_ar2_concentration(
                        state.current_concentration,
                        state.prev_concentration,
                        no_whiff_mean,
                        0.05 * no_whiff_std
                    )
                    new_concentration = np.clip(new_concentration, 0.7, 1.0)
                    
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
            
            all_concentrations.append(concentrations)
            all_predictions.append(predictions)

        # Combine results
        final_concentrations = np.concatenate(all_concentrations)
        final_predictions = np.concatenate(all_predictions)
        
        # Apply final smoothing
        final_concentrations = gaussian_filter(final_concentrations, sigma=0.8)
        
        df_result['predicted_odor'] = final_concentrations
        df_result['whiff_predicted'] = final_predictions
        
        return df_result

def main(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    dirname = '../data/simulator/hws/'
    """Example usage of SimpleOdorPredictor"""
    hmap_data = np.load(str(dirname) + "hmap.npz")
    fdf = pd.read_hdf(str(dirname) + 'whiff.h5')
    fdf_nowhiff = pd.read_hdf(str(dirname) + 'nowhiff.h5')
    
    predictor = SimpleOdorPredictor(
        fitted_p_heatmap=hmap_data['fitted_heatmap'],
        xedges=hmap_data['xedges'],
        yedges=hmap_data['yedges'],
        fdf=fdf,
        fdf_nowhiff=fdf_nowhiff
    )
    
    return predictor.predict_odor_concentration(x, y)



class CenterlineInferringAgent:
    """
    Implementation of centerline-inferring tracking algorithm that works with external odor input.
    Operates in 2D (x,y) space and integrates with provided odor prediction system.
    """
    def __init__(
            self,
            tau: float = 0.42,      # Timescale of turning (from paper)
            noise: float = 1.9,      # Noise amplitude (from paper) 
            bias: float = 0.25,      # Base casting bias (from paper)
            threshold: float = 0.8,   # Odor threshold for detection
            hit_trigger: str = 'peak',
            hit_influence: float = 1.0,  # Influence of hits on centerline inference
            tau_memory: float = 1.0,    # Memory decay timescale
            k_0: float = 1.0,           # Initial uncertainty
            k_s: float = 0.1,           # Hit certainty
            bounds: Optional[List[Tuple[float, float]]] = None  # Environment bounds
        ):
        self.tau = tau
        self.noise = noise
        self.bias = bias
        self.threshold = threshold
        self.hit_trigger = hit_trigger
        self.hit_influence = hit_influence
        self.tau_memory = tau_memory
        self.bounds = bounds
        
        # Covariance matrices
        self.k_0 = np.eye(2) * k_0
        self.k_s = np.eye(2) * k_s
        self.k_0_inv = np.linalg.inv(self.k_0)
        self.k_s_inv = np.linalg.inv(self.k_s)

    def reflect_if_out_of_bounds(self, v: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply boundary conditions if specified"""
        if self.bounds is None:
            return v, x
            
        v_new = v.copy()
        x_new = x.copy()
        
        for dim in range(2):
            if x[dim] < self.bounds[dim][0]:
                v_new[dim] *= -1
                x_new[dim] = 2 * self.bounds[dim][0] - x[dim]
            elif x[dim] > self.bounds[dim][1]:
                v_new[dim] *= -1
                x_new[dim] = 2 * self.bounds[dim][1] - x[dim]
                
        return v_new, x_new

    def update_centerline_posterior(self, centerline_mu: np.ndarray, 
                                  centerline_k: np.ndarray, 
                                  hit_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update centerline estimate given new hit"""
        # Update covariance
        k_inv_prev = np.linalg.inv(centerline_k)
        k_inv = k_inv_prev + self.k_s_inv
        k = np.linalg.inv(k_inv)
        
        # Update mean
        temp = self.k_s_inv.dot(hit_x) + k_inv_prev.dot(centerline_mu)
        mu = k.dot(temp)
        
        return mu, k

    def decay_centerline_posterior(self, centerline_mu: np.ndarray,
                                 centerline_k: np.ndarray,
                                 dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Decay certainty over time"""
        d_centerline_mu = (dt / self.tau_memory) * (-centerline_mu)
        d_centerline_k = (dt / self.tau_memory) * (-centerline_k + self.k_0)
        
        return centerline_mu + d_centerline_mu, centerline_k + d_centerline_k

    def bias_from_centerline_distr(self, x: np.ndarray, mu: np.ndarray, 
                                  k: np.ndarray) -> np.ndarray:
        """Calculate bias vector based on current centerline estimate"""
        # Get crosswind component
        cw = mu - x
        cw /= np.linalg.norm(cw)
        
        # Calculate certainty
        certainty = np.linalg.det(np.linalg.inv(k))
        
        # Combine for final bias
        bias = cw * (self.hit_influence * certainty)
        return bias

    def track(self, odor_predictor, start_pos: np.ndarray, duration: float, dt: float) -> Dict:
        """
        Run centerline-inferring tracking using provided odor prediction system.
        
        Args:
            odor_predictor: Object with predict_odor_concentration method that takes x,y arrays
            start_pos: Starting position [x,y]
            duration: Duration to track for (seconds)
            dt: Time step size (seconds, should be 1/200 for 200Hz)
            
        Returns:
            Dictionary containing trajectory and state information
        """
        n_steps = int(duration / dt)
        ts = np.arange(n_steps) * dt
        
        # Storage arrays
        centerline_mus = np.zeros((n_steps, 2))  # Centerline estimates
        centerline_ks = np.zeros((n_steps, 2, 2))  # Certainty matrices
        bs = np.zeros((n_steps, 2))  # Bias vectors
        vs = np.zeros((n_steps, 2))  # Velocities
        xs = np.zeros((n_steps, 2))  # Positions
        odors = np.zeros(n_steps)   # Odor concentrations
        hits = np.zeros(n_steps)    # Detection events
        
        # Tracking state
        last_odor = 0
        in_puff = False
        hit_occurred = False
        
        # Position storage for batch predictions
        positions_x = []
        positions_y = []
        
        for t_ctr in range(n_steps):
            if t_ctr == 0:
                # Initialize state
                centerline_mu = np.zeros(2)
                centerline_k = self.k_0.copy()
                b = self.bias_from_centerline_distr(start_pos, centerline_mu, centerline_k)
                v = np.zeros(2)
                x = start_pos.copy()
            else:
                # Random walk with bias
                eta = np.random.normal(0, self.noise, (2,))
                b = self.bias_from_centerline_distr(x, centerline_mu, centerline_k)
                
                # Update velocity and position
                v += (dt / self.tau) * (-v + eta + b)
                x += v * dt
            
            # Store positions for batch prediction
            positions_x.append(x[0])
            positions_y.append(x[1])
            
            # Get odor predictions every 20 steps (10Hz) or at end
            if (t_ctr + 1) % 20 == 0 or t_ctr == n_steps - 1:
                pos_array_x = np.array(positions_x)
                pos_array_y = np.array(positions_y)
                
                # Get predictions for stored positions
                result = odor_predictor.predict_odor_concentration(pos_array_x, pos_array_y)
                
                # Update odor history
                start_idx = t_ctr - len(positions_x) + 1
                odors[start_idx:t_ctr+1] = result['predicted_odor'].values
                
                # Clear position storage
                positions_x = []
                positions_y = []
            
            # Apply boundary conditions
            v, x = self.reflect_if_out_of_bounds(v, x)
            
            # Current odor reading
            odor = odors[t_ctr]
            
            # Detect hits based on trigger type
            hit = 0
            if self.hit_trigger == 'entry':
                if odor >= self.threshold and not in_puff:
                    hit = 1
                    in_puff = True
            elif self.hit_trigger == 'peak':
                if odor >= self.threshold:
                    if odor <= last_odor and not hit_occurred:
                        hit = 1
                        hit_occurred = True
                    last_odor = odor
            
            # Reset state if below threshold
            if odor < self.threshold:
                last_odor = 0
                in_puff = False
                hit_occurred = False
            
            # Store current state
            centerline_mus[t_ctr] = centerline_mu
            centerline_ks[t_ctr] = centerline_k
            bs[t_ctr] = b
            vs[t_ctr] = v
            xs[t_ctr] = x
            hits[t_ctr] = hit
            
            # Update centerline estimate
            if hit:
                # Update posterior if hit occurred
                centerline_mu, centerline_k = self.update_centerline_posterior(
                    centerline_mu, centerline_k, x)
            else:
                # Decay posterior if no hit
                centerline_mu, centerline_k = self.decay_centerline_posterior(
                    centerline_mu, centerline_k, dt)
        
        return {
            'centerline_mus': centerline_mus,  # Centerline estimates
            'centerline_ks': centerline_ks,    # Certainty matrices
            'bs': bs,                          # Bias vectors
            'vs': vs,                          # Velocities
            'xs': xs,                          # Positions
            'odors': odors,                    # Odor concentrations
            'hits': hits,                      # Hit detections
            'ts': ts,                          # Timestamps
        }
    













### BACKUP

# def moving_average(data, window_size):
#     if window_size < 1 or len(data) < window_size:
#         return data  # Return data as is if the window size is too large for the data length
#     window = np.ones(int(window_size))/float(window_size)
#     return np.convolve(data, window, 'same')

# i = 0
# while i < len(df_test):
#     dist_along, nearest_from = df_test.iloc[i][['distance_along_streakline', 'nearest_from_streakline']]
#     whiff_predicted, whiff_prob = predict_whiff_from_probability(dist_along, nearest_from, 
#                                                                  fitted_p_heatmap_1, xedges, yedges)
#     # whiff_predicted, whiff_prob = predict_whiff_from_probability(dist_along, nearest_from, interpolator)
    
#     if whiff_predicted and np.min(distances[i]) <= distance_threshold:
#         nearest_whiff_idx = np.argmin(distances[i])
#         mean_concentration = whiff_means[nearest_whiff_idx]
#         std_dev_whiff = whiff_stds[nearest_whiff_idx]
#         duration = int(whiff_duration[nearest_whiff_idx] * rows_per_second)
#         rows_to_fill = min(duration, len(df_test) - i)

#         generated_concentrations = np.random.normal(mean_concentration, std_dev_whiff, rows_to_fill)
#         generated_concentrations = np.clip(generated_concentrations, 4.5, 10.2)
#         odor_concentration_samples[i:i + rows_to_fill] = generated_concentrations
#         whiff_prediction_samples[i:i + rows_to_fill] = 1
#         i += rows_to_fill  # Move index by the number of filled rows
#     else:
#         i += 1  

# # Step 2: Address No Whiff Regions
# i = 0
# while i < len(df_test):
#     if odor_concentration_samples[i] == base_odor_level:
#         nearest_no_whiff_idx = np.argmin(distances_nowhiff[i])
#         no_whiff_mean = nowhiff_means[nearest_no_whiff_idx]
#         no_whiff_std = nowhiff_wsd[nearest_no_whiff_idx]

#         start = i
#         while i < len(df_test) and odor_concentration_samples[i] == base_odor_level:
#             i += 1
#         end = i

#         no_whiff_concentrations = np.random.normal(no_whiff_mean, no_whiff_std, end - start)
#         no_whiff_concentrations = np.clip(no_whiff_concentrations, 0.58, 1)  # Ensure values are within realistic bounds
#         # odor_concentration_samples[start:end] = no_whiff_concentrations
#         # Apply moving average smoothing
#         smoothed_concentrations = moving_average(no_whiff_concentrations, window_size=5)

#         odor_concentration_samples[start:end] = smoothed_concentrations
#     else:
#         i += 1

# # Update the DataFrame with the results
# df_test['predicted_odor'] = odor_concentration_samples
# df_test['whiff_predicted'] = whiff_prediction_samples
