import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional


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
        self.z_prev    = z_init

        # For reference/tracking we also keep actual odor in [0..10]
        # (initialized to the base_odor_level).
        self.current_concentration = config.base_odor_level
        self.prev_concentration    = config.base_odor_level

        # For other state logic
        self.recent_history        = [0] * 1000
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
    warmup_steps: int = 1000
    low_threshold: float = 0.05
    history_length: int = 7
    # Transition matrix for whiff states
    transition_matrix: np.ndarray = np.array([[0.15, 0.85],
                                              [0.15, 0.85]])

class ParallelOdorPredictor:
    def __init__(self, fitted_p_heatmap, xedges, yedges, fdf, fdf_nowhiff):
        self.config = OdorConfig()
        self.fitted_p_heatmap = fitted_p_heatmap
        self.xedges = xedges
        self.yedges = yedges
        self.fdf = fdf
        self.fdf_nowhiff = fdf_nowhiff
        self.setup_data()

    def get_spatial_probability(self, x, y):
        x_bin = np.digitize(x, self.xedges) - 1
        y_bin = np.digitize(y, self.yedges) - 1
        
        if (x_bin < 0 or x_bin >= self.fitted_p_heatmap.shape[0] or 
            y_bin < 0 or y_bin >= self.fitted_p_heatmap.shape[1]):
            return 0.0
            
        return self.fitted_p_heatmap[x_bin, y_bin]

    def predict_odor_concentration(self, x, y):
        if isinstance(x, pd.Series):
            x = x.values
        if isinstance(y, pd.Series):
            y = y.values
            
        df_test = pd.DataFrame({
            'distance_along_streakline': x,
            'nearest_from_streakline': y,
            'spatial_prob': [self.get_spatial_probability(xi, yi) for xi, yi in zip(x, y)]
        })
        
        self.df_test = df_test
        result = self.predict()
        return result

    def setup_data(self):
        distance_bins = np.arange(0, 51, 5) 
        nearest_bins = np.arange(-15, 16, 5) 
        
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

    # ------------------------------------
    # 3c) Update whiff posterior
    # ------------------------------------
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
        time_since_last_whiff = min(1.5, time_since_whiff ) if time_since_whiff > 50 else 1.0
        recent_whiff_memory = (1 + (num_recent_whiffs ) * scaler) * time_since_last_whiff
                
        posterior = ((prior_prob * scaler)
                     * self.config.transition_matrix[whiff_state][1]
                     * recent_whiff_memory)
                    #  * concentration_factor)
        
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
        """
        Perform an AR(2)-like update, but in 'z' (logit) space, so that
        when we transform back via inv_logit, the odor is guaranteed in [0,10].
        
        distance is used to modulate noise or AR coefficients if desired.
        """
        # Distance factor to damp or intensify fluctuations
        distance_factor = np.exp(-distance / 50.0)  # decays with distance

        # Possibly adjust AR(1), AR(2), etc. coefficients
        ar1_local = self.config.ar1 * (1 + 0.1 * distance_factor)
        ar2_local = self.config.ar2 * (1 - 0.1 * distance_factor)
        
        # Base random noise
        noise = base_noise_scale * (1 + 2 * distance_factor) * np.random.randn()
        
        # Optional “jumps”
        if np.random.rand() < jump_prob:
            jump_size = np.random.uniform(-1, 1) * base_noise_scale * 3
            noise += jump_size

        # AR(2) update in unbounded space
        z_next = 0.85 * (ar1_local * (z_current - z_target)
                         + ar2_local * (z_prev - z_target)) \
                 + z_target + noise

        return z_next

    # ------------------------------------
    # 3e) Main loop over data
    # ------------------------------------
    def process_segment(self, start_idx: int, end_idx: int,
                        state: OdorStateManager) -> Tuple[np.ndarray, np.ndarray]:
        segment = self.df_test.iloc[start_idx:end_idx]

        concentrations = np.full(len(segment), self.config.base_odor_level)
        predictions = np.zeros(len(segment), dtype=int)

        test_locations    = segment[['distance_along_streakline','nearest_from_streakline']].values
        whiff_locations   = self.fdf[['avg_distance_along_streakline','avg_nearest_from_streakline']].values
        nowhiff_locations = self.fdf_nowhiff[['avg_distance_along_streakline','avg_nearest_from_streakline']].values

        # Distances from (0,0) if you want to scale noise by how far we are
        distances_from_source = np.array([
            self.calculate_distance_from_source(x, y) 
            for x, y in test_locations
        ])
        distances = cdist(test_locations, whiff_locations)
        distances_nowhiff = cdist(test_locations, nowhiff_locations)

        i = 0
        while i < len(segment):
            # Skip warmup to let the AR state stabilize
            if start_idx + i < self.config.warmup_steps:
                i += 1
                continue

            prior_prob = segment.spatial_prob.iloc[i]
            posterior = self.update_whiff_posterior(prior_prob, state)

            # State transitions for whiff on/off
            if state.in_whiff_state:
                state.state_duration += 1
                # After min_duration, we might exit whiff with some prob
                min_duration = 0.1 * self.config.rows_per_second
                if state.state_duration > min_duration:
                    continue_prob = 0.5 * prior_prob
                    state.in_whiff_state = (np.random.rand() < continue_prob)
            else:
                # Chance to enter whiff
                # transition_prob = posterior * (1 + max(0, concentration_factor) * 0.3)
                transition_prob = posterior
                state.in_whiff_state = (np.random.rand() < transition_prob * 0.5)
                if state.in_whiff_state:
                    state.state_duration = 0

            # If whiff and close enough, generate whiff concentrations
            if state.in_whiff_state and (np.min(distances[i]) <= self.config.distance_threshold):
                nearest_idx = np.argmin(distances[i])
                mean_concentration = self.fdf.mean_concentration.values[nearest_idx]
                std_dev_whiff      = self.fdf.std_whiff.values[nearest_idx]
                duration           = int(self.fdf.length_of_encounter.values[nearest_idx] *
                                         self.config.rows_per_second)
                rows_to_fill = min(duration, len(segment) - i)

                # Convert the whiff's mean concentration to z_target
                z_target = logit(mean_concentration, 0, 10)

                for j in range(rows_to_fill):
                    dist_here = distances_from_source[i + j] if (i+j) < len(distances_from_source) else 0
                    # AR(2) update in z, then invert to odor
                    z_next = self.update_ar2_in_zspace(
                        state.z_current, state.z_prev, z_target,
                        distance=dist_here,
                        base_noise_scale=0.4 * std_dev_whiff,
                        jump_prob=0.1
                    )
                    odor_next = inv_logit(z_next, 0, 10)

                    # Update state
                    state.z_prev = state.z_current
                    state.z_current = z_next
                    state.prev_concentration = state.current_concentration
                    state.current_concentration = odor_next

                    concentrations[i+j] = odor_next
                    predictions[i+j]    = 1

                    # Update rolling histories
                    state.recent_concentrations.append(odor_next)
                    state.recent_concentrations.pop(0)
                    state.recent_history.append(1)
                    state.recent_history.pop(0)

                i += rows_to_fill

                dist_along = segment.distance_along_streakline.iloc[i-1]
                dist_from = segment.nearest_from_streakline.iloc[i-1]
                intermittency = self.generate_intermittency(dist_along, dist_from, state)
                state.recent_intermittencies.append(intermittency)
                state.recent_intermittencies.pop(0)

                # Convert intermittency in seconds to # of rows
                intermittency_duration = int(intermittency * self.config.rows_per_second * 0.9)
                i += intermittency_duration

            else:
                # No-whiff update
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
    # ------------------------------------
    # 3f) High-level driver
    # ------------------------------------
    def predict(self) -> pd.DataFrame:
        # We split data into segments so that state carries over between segments
        segment_size = 2000
        total_segments = (
            len(self.df_test) // segment_size +
            (1 if len(self.df_test) % segment_size else 0)
        )

        all_concentrations = []
        all_predictions = []

        # Create the state manager
        state = OdorStateManager(self.config, self.fdf.odor_intermittency.values)

        for seg_idx in range(total_segments):
            start_idx = seg_idx * segment_size
            end_idx   = min((seg_idx + 1) * segment_size, len(self.df_test))

            concentrations, predictions = self.process_segment(start_idx, end_idx, state)
            all_concentrations.append(concentrations)
            all_predictions.append(predictions)

        # Concatenate results
        final_concentrations = np.concatenate(all_concentrations)
        final_predictions = np.concatenate(all_predictions)
        final_concentrations = gaussian_filter(final_concentrations, sigma=0.8)

        self.df_test.loc[:, 'predicted_odor'] = final_concentrations
        self.df_test.loc[:, 'whiff_predicted'] = final_predictions
        return self.df_test


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