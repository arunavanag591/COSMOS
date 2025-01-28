import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from dataclasses import dataclass

def logit(x, lower=0.0, upper=10.0, eps=1e-8):
    x_clamped = np.clip(x, lower + eps, upper - eps)
    scale = upper - lower
    ratio = (x_clamped - lower) / scale
    return np.log(ratio / (1 - ratio))

def inv_logit(z, lower=0.0, upper=10.0):
    scale = upper - lower
    return lower + scale / (1.0 + np.exp(-z))

@dataclass
class OdorConfig:
    rows_per_second: int = 200
    base_odor_level: float = 0.6
    distance_threshold: float = 3
    ar1: float = 0.98
    ar2: float = -0.02
    warmup_steps: int = 1000
    low_threshold: float = 0.05
    history_length: int = 7
    transition_matrix: np.ndarray = np.array([[0.15, 0.85],
                                            [0.15, 0.85]])

class OdorStateManager:
    def __init__(self, config, whiff_intermittency):
        z_init = logit(config.base_odor_level, 0, 10)
        self.z_current = z_init
        self.z_prev = z_init
        self.current_concentration = config.base_odor_level
        self.prev_concentration = config.base_odor_level
        self.recent_history = [0] * 1000
        self.recent_concentrations = [config.base_odor_level] * 10
        self.recent_intermittencies = list(np.random.choice(whiff_intermittency, 5))
        self.in_whiff_state = False
        self.state_duration = 0

class ParallelOdorPredictor:
    def __init__(self, fitted_p_heatmap, xedges, yedges, fdf, fdf_nowhiff):
        self.config = OdorConfig()
        self.fitted_p_heatmap = fitted_p_heatmap
        self.xedges = xedges
        self.yedges = yedges
        self.fdf = fdf
        self.fdf_nowhiff = fdf_nowhiff
        
        # State management
        self.state = OdorStateManager(self.config, self.fdf.odor_intermittency.values)
        
        # Current whiff management
        self.current_whiff_duration = 0
        self.whiff_timer = 0
        self.intermittency_remaining = 0
        
        # Setup binned data for intermittency
        self.setup_data()
    
    def setup_data(self):
        """Initialize binned data for intermittency generation"""
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

    def get_spatial_probability(self, x, y):
        """Get probability from heatmap for current location"""
        x_bin = np.digitize(x, self.xedges) - 1
        y_bin = np.digitize(y, self.yedges) - 1
        if (x_bin < 0 or x_bin >= self.fitted_p_heatmap.shape[0] or
            y_bin < 0 or y_bin >= self.fitted_p_heatmap.shape[1]):
            return 0.0
        return self.fitted_p_heatmap[x_bin, y_bin]

    def update_whiff_posterior(self, prior_prob: float, state: OdorStateManager) -> float:
        whiff_state = 1 if state.in_whiff_state else 0
        num_recent_whiffs = sum(state.recent_history[-20:])
        
        time_since_whiff = 0
        for i in range(len(state.recent_history)-1, -1, -1):
            if state.recent_history[i]:
                break
            time_since_whiff += 1
        
        scaler = 0.25
        time_since_last_whiff = min(1.5, time_since_whiff) if time_since_whiff > 50 else 1.0
        recent_whiff_memory = (1 + (num_recent_whiffs) * scaler) * time_since_last_whiff
        
        posterior = ((prior_prob * scaler)
                    * self.config.transition_matrix[whiff_state][1]
                    * recent_whiff_memory)
        
        return posterior

    def generate_intermittency(self, distance_along, distance_from, state, default=0.05):
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

    def update_ar2_in_zspace(self, z_current, z_prev, z_target, distance,
                            base_noise_scale=0.1, jump_prob=0.05):
        distance_factor = np.exp(-distance / 50.0)
        ar1_local = self.config.ar1 * (1 + 0.1 * distance_factor)
        ar2_local = self.config.ar2 * (1 - 0.1 * distance_factor)
        
        noise = base_noise_scale * (1 + 2 * distance_factor) * np.random.randn()
        
        if np.random.rand() < jump_prob:
            jump_size = np.random.uniform(-1, 1) * base_noise_scale * 3
            noise += jump_size
        
        z_next = 0.85 * (ar1_local * (z_current - z_target) +
                         ar2_local * (z_prev - z_target)) + z_target + noise
        return z_next

    def update_ar2_concentration(self, current, prev, target, noise_scale):
        noise = noise_scale * (np.random.randn() - 0.5) * 0.5
        x_next = (0.85 * (self.config.ar1 * (current - target) +
                         self.config.ar2 * (prev - target)) + target + noise)
        return x_next

    def step_update(self, x: float, y: float, dt: float = 0.005) -> float:
        """Single step update for real-time prediction"""
        pos = np.array([[x, y]])
        whiff_locations = self.fdf[['avg_distance_along_streakline', 'avg_nearest_from_streakline']].values
        nowhiff_locations = self.fdf_nowhiff[['avg_distance_along_streakline', 'avg_nearest_from_streakline']].values
        
        dist_whiff = cdist(pos, whiff_locations)[0]
        min_dist = np.min(dist_whiff)
        nearest_whiff_idx = np.argmin(dist_whiff)
        
        # If we're in intermittency period, continue no-whiff state
        if self.intermittency_remaining > 0:
            self.intermittency_remaining -= 1
            nearest_idx = np.argmin(cdist(pos, nowhiff_locations)[0])
            no_whiff_mean = self.fdf_nowhiff.wc_nowhiff.values[nearest_idx]
            no_whiff_std = self.fdf_nowhiff.wsd_nowhiff.values[nearest_idx]
            
            new_concentration = self.update_ar2_concentration(
                self.state.current_concentration,
                self.state.prev_concentration,
                no_whiff_mean,
                0.05 * no_whiff_std
            )
            new_concentration = np.clip(new_concentration, 0.6, 1.0)
            
            self.state.prev_concentration = self.state.current_concentration
            self.state.current_concentration = new_concentration
            return new_concentration
        
        # Check for whiff state transitions
        if not self.state.in_whiff_state and min_dist <= self.config.distance_threshold:
            spatial_prob = self.get_spatial_probability(x, y)
            posterior = self.update_whiff_posterior(spatial_prob, self.state)
            
            if np.random.rand() < posterior * 0.5:
                self.state.in_whiff_state = True
                self.state.state_duration = 0
                self.current_whiff_duration = int(
                    self.fdf.length_of_encounter.values[nearest_whiff_idx] * 
                    self.config.rows_per_second
                )
        
        # Handle active whiff
        if self.state.in_whiff_state and min_dist <= self.config.distance_threshold:
            self.state.state_duration += 1
            
            if self.state.state_duration >= self.current_whiff_duration:
                # End whiff and calculate intermittency
                self.state.in_whiff_state = False
                dist_along = whiff_locations[nearest_whiff_idx, 0]
                dist_from = whiff_locations[nearest_whiff_idx, 1]
                intermittency = self.generate_intermittency(dist_along, dist_from, self.state)
                self.state.recent_intermittencies.append(intermittency)
                self.state.recent_intermittencies.pop(0)
                self.intermittency_remaining = int(intermittency * self.config.rows_per_second * 0.9)
            
            # Generate whiff concentration
            mean_concentration = self.fdf.mean_concentration.values[nearest_whiff_idx]
            std_dev_whiff = self.fdf.std_whiff.values[nearest_whiff_idx]
            z_target = logit(mean_concentration, 0, 10)
            
            dist_from_source = np.sqrt(x**2 + y**2)
            z_next = self.update_ar2_in_zspace(
                self.state.z_current,
                self.state.z_prev,
                z_target,
                distance=dist_from_source,
                base_noise_scale=0.15 * std_dev_whiff
            )
            
            new_concentration = inv_logit(z_next, 0, 10)
            
            # Update states
            self.state.z_prev = self.state.z_current
            self.state.z_current = z_next
            self.state.prev_concentration = self.state.current_concentration
            self.state.current_concentration = new_concentration
            
            # Update histories
            self.state.recent_concentrations.append(new_concentration)
            self.state.recent_concentrations.pop(0)
            self.state.recent_history.append(1)
            self.state.recent_history.pop(0)
            
            return new_concentration
        
        # No whiff - background concentration
        nearest_idx = np.argmin(cdist(pos, nowhiff_locations)[0])
        no_whiff_mean = self.fdf_nowhiff.wc_nowhiff.values[nearest_idx]
        no_whiff_std = self.fdf_nowhiff.wsd_nowhiff.values[nearest_idx]
        
        new_concentration = self.update_ar2_concentration(
            self.state.current_concentration,
            self.state.prev_concentration,
            no_whiff_mean,
            0.05 * no_whiff_std
        )
        new_concentration = np.clip(new_concentration, 0.6, 1.0)
        
        # Update states and histories
        self.state.prev_concentration = self.state.current_concentration
        self.state.current_concentration = new_concentration
        self.state.recent_concentrations.append(new_concentration)
        self.state.recent_concentrations.pop(0)
        self.state.recent_history.append(0)
        self.state.recent_history.pop(0)
        
        return new_concentration