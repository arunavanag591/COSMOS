import numpy as np
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
    warmup_steps: int = 100         # No warmup delay (set >0 if desired)
    low_threshold: float = 0.05
    history_length: int = 7
    transition_matrix: np.ndarray = np.array([[0.15, 0.85],
                                               [0.15, 0.85]])

class OdorStateManager:
    def __init__(self, config, whiff_intermittency):
        # Initialize in z-space via the logit of base concentration.
        z_init = logit(config.base_odor_level, 0, 10)
        self.z_current = z_init
        self.z_prev = z_init
        self.current_concentration = config.base_odor_level
        self.prev_concentration = config.base_odor_level
        
        # These histories (recent concentrations, whiff flags, intermittency values)
        # are used to update the AR(2) process and decide transitions.
        self.recent_history = [0] * 1000
        self.recent_concentrations = [config.base_odor_level] * 10
        self.recent_intermittencies = list(np.random.choice(whiff_intermittency, 5))
        
        self.in_whiff_state = False
        self.state_duration = 0  # How long we have been in the current whiff

class ParallelOdorPredictor:
    def __init__(self, fitted_p_heatmap, xedges, yedges, fdf, fdf_nowhiff):
        self.config = OdorConfig()
        self.fitted_p_heatmap = fitted_p_heatmap
        self.xedges = xedges
        self.yedges = yedges
        self.fdf = fdf
        self.fdf_nowhiff = fdf_nowhiff
        
        self.state = OdorStateManager(self.config, self.fdf.odor_intermittency.values)
        self.steps_processed = 0
        
        # For whiff generation
        self.current_whiff_duration = 0
        self.whiff_values = []
        self.whiff_index = 0
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
        """Return the probability from the fitted heatmap given the spatial coordinates."""
        x_idx = np.digitize([x], self.xedges)[0] - 1
        y_idx = np.digitize([y], self.yedges)[0] - 1
        x_idx = np.clip(x_idx, 0, len(self.xedges) - 2)
        y_idx = np.clip(y_idx, 0, len(self.yedges) - 2)
        return self.fitted_p_heatmap[x_idx, y_idx]

    def update_whiff_posterior(self, prior_prob, state):
        """Update the whiff transition probability based on recent state."""
        whiff_state = 1 if state.in_whiff_state else 0
        num_recent_whiffs = sum(state.recent_history[-20:])
        time_since_whiff = 0
        for i in range(len(state.recent_history) - 1, -1, -1):
            if state.recent_history[i]:
                break
            time_since_whiff += 1
        scaler = 2  # parameter you can adjust
        time_since_last_whiff = min(1.5, time_since_whiff) if time_since_whiff > 50 else 1.0
        recent_whiff_memory = (1 + num_recent_whiffs * scaler) * time_since_last_whiff
        posterior = ((prior_prob * scaler)
                     * self.config.transition_matrix[whiff_state][1]
                     * recent_whiff_memory)
        return posterior

    def generate_intermittency(self, distance_along, distance_from, state, default=0.05):
        """Return a random intermittency value from the appropriate binned data."""
        last_values = np.array(state.recent_intermittencies[-self.config.history_length:])
        low_frequency = np.mean(last_values < self.config.low_threshold)
        for (sd, ed, sn, en), values in self.bin_data_dict.items():
            if (sd <= distance_along < ed) and (sn <= distance_from < en):
                if len(values) > 0:
                    if low_frequency > 0.5:
                        median_val = np.median(values)
                        subset = values[values < median_val]
                        intermittency = np.random.choice(subset) if len(subset) > 0 else np.random.choice(values)
                    else:
                        intermittency = np.random.choice(values)
                    return np.clip(intermittency, np.min(values), np.max(values))
        return default

    def update_ar2_in_zspace(self, z_current, z_prev, z_target, distance, base_noise_scale=0.1, jump_prob=0.03):
        """Perform the AR(2) update in unbounded (z) space."""
        distance_factor = np.exp(-distance / 50.0)
        ar1_local = self.config.ar1 * (1 + 0.1 * distance_factor)
        ar2_local = self.config.ar2 * (1 - 0.1 * distance_factor)
        noise = base_noise_scale * (1 + 2 * distance_factor) * np.random.randn()
        if np.random.rand() < jump_prob:
            jump_size = np.random.uniform(-1, 1) * base_noise_scale * 3
            noise += jump_size
        z_next = 0.85 * (ar1_local * (z_current - z_target) + ar2_local * (z_prev - z_target)) + z_target + noise
        return z_next

    def update_ar2_concentration(self, current, prev, target, noise_scale):
        """Background AR(2) update (no whiff) for concentration."""
        noise = noise_scale * (np.random.randn() - 0.5) * 0.5
        new_val = 0.85 * (self.config.ar1 * (current - target) + self.config.ar2 * (prev - target)) + target + noise
        return new_val

    def generate_whiff_segment(self, mean_concentration, std_dev, duration_steps):
        """Generate whiff segment with plateaus and minimal AR(2) variation"""
        # Create base plateau
        plateau = np.ones(duration_steps) * mean_concentration
        
        # Add small AR(2) variations to make it more natural while maintaining plateau
        signal = np.zeros(duration_steps)
        z_current = logit(mean_concentration, 0, 10)
        z_prev = z_current
        z_target = z_current
        
        for i in range(duration_steps):
            z_next = self.update_ar2_in_zspace(z_current, z_prev, z_target, 
                                            distance=0.1,  # Small distance for minimal variation
                                            base_noise_scale=0.05)  # Reduced noise
            signal[i] = inv_logit(z_next, 0, 10)
            z_prev = z_current
            z_current = z_next
        
        # Maintain plateau character while allowing small variations
        signal = 0.9 * plateau + 0.1 * signal
        return np.clip(signal, mean_concentration - std_dev * 0.2, 
                    mean_concentration + std_dev * 0.2).tolist()

    def step_update(self, x, y, dt=0.005):
        """Step update with AR(2) process for background"""
        self.steps_processed += 1
        if self.steps_processed < self.config.warmup_steps:
            return self.config.base_odor_level

        pos = np.array([[x, y]])
        whiff_locations = self.fdf[['avg_distance_along_streakline',
                                'avg_nearest_from_streakline']].values
        nowhiff_locations = self.fdf_nowhiff[['avg_distance_along_streakline',
                                            'avg_nearest_from_streakline']].values
        
        dist_whiff = cdist(pos, whiff_locations)[0]
        min_dist = np.min(dist_whiff)
        dist_from_source = np.sqrt(x**2 + y**2)
        
        prior_prob = self.get_spatial_prob(x, y)
        posterior = self.update_whiff_posterior(prior_prob, self.state)

        should_check_whiff = (
            not self.state.in_whiff_state or 
            self.whiff_index >= len(self.current_whiff_values)
        )
        
        if should_check_whiff and min_dist <= self.config.distance_threshold:
            distance_factor = np.exp(-dist_from_source / 20.0)
            transition_prob = posterior * (1 + distance_factor)
            
            if np.random.rand() < transition_prob:
                nearest_idx = np.argmin(dist_whiff)
                self.state.in_whiff_state = True
                
                mean_concentration = self.fdf.mean_concentration.values[nearest_idx]
                std_dev = self.fdf.std_whiff.values[nearest_idx]
                base_duration = self.fdf.length_of_encounter.values[nearest_idx]
                
                duration_factor = 1 + 2 * distance_factor
                duration_steps = int(base_duration * duration_factor * self.config.rows_per_second)
                
                # Generate whiff with subtle AR(2) variations
                whiff_values = self.generate_whiff_segment(mean_concentration, std_dev, duration_steps)
                
                # Add no-whiff period
                intermittency = self.generate_intermittency(
                    whiff_locations[nearest_idx, 0],
                    whiff_locations[nearest_idx, 1],
                    self.state
                )
                blank_duration = int(intermittency * self.config.rows_per_second)
                
                # Use AR(2) for no-whiff period
                blank_values = []
                current = self.config.base_odor_level
                prev = current
                for _ in range(blank_duration):
                    new_val = self.update_ar2_concentration(
                        current, prev, self.config.base_odor_level, 0.02
                    )
                    blank_values.append(new_val)
                    prev = current
                    current = new_val
                
                self.current_whiff_values = whiff_values + blank_values
                self.whiff_index = 0
                
                self.state.recent_intermittencies.append(intermittency)
                self.state.recent_intermittencies.pop(0)

        if self.state.in_whiff_state and min_dist <= self.config.distance_threshold * 1.2:
            if self.whiff_index < len(self.current_whiff_values):
                new_concentration = self.current_whiff_values[self.whiff_index]
                self.whiff_index += 1
                
                self.state.recent_concentrations.append(new_concentration)
                self.state.recent_concentrations.pop(0)
                self.state.recent_history.append(1)
                self.state.recent_history.pop(0)
                
                if self.whiff_index >= len(self.current_whiff_values):
                    self.state.in_whiff_state = False
            else:
                self.state.in_whiff_state = False
                new_concentration = self.config.base_odor_level
        else:
            # No-whiff background concentration
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
            
            # Smoothing
            if self.steps_processed >= 2:
                window_data = np.append(self.state.recent_concentrations[-10:], new_concentration)
                window = np.ones(10) / 10.0
                new_concentration = np.convolve(window_data, window, mode='valid')[-1]
            
            # Update states
            self.state.prev_concentration = self.state.current_concentration
            self.state.current_concentration = new_concentration
            self.state.recent_concentrations.append(new_concentration)
            self.state.recent_concentrations.pop(0)
            self.state.recent_history.append(0)
            self.state.recent_history.pop(0)
        
        return new_concentration