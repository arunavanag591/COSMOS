from dataclasses import dataclass
import h5py
import scipy.io as sio
import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
import os

from scipy.interpolate import RectBivariateSpline
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import partial

@dataclass
class Cfd_rigolli:
    dirname: str
    scale_length_x: float = 1#1.4             # tried 15/40
    scale_length_y: float = 1#3.7             # tried 3/8
    scale_wind_v: float = 50/32 * 1e-2        # 1e-2 to convert cm/s to m/s
    scale_time: float = .15/.01               # was 1/.15
    
    def __post_init__(self):
        files = ['nose_data.mat','ground_data.mat','downwind_v.mat','crosswind_v.mat','vertical_v.mat']
        self.keys = ['nose','ground','uu','vv','ww']
        fnames = [f"{self.dirname}/{f}" for f in files]
        for f in fnames:
            if not os.path.exists(f):
                raise ValueError(f"File {f} does not exist")
        
        self.hdf_files = [h5py.File(f) for f in fnames]
        self.coord = sio.loadmat(f"{self.dirname}/coordinates.mat")
        self.x_coords = self.coord['X'].flatten() * self.scale_length_x
        self.y_coords = self.coord['Y'].flatten() * self.scale_length_y
        self.max_time = (self.hdf_files[0]['nose'].shape[0] - 1) / self.scale_time
        
    def get_odor_at_position_and_time(self, position, time):
        time = int(time * self.scale_time)
        mat = self.hdf_files[0]['nose'][time + 1,:,:]    # time + 1 to match odor data with wind data. (check emails)
        mat[mat<=0] = 1e-20
        mat = np.log10(mat) + 20
        f = rgi((self.y_coords, self.x_coords), mat, bounds_error=False, fill_value=0, method='linear')
        return f(position[::-1]) # reverse the position because the matrix is in y,x format
    
    def get_wind_at_position_and_time(self, position, time):
        time = int(time * self.scale_time)
        position = position[::-1] # reverse the position because the matrix is in y,x format
        positionY = position[0] if position[0] < self.y_coords[-1] else self.y_coords[-1]
        positionY = positionY if positionY > self.y_coords[0] else self.y_coords[0]
        positionX = position[1] if position[1] < self.x_coords[-1] else self.x_coords[-1]
        positionX = positionX if positionX > self.x_coords[0] else self.x_coords[0]
        position = (positionY, positionX)
        f1 = rgi((self.y_coords, self.x_coords), self.hdf_files[2]['uu'][time,:,:], bounds_error=True, fill_value=None)
        f2 = rgi((self.y_coords, self.x_coords), self.hdf_files[3]['vv'][time,:,:], bounds_error=True, fill_value=None)
        f3 = rgi((self.y_coords, self.x_coords), self.hdf_files[4]['ww'][time,:,:], bounds_error=True, fill_value=None)
        wind = [f1(position), f2(position), f3(position)]
        return np.array(wind) * self.scale_wind_v
            
    def get_mat_at_time(self, time, key):
        time = int(time * self.scale_time)
        mat = self.hdf_files[self.keys.index(key)][key][time + 1,:,:]
        if key == 'nose':
            mat[mat<=0] = 1e-20
            mat = np.log10(mat) + 20
        elif key == 'ground':
            mat[mat<=0] = 1e-20
            mat = np.log10(mat) + 20
        else:
            mat = mat * self.scale_wind_v
            
        return mat
    





@dataclass
class Cfd_rigolli_BDCATS:
    dirname: str
    scale_length_x: float = 1
    scale_length_y: float = 1
    scale_wind_v: float = 50/32 * 1e-2  # 1e-2 to convert cm/s to m/s
    scale_time: float = .15/.01
    epsilon: float = 0.1  # DBSCAN parameter for clustering
    min_pts: int = 5  # DBSCAN parameter for minimum points in a cluster
    prefetch_count: int = 3  # Number of future time steps to prefetch
    
    def __post_init__(self):
        # Load files
        files = ['nose_data.mat', 'ground_data.mat', 'downwind_v.mat', 'crosswind_v.mat', 'vertical_v.mat']
        self.keys = ['nose', 'ground', 'uu', 'vv', 'ww']
        fnames = [f"{self.dirname}/{f}" for f in files]
        
        for f in fnames:
            if not os.path.exists(f):
                raise ValueError(f"File {f} does not exist")
        
        # Use parallel I/O similar to BD-CATS when possible
        self.hdf_files = [h5py.File(f) for f in fnames]
        self.coord = sio.loadmat(f"{self.dirname}/coordinates.mat")
        self.x_coords = self.coord['X'].flatten() * self.scale_length_x
        self.y_coords = self.coord['Y'].flatten() * self.scale_length_y
        self.max_time = (self.hdf_files[0]['nose'].shape[0] - 1) / self.scale_time
        
        # Create spatial partition structures for efficient querying
        self._build_spatial_index()
        
        # Setup caching system
        self.cache = {}  # Dictionary to store cached matrices: {(key, time_idx): matrix}
        self.cache_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _build_spatial_index(self):
        """Build spatial indexing structure (KD-tree) for efficient querying (BD-CATS approach)."""
        # Create a grid of all points
        y_grid, x_grid = np.meshgrid(self.y_coords, self.x_coords, indexing='ij')
        points = np.column_stack((y_grid.flatten(), x_grid.flatten()))
        
        # Create KD-tree for efficient nearest neighbor search
        # This follows BD-CATS's approach for geometric partitioning
        self.kdtree = KDTree(points)
        
        # Store indices map for quick lookup
        self.grid_indices = {}
        for i, (y, x) in enumerate(points):
            self.grid_indices[(y, x)] = i
    
    def _load_mat(self, key, time_idx):
        """
        Loads a matrix from the HDF5 file for the given key and time index.
        This method applies pre-processing and caching.
        """
        try:
            # Check if already in cache
            with self.cache_lock:
                if (key, time_idx) in self.cache:
                    return self.cache[(key, time_idx)]
            
            # Not in cache, load it
            mat = self.hdf_files[self.keys.index(key)][key][time_idx + 1, :, :]
            
            if key in ['nose', 'ground']:
                mat[mat <= 0] = 1e-20
                mat = np.log10(mat) + 20
            else:
                mat = mat * self.scale_wind_v
                
            # Cache the result
            with self.cache_lock:
                self.cache[(key, time_idx)] = mat
            
            return mat
        except Exception as e:
            print(f"Error loading {key} at time index {time_idx}: {e}")
            return None
    
    def prefetch(self, key, current_time_idx):
        """
        Schedule background loading of the next few time steps for a given key.
        """
        for t in range(current_time_idx + 1, current_time_idx + 1 + self.prefetch_count):
            if t < 0 or t > int(self.max_time * self.scale_time):
                continue  # Skip invalid time indices
                
            with self.cache_lock:
                if (key, t) in self.cache:
                    continue  # Already cached
                    
            self.executor.submit(self._load_mat, key, t)
    
    # def get_odor_at_position_and_time(self, position, time):
    #     """Get odor concentration using BD-CATS-inspired nearest neighbor search with caching."""
    #     time_idx = int(time * self.scale_time)
        
    #     # Try to get from cache or load
    #     mat = self._load_mat('nose', time_idx)
        
    #     # Trigger prefetching for future time steps
    #     self.prefetch('nose', time_idx)
        
    #     # Use KD-tree for efficient nearest neighbor search (BD-CATS approach)
    #     # with bicubic spline interpolation for accuracy
    #     reverse_position = position[::-1]  # Reverse for y,x format
        
    #     # Create interpolator
    #     spline = RectBivariateSpline(self.y_coords, self.x_coords, mat)
        
    #     # Interpolate at the given position
    #     # Note: spline expects (y, x) but we store coordinates in (x, y)
    #     return spline(reverse_position[0], reverse_position[1])[0, 0]
    
    def get_odor_at_position_and_time(self, position, time):
        """Get odor concentration using BD-CATS-inspired nearest neighbor search with caching."""
        time_idx = int(time * self.scale_time)
        
        # Try to get from cache or load
        mat = self._load_mat('nose', time_idx)
        
        # Trigger prefetching for future time steps
        self.prefetch('nose', time_idx)
        
        # Use the same interpolation approach as the original implementation
        # for consistency and to prevent negative values
        f = rgi((self.y_coords, self.x_coords), mat, bounds_error=False, fill_value=0, method='linear')
        return f(position[::-1])  # reverse the position because the matrix is in y,x format
    
    def get_wind_at_position_and_time(self, position, time):
        """Get wind vector using BD-CATS-inspired approach with caching."""
        time_idx = int(time * self.scale_time)
        
        # Get matrices from cache or load them
        uu = self._load_mat('uu', time_idx)
        vv = self._load_mat('vv', time_idx)
        ww = self._load_mat('ww', time_idx)
        
        # Trigger prefetching
        self.prefetch('uu', time_idx)
        self.prefetch('vv', time_idx)
        self.prefetch('ww', time_idx)
        
        # Adjust position ordering and apply boundary constraints
        position = position[::-1]  # Reverse for y,x format
        positionY = min(max(position[0], self.y_coords[0]), self.y_coords[-1])
        positionX = min(max(position[1], self.x_coords[0]), self.x_coords[-1])
        
        # Create spline interpolators
        spline_uu = RectBivariateSpline(self.y_coords, self.x_coords, uu)
        spline_vv = RectBivariateSpline(self.y_coords, self.x_coords, vv)
        spline_ww = RectBivariateSpline(self.y_coords, self.x_coords, ww)
        
        # Interpolate and return wind vector
        wind = np.array([
            spline_uu(positionY, positionX)[0, 0],
            spline_vv(positionY, positionX)[0, 0],
            spline_ww(positionY, positionX)[0, 0]
        ])
        
        return wind
    
    def get_mat_at_time(self, time, key):
        """Get matrix at specified time with proper scaling and caching."""
        time_idx = int(time * self.scale_time)
        
        # Get from cache or load
        mat = self._load_mat(key, time_idx)
        
        # Trigger prefetching
        self.prefetch(key, time_idx)
        
        return mat
    
    def find_high_density_regions(self, time, key, threshold):
        """
        Find high-density regions using DBSCAN, similar to BD-CATS approach.
        This is useful for identifying important features like vortices or high odor concentrations.
        """
        mat = self.get_mat_at_time(time, key)
        
        # Find coordinates where values exceed threshold
        y_indices, x_indices = np.where(mat > threshold)
        if len(y_indices) == 0:
            return []
        
        # Create points array
        points = np.column_stack((
            self.y_coords[y_indices],
            self.x_coords[x_indices]
        ))
        
        # Use DBSCAN for clustering (core of BD-CATS)
        db = DBSCAN(eps=self.epsilon, min_samples=self.min_pts).fit(points)
        
        # Extract clusters
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        clusters = []
        for i in range(n_clusters):
            cluster_points = points[labels == i]
            cluster_values = np.array([mat[y_idx, x_idx] for y_idx, x_idx in 
                                      zip(y_indices[labels == i], x_indices[labels == i])])
            
            clusters.append({
                'points': cluster_points,
                'values': cluster_values,
                'center': np.mean(cluster_points, axis=0),
                'max_value': np.max(cluster_values),
                'avg_value': np.mean(cluster_values),
                'size': len(cluster_points)
            })
        
        return sorted(clusters, key=lambda x: x['max_value'], reverse=True)
    
    def parallel_process_time_steps(self, start_time, end_time, key, operation):
        """Process multiple time steps in parallel using BD-CATS approach with caching."""
        time_steps = np.arange(start_time, min(end_time, self.max_time), 1/self.scale_time)
        
        # Prefetch data for all time steps
        for t in time_steps:
            time_idx = int(t * self.scale_time)
            self.prefetch(key, time_idx)
        
        with mp.Pool(processes=mp.cpu_count()-4) as pool:
            func = partial(operation, key=key)
            results = pool.map(func, time_steps)
            
        return results