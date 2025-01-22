from dataclasses import dataclass
import h5py
import scipy.io as sio
import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
import os

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