import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import cdist
from multiprocessing import Pool
import multiprocessing as mp


def compute_distance(streakline, odor_position, nearest_from_streakline):
    source = np.array([[0,0]])
    streakline = np.concatenate((streakline, source))
    distance = cdist(odor_position, streakline).flatten()
    nearest_from_streakline.append(np.min(distance))

def find_streakline(df, dt):
    eastwest = [np.sum(df.windx[j:]) * dt for j in range(0, len(df))]
    northsouth = [np.sum(df.windy[j:]) * dt for j in range(0, len(df))]
    return eastwest, northsouth

def process_file(folder_path, filename):
    df = pd.read_hdf(folder_path + filename)
    source = np.array([[0,0]])
    odor_position = np.array([[df.x[i], df.y[i]] for i in range(len(df))])
    distance_from_source = np.array([cdist(odor_position, source)]).flatten()
    df['distance_from_source'] = distance_from_source

    dt = 0.3
    eastwest, northsouth = find_streakline(df, dt)
    nearest_from_streakline = [0]

    for i in range(len(eastwest) - 1, 0, -1):
        odor_pos = [odor_position[i]]
        eastwest = np.resize(np.array([eastwest - df.windx[i] * dt]), (1, i)).flatten()
        northsouth = np.resize(np.array([northsouth - df.windy[i] * dt]), (1, i)).flatten()
        wind_pos = np.vstack([eastwest, northsouth]).T
        compute_distance(wind_pos, odor_pos, nearest_from_streakline)

    df['nearest_from_streakline'] = nearest_from_streakline[::-1]

    squared_difference = np.square(distance_from_source) - np.square(df.nearest_from_streakline)
    squared_difference = np.abs(squared_difference)
    df['distance_along_streakline'] = np.sqrt(squared_difference)
    df.to_hdf(folder_path + filename, key='data', mode='w')

# Set folder path
folder_path = '/home/beast/An/data/Sept13Plumes/plume2/train/'

all_filenames = [filename for filename in os.listdir(folder_path) if filename.startswith("diag") and 
                 filename.endswith(".h5")]
# Sort filenames numerically
sorted_filenames = sorted(all_filenames, key=lambda x: int(x[4:-3]))

def main():
    folder_path = '/home/beast/An/data/Sept13Plumes/plume2/train/'
    all_filenames = [filename for filename in os.listdir(folder_path) if filename.startswith("diag") and filename.endswith(".h5")]
    sorted_filenames = sorted(all_filenames, key=lambda x: int(x[4:-3]))

    # Create a multiprocessing Pool
    pool = Pool(mp.cpu_count()-1)

    # Use pool.map to apply the process_file function to each filename
    pool.starmap(process_file, [(folder_path, filename) for filename in sorted_filenames])

if __name__ == "__main__":
    main()
