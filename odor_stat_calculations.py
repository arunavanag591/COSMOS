# dataframes
import pandas as pd

#suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.TimeSeries = pd.Series 

from itertools import groupby
from operator import itemgetter

#math
import numpy as np
from scipy.spatial.distance import cdist



#misc
np.set_printoptions(suppress=True)


def get_time_col(df,dt):

    time = []
    time.append(0)
    for i in range(1,len(df)):
        time.append(time[i-1]+dt)
        
    df['time'] = time

def calculate_distance_from_source(df):
    source = np.array([[0,0]])
    odor_position = np.array([[df.x[i],df.y[i]] for i in range (len(df))]) 

    distance_from_source = np.array([cdist(odor_position,source)]).flatten()
    df['distance_from_source'] = distance_from_source
    

def get_index(data,th):
    idx = []
    for i in range(len(data)):
        if (data[i]>th):
            idx.append(i)
    
    index = []
    for k, g in groupby(enumerate(idx),lambda ix : ix[0] - ix[1]):
        index.append((list((map(itemgetter(1), g)))))
    return index



def scale_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (((data - min_val) / (max_val - min_val)) * 10)



def avg_distance(df,index,fdf): #input ; location ; storage
  
    #Distance
    i = 0
    avg_dist_source = []
    while i<len(index):
        avg_dist_source.append(np.mean(df.distance_from_source[index[i]])) ## _ is declination corrected distance
        i+=1
    fdf['avg_dist_from_source']=avg_dist_source
    fdf['log_avg_dist_from_source']= np.log10(fdf.avg_dist_from_source)


def mean_conc(df,index,fdf):
    #Distance
    i = 0
    mean_concentration = []
    while i<len(index):
        mean_concentration.append(np.mean(df.odor[index[i]])) 
        i+=1
    fdf['mean_concentration']=mean_concentration
  
def mean_conc_p(df,index,fdf):
    #Distance
    i = 0
    mean_concentration = []
    while i<len(index):
        mean_concentration.append(np.mean(df.predicted_odor[index[i]])) 
        i+=1
    fdf['mean_concentration_p']=mean_concentration
        
def whiff_blank_duration(df,index,fdf):
    # time of the encounters
    i = 0
    length_of_encounter = []
    dt = df.time[1]-df.time[0]   ## dt is constant, dt * length gives length of time
    while i < len(index):
        length_of_encounter.append(dt*(len(index[i])))
        i+=1
    fdf['length_of_encounter'] = length_of_encounter

    #time between the encounters
    i = 0
    intermittency = []
    while i < len(index):
        if i < (len(index)-1):
            intermittency.append((index[i+1][0] - index[i][-1])*dt)
            i+=1
        else:
            intermittency.append(0)
            i+=1
    fdf['odor_intermittency'] = intermittency
    fdf['log_whiff']=np.log10(fdf.length_of_encounter)
    fdf['log_blank']=np.log10(fdf.odor_intermittency)