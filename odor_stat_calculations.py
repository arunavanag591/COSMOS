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
from scipy import signal



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



def avg_distance(df,index,dat): #input ; location ; storage
  
    #Distance
    i = 0
    avg_dist_source = []
    while i<len(index):
        avg_dist_source.append(np.mean(df.distance_from_source[index[i]])) ## _ is declination corrected distance
        i+=1
    dat['avg_dist_from_source']=avg_dist_source
    dat['log_avg_dist_from_source']= np.log10(dat.avg_dist_from_source)


def whiff_blank_duration(df,index,dat):
    # time of the encounters
    i = 0
    length_of_encounter = []
    dt = df.time[1]-df.time[0]   ## dt is constant, dt * length gives length of time
    while i < len(index):
        length_of_encounter.append(dt*(len(index[i])))
        i+=1
    dat['length_of_encounter'] = length_of_encounter

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
    dat['odor_intermittency'] = intermittency
    dat['log_whiff']=np.log10(dat.length_of_encounter)
    dat['log_blank']=np.log10(dat.odor_intermittency)
    
## Actual
def mean_conc(df,index,dat):
    #Distance
    i = 0
    mean_concentration = []
    while i<len(index):
        mean_concentration.append(np.mean(df.odor[index[i]])) 
        i+=1
    dat['mean_concentration']=mean_concentration

        
    
def std_whiff(dataframe,index,dat):
    odor = dataframe.odor
    i = 0
    std_whiff = []
    while i<len(index):
        std_whiff.append(np.std(odor[index[i]])) 
        i+=1
    dat['std_whiff']=std_whiff
    

def mean_avg(dataframe,index,dat):
    odor = dataframe.odor
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=200)
    ma = odor.rolling(window=indexer, min_periods=1).mean()
    # df['ma_inter']=ma
    moving_avg = []
    i=0
    while i<len(index):
        moving_avg.append(np.mean(ma[index[i]])) 
        i+=1
    dat['whiff_ma']=moving_avg
    
    
def encounter_frequency(df,index,fdf,kernel_size,tau): 
    # binary vector
    start = []
    for i in range (len(index)):
        start.append(index[i][0])
    df['efreq'] = np.zeros(len(df))
    df.efreq.iloc[start] = 1

    ## encounter frequency
    def exp_ker(t, tau):
        return np.exp(-t/tau)/tau

    dt = df.time[1]-df.time[0]
    t = np.arange(0,kernel_size,dt)
    # t=df.time[:10]
    kernel = exp_ker(t,tau)

    filtered = signal.convolve(df.efreq, kernel, mode='full', method='auto')
    filtered = filtered[:-(len(t)-1)]
    df['encounter_frequency']=filtered

    #Average Encounter Frequency
    i = 0
    wfreq = []
    while i<len(index):
        wfreq.append(np.mean(df.encounter_frequency[index[i]]))
        i+=1
    fdf['wf'] = wfreq
    return wfreq


def mean_t(dataframe,index,dat):
  i = 0
  avg_time = []
  while i<len(index):
      avg_time.append(np.mean(dataframe.time[index[i]])) 
      i+=1
  dat['mean_time']=avg_time


##################################################################
############################# Predicted
  
def predicted_mean_conc(df,idx_predicted,dat):
    #Distance
    i = 0
    mean_concentration = []
    while i<len(idx_predicted):
        mean_concentration.append(np.mean(df.predicted_odor[idx_predicted[i]])) 
        i+=1
    dat['mean_concentration']=mean_concentration

def std_whiff_predicted(dataframe,idx_predicted,dat):
    odor = dataframe.predicted_odor
    i = 0
    std_whiff = []
    while i<len(idx_predicted):
        std_whiff.append(np.std(odor[idx_predicted[i]])) 
        i+=1
    dat['std_whiff']=std_whiff

def mean_avg_predicted(dataframe,idx_predicted,dat):
    odor = dataframe.predicted_odor
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=200)
    ma = odor.rolling(window=indexer, min_periods=1).mean()
    # df['ma_inter']=ma
    moving_avg = []
    i=0
    while i<len(idx_predicted):
        moving_avg.append(np.mean(ma[idx_predicted[i]])) 
        i+=1
    dat['whiff_ma']=moving_avg
    

def encounter_frequency_predicted(dataframe,idx_predicted,fdf,kernel_size,tau): 
    # binary vector
    start = []
    for i in range (len(idx_predicted)):
        start.append(idx_predicted[i][0])
    dataframe['efreq'] = np.zeros(len(dataframe))
    dataframe.efreq.iloc[start] = 1

    ## encounter frequency
    def exp_ker(t, tau):
        return np.exp(-t/tau)/tau

    dt = dataframe.time[1]-dataframe.time[0]
    t = np.arange(0,kernel_size,dt)
    # t=df.time[:10]
    kernel = exp_ker(t,tau)

    filtered = signal.convolve(dataframe.efreq, kernel, mode='full', method='auto')
    filtered = filtered[:-(len(t)-1)]
    dataframe['encounter_frequency']=filtered

    #Average Encounter Frequency
    i = 0
    wfreq = []
    while i<len(idx_predicted):
        wfreq.append(np.mean(dataframe.encounter_frequency[idx_predicted[i]]))
        i+=1
    fdf['wf'] = wfreq
    return wfreq



## meta statistic calculation

def get_timed_rows(dataframe,duration_of_encounters):
    x = dataframe.sample(1)
    A = x.mean_time.values.round(0) - duration_of_encounters
    B = x.mean_time.values.round(0)
    timed_rows = dataframe.loc[(dataframe.mean_time > A[0]) & (dataframe.mean_time < B[0])]
    return timed_rows
    
def get_timed_encounter_stats(dataframe, distance_class, duration_of_encounters):
    df_q = dataframe.query('type == ' + str(distance_class))   
    df_q.reset_index(inplace=True, drop=True)     
            
    Nrows = get_timed_rows(df_q,duration_of_encounters)
    avg_dist = np.mean(Nrows.avg_dist_from_source)
    
    mean_time_whiff=np.mean(Nrows.mean_time)
#     mean_conc=np.mean(Nrows_cont.odor)
    pack_data=np.vstack([Nrows.mean_concentration,Nrows.wf,Nrows.log_whiff,Nrows.whiff_ma,Nrows.std_whiff])
    return pack_data,avg_dist,len(Nrows),mean_time_whiff


def gather_stat_timed(dataframe, distance_class, duration_of_encounters,X,y,D,N,T):
    for i in range(500):
        xx,dx,n,t=get_timed_encounter_stats(dataframe,
                                               distance_class, duration_of_encounters)
        X.append(xx)
        D.append(dx)
        y.append(distance_class)
        N.append(n)
        T.append(t)
        
    return X,y,D,N,T