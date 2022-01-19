import multiprocessing

import numpy as np
import os
import glob
import pandas as pd
from tqdm.notebook import tqdm
import time
from itertools import repeat

import warnings
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)
warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
import mspt.diff.diffusion_analysis as diff


def get_csv_file_size(row):
    return os.path.getsize(row['file_data'])

def get_csv_files(directory):
    initialdir = os.path.normpath(directory)
    
    csv_files = list()

    for subdir, dirs, files in os.walk(os.path.normpath(initialdir)):
        csv = glob.glob(subdir  + '/*.csv')
        if csv:
            tmp = glob.glob(subdir  + '/*.csv')
            
            if 'trajectories' in tmp[0]:
                csv_files.append(tmp)
                
    # if path and filename get too long (>261 characters), use this
    csv_files_mod = list()
    for i in csv_files:
        if len(i[0]) >=260:
            tmp = '\\\\?\\'+i[0]
            csv_files_mod.append([tmp])
        else:
            csv_files_mod.append([i[0]])
            
    df = pd.DataFrame(csv_files_mod,columns=['file_data'])
    
    df_temp = df['file_data'].str.split(pat='\\',expand=True)
    df_temp = df_temp.add_prefix('folder_')
    
    df = pd.concat((df_temp,df),axis=1)

    df.loc[:,'csv file size'] = df.apply(get_csv_file_size,axis=1)
    df = df.sort_values(by='csv file size',ascending=False)
    
    list_of_csv = list(df.loc[:,'file_data'])
    return list_of_csv


def fit_trajectories(list_of_csv, output_file, frame_rate=199.8, pixel_size=84.4, parallel=True, processes=(os.cpu_count()-1) ):
    """Analyze trajectories.

    Keyword arguments:
    list_of_csv -- the real part (default 0.0)
    output_file -- the imaginary part (default 0.0)
    """
    if output_file:
        counter = 2
        filename, extension = os.path.splitext(output_file)
        while os.path.exists(output_file):
            output_file = filename + "_" + str(counter) + extension
            counter += 1
            
        dfs_store = pd.HDFStore(output_file)
    else:
        dfs_store = pd.HDFStore(output_file)

    
    if parallel:
        with tqdm(total=len(list_of_csv), desc='Analyzing trajectories...', unit=' csv files') as pbar:
            with multiprocessing.Pool(processes=processes) as pool:

                results_object = pool.starmap_async(diff.do_work, zip(list_of_csv,repeat(frame_rate),repeat(pixel_size)),chunksize=1)
                remaining = len(list_of_csv)
                while not results_object.ready():
                    remaining_now = results_object._number_left
                    if remaining_now != remaining:
                        pbar.update(remaining-remaining_now)
                    remaining = remaining_now
                    time.sleep(0.1)
                result = results_object.get()
                
            if pbar.n < 100:
                pbar.update(100-pbar.n)
  
        for i, res in enumerate(result):
            dfs_store[list_of_csv[i]] = res
    
        pool.close()
        pool.join()
    
    else:
        with tqdm(total=len(list_of_csv), desc='Analyzing trajectories...', unit=' csv files') as pbar:
            for filename in list_of_csv:
                dfs_store[filename] = diff.do_work(filename)
                pbar.update(1)
                
    dfs_store.close()
    return print('Saved trajectory analysis results to: {}'.format(output_file))


def fit_trajectories_all_files_async(list_of_csv, output_file, frame_rate=199.8, pixel_size=84.4, parallel=True, processes=(os.cpu_count()-1) ):
    """Analyze trajectories.

    Keyword arguments:
    list_of_csv -- the real part (default 0.0)
    output_file -- the imaginary part (default 0.0)
    """
    if output_file:
        counter = 2
        filename, extension = os.path.splitext(output_file)
        while os.path.exists(output_file):
            output_file = filename + "_" + str(counter) + extension
            counter += 1
            
        dfs_store = pd.HDFStore(output_file)
    else:
        dfs_store = pd.HDFStore(output_file)

    
    if parallel:
        with tqdm(total=len(list_of_csv), desc='Analyzing trajectories...', unit=' csv files') as pbar:
            with multiprocessing.Pool(processes=processes) as pool:

                results_object = pool.starmap_async(diff.fit_JDD_MSD, zip(list_of_csv,repeat(frame_rate),repeat(pixel_size)),chunksize=1)
                remaining = len(list_of_csv)
                while not results_object.ready():
                    remaining_now = results_object._number_left
                    if remaining_now != remaining:
                        pbar.update(remaining-remaining_now)
                    remaining = remaining_now
                    time.sleep(0.1)
                result = results_object.get()
                
            if pbar.n < 100:
                pbar.update(100-pbar.n)
  
        for i, res in enumerate(result):
            dfs_store[list_of_csv[i]] = res
    
        pool.close()
        pool.join()
    
    else:
        with tqdm(total=len(list_of_csv), desc='Analyzing trajectories...', unit=' csv files') as pbar:
            for filename in list_of_csv:
                dfs_store[filename] = diff.do_work(filename)
                pbar.update(1)
                
    dfs_store.close()
    return print('Saved trajectory analysis results to: {}'.format(output_file))


def apply_calibration(df, slope=28191.37194436, offset=-20.47852753):
    return slope * df['med_c'] + offset


def get_frames(row):
    if row['len'] % 2 == 1:
        fr_min = int(row['centre frame'] - (row['len'] - 1)/2.0)
        fr_max = int(row['centre frame'] + (row['len'] - 1)/2.0)
    else:
        fr_min = int(np.floor(row['centre frame']) - (row['len'] - 2)/2.0)
        fr_max = int(np.ceil(row['centre frame']) + (row['len'] - 2)/2.0)
        
    frames = np.arange(fr_min,fr_max+1,1)
    
    return frames

def calc_median_concurrent_trajectories(row, counts_dict):
    frames = row['frame_indices']
    particles = [counts_dict[key] for key in frames]
    return np.median(np.asarray(particles))
        

def calc_particle_number_linked(df):
    frame_indices = df.apply(get_frames,axis=1)
    frame_indices_mod = np.concatenate((frame_indices.values),axis=0)
    
    unique, counts = np.unique(frame_indices_mod, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    
    frame_indices = pd.DataFrame(frame_indices)
    frame_indices.columns = ['frame_indices']
    
    return frame_indices.apply(calc_median_concurrent_trajectories,axis=1,args=(counts_dict,))

if __name__ == '__main__':
    fit_trajectories()
    
