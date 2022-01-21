import multiprocessing as mp

import numpy as np
import os
import glob
import pandas as pd
from tqdm.notebook import tqdm

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
    '''
    

    Parameters
    ----------
    list_of_csv : list
        DESCRIPTION.
    output_file : HDF5 store
        DESCRIPTION.
    frame_rate : float, optional
        DESCRIPTION. The default is 199.8.
    pixel_size : float, optional
        DESCRIPTION. The default is 84.4.
    parallel : bool, optional
        DESCRIPTION. The default is True.
    processes : int, optional
        DESCRIPTION. The default is (os.cpu_count()-1).

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if os.path.exists(output_file):
        counter = 2
        filename, extension = os.path.splitext(output_file)
        while os.path.exists(output_file):
            output_file = filename + "_" + str(counter) + extension
            counter += 1


    pool = mp.Pool(processes)

    for csv_idx, csv_file in enumerate(list_of_csv):
        traj_data = pd.read_csv(csv_file, usecols=['frame',
                                                     'contrast',
                                                     'x',
                                                     'y',
                                                     'sigma x',
                                                     'sigma y',
                                                     'particle'])

        particle_id = traj_data['particle'].unique()
    
        useful_chunk_size = len(particle_id) // (processes*10)
        number_of_chunks = len(particle_id) // useful_chunk_size
        
        particle_ids_split = np.array_split(particle_id, number_of_chunks)
    
        with tqdm(total=particle_id.shape[0], desc='Fitting trajectories...', unit='trajectories') as pbar:

            result_objects = [pool.apply_async(diff.fit_JDD_MSD,
                                               args=(chunk,
                                                     traj_data,
                                                     frame_rate,
                                                     pixel_size),
                                                     callback=lambda _: pbar.update(chunk.shape[0])) for chunk in particle_ids_split]
        
            fits_list = list()
            for i in range(len(result_objects)):
                fits_list.append(result_objects[i].get())
    
    
        df_JDD_MSD = pd.concat(fits_list,axis=0) 
        
        with pd.HDFStore(output_file) as hdf_store:
            hdf_store[csv_file] = df_JDD_MSD
        
        print('Saved trajectory analysis results of file {} ({}/{}) to HDF5'.format(csv_file, str(csv_idx+1), str(len(list_of_csv))))
    
    
    pool.close()
    pool.join()
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
