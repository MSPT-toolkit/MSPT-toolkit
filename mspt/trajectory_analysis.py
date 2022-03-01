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
    '''
    Find files with csv file extension and string pattern 'trajectories' in
    directory.

    Parameters
    ----------
    directory : str
        Directory path.

    Returns
    -------
    list_of_csv : list
        List of csv files in the directory and its subdirectories.

    '''
    initialdir = os.path.normpath(directory)
    
    csv_files = list()

    for subdir, dirs, files in os.walk(os.path.normpath(initialdir)):
        csv = glob.glob(subdir  + '/*.csv')
        if csv:
            tmp = glob.glob(subdir  + '/*.csv')
            
            if 'trajectories' in tmp[0]:
                csv_files.append(tmp)
                
    # if path and filename get too long (>261 characters), use this
    # workaround in Windows
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


def fit_trajectories(list_of_csv,
                     output_file,
                     frame_rate=199.8,
                     pixel_size=84.4, 
                     n_timelags_MSD=None,
                     n_timelags_JDD=None,
                     parallel=True,
                     processes=( os.cpu_count()-1 ) ):
    '''
    Extract diffusion coefficients from single particle trajectories.
    
    Fits mean squared displacement (MSD) and one- and two-component jump
    distance distributions (JDD) to trajectories. For details regarding the
    diffusion analysis, see Heermann et al., Nature Methods (2021)
    (https://doi.org/10.1038/s41592-021-01260-x), and the underlying MSD and
    JDD papers (Michalet, 2010 (10.1103/PhysRevE.82.041914) [1], and Weimann et
    al., 2013 [https://doi.org/10.1371/journal.pone.0064287])

    Parameters
    ----------
    list_of_csv : list
        List of csv files containing trajectory information.
    output_file : HDF5
        HDF5 store where trajectory fitting results are stored in containers
        with the name of each csv file as keys.
    frame_rate : float, optional
        Frame rate of movie acquisition. The default is 199.8.
    pixel_size : float, optional
        Pixel size of camera in nm. The default is 84.4.
    n_timelags_MSD : None or int, optional
        Number of time lags to consider for linear mean squared displacement
        fitting. If None, the number of time lags is estimated based on ref [1].
        The default is None.
    n_timelags_JDD : None or int, optional
        Number of time lags to consider for jump distance distribution fitting.
        If None, the same number of time lags is considered as in MSD fitting.
        The default is None.
    parallel : bool, optional
        Enable or disable parallelization. The default is True.
    processes : int, optional
        Number of worker processes. The default is ( os.cpu_count()-1 ).

    Returns
    -------
    None

    '''
    if os.path.exists(output_file):
        counter = 2
        filename, extension = os.path.splitext(output_file)
        while os.path.exists(output_file):
            output_file = filename + "_" + str(counter) + extension
            counter += 1

    if parallel:
        with mp.Pool(processes) as pool:
    
            
            for csv_idx, csv_file in enumerate(list_of_csv):
                traj_data = pd.read_csv(csv_file, usecols=['frame',
                                                           'contrast',
                                                           'x',
                                                           'y',
                                                           'sigma x',
                                                           'sigma y',
                                                           'particle'])
        
                particle_id = traj_data['particle'].unique()
            
                # useful_chunk_size = particle_id .shape[0] // ((mp.cpu_count()-1)*10)
                # number_of_chunks = particle_id .shape[0] // useful_chunk_size
                number_of_chunks = (mp.cpu_count()-1)*10
                
                particle_ids_split = np.array_split(particle_id, number_of_chunks)
            
                with tqdm(total=particle_id.shape[0], unit='trajectories') as pbar:

                    pbar.set_description("Fitting trajectories... file {}/{}".format(str(csv_idx+1),str(len(list_of_csv))))
                    
                    result_objects = [pool.apply_async(diff.fit_JDD_MSD,
                                                       args=(chunk,
                                                             traj_data,
                                                             frame_rate,
                                                             pixel_size,
                                                             n_timelags_MSD,
                                                             n_timelags_JDD),
                                                             callback=lambda _: pbar.update(chunk.shape[0])) for chunk in particle_ids_split]
                
                    fits_list = list()
                    for i in range(len(result_objects)):
                        fits_list.append(result_objects[i].get())
            
                    if pbar.n < particle_id.shape[0]:
                        pbar.update(particle_id.shape[0]- pbar.n)
                        
                        
                df_JDD_MSD = pd.concat(fits_list,axis=0) 
                
                with pd.HDFStore(output_file) as hdf_store:
                    hdf_store[csv_file] = df_JDD_MSD
                
                # print('Saved trajectory analysis results of file {} ({}/{}) to HDF5'.format(csv_file,
                #                                                                             str(csv_idx+1),
                #                                                                             str(len(list_of_csv))))
    
    else:
        for csv_idx, csv_file in enumerate(list_of_csv):
            
            traj_data = pd.read_csv(csv_file, usecols=['frame',
                                                         'contrast',
                                                         'x',
                                                         'y',
                                                         'sigma x',
                                                         'sigma y',
                                                         'particle'])
    
            particle_id = traj_data['particle'].unique()
            
            df_JDD_MSD = diff.fit_JDD_MSD(particle_id,
                                          traj_data,
                                          frame_rate,
                                          pixel_size,
                                          n_timelags_MSD,
                                          n_timelags_JDD)
    
            with pd.HDFStore(output_file) as hdf_store:
                hdf_store[csv_file] = df_JDD_MSD
            
            # print('Saved trajectory analysis results of file {} ({}/{}) to HDF5'.format(csv_file,
            #                                                                             str(csv_idx+1), 
            #                                                                             str(len(list_of_csv))))
            
            
    return print('Saved trajectory analysis results to: {}'.format(output_file))




def apply_calibration(df, slope=28191.37194436, offset=-20.47852753):
    '''
    Convert contrast to mass using a linear relationship.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing trajectory fitting results.
    slope : float, optional
        Slope of the linear contrast-to-mass relationship determined by
        calibration measurement of standards with known mass in kDa.
        The default is 28191.37194436 [kDa].
    offset : float, optional
        y-intercept of the linear contrast-to-mass relationship in kDa.
        The default is -20.47852753 [kDa].

    Returns
    -------
    pandas Series
        Median mass of trajectories calculated as slope*df['med_c'] + offset.

    '''
    return slope * df['med_c'] + offset


def get_frames(row):
    if row['len'] % 2 == 1:
        fr_min = int(row['center frame'] - (row['len'] - 1)/2.0)
        fr_max = int(row['center frame'] + (row['len'] - 1)/2.0)
    else:
        fr_min = int(np.floor(row['center frame']) - (row['len'] - 2)/2.0)
        fr_max = int(np.ceil(row['center frame']) + (row['len'] - 2)/2.0)
        
    frames = np.arange(fr_min,fr_max+1,1)
    
    return frames

def calc_median(frames, counts_dict):
    # frames = row['frame_indices']
    particles = [counts_dict[frame] for frame in frames]
    return np.median(np.asarray(particles))
        

def calc_particle_number_linked(df):
    '''
    Calculate membrane-associated trajectory numbers.
    
    Each trajectory is assigned to an apparent membrane crowdedness determined
    as the median of all trajectories detected during the trajectory’s lifetime.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing trajectory fitting results.

    Returns
    -------
    pandas Series
        Trajectory-wise membrane crowdedness with regard to trajectory numbers.

    '''
    frame_indices = df.apply(get_frames,axis=1)
    frame_indices_mod = np.concatenate((frame_indices.values),axis=0)
    
    unique, counts = np.unique(frame_indices_mod, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    
    frame_indices = pd.Series(frame_indices, name='frame_indices')
    
    return frame_indices.apply(calc_median,args=(counts_dict,))


def calc_particle_number_detected(df, csv_file):
    '''
    Calculate membrane-associated detection numbers.
    
    Each trajectory is assigned to an apparent membrane crowdedness determined
    as the median of all detections during the trajectory’s lifetime.   

    Parameters
    ----------
    df : DataFrame
        DataFrame containing trajectory fitting results.
    csv_file : str
        CSV file containing trajectory information returned by trackpy.

    Returns
    -------
    pandas Series
        Trajectory-wise membrane crowdedness with regard to detection numbers.

    '''
    if csv_file.startswith('/'): # remove leading slash added py pandas HDFStore
        csv_file = csv_file[1:]
        
    detections_folder = os.path.dirname(os.path.dirname(os.path.normpath(csv_file)))
    detections_csv = glob.glob(detections_folder + './*.csv')

    assert len(detections_csv) == 1, 'No or multiple detection CSV file(s) found in {}'.format(detections_folder)
    
    dets_df = pd.read_csv(detections_csv[0], usecols=['frame', 'contrast'])

    unique_dets, counts_dets = np.unique(dets_df['frame'].values, return_counts=True)
    counts_dict_dets = dict(zip(unique_dets, counts_dets))
    
    frame_indices_trajs = df.apply(get_frames,axis=1)
    frame_indices_trajs.name = 'frame_indices'
    
    return frame_indices_trajs.apply(calc_median,args=(counts_dict_dets,))


if __name__ == '__main__':
    fit_trajectories()
