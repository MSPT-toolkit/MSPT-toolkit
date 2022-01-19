import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
import multiprocessing as mp

import mspt.particle_detection as detect
import mspt.psf.peak_fit as psf


def ROI_generator(full_image, centre_coordinates, ROI_size):
    ROI = full_image[centre_coordinates[0]-(ROI_size//2):centre_coordinates[0]+(ROI_size//2)+1, centre_coordinates[1]-(ROI_size//2):centre_coordinates[1]+(ROI_size//2)+1]
    
    return ROI

def frame_fit(img, thresh, method, DoG_estimates):
    fitted_particle_list = []
    candidate_spots = detect.Pfind_simple(img, thresh)
    if not len(candidate_spots) == 0:
        for pix in candidate_spots:

            ROI_size = 13
            ROI = ROI_generator(img, pix, ROI_size)
            
            if ROI.shape[0] == ROI_size and ROI.shape[1] == ROI_size:
                try:

                    fit_params = psf.fit_peak_DoG_mle(ROI, T_guess=DoG_estimates['T'], s_guess=DoG_estimates['s'], sigma_guess=DoG_estimates['sigma'], method=method, full_output=False)

                    if fit_params[8]: # fit successful
                        list_entry = [fit_params[0],        # contrast
                                      fit_params[1]+pix[0], # x position
                                      fit_params[2]+pix[1], # y position
                                      fit_params[3],        # T
                                      fit_params[4],        # s
                                      fit_params[5],        # offset
                                      fit_params[6],        # sigma x
                                      fit_params[7],        # sigma y
                                      fit_params[9]]        # residual
                        fitted_particle_list.append(list_entry)
                        
                except Exception:
                    pass
    fitted_particle_list = np.asarray(fitted_particle_list)
    
    return fitted_particle_list


def particle_fitter_parallel(data, halfsize, thresh, frame_range=[], used_CPUs=( mp.cpu_count()-1 ), method='trust-ncg', DoG_estimates={ 'T' : 0.1423, 's' : 2.1436, 'sigma' : 1.2921 }):
    '''
    Identify and localize single particles in mass photometry movies.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    halfsize : TYPE
        DESCRIPTION.
    thresh : TYPE
        DESCRIPTION.
    frame_range : TYPE, optional
        DESCRIPTION. The default is [].
    used_CPUs : TYPE, optional
        DESCRIPTION. The default is ( mp.cpu_count()-1 ).
    DoG_estimates : TYPE, optional
        DESCRIPTION. The default is { 'T' : 0.1423, 's' : 2.1436, 'sigma' : 1.2921 }.

    Returns
    -------
    fits_DF : TYPE
        DESCRIPTION.

    '''
    
    if not len(frame_range) == 0:
        span = range(frame_range[0], frame_range[1], 1)
    else:
        span = range(halfsize, data.shape[0]-halfsize-1, 1)
        
    # if all_frames:
    #     span = range(halfsize, data.shape[0]-halfsize-1, 1)
    # else:
    #     span = range(frame_range[0], frame_range[1], 1)

    useful_chunk_size = len(span)//(used_CPUs*10)
    
    if np.remainder(len(span), useful_chunk_size) != 0:
        number_of_chunks = (len(span)//useful_chunk_size)+1
        pbar = tqdm(total=number_of_chunks, desc='Fitting particles...', unit='video chunks')
        last_processed_chunk = fit_particles_in_chunk(data[span[(number_of_chunks-2)*useful_chunk_size]:span[-1]+1], span[(number_of_chunks-2)*useful_chunk_size], thresh, method, DoG_estimates)
        pbar.update(1)
        pool = mp.Pool(used_CPUs)
        result_objects = [pool.apply_async(fit_particles_in_chunk, args=(data[span[chunk*useful_chunk_size]:span[(chunk+1)*useful_chunk_size]], span[chunk*useful_chunk_size], thresh,  method, DoG_estimates), callback=lambda _: pbar.update(1)) for chunk in range(number_of_chunks-1)]
        
        fits_all_array = np.empty((0, 10))
        for i in range(len(result_objects)-1):
            fits_all_array = np.concatenate((fits_all_array, result_objects[i].get()))
        
        fits_all_array = np.concatenate((fits_all_array, last_processed_chunk))
        
        pool.close()
        pool.join()
        
    else:
        number_of_chunks = len(span)//useful_chunk_size
        pbar = tqdm(total=number_of_chunks, desc='Fitting particles...', unit='video chunks')
        last_processed_chunk = fit_particles_in_chunk(data[span[(number_of_chunks-2)*useful_chunk_size]:span[-1]+1], span[(number_of_chunks-2)*useful_chunk_size], thresh, method, DoG_estimates)
        pbar.update(1)
        pool = mp.Pool(used_CPUs)
        result_objects = [pool.apply_async(fit_particles_in_chunk, args=(data[span[chunk*useful_chunk_size]:span[(chunk+1)*useful_chunk_size]], span[chunk*useful_chunk_size], thresh, method, DoG_estimates), callback=lambda _: pbar.update(1)) for chunk in range(number_of_chunks-1)]
        
        fits_all_array = np.empty((0, 10))
        for i in range(len(result_objects)-1):
            fits_all_array = np.concatenate((fits_all_array, result_objects[i].get()))
        
        fits_all_array = np.concatenate((fits_all_array, last_processed_chunk))
        
        pool.close()
        pool.join()
    
    fits_DF = pd.DataFrame(fits_all_array, columns=('frame', 'contrast', 'x', 'y', 'T','s','offset','sigma x', 'sigma y','residual'))

    return fits_DF


def fit_particles_in_chunk(chunk, start_frame_index, thresh, method, DoG_estimates):

    fits_all_array = np.empty((0, 10))
    for i, frame in enumerate(chunk): # Loop through frames

        fitted_particles = frame_fit(frame, thresh, method, DoG_estimates) # Find and fit particles,
        if len(fitted_particles.shape) == 2: # Ensure it doesn't break if no particles detected
            fp_array = np.zeros((fitted_particles.shape[0], fitted_particles.shape[1]+1))
            for c, j in enumerate(fitted_particles):
                fp_array[c] = np.concatenate((np.reshape(i+start_frame_index, (1,)), j))
            fits_all_array = np.concatenate((fits_all_array, fp_array))# Store successful fits and corresponding frame in array
    
    return fits_all_array