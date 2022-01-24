import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
import multiprocessing as mp

import mspt.particle_detection as detect
import mspt.psf.peak_fit as psf


def ROI_generator(full_image, centre_coordinates, ROI_size):
    '''Return array of size (ROI_size, ROI_size) 
    

    Parameters
    ----------
    full_image : ndarray
        DESCRIPTION.
    centre_coordinates : tuple
        DESCRIPTION.
    ROI_size : int
        DESCRIPTION.

    Returns
    -------
    ROI : ndarray
        DESCRIPTION.

    '''
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


# Pass shared memory array information to Pool workers
arr_dict = {}
def pass_shared_arr(shared_memory_array, array_shape):
    arr_dict['shared_memory_array'] = shared_memory_array
    arr_dict['array_shape'] = array_shape


def particle_fitter_parallel(movie,
                             halfsize,
                             thresh,
                             frame_range=[],
                             processes=( mp.cpu_count()-1 ),
                             method='trust-ncg',
                             DoG_estimates={'T' : 0.1423, 's' : 2.1436, 'sigma' : 1.2921}):
    '''
    Identify and localize single particles in movie.

    Identifies local maxima in the image and excises a ROI of 13x13 pixels 
    around each of the candidate pixels. The ROIs are then fitted by the model
    PSF to extract particle contrast and location at subpixel resolution. The
    model PSF consists of the difference of two concentric Gaussian (DoG)
    functions. For details, see Supplementary Material of Young et al.,
    Science 2018. (10.1126/science.aar5839)

    Parameters
    ----------
    movie : ndarray
        Movie file with dimensions (frames, x, y).
    halfsize : int
        Size of the centered median or mean window in each direction.
    thresh : float
        Threshold paramter to mask candidate spots.
    frame_range : [] or [int, int], optional
        Restrict analysis to certain frames, e.g. [0, 2000]. To analyze whole
        movie, set empty list. The default is [].
    processes : int, optional
        Number of worker processes. The default is ( mp.cpu_count()-1 ).
    method : str, optional
        Type of solver of scipy.optimize.minimize. The default is 'trust-ncg'.
    DoG_estimates : dict, optional
        Initial guesses for PSF parameters used for peak fitting. The default
        is {'T' : 0.1423, 's' : 2.1436, 'sigma' : 1.2921}.

    Returns
    -------
    fits_df : DataFrame
        DataFrame containing particle localizations.

    '''
    cands = detect.identify_candidates(movie,
                                       halfsize,
                                       sig=1.5,
                                       thresh=thresh,
                                       frame_range=frame_range,
                                       processes=processes,
                                       lmax_size=7)

    # Get dimensions of movie data
    movie_shape = movie.shape
    # Create shared memomy array
    shared_arr = mp.RawArray('d', movie_shape[0] * movie_shape[1] * movie_shape[2])
    shared_arr_np = np.frombuffer(shared_arr, dtype=np.float64).reshape(movie_shape)
    # Copy movie data to shared array.
    np.copyto(shared_arr_np, movie)


    useful_chunk_size = cands.shape[0] // (processes*10)
    number_of_chunks = cands.shape[0] // useful_chunk_size
    
    cands_split = np.array_split(cands, number_of_chunks)
    
    with tqdm(total=cands.shape[0], desc='Fitting particles...', unit='candidate spots') as pbar:
        
        with mp.Pool(processes=processes, initializer=pass_shared_arr, initargs=(shared_arr, movie_shape)) as pool:
    
            result_objects = [pool.apply_async(fit_candidates,
                                               args=(chunk,
                                                     method,
                                                     DoG_estimates),
                                                     callback=lambda _: pbar.update(chunk.shape[0])) for chunk in cands_split]
            
            fits_all_list = list()
            for i in range(len(result_objects)):
                fits_all_list.append(result_objects[i].get())
        
        if pbar.n < cands.shape[0]:
            pbar.update(cands.shape[0]- pbar.n)
                
    fits_all = np.concatenate(fits_all_list)
    
    fits_df = pd.DataFrame(fits_all, columns=('frame', 'contrast', 'x', 'y', 'T','s','offset','sigma x', 'sigma y','residual'))

    return fits_df


def fit_candidates(candidate_spots,  method, DoG_estimates):
    fitted_particle_list = []
    
    movie = np.frombuffer(arr_dict['shared_memory_array']).reshape(arr_dict['array_shape'])
    
    for frame, *pix in candidate_spots:

        ROI_size = 13
        ROI = ROI_generator(movie[ frame ,:,:], pix, ROI_size)

        if ROI.shape[0] == ROI_size and ROI.shape[1] == ROI_size:
            try:

                fit_params = psf.fit_peak_DoG_mle(ROI,
                                                  T_guess=DoG_estimates['T'],
                                                  s_guess=DoG_estimates['s'],
                                                  sigma_guess=DoG_estimates['sigma'],
                                                  method=method,
                                                  full_output=False)

                if fit_params[8]: # fit successful
                    list_entry = [frame,
                                  fit_params[0],        # contrast
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