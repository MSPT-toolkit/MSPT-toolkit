import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
import multiprocessing as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ipywidgets import interact
import ipywidgets as widgets

import mspt.particle_detection as detect
import mspt.loc.peak_fit as psf

def ROI_generator(full_image, centre_coordinates, ROI_size):
    '''Return array of size (ROI_size, ROI_size) 
    

    Parameters
    ----------
    full_image : ndarray
        Image.
    centre_coordinates : (int, int)
        Center coordinates of ROI.
    ROI_size : int
        Size of returned ROI.

    Returns
    -------
    ROI : ndarray
        Excised array of size (ROI_size, ROI_size) around center coordinates.

    '''
    ROI = full_image[centre_coordinates[0]-(ROI_size//2):centre_coordinates[0]+(ROI_size//2)+1,
                     centre_coordinates[1]-(ROI_size//2):centre_coordinates[1]+(ROI_size//2)+1]
    
    return ROI

def frame_fit(img, thresh, method, DoG_estimates,candidate_spots=None):
    fitted_particle_list = []
    try:
        candidate_spots
    except NameError: 
        candidate_spots = detect.Pfind_simple(img, thresh)
        
    if not len(candidate_spots) == 0:
        for pix in candidate_spots:

            ROI_size = 13
            ROI = ROI_generator(img, pix, ROI_size)
            
            if ROI.shape[0] == ROI_size and ROI.shape[1] == ROI_size:
                try:

                    fit_params = psf.fit_peak_DoG_mle(ROI,
                                                      T_guess=DoG_estimates['T'],
                                                      s_guess=DoG_estimates['s'],
                                                      sigma_guess=DoG_estimates['sigma'],
                                                      method=method,
                                                      full_output=False)

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


def particle_fitter(movie,
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

    # useful_chunk_size = cands.shape[0] // (processes*10)
    # number_of_chunks = cands.shape[0] // useful_chunk_size
    number_of_chunks = processes*10
    
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


def frame_slider_view_cands_dets(frames,
                                 vmin=-0.01,
                                 vmax=0.01,
                                 method='trust-ncg',
                                 DoG_estimates={'T' : 0.1423, 's' : 2.1436, 'sigma' : 1.2921},
                                 figsize=(9.5, 9.5*35./128.)):
    '''
    Browse through movie interactively with frame slider, detection threshold
    selection and successful particle fits.

    Parameters
    ----------
    frames : ndarray
        Movie file with dimensions (frames, x, y).
    vmin : float, optional
        Minimum contrast value that the colormap covers. The default is -0.01.
    vmax : float, optional
        Maximum contrast value that the colormap covers. The default is 0.01.
    method : str, optional
        Type of solver of scipy.optimize.minimize. The default is 'trust-ncg'.
    DoG_estimates : dict, optional
        Initial guesses for PSF parameters used for peak fitting. The default is {'T' : 0.1423, 's' : 2.1436, 'sigma' : 1.2921}.
    figsize : (float, float), optional
        Size of figure frame in inches. The default is (9.5, 9.5*35./128.).

    Returns
    -------
    None.

    '''
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes((0.05,0.1, 0.8, 0.8))
    
    im = ax.imshow(frames[0,:,:], interpolation="None", vmin=vmin, vmax=vmax, cmap='binary_r')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.2)
    fig.colorbar(im, cax=cax)
    
    legend_elements = [Line2D([0], [0], color='#ff8859', linewidth=2.5, linestyle=(0, (2.5, 2.5)), label='candidate spot'),
                       Line2D([0], [0], color='r',       linewidth=2.5, linestyle='-',             label='fitted particle')]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.175), frameon=False, fancybox=False, shadow=False, ncol=2)
    
    #fig.canvas.draw_idle()
    
    def view_frame_cands_dets(frame, thresh):
        while ax.patches:
            ax.patches.pop();
        im.set_data(frames[frame,:,:]);
        
        cands_found = detect.Pfind_simple(frames[frame,:,:], thresh);
        [ax.add_patch(plt.Circle((j[1], j[0]), radius=3, fill=False, edgecolor='#ff8859', linewidth=2.5, linestyle=(0, (2.5, 2.5)) )) for j in cands_found];
        
        
        detections = frame_fit(frames[frame,:,:], thresh, method, DoG_estimates, candidate_spots=cands_found);
        [ax.add_patch(plt.Circle((j[2], j[1]), radius=3, fill=False, edgecolor='r', linewidth=2.5, linestyle='-' )) for j in detections];
        
        fig.canvas.draw_idle();
        fig.canvas.flush_events();
      
    values_thresh = np.arange(0.0001, 0.01+0.00001, 0.00001)
        
    interact(view_frame_cands_dets,
             frame=widgets.IntSlider(min=0, max=len(frames)-1, step=1, value=0, layout=widgets.Layout(width='90%', position='top', continuous_update=False)),
             thresh=widgets.SelectionSlider(options=[("%g"%i,i) for i in values_thresh], value=0.0005, layout=widgets.Layout(width='90%', position='top'), continuous_update=False));