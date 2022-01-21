import numpy as np
import scipy.ndimage.filters as fi
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ipywidgets import interact
import ipywidgets as widgets
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import multiprocessing as mp

# A simple particle finding algorithm that takes an image as input and returns list of candidate pixel #
#   locations (and optionally the corresponding image map). Applies a LoG filter (of sigma = sig) to   #
#    smooth image and highlight rapid changes in pixel intensity, and thresholds the filtered image.   #
#     Pixels that exceed the threshold and are local maxima (within area specified by lmax_size in     #
#                   the smoothed image) are returned as possible particle locations.                   #
def Pfind_simple(img, thresh=0.00055, sig=1.5, lmax_size=7, return_candmap=False):
    filt_img = fi.gaussian_laplace(img, sig)

    map_thresh = filt_img >= thresh
    
    local_max = fi.maximum_filter(filt_img, size=lmax_size) == filt_img
    
    cand_img = local_max*map_thresh
    
    candidate_locs = []
    
    candidate_locs = np.argwhere(cand_img)
                
    if return_candmap:
        return candidate_locs, cand_img
    else:
        return candidate_locs

# Pfind_simple for 3D array of type [frame, x, y]
def Pfind_simple_movie(movie, start_frame_index, sig, thresh, lmax_size):
    filt_movie = np.empty_like(movie)
    
    for frame in np.arange(movie.shape[0]):
        filt_movie[frame,:,:] = fi.gaussian_laplace(movie[frame,:,:], sig)

    map_thresh = (filt_movie >= thresh)
    
    local_max = fi.maximum_filter(filt_movie, size=(0,lmax_size,lmax_size)) == filt_movie
    
    cand_movie = local_max*map_thresh
    
    candidate_locs = np.argwhere(cand_movie)
    
    candidate_locs[:,0] = candidate_locs[:,0] + start_frame_index
    
    return candidate_locs


def identify_candidates(data,
                        halfsize,
                        sig=1.5,
                        thresh=0.00055,
                        frame_range=[],
                        processes=(mp.cpu_count()-1),
                        lmax_size=7):

    if len(frame_range) == 0:
        span = range(halfsize, data.shape[0]-halfsize-1, 1)
    else:
        span = range(frame_range[0], frame_range[1], 1)

    useful_chunk_size = len(span) // (processes*10)
    number_of_chunks = len(span) // useful_chunk_size
    
    frames_split = np.array_split(span, number_of_chunks)

    with tqdm(total=len(span), desc='Identifying particle candidates...', unit='frames') as pbar:
                
        with mp.Pool(processes) as pool:
    
            result_objects = [pool.apply_async(Pfind_simple_movie,
                                               args=(data[chunk,:,:],
                                                     chunk[0],
                                                     sig,
                                                     thresh,
                                                     lmax_size),
                                                     callback=lambda _: pbar.update(len(chunk))) for chunk in frames_split]
                
            candidate_list = list()
            for i in range(len(result_objects)):
                candidate_list.append(result_objects[i].get())
            
        if pbar.n < len(span):
            pbar.update(len(span)- pbar.n)
                
    candidate_locs = np.concatenate(candidate_list) # cols: frame, x, y
    
    print('{} particle candidates identified'.format(candidate_locs.shape[0]))

    return candidate_locs


def frame_slider_view_cands_simple_filter(frames, vmin=0.01, vmax=0.01):
    def viewframe_wcands(frame, T1):
        plot_cache = {}
        img = frames[frame];
        
        cands_found = Pfind_simple(img, T1);

        plot_cache["fig"], (ax) = plt.subplots(1, 1, figsize=(12, 5));
        plot_cache["im"] = ax.imshow(img, vmin=vmin, vmax=vmax, cmap='binary_r');

        [ax.add_patch(plt.Circle((j[1], j[0]), radius=3, fill=False, edgecolor='r', linewidth=2)) for j in cands_found];
        
        divider = make_axes_locatable(ax);
        cax = divider.append_axes("right", size="2%", pad=0.2);
        plt.colorbar(plot_cache["im"], cax=cax);
    
    values_T1 = np.linspace(0.0001, 0.01, 100)
    
    interact(viewframe_wcands, frame=widgets.IntSlider(min=0, max=len(frames)-1, step=1, value=0), T1=widgets.SelectionSlider(options=[("%g"%i,i) for i in values_T1]));