import numpy as np
import scipy.ndimage.filters as fi
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ipywidgets import interact
import ipywidgets as widgets
import matplotlib.pyplot as plt


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
    
    for i in range(cand_img.shape[0]):
        for j in range(cand_img.shape[1]):
            if cand_img[i, j] != 0:
                candidate_locs.append((i, j))
                
    if return_candmap:
        return candidate_locs, cand_img
    else:
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