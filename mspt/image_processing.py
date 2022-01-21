import tkinter as tk
from tkinter import filedialog
import h5py
from ipywidgets import interact
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.notebook import tqdm
import os
import multiprocessing as mp


try:
    import torch as th
    if not th.cuda.is_available():
        print('pytorch/CUDA not available or not correctly installed')
except ImportError:
    print('pytorch/CUDA not available or not correctly installed')
    




def load_mp(homedir = "D:"):
    root = tk.Tk()
    filename =  filedialog.askopenfilename(initialdir = homedir,
                                           title = "Select mp file",
                                           filetypes = (("mp files","*.mp"),("all files","*.*")))
    root.withdraw()
    
    mpfile = h5py.File(filename, 'r')
    frames = np.asarray(mpfile['frame'])
    
    print('Loaded %s' %filename)
    
    return frames, filename

def load_mp_nodialog(filename):
    mpfile = h5py.File(filename, 'r')
    frames = np.asarray(mpfile['frame'])
    
    print('Loaded %s' %filename)
    
    return frames, filename


_fileDialogLastDir=None

def fileDialog(initialdir=None):
    """
    Opens file dialog and returns filename path that user selects.

    Args:
       initialdir (str):: Directory that is opened when file dialog is opened for the first time
    
    Returns:
       path (str):: File path

    """
    global _fileDialogLastDir
    # Create Tk root
    root = tk.Tk()
    # Hide the main window
    root.withdraw()
    root.call("wm", "attributes", ".", "-topmost", True)
    enterdir = None
    if _fileDialogLastDir is None:
        if not initialdir is None:
            enterdir = initialdir
        else:
            enterdir = "."
    else:
        enterdir = _fileDialogLastDir
    assert not (enterdir is None)
    path = filedialog.askopenfilename(initialdir=enterdir, multiple=False)
    _fileDialogLastDir = os.path.dirname(path)
    get_ipython().run_line_magic("gui", "tk")
    return path


_directoryDialogLastDir = None

def directoryDialog(initialdir=None):
    """
    Opens directory dialog and returns directory path that user selects.

    Args:
       initialdir (str):: Directory that is opened when file dialog is opened for the first time
    
    Returns:
       path (str):: Directory path

    """
    global _directoryDialogLastDir
    # Create Tk root
    root = tk.Tk()
    # Hide the main window
    root.withdraw()
    root.call("wm", "attributes", ".", "-topmost", True)
    enterdir = None
    if _directoryDialogLastDir is None:
        if not initialdir is None:
            enterdir = initialdir
        else:
            enterdir = "."
    else:
        enterdir = _directoryDialogLastDir
    assert not (enterdir is None)
    path = filedialog.askdirectory(initialdir=enterdir)
    _directoryDialogLastDir = path
    get_ipython().run_line_magic("gui", "tk")
    return path


def find_filepaths(directory, extension="mp", exclude=None):
    """
    Find files with specified extension type in directory.

    Returns a list of paths to all files of the chosen extension type 
    within a directory. Optionally feed a string to the exclude argument
    in order to exclude files that contain this text patch.
    
    Keyword arguments:
    directory -- directory path
    extension -- file extension (default 'mp')
    exclude   -- string pattern to filter files
    """

    filepaths = []
    for root, dirs, files in os.walk(directory):
        for fn in files:
            if exclude is not None:
                if fn.endswith(f".{extension}") and exclude not in fn:
                    filepaths.append(os.path.normpath(os.path.join(root, fn)))
            else:
                if fn.endswith(f".{extension}"):
                    filepaths.append(os.path.normpath(os.path.join(root, fn)))
    return filepaths



def frame_averager(input_frame_sequence, navg=1):
    navg = int(navg)
    assert navg > 0, "navg must be a positive integer"
    
    av_frames = np.mean(input_frame_sequence.reshape(navg,
                                                     len(input_frame_sequence)//navg,
                                                     input_frame_sequence.shape[1],
                                                     input_frame_sequence.shape[2],
                                                     order='F'),
                                                     axis=0)

    return av_frames



def median_filter_frames(video_chunk, starting_frame_number, full_video, window_half_size):
    processed_frames = np.zeros_like(video_chunk)
    for frame_number, frame in enumerate(video_chunk):
        if frame_number+starting_frame_number >= window_half_size and frame_number+starting_frame_number < len(full_video)-window_half_size-1:
            processed_frames[frame_number] = frame/np.median(full_video[frame_number+starting_frame_number-window_half_size:frame_number+starting_frame_number+window_half_size+1], axis=0)-1.0
    return processed_frames


arr_dict = {}
# Pass shared memory array information to Pool workers
def pass_shared_arr(shared_memory_array, array_shape):
    arr_dict['shared_memory_array'] = shared_memory_array
    arr_dict['array_shape'] = array_shape
    
    
def median_filter_frames_shared_arr(frames, window_half_size):
    
    full_video = np.frombuffer(arr_dict['shared_memory_array']).reshape(arr_dict['array_shape'])
    
    processed_frames = np.zeros((len(frames), full_video.shape[1], full_video.shape[2]))
    starting_frame_number = frames[0]
    
    for frame_idx, frame in enumerate(frames):
        if frame >= window_half_size and frame < full_video.shape[0] - window_half_size - 1:
            processed_frames[frame_idx,:,:] = ( full_video[frame,:,:] / 
                                                np.median(full_video[frame-window_half_size:frame+window_half_size+1,:,:], axis=0) - 1.0 )
            
    return starting_frame_number, processed_frames


def continuous_bg_remover(raw_frame_sequence, navg=1, window_half_size=5, mode = 'mean', parallel = 0, GPU = 0):
    # mode 'mean': continuous background removal as used for mass photometry. Generates mean images of window_half_size frames before (mean_before) and after (mean_after) the central frame and generates the new frame by calculating mean_after/mean_before.
    #mode 'median': continuous background removal using a sliding median window. Generates a median image starting at window_half_size frames before and ending window_half_size frames after the central frame and divides the central frame by this median image.
    
    assert mode == 'mean' or mode == 'median', 'continuous_bg_mode not recognised, choose between mean or median'
    
    av_frames = frame_averager(raw_frame_sequence, navg=navg)
    
    if mode == 'mean':
        
        processed_frames = np.zeros_like(av_frames)
        for frame_number, frame in enumerate(tqdm(av_frames, desc='Generating frames...', unit='frames')):
            if frame_number > window_half_size and frame_number < len(av_frames)-window_half_size-1:
                processed_frames[frame_number] = np.mean(av_frames[frame_number+1:frame_number+window_half_size+1], axis=0)/np.mean(av_frames[frame_number-window_half_size-1:frame_number], axis=0)-1.
    
    elif mode == 'median':
        
        if parallel == 0 and GPU == 0:
            processed_frames = np.zeros_like(av_frames)
            for frame_number, frame in enumerate(tqdm(av_frames, desc='Generating frames...', unit='frames')):
                if frame_number >= window_half_size and frame_number < len(av_frames)-window_half_size-1:
                    processed_frames[frame_number] = frame/np.median(av_frames[frame_number-window_half_size:frame_number+window_half_size+1], axis=0)-1.0
        
        if parallel == 0 and GPU == 1:
            cuda0 = th.device('cuda:0')
            processed_frames = np.zeros_like(av_frames)
            pbar = tqdm(total=len(processed_frames), desc='Generating frames...', unit='frames')
            
            av_frames = th.from_numpy(av_frames).to(cuda0)
            processed_frames = th.from_numpy(processed_frames).to(cuda0)
            
            for frame_number, frame in enumerate(av_frames):
                if frame_number >= window_half_size and frame_number < len(av_frames)-window_half_size-1:
                    processed_frames[frame_number] = th.div(frame, th.median(av_frames[frame_number-window_half_size:frame_number+window_half_size+1], dim=0).values)-1.0
                pbar.update(1)
            
            processed_frames = processed_frames.to(device='cpu').numpy()
            
            del av_frames, frame, pbar
            th.cuda.empty_cache()
        
        elif parallel == 1 and GPU == 0:
            
            useful_chunk_size = av_frames.shape[0] // ((mp.cpu_count()-1)*10)
            number_of_chunks = av_frames.shape[0] // useful_chunk_size
            
            frames_split = np.array_split(np.arange(av_frames.shape[0]), number_of_chunks)
            
            # Get dimensions of movie data
            movie_shape = av_frames.shape
            # Create shared memomy array
            shared_arr = mp.RawArray('d', movie_shape[0] * movie_shape[1] * movie_shape[2])
            shared_arr_np = np.frombuffer(shared_arr, dtype=np.float64).reshape(movie_shape)
            # Copy movie data to shared array.
            np.copyto(shared_arr_np, av_frames)
            
            
            with tqdm(total=av_frames.shape[0], desc='Generating frames...', unit='frames') as pbar:
                
                with mp.Pool(processes=(mp.cpu_count()-1), initializer=pass_shared_arr, initargs=(shared_arr, movie_shape)) as pool:
            
                    result_objects = [pool.apply_async(median_filter_frames_shared_arr,
                                                       args=(chunk,
                                                             window_half_size),
                                                             callback=lambda _: pbar.update(chunk.shape[0])) for chunk in frames_split]
                    
                    processed_movie_list = list()
                    for i in range(len(result_objects)):
                        processed_movie_list.append(result_objects[i].get())
                
                # Sort movie chunks according to start frame index
                processed_movie_list.sort(key=lambda x: x[0])
                # Remove start index from list of lists
                processed_movie_list = [i[1] for i in processed_movie_list]
                # Concatenate list
                processed_frames = np.concatenate(processed_movie_list)
                
                if pbar.n < av_frames.shape[0]:
                    pbar.update(av_frames.shape[0] - pbar.n)
            

    return processed_frames


def mp_reader(batch_mode = False, file_to_load = '', homedir = 'D:', frame_range = [], mode = 'raw', navg = 1, window_length = 1,  parallel = 0, GPU = 0):
    """Load mp movie file.

    Loads an mp movie file into a numpy 3D array and applies background removal
    strategies commonly used for iSCAT:
    - Mode 'raw': Loads the raw movie. For frame averaging, give an navg.
    - Mode 'continuous_mean': Loads the raw movie and applies a continuous 
      background removal as used for mass photometry. Generates mean images
      of navg frames before (mean_before) and after (mean_after) the central
      frame and generates the new frame by calculating mean_after/mean_before.
    - Mode 'continuous_median': Loads the raw movie and applies a continuous
      background removal using a sliding median window. This is useful for
      movies of freely diffusing particles. Generates a median image starting
      at median_length/2 frames before and ending median_length/2 frames after
      the central frame and divides the central frame by this median image.
    
    Keyword arguments:
    batch mode    -- enable or disable batch mode (bool)
    file_to_load  -- file path (optional, applies if batch_mode=True)
    homedir       -- initial directory of file dialog (optional,
                     applies if batch_mode=False)
    frame_range   -- frames to load and analyze. If empty, all frames are
                     processed (tuple of ints)
    mode          -- backround removal strategy ('raw', 'continuous_mean',
                     or 'continuous_median')
    navg          -- frame averaging before image processing (int)
    window_length -- size of the mean or median window (int).
    parallel      -- enable multiprocessing (bool)
    GPU           -- enable CUDA (if available)
    """
    
    assert mode == 'raw' or mode == 'continuous_mean' or mode == 'continuous_median', 'mode not recognised, choose between raw, continuous_mean or continuous_median'
    
    if not batch_mode:
        raw_frames, filename = load_mp(homedir)
        
    else:
        raw_frames, filename = load_mp_nodialog(file_to_load)
    
    if not len(frame_range) == 0:
        raw_frames = raw_frames[frame_range[0]:frame_range[1]]
    
    if mode == 'raw':
        
        av_frames = frame_averager(raw_frames, navg=navg)
        
        return av_frames, filename
    
    elif mode == 'continuous_mean':
        
        processed_frames = continuous_bg_remover(raw_frames, window_half_size=window_length//2, mode='mean')
        
        return processed_frames, filename
    
    elif mode == 'continuous_median':
        
        processed_frames = continuous_bg_remover(raw_frames, navg=navg, window_half_size=window_length//2, mode='median', parallel = parallel, GPU = GPU)
        
        return processed_frames, filename


def frame_slider(frames, vmin=0.01, vmax=0.01):
    def view_frame(frame):
        plot_cache = {}
        img = frames[frame];
        
        plot_cache["fig"], (ax) = plt.subplots(1, 1, figsize=(12*9/12, 5*9/12));
        plot_cache["im"] = ax.imshow(img, vmin=vmin, vmax=vmax, cmap='binary_r');
        
        
        divider = make_axes_locatable(ax);
        cax = divider.append_axes("right", size="2%", pad=0.2);
        plt.colorbar(plot_cache["im"], cax=cax);
        #plt.figure('This title does not last');
        #plot_cache["fig"].canvas.toolbar_visible = False;
        
    interact(view_frame, frame=widgets.IntSlider(min=0, max=len(frames)-1, step=1, value=0));
