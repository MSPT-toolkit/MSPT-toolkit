import os
import h5py
import numpy as np
import multiprocessing as mp
import tkinter as tk
from tkinter import filedialog
from ipywidgets import interact
import ipywidgets as widgets
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import bottleneck as bn
    bn_available = True
except ImportError:
    bn_available = False
    print('bottleneck not available or not correctly installed')
    
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

    # Handle different data structures of files acquired with Refeyn OneMP or TwoMP 
    frames = mpfile.get('frame') # Refeyn OneMP
    # instrument = mpfile.get('device_info/InstrumentName')[()].decode()
    if frames == None:
        frames = mpfile.get('movie/frame') # Refeyn TwoMP
        # instrument = mpfile.get('movie/device_info/InstrumentName')[()].decode()
        if frames == None:
            raise KeyError('Unsupported data structure') # Unknown data structure
            
    frames = np.asarray(frames)
            
    print('Loaded {}'.format(filename))
    
    return frames, filename

def load_mp_nodialog(filename):
    mpfile = h5py.File(filename, 'r')
    
    # Handle different data structures of files acquired with Refeyn OneMP or TwoMP 
    frames = mpfile.get('frame') # Refeyn OneMP
    # instrument = mpfile.get('device_info/InstrumentName')[()].decode()
    if frames == None:
        frames = mpfile.get('movie/frame') # Refeyn TwoMP
        # instrument = mpfile.get('movie/device_info/InstrumentName')[()].decode()
        if frames == None:
            raise KeyError('Unsupported data structure') # Unknown data structure
            
    frames = np.asarray(frames)
        
    print('Loaded {}'.format(filename))
    
    return frames, filename


_fileDialogLastDir=None

def fileDialog(initialdir=None):
    '''
    Opens file dialog and returns filename path that user selects.

    Parameters
    ----------
    initialdir : str, optional
        Directory that is opened when file dialog is opened for the first time.
        The default is None.

    Returns
    -------
    path : str
        File path.

    '''
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
    #get_ipython().run_line_magic("gui", "tk")
    print('Loaded {}'.format(path))
    return path


_directoryDialogLastDir = None

def directoryDialog(initialdir=None):
    '''
    Opens directory dialog and returns directory path that user selects.

    Parameters
    ----------
    initialdir : str, optional
        Directory that is opened when file dialog is opened for the first time.
        The default is None.

    Returns
    -------
    path : str
        Directory path.

    '''  
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
    #get_ipython().run_line_magic("gui", "tk")
    return path


def find_filepaths(directory, extension="mp", exclude=None):
    '''
    Find files with specified extension type in directory.
    
    Returns a list of paths to all files of the chosen extension type 
    within a directory. Optionally, feed a string to the exclude argument
    in order to exclude files that contain this text patch.

    Parameters
    ----------
    directory : str
        Directory path.
    extension : str, optional
        File extension. The default is "mp".
    exclude : str, optional
        String pattern to filter files. The default is None.

    Returns
    -------
    filepaths : list
        List of filepaths.

    '''
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
    
    if navg == 1: # No frame averaging
        av_frames = input_frame_sequence.astype(np.int32) # Convert to int32 as int16 dtype of MP files is not supported by bottleneck
        dtype = np.int32
        typecode = 'I' # Required to initialize the shared array for multiprocessing
    else:
        av_frames = np.mean(input_frame_sequence.reshape(navg,
                                                         len(input_frame_sequence)//navg,
                                                         input_frame_sequence.shape[1],
                                                         input_frame_sequence.shape[2],
                                                         order='F'),
                                                         axis=0) # Result is float64
        dtype = np.float64
        typecode = 'd' # Required to initialize the shared array for multiprocessing

    return av_frames, dtype, typecode



def median_filter_frames(video_chunk, starting_frame_number, full_video, window_half_size):
    processed_frames = np.zeros_like(video_chunk)
    for frame_number, frame in enumerate(video_chunk):
        if frame_number+starting_frame_number >= window_half_size and frame_number+starting_frame_number < len(full_video)-window_half_size:
            processed_frames[frame_number] = frame/np.median(full_video[frame_number+starting_frame_number-window_half_size:frame_number+starting_frame_number+window_half_size+1], axis=0)-1.0
    return processed_frames


arr_dict = {}
# Pass shared memory array information to Pool workers
def pass_shared_arr(shared_memory_array, array_shape, dtype):
    arr_dict['shared_memory_array'] = shared_memory_array
    arr_dict['array_shape'] = array_shape
    arr_dict['dtype'] = dtype


if bn_available == True:
    # Use move median function implemented in bottleneck
    def moving_median_filter(frames, window_half_size, full_video):
        
        if full_video is None:
            full_video = np.frombuffer(arr_dict['shared_memory_array'],
                                       dtype=arr_dict['dtype']).reshape(arr_dict['array_shape'])
        
        window_size = window_half_size * 2 + 1
        starting_frame_number = frames[0]
        
        # Specify frames for bottlenecks moving median as it is not using a centered moving window
        frames_move_median = np.arange(frames[0] - window_half_size,
                                       frames[-1] + window_half_size +1,
                                       1)
        frames_move_median = np.where(frames_move_median>=full_video.shape[0],
                                      frames_move_median-full_video.shape[0],
                                      frames_move_median)
        
        array_move_median = full_video[frames_move_median]
            
        processed_frames = ( full_video[frames,:,:] / 
                             bn.move_median(array_move_median, window_size, axis=0)[window_size-1:] - 1)
        # Set leading #window_half_size frames to NaNs as no median of length `window_size` can be calculated
        if np.any(frames_move_median<0):
            processed_frames[:np.sum(frames_move_median<0)] = np.nan
        # Set trailing #window_half_size frames to NaNs as no median of length `window_size` can be calculated
        if (frames[-1] + window_half_size +1)>full_video.shape[0]:
            processed_frames[(full_video.shape[0]) - (frames[-1] + window_half_size +1):] = np.nan
        
        return starting_frame_number, processed_frames
    
else:
    # Use custom numpy function if bottleneck is not installed
    def moving_median_filter(frames, window_half_size, full_video):
        
        if full_video is None:
            full_video = np.frombuffer(arr_dict['shared_memory_array'],
                                       dtype=arr_dict['dtype']).reshape(arr_dict['array_shape'])
        
        processed_frames = np.full((len(frames), full_video.shape[1], full_video.shape[2]),np.nan,dtype=np.float64)
        starting_frame_number = frames[0]
        
        for frame_idx, frame in enumerate(frames):
            if frame >= window_half_size and frame < full_video.shape[0] - window_half_size:
                processed_frames[frame_idx,:,:] = ( full_video[frame,:,:] / 
                                                    np.median(full_video[frame-window_half_size:frame+window_half_size+1,:,:], axis=0) - 1.0 )
                
        return starting_frame_number, processed_frames


def continuous_bg_remover(raw_frames, navg=1, window_half_size=5, mode = 'mean', parallel = 0, GPU = 0):
    # mode 'mean': continuous background removal as used for mass photometry. Generates mean images of window_half_size frames before (mean_before) and after (mean_after) the central frame and generates the new frame by calculating mean_after/mean_before.
    #mode 'median': continuous background removal using a sliding median window. Generates a median image starting at window_half_size frames before and ending window_half_size frames after the central frame and divides the central frame by this median image.
    
    assert mode == 'mean' or mode == 'median', 'continuous_bg_mode not recognised, choose between mean or median'
    

    av_frames, dtype, typecode = frame_averager(raw_frames, navg=navg) # Result is either int32 (navg=1) or float64 (else)

        
    if mode == 'mean':
        
            
        processed_frames = np.full(av_frames.shape,np.nan,dtype=np.float64)
        for frame_number, frame in enumerate(tqdm(av_frames, desc='Generating frames...', unit='frames')):
            if frame_number > window_half_size and frame_number < len(av_frames)-window_half_size:
                processed_frames[frame_number] = np.mean(av_frames[frame_number+1:frame_number+window_half_size+1], axis=0)/np.mean(av_frames[frame_number-window_half_size:frame_number], axis=0)-1.

            
    elif mode == 'median':
        
        if parallel == 0 and GPU == 0:
            processed_frames = np.full(av_frames.shape,np.nan,dtype=np.float64)
            number_of_chunks = 100
            frames_split = np.array_split(np.arange(av_frames.shape[0]), number_of_chunks)
            with tqdm(total=av_frames.shape[0], desc='Generating frames...', unit='frames') as pbar:
                
                for chunk in frames_split:
                    starting_frame, processed_chunk = moving_median_filter(chunk, window_half_size,av_frames)
                    processed_frames[starting_frame:starting_frame+chunk.size,:,:] = processed_chunk
                    pbar.update(chunk.size)
                    
        
        elif parallel == 0 and GPU == 1:
            cuda0 = th.device('cuda:0')
            processed_frames = np.zeros_like(av_frames)
            pbar = tqdm(total=len(processed_frames), desc='Generating frames...', unit='frames')
            
            av_frames = th.from_numpy(av_frames).to(cuda0)
            processed_frames = th.from_numpy(processed_frames).to(cuda0)
            
            for frame_number, frame in enumerate(av_frames):
                if frame_number >= window_half_size and frame_number < len(av_frames)-window_half_size:
                    processed_frames[frame_number] = th.div(frame, th.median(av_frames[frame_number-window_half_size:frame_number+window_half_size+1], dim=0).values)-1.0
                pbar.update(1)
            
            processed_frames = processed_frames.to(device='cpu').numpy()
            
            del av_frames, frame, pbar
            th.cuda.empty_cache()
        
        elif parallel == 1 and GPU == 0:
            # useful_chunk_size = av_frames.shape[0] // ((mp.cpu_count()-1)*10)
            # number_of_chunks = av_frames.shape[0] // useful_chunk_size
            number_of_chunks = (mp.cpu_count()-1)*10
            
            frames_split = np.array_split(np.arange(av_frames.shape[0]), number_of_chunks)
            
            # Get dimensions of movie data
            movie_shape = av_frames.shape
            # Create shared memomy array
            shared_arr = mp.RawArray(typecode, movie_shape[0] * movie_shape[1] * movie_shape[2])
            shared_arr_np = np.frombuffer(shared_arr, dtype=dtype).reshape(movie_shape)
            # Copy movie data to shared array.
            np.copyto(shared_arr_np, av_frames)
            
            
            with tqdm(total=av_frames.shape[0], desc='Generating frames...', unit='frames') as pbar:
                
                with mp.Pool(processes=(mp.cpu_count()-1), initializer=pass_shared_arr, initargs=(shared_arr, movie_shape, dtype)) as pool:
            
                    result_objects = [pool.apply_async(moving_median_filter,
                                                       args=(chunk,
                                                             window_half_size,
                                                             None),
                                                             callback=lambda _: pbar.update(chunk.size)) for chunk in frames_split]
                    
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
    '''
    Load mp movie file.
    
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

    Parameters
    ----------
    batch_mode : bool, optional
        Enable batch mode. The default is False.
    file_to_load : str, optional
        File path (optional, applies if batch_mode=True). The default is ''.
    homedir : str, optional
        Initial directory of the file dialog. Applies only if batch_mode=False.
        The default is 'D:'.
    frame_range : [] or [int, int], optional
        Frames to load and analyze. If empty, all frames are processed.
        The default is [].
    mode : str, optional
        Backround removal strategy. Choose between 'raw', 'continuous_mean', or
        'continuous_median'. The default is 'raw'.
    navg : int, optional
        Frame averaging before image processing. The default is 1.
    window_length : int, optional
        Size of the moving mean or median window. The default is 1.
    parallel : bool, optional
        Enable multiprocessing. Applies only if mode='continuous_median' and
        GPU=False. The default is 0.
    GPU : bool, optional
        Enable CUDA (if available). . Applies only if mode='continuous_median'
        parallel=False. The default is 0.

    Returns
    -------
    ndarray
        Processed or raw (if mode='raw') movie as ndarray.
    filename : str
        File path.

    '''
    
    assert mode == 'raw' or mode == 'continuous_mean' or mode == 'continuous_median', 'mode not recognised, choose between raw, continuous_mean or continuous_median'
    
    if not batch_mode:
        raw_frames, filename = load_mp(homedir)
        
    else:
        raw_frames, filename = load_mp_nodialog(file_to_load)
    
    if not len(frame_range) == 0:
        raw_frames = raw_frames[frame_range[0]:frame_range[1]]
    
    if mode == 'raw':
        
        if navg == 1:

            return raw_frames, filename
        
        else:
            av_frames, _, _ = frame_averager(raw_frames, navg=navg)
            
            return av_frames, filename
    
    elif mode == 'continuous_mean':
        
        processed_frames = continuous_bg_remover(raw_frames, window_half_size=window_length//2, mode='mean')
        
        return processed_frames, filename
    
    elif mode == 'continuous_median':
        
        processed_frames = continuous_bg_remover(raw_frames, navg=navg, window_half_size=window_length//2, mode='median', parallel = parallel, GPU = GPU)
        
        return processed_frames, filename


def frame_slider(frames,
                 vmin=-0.01,
                 vmax=0.01,
                 figsize=(9.5, 9.5*35./128.)):
    '''
    Browse through movie interactively with frame slider.

    Parameters
    ----------
    frames : ndarray
        Movie file with dimensions (frames, x, y).
    vmin : float, optional
        Minimum contrast value that the colormap covers. The default is -0.01.
    vmax : float, optional
        Maximum contrast value that the colormap covers. The default is 0.01.
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
    
    #fig.canvas.draw_idle()
    
    def view_frame(frame):
        im.set_data(frames[frame,:,:]);
        fig.canvas.draw_idle();
        fig.canvas.flush_events();
        
    interact(view_frame, frame=widgets.IntSlider(min=0, max=len(frames)-1, step=1, value=0,layout=widgets.Layout(width='90%', position='top')));