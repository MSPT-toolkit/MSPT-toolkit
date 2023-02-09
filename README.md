# MSPT: Mass-Sensitive Particle Tracking


#### This toolkit provides a complete single particle tracking analysis routine for mass photometry movies, including:

- [x] Rolling median-based background removal
- [x] Particle detection and localization
- [x] Linking of localizations into trajectories using [`trackpy`](http://soft-matter.github.io/trackpy/v0.5.0/)
- [x] Mean squared displacement and jump distance distribution analysis of the trajectories
- [x] Visualizing the correlation of molecular mass and diffusion coefficient of single particles
- [x] Fast batch-mode processing

<br/>

<div align="center">
  <img src="https://user-images.githubusercontent.com/80796346/156175815-16fd5b1a-b6b8-402f-92c9-ce98f8e5f645.png" width="50%" align="center" alt="Workflow diagram">
</div>

<div align="center">
  <b>Data analysis workflow</b>. Reprinted from <a href="https://dx.doi.org/10.3791/63583">Steiert et al.</a>, <em>J. Vis. Exp.</em> (2022).
</div>


#### A detailed protocol describing the experimental and data analysis workflow can be found here:
* [`Mass-Sensitive Particle Tracking to Characterize Membrane-Associated Macromolecule Dynamics`](https://dx.doi.org/10.3791/63583) <br/> _Journal of Visualized Experiments_ (2022).


#### The MSPT-toolkit was used for data analysis in:
* [`Mass-Sensitive Particle Tracking to Elucidate the Membrane-Associated MinDE Reaction Cycle`](https://doi.org/10.1038/s41592-021-01260-x) <br/> _Nature Methods_ (2021).

##
### Installation remarks

If you are new to Python, installing the [`Anaconda Distribution`](https://www.anaconda.com/products/distribution) is highly recommended.

#### Required packages

<ul>
  <b>trackpy</b>
    
  [`Trackpy`](http://soft-matter.github.io/trackpy/v0.5.0/) is used to link particle detections from consecutive frames into trajectories.
  For installation instructions, click [`here`](http://soft-matter.github.io/trackpy/v0.5.0/installation.html).

  <b>fastkde</b> or <b>scikit-learn</b> (for plotting)
  <br>To compare the distributions of molecular mass and diffusion coefficient, two-dimensional kernel density estimations are generated using [`fastKDE`](https://github.com/LBL-EESA/fastkde). For installation instructions, click [`here`](https://github.com/LBL-EESA/fastkde#how-do-i-get-set-up). Alternatively, [`sklearn.neighbors.KernelDensity`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html) could be used.

   <b>bottleneck</b> (optional but recommended, included in Anaconda)
  <br> The package [`bottleneck`](https://bottleneck.readthedocs.io/en/latest/) speeds up the rolling median-based background removal on CPUs more than 10-fold. This option is usually faster than running the background removal on GPU. For installation instructions, click [`here`](https://bottleneck.readthedocs.io/en/latest/installing.html).
  
  <b>CUDA version</b> (optional) 
  <br>To be able to perform image processing on a [`CUDA-capable`](https://developer.nvidia.com/cuda-zone) GPU, [`pytorch`](https://pytorch.org/) is required. Follow the instructions [`here`](https://pytorch.org/get-started/locally/) for details regarding installation.
</ul>

##
### Usage

The data analysis workflow is integrated in the Jupyter notebook `MSPT analysis.ipynb` which contains descriptions about the expected input and output of each step as well as required parameters. In the complementary notebook  `Movie visualization.ipynb`, raw or processed movies can be inspected interactively, for example to examine the membrane crowdedness or the effect of different threshold settings on particle detection.

##
### ❗ Important note

File compression needs to be turned off in the acquisition software before movies are saved to disk. Instructions on how to turn off file compression can be found in the manufacturer's user manual of the acquisition software.
<br/>
If movie recordings were saved with compression and you want to recover them, please get in touch with Refeyn for a decompression module.