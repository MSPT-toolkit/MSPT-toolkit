import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from fastkde import fastKDE
from IPython.display import clear_output

def generate_2D_KDE(data,
                x='median_mass',
                y='Deff_global',
                x_range=(0,200),
                y_range=(-1,1),
                figsize=(5,5),
                traj_length=5,
                density=None,
                n_levels=12,
                cmap=mpl.cm.gray_r,
                alpha=1.0,
                show=True):
    '''
    Calculate 2D-KDE to visualize the distribution of molecular mass and diffusion of single particles.
    
    KDE is performed with the package fastkde (https://pypi.org/project/fastkde/).
    As diffusion readout, one can choose between diffusion coefficients 
    determined by mean squared displacement (MSD) and jump distance
    distribution (JDD) analysis. For details, see Heermann et al.,
    Nature Methods (2021). (https://doi.org/10.1038/s41592-021-01260-x)
    
    Parameters
    ----------
    data : pandas DataFrame
        DataFrame containing the mass and diffusion information obtained
        from MSPT analysis.
    x : str, optional
        x variable. Choose between 'median_mass', 'mean_mass', or their
        contrast analogues ('med_c, 'mean_c'). The default is 'median_mass'.
    y : str, optional
        y variable. Choose between 'D_MSD' (MSD fit), 'D_JDD' (JDD fit), 
        'Deff_JDD_2c' (2-component JDD fit), 'Deff_JDD' (1 or 2-component JDD fit),
        'D_MSD_JDD' (global MSD and JDD fit), 'Deff_MSD_JDD_2c'
        (global MSD and JDD fit, 2 components), 'Deff_MSD_JDD'
        (global MSD and JDD fit, 1 or 2 components).
        The default is 'D_MSD'.
    x_range : (float, float), optional
        x axis limit. The default is (0,200).
    y_range : (float, float), optional
        y axis limit in log scale. The default is (-1,1).
    figsize : (float, float), optional
        Size of figure frame in inches. The default is (5,5).
    traj_length : int or None, optional
        Minimum trajectory length in frames. The default is 5.
    density : int, (int, int), or None, optional
        Upper limit (if int) or interval (if tuple) of membrane particle
        density in units of trajectory numbers. Each trajectory is assigned
        to an apparent membrane protein density determined as the median number
        of all trajectories detected during the trajectory’s lifetime.
        The default is None.
    n_levels : int, optional
        Determines the number of contour lines. The default is 12.
    cmap : str or Colromap, optional
        A Colormap instance or registered colormap name.
        The default is mpl.cm.gray_r.
    alpha : float, optional
        The alpha blending value, between 0 (transparent) and 1 (opaque).
        The default is 1.0.
    show : bool, optional
        Display figure. The default is True.

    Returns
    -------
    fig : Figure instance

    list
        axes objects of the plot. The list contains the main axis containing
        the 2D-KDE image (axs0), marginal distributions on top (axs1) and
        right (axs2), and the colorbar (cax).
    df_kde : pandas DataFrame
        DataFrame containing 2D-KDE result.
    data_plotted : pandas DataFrame
        filtered DataFrame.

    '''
    assert x in ('median_mass',
                 'mean_mass',
                 'med_c',
                 'mean_c'), 'Choose between median_mass, mean_mass, med_c, or mean_c'
    assert y in ('D_MSD',
                 'D_JDD',
                 'Deff_JDD_2c',
                 'Deff_JDD',
                 'D_MSD_JDD',
                 'Deff_MSD_JDD_2c',
                 'Deff_MSD_JDD'), 'Choose between D_MSD, D_JDD, Deff_JDD_2c, Deff_JDD, D_MSD_JDD, Deff_MSD_JDD_2c, or Deff_MSD_JDD'
    
    fig = plt.figure(figsize=figsize)
    
    im_fraction = 0.55
    axs0 = fig.add_axes((0.25,
                         0.2,
                         im_fraction*5.0/6.0,
                         im_fraction*5.0/6.0))
    
    axs1 = fig.add_axes((0.25,
                         0.2+im_fraction*5.0/6.0,
                         im_fraction*5.0/6.0,
                         im_fraction*1.0/6.0),
                        xticklabels=[])
    
    axs2 = fig.add_axes((0.25 + im_fraction*5.0/6.0,
                         0.2,
                         im_fraction*1.0/6.0,
                         im_fraction*5.0/6.0),
                        yticklabels=[])
    
    cax = fig.add_axes([0.825,
                        0.2,
                        0.025,
                        im_fraction*5.0/6.0])
    
    mmin,mmax = x_range
    Dmin,Dmax = y_range
    
    dat = data.copy()
    
    if density:
        if len(density) == 1:
            dat = dat[dat['particle number (linked)'] <= density]
        if len(density) == 2:
            dat = dat[dat['particle number (linked)'].between( density[0],density[1], inclusive='both') ]

    if traj_length:
        dat = dat[dat['len'] >= traj_length]

    # Filter out results with a negative median mass
    dat = dat[dat['median_mass']>=0]

    # Filter out unsuccessful fits
    if y == 'D_JDD':
        dat = dat[dat['fit_JDD_success']==1]
    elif y == 'Deff_JDD_2c':
        dat = dat[dat['fit_JDD_2c_success']==1]
    elif y == 'Deff_JDD':
        dat = dat[(dat['fit_JDD_success']==1) | (dat['fit_JDD_2c_success']==1)]
    elif y == 'D_MSD_JDD':
        dat = dat[dat['fit_MSD_JDD_1c_success']==1]
    elif y == 'Deff_MSD_JDD_2c':
        dat = dat[dat['fit_MSD_JDD_2c_success']==1]
    elif y == 'Deff_MSD_JDD':
        dat = dat[(dat['fit_MSD_JDD_1c_success']==1) | (dat['fit_MSD_JDD_2c_success']==1)]
    
    # Filter out NaNs in y variable originating from non-physical fit results
    dat = dat[~dat[y].isna()]
    
    # Filter out very slow particles (and negative diffusion coeffs)
    dat = dat[dat[y]>0.0001]
    
    # Filter out particles where first 3 mean squared displacements are not
    # monotonously increasing
    dat = dat[dat['MSD_check']==True]

    # Number of datapoints to calculate KDE
    numPoints = 2**12+1
    
    # Create 512x512 matrix 
    mass_center = 0.
    D_center = 0.
    num_x = (mmax-mmin)/512. * numPoints
    num_y = (Dmax-Dmin)/512. * numPoints

    # 2D KDE
    print('Calculating KDE...')
    myPDF,axes = fastKDE.pdf(dat[x].values, 
                             np.log10(dat[y].values), # calculate diffusion KDE in log-space
                             axes=[np.linspace(mass_center-num_x/2., mass_center+num_x/2.,numPoints),
                                   np.linspace(D_center-num_y/2., D_center+num_y/2.,numPoints)],
                             positiveShift=True,
                             doApproximateECF=True,
                             ecfPrecision=2,
                             logAxes=[False,False])
    # 1D KDE of x variable
    # myPDF_x,axes_x = fastKDE.pdf(dat[x].values,
    #                              axes=[np.linspace(mass_center-num_x/2, mass_center+num_x/2,numPoints)],
    #                              positiveShift=True, 
    #                              doApproximateECF=True,
    #                              ecfPrecision=2,
    #                              logAxes=False)
    # 1D KDE of y variable
    # myPDF_y,axes_y = fastKDE.pdf(np.log10(dat[y].values), # calculate diffusion KDE in log-space
    #                              axes=[np.linspace(D_center-num_y/2, D_center+num_y/2,numPoints)],
    #                              positiveShift=True,
    #                              doApproximateECF=True,
    #                              ecfPrecision=2,
    #                              logAxes=False)
    clear_output(wait=True)
    v1,v2 = axes
    
    df_kde = pd.DataFrame(myPDF,index=v2,columns=v1)
    
    df_kde = df_kde.loc[df_kde.index[df_kde.index>=Dmin], df_kde.columns[df_kde.columns>=mmin]]
    df_kde = df_kde.loc[df_kde.index[df_kde.index<=Dmax], df_kde.columns[df_kde.columns<=mmax]]
    
    axs1.set_title('n = {}'.format(str(dat.shape[0])))
    print('n = {} trajectories, \
          shape of KDE: {}, \
          relative sum of KDE within plotting frame: {}'.format(str(dat.shape[0]),
                                                                str(df_kde.shape),
                                                                str(round(np.sum(df_kde.values)/np.sum(myPDF),2))))
        
    df_kde = df_kde * np.sum(myPDF)/np.sum(df_kde.values)
        


    # Normalize probability densities
    zi = (df_kde.values-df_kde.values.min())/(df_kde.values.max() - df_kde.values.min())
    

    levels = np.linspace(0,1,n_levels+1)

    colors = np.zeros((n_levels,4))
    colors = cmap(np.linspace(0.0,1.0,n_levels))
    colors[:,-1] = alpha
    
    # Filled contour plot
    cs = axs0.contourf(zi,
                       extent=[mmin, mmax, Dmin, Dmax],
                       origin='lower',
                       levels=levels,
                       colors=colors,
                       antialiased=True)
    
    for c in cs.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000000000001) # see https://github.com/matplotlib/matplotlib/issues/9574

   
    line_colors = colors.copy()
    line_colors[:,:-1] = [0,0,0]
    line_colors[:,-1] = np.linspace(0,1,n_levels)

    # Plot marginal distributions
    axs1.plot(df_kde.columns,
              df_kde.sum(axis=0)*np.mean(np.diff(df_kde.index.values)),
              color=colors[-1,:],
              linestyle='-',
              linewidth=0.75,
              zorder=-1)
    axs2.plot(df_kde.sum(axis=1)*np.mean(np.diff(df_kde.columns.values)),
              df_kde.index,
              color=colors[-1,:],
              linestyle='-',
              linewidth=0.75,
              zorder=-1)
    
    
    ### Main image ###
    axs0.set_xlim([mmin,mmax])
    axs0.set_ylim([Dmin,Dmax])
    
    axs0.set_xlabel('mass [kDa]',color='black')
    axs0.set_ylabel('D [um2/s]',color='black')
    
    axs0.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    axs0.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    
    logticks = axs0.get_yticks().tolist()
    axs0.yaxis.set_major_locator(mpl.ticker.FixedLocator(logticks))
    axs0.set_yticklabels([10**x for x in logticks])
    
    # Workaround to force log-scaled minor ticks
    minor_logticks = np.concatenate((np.log10(np.linspace(2,9,8)/1000.),
                                     np.log10(np.linspace(2,9,8)/100.),
                                     np.log10(np.linspace(2,9,8)/10.),
                                     np.log10(np.linspace(2,9,8)))) 
    
    axs0.yaxis.set_minor_locator(mpl.ticker.FixedLocator(minor_logticks)) 
    ##################
    
    ### Marginal distribution on top ###
    axs1.set_xlim([mmin,mmax])
    axs1.set_ylim(bottom=0)
    

    axs1.yaxis.set_minor_locator(plt.NullLocator())
    axs1.yaxis.set_major_locator(plt.NullLocator())    
    
    axs1.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        
    axs1.spines['right'].set_visible(False)
    axs1.spines['top'].set_visible(False)
    ##################
    
    ### Marginal distribution on the right ###
    axs2.set_ylim([Dmin,Dmax])
    axs2.set_xlim(left=0)
    
    axs2.set_yticklabels([])    
    
    axs2.spines['top'].set_visible(False)
    axs2.spines['right'].set_visible(False)
    
    axs2.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    axs2.yaxis.set_minor_locator(mpl.ticker.FixedLocator(minor_logticks))
    
    axs2.xaxis.set_minor_locator(plt.NullLocator())
    axs2.xaxis.set_major_locator(plt.NullLocator())
    ##################
    
    ### Colorbar ###
    kw = {'spacing':  'proportional' }
    cbar = fig.colorbar(cs,cax=cax, **kw)
    cax.set_title('density\n[norm.]')
    cax.xaxis.set_minor_locator(plt.NullLocator())
    cax.xaxis.set_major_locator(plt.NullLocator())

    cbar.set_ticks(levels[0::3])
    
    cax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(3))
    ##################
    
    
    if show == True:
        plt.show()

    return fig, [axs0,axs1,axs2,cax], df_kde, dat