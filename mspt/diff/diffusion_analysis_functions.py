import numpy as np
from numba import jit

@jit(nopython=True,nogil=False,cache=True)
def calc_msd(x,y):
    length = len(x)
    
    MSD = np.empty((length,),dtype=np.float64)
    MSD[0] = 0.
    for lag_tau in np.arange(1, length, 1):
        temp_array =  np.full((length - lag_tau,),np.nan,dtype=np.float64)
        for idx in np.arange(length - lag_tau):
            temp_array[idx] = (x[idx] - x[idx + lag_tau])**2 +  \
                              (y[idx] - y[idx + lag_tau])**2
        MSD[lag_tau] = np.nanmean(temp_array)
    return MSD

@jit(nopython=True,nogil=False,cache=True)
def calc_jd_nth(x,y,n=1):
    length = len(x)

    jump_distance =  np.empty((length - n,),dtype=np.float64)
    for idx in np.arange(length - n):
        jump_distance[idx] = np.sqrt( (x[idx] - x[idx + n])**2 +
                                      (y[idx] - y[idx + n])**2)
                              
    return jump_distance

@jit(nopython=True,nogil=False,cache=True)
def jdd_cumul_off(parms, r, delta_t):
    D_coeff = parms[0]
    off = parms[1]
    return (1.0 - np.exp(-(r**2/(4.0 * (D_coeff *delta_t + off)))))

@jit(nopython=True,nogil=False,cache=True)
def jdd_cumul_off_2c(parms, r, delta_t):
    D_coeff_1 = parms[0]
    D_coeff_2 = parms[1]
    A_1 = parms[2]
    A_2 = 1.0 - A_1
    off = parms[3]
    return (1.0 - A_1 * np.exp( -(r**2/(4.0 * (D_coeff_1 *delta_t + off))) ) - 
            A_2 * np.exp( -(r**2/(4.0 * (D_coeff_2 *delta_t + off))) ))

@jit(nopython=True,nogil=False,cache=True)
def fit_jdd_cumul_off(parms,JDDs,length,delta_t,n_delta_t):
    residuals = np.empty((length*n_delta_t-sum(range(n_delta_t))), dtype=np.float64)
    
    i = 0
    for nlag in np.arange(1,n_delta_t+1,1):
        jdd_sorted = JDDs[nlag-1]

        residuals[i:i+jdd_sorted.size] = ( ((np.arange(jdd_sorted.size)/(jdd_sorted.size-1) ) -
                            jdd_cumul_off(parms, jdd_sorted, nlag*delta_t)) * float((length/(jdd_sorted.size))) )
        
        i += jdd_sorted.size

    return residuals

@jit(nopython=True,nogil=False,cache=True)
def fit_jdd_cumul_off_2c(parms,JDDs,length,delta_t,n_delta_t):
    residuals = np.empty((length*n_delta_t-sum(range(n_delta_t))), dtype=np.float64)
    
    i = 0
    for nlag in np.arange(1,n_delta_t+1,1):
        jdd_sorted = JDDs[nlag-1]

        residuals[i:i+jdd_sorted.size] = ( ((np.arange(jdd_sorted.size)/(jdd_sorted.size-1) ) -
                            jdd_cumul_off_2c(parms, jdd_sorted, nlag*delta_t)) * float((length/(jdd_sorted.size))) )
        
        i += jdd_sorted.size

    return residuals


def jdd_cumul(D_coeff, r, delta_t):
    return (1.0 - np.exp(-(r**2/(4.0 * D_coeff *delta_t))))


def fit_jdd_cumul(D_coeff,data,r,delta_t):
    return (data - jdd_cumul(D_coeff, r, delta_t))


def jdd_cumul_2pop(parms, r, delta_t):
    D_coeff_1 = parms[0]
    D_coeff_2 = parms[1]
    A_1 = parms[2]
    A_2 = parms[3]
    return (1.0 - A_1 * np.exp(-(r**2/(4.0 * D_coeff_1 *delta_t))) - A_2 * np.exp(-(r**2/(4.0 * D_coeff_2 *delta_t))))


def fit_jdd_cumul_2pop(parms,data,r,delta_t):
    resid = (data - jdd_cumul_2pop(parms, r, delta_t))
    restraint = np.zeros((1,))
    restraint[0] = (parms[2]+parms[3]) - 1.0
    return np.concatenate((resid, restraint*len(resid)))


def lin_fit_msd_offset_residuals(parms,data,delta_t):
    model = 4*parms[0]*(np.arange(1,len(data)+1,1)*delta_t) + 4. * parms[1]
    return data - model


@jit(nopython=True,nogil=False,cache=True)
def lin_fit_msd_offset(data,delta_t):
    '''
    Least squares fit of MSD by linear equation 'MSD = slope*time_lag + offset'.

    '''
    length = data.shape[0]
    time_lag = np.arange(1,length+1,1)*delta_t
    
    (slope, offset), SSR = np.linalg.lstsq(np.vstack( (time_lag, np.ones(length, )) ).T,
                                           data)[:2]

    return [slope, offset, SSR[0]]


def lin_fit_msd_offset_iterative(data,delta_t,max_it=10):
    '''
    Fit mean squared displacement with optimal number of time lags.
    
    For details, see Michalet, Phys Rev E Stat Nonlin Soft Matter Phys (2010).
    (10.1103/PhysRevE.82.041914 and Erratum https://doi.org/10.1103/PhysRevE.83.059904)

    Parameters
    ----------
    data : ndarray
        MSD data points, starting at first time lag.
    delta_t : float
        Time lag or frame duration .
    max_it : int, optional
        Upper limit of iterations to find optimal number of MSD points to be 
        included in fit. The default is 10.

    Returns
    -------
    list
        List containing diffusion coefficient, squared localization uncertainty,
        sum of squared residuals, and optimal number of MSD points used for the fit.

    '''
    p_min = []
    p_min.append(3)
    
    i=0
    while i < max_it:
        # Restrict number of MSD time lags
        MSD = data[:p_min[-1]]
        # Fit MSD linearly
        slope, offset, SSR = lin_fit_msd_offset(MSD,delta_t)
        
        # reduced localization error, see 10.1103/PhysRevE.82.041914
        x = np.abs( offset/(slope*delta_t) )
        
        ### Assign number of time lags included in fit
        p_est = p_min[-1]
        
        # Update number of time lags included in fit
        # See https://doi.org/10.1103/PhysRevE.83.059904
        p_new = int( np.round_(2. + 2.3*x**0.52, decimals=0) )
        # Force at least two points to be included
        if p_new <= 2:
            p_new = 2
        # Force max number of points included
        elif p_new > len(data):
            p_new = len(data)
            
        p_min.append(p_new)

        # Break if estimated number of points did not change
        if p_min[-1] == p_min[-2]:
            break
        
        i+=1   
    
    return [slope/4., offset/4., SSR, p_est]

@jit(nopython=True,nogil=False,cache=True)
def fit_msd_jdd_cumul_off_global(parms,JDDs,MSD,length,delta_t,n_delta_t):   
    resid_len_JDD = length*n_delta_t-sum(range(n_delta_t))
    residuals_JDD = np.empty((resid_len_JDD), dtype=np.float64)

    i = 0
    for nlag in np.arange(1,n_delta_t+1,1):
        jdd_sorted = JDDs[nlag-1]
        
        residuals_JDD[i:i+jdd_sorted.size] = ( ((np.arange(jdd_sorted.size)/(jdd_sorted.size-1) ) -
                            jdd_cumul_off(parms, jdd_sorted, nlag*delta_t)) * float((length/(jdd_sorted.size))) )
        
        i += jdd_sorted.size       
        
        
    residuals_MSD = ( (MSD - (4*parms[0]*(np.arange(1,MSD.size+1,1)*delta_t) + 4. * parms[1])) * \
                     (resid_len_JDD - n_delta_t)/MSD.size  * 0.5/np.mean(MSD) )
    
    residuals = np.empty((resid_len_JDD + MSD.size), dtype=np.float64)
    residuals[:resid_len_JDD] = residuals_JDD
    residuals[resid_len_JDD:] = residuals_MSD
    
    return residuals

@jit(nopython=True,nogil=False,cache=True)
def fit_msd_jdd_cumul_off_global_2c(parms,JDDs,MSD,length,delta_t,n_delta_t):
    resid_len_JDD = length*n_delta_t-sum(range(n_delta_t))
    residuals_JDD = np.empty((resid_len_JDD), dtype=np.float64)

    i = 0
    for nlag in np.arange(1,n_delta_t+1,1):
        jdd_sorted = JDDs[nlag-1]
        
        residuals_JDD[i:i+jdd_sorted.size] = ( ((np.arange(jdd_sorted.size)/(jdd_sorted.size-1) ) -
                            jdd_cumul_off_2c(parms, jdd_sorted, nlag*delta_t)) * float((length/(jdd_sorted.size))) ) # Weight each time lag equally
        
        i += jdd_sorted.size


    Dapp_MSD = parms[2] * parms[0] +  (1.0 - parms[2]) * parms[1]
    residuals_MSD = ( (MSD -  (4*Dapp_MSD*(np.arange(1,MSD.size+1,1)*delta_t) + 4. * parms[3])) * \
                     (resid_len_JDD - n_delta_t)/MSD.size  * 0.5/np.mean(MSD) ) # Weight JDD and MSD fit approx. equally
    
    residuals = np.empty((resid_len_JDD + MSD.size), dtype=np.float64)
    residuals[:resid_len_JDD] = residuals_JDD
    residuals[resid_len_JDD:] = residuals_MSD
        
    return residuals
