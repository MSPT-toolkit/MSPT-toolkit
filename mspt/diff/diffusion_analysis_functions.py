import numpy as np

def calc_msd(x,y):
    coords = np.transpose(np.asarray([x,y]))
    MSD = np.zeros((coords.shape[0]-1,))
    for lag_tau in np.arange(coords.shape[0]-1):
        temp_array =  np.full((coords.shape[0] - lag_tau,),np.nan)
        for idx in np.arange(coords.shape[0] - lag_tau):
            temp_array[idx] = (x[idx] - x[idx + lag_tau])**2 +  \
                              (y[idx] - y[idx + lag_tau])**2
        MSD[lag_tau] = np.nanmean(temp_array)
    return MSD


def calc_jd(x,y):
    coords = np.transpose(np.array([x,y]))
    jump_difference = np.diff(coords,axis=0)
    jump_distance = np.sqrt(np.sum(jump_difference**2,axis=1))
    return jump_distance


def calc_jd_nth(x,y,n=1):
    coords = np.transpose(np.array([x,y]))
    jump_difference = coords[n:] - coords[:-n]
    jump_distance = np.sqrt(np.sum(jump_difference**2,axis=1))
    return jump_distance


def jdd_cumul_off(parms, r, delta_t):
    D_coeff = parms[0]
    off = parms[1]
    return (1.0 - np.exp(-(r**2/(4.0 * (D_coeff *delta_t + off)))))


def jdd_cumul_off_2c(parms, r, delta_t):
    D_coeff_1 = parms[0]
    D_coeff_2 = parms[1]
    A_1 = parms[2]
    A_2 = 1.0 - A_1
    off = parms[3]
    return (1.0 - A_1 * np.exp( -(r**2/(4.0 * (D_coeff_1 *delta_t + off))) ) - 
            A_2 * np.exp( -(r**2/(4.0 * (D_coeff_2 *delta_t + off))) ))


def fit_jdd_cumul_off(parms,x,y,delta_t,n_delta_t):
    residuals = list()
    for nlag in np.arange(1,n_delta_t+1,1):
        jdd = calc_jd_nth(x,y,n=nlag)
        jdd_sorted = np.sort(jdd)

        residuals.append( ((np.arange(jdd_sorted.size+1)/jdd_sorted.size) -
                            jdd_cumul_off(parms, np.concatenate([[0.0],jdd_sorted]), nlag*delta_t)) * float((len(x)/(jdd_sorted.size+1))) )
    
    residuals = np.concatenate((residuals))

    return residuals


def fit_jdd_cumul_off_2c(parms,x,y,delta_t,n_delta_t):
    residuals = list()
    for nlag in np.arange(1,n_delta_t+1,1):
        jdd = calc_jd_nth(x,y,n=nlag)
        jdd_sorted = np.sort(jdd)

        residuals.append( ((np.arange(jdd_sorted.size+1)/jdd_sorted.size) - 
                            jdd_cumul_off_2c(parms, np.concatenate([[0.0],jdd_sorted]), nlag*delta_t)) * float((len(x)/(jdd_sorted.size+1))) )
         
    residuals = np.concatenate((residuals))

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


def lin_fit_msd_offset(parms,data,delta_t):
    model = 4*parms[0]*(np.arange(1,len(data)+1,1)*delta_t) + 4. * parms[1]
    return data - model


def fit_msd_jdd_cumul_off_global(parms,x,y,delta_t,n_delta_t):
    
    residuals_JDD = list()

    for nlag in np.arange(1,n_delta_t+1,1):
        jdd = calc_jd_nth(x,y,n=nlag)
        jdd_sorted = np.sort(jdd)
        
        residuals_JDD.append( ((np.arange(jdd_sorted.size+1)/jdd_sorted.size) - 
                                jdd_cumul_off(parms, np.concatenate([[0.0],jdd_sorted]), nlag*delta_t)) * float(((len(x)-1)/(jdd_sorted.size))) )
        
        
    residuals_JDD = np.concatenate((residuals_JDD))

    MSD = calc_msd(x,y)
    residuals_MSD = ( (MSD[1:5] -  (4*parms[0]*(np.arange(1,len(MSD[1:5])+1,1)*delta_t) + 4. * parms[1])) * ((len(x)-1) * n_delta_t)/len(MSD[1:5])  * 0.5/np.mean(MSD[1:5]) )
    
    return np.concatenate((residuals_JDD, residuals_MSD))


def fit_msd_jdd_cumul_off_global_2c(parms,x,y,delta_t,n_delta_t):
    
    residuals_JDD = list()

    for nlag in np.arange(1,n_delta_t+1,1):
        jdd = calc_jd_nth(x,y,n=nlag)
        jdd_sorted = np.sort(jdd)
        
        residuals_JDD.append( ((np.arange(jdd_sorted.size+1)/jdd_sorted.size) -
                                jdd_cumul_off_2c(parms, np.concatenate([[0.0],jdd_sorted]), nlag*delta_t)) * float(((len(x)-1)/(jdd_sorted.size))) )
        
        
    residuals_JDD = np.concatenate((residuals_JDD))

    MSD = calc_msd(x,y)
    Dapp_MSD = parms[2] * parms[0] +  (1.0 - parms[2]) * parms[1]
    residuals_MSD = ( (MSD[1:5] -  (4*Dapp_MSD*(np.arange(1,len(MSD[1:5])+1,1)*delta_t) + 4. * parms[3])) * ((len(x)-1) * n_delta_t)/len(MSD[1:5])  * 0.5/np.mean(MSD[1:5]) )
    
    return np.concatenate((residuals_JDD,residuals_MSD))
