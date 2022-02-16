"""
    mspt.diff.diffusion_analysis_jacobians.py
    
    ~~~~~~~~~~~~~~~~~~~~
    
    Supply analytical Jacobian matrices for least squares fitting to scipy.optimize
"""
import numpy as np
from numba import jit



@jit(nopython=True,nogil=False,cache=True)
def jdd_jac(parms,JDDs,length,delta_t,n_delta_t):
    Dcoeff = parms[0]
    off = parms[1]
    
    resid_len_JDD = length*n_delta_t-sum(range(n_delta_t))
    Js_dDcoeff = np.empty((resid_len_JDD), dtype=np.float64)
    Js_off = np.empty((resid_len_JDD), dtype=np.float64)

    i = 0
    for nlag in np.arange(1,n_delta_t+1,1):

        jdd_sorted = JDDs[nlag-1]
        
        Js_dDcoeff[i:i+jdd_sorted.size] = ( ((np.exp(-(jdd_sorted**2/(4.0*(off+Dcoeff * nlag*delta_t)))) * \
                             jdd_sorted**2 * nlag*delta_t)/(4.0*(off+Dcoeff * nlag*delta_t)**2)) * \
                             float(((length)/(jdd_sorted.size))) )
            
        Js_off[i:i+jdd_sorted.size] = ( ((np.exp(-(jdd_sorted**2/(4.0*(off+Dcoeff * nlag*delta_t)))) * \
                             jdd_sorted**2)/(4.0*(off+Dcoeff * nlag*delta_t)**2)) * \
                             float(((length)/(jdd_sorted.size))) )
    
        i += jdd_sorted.size
    
    
    J = np.empty((resid_len_JDD, len(parms)))
    J[:,0] = Js_dDcoeff
    J[:,1] = Js_off
    return J

@jit(nopython=True,nogil=False,cache=True)
def msd_jdd_jac(parms,JDDs,MSD,length,delta_t,n_delta_t):
    Dcoeff = parms[0]
    off = parms[1]
    
    resid_len_JDD = length*n_delta_t-sum(range(n_delta_t))
    Js_dDcoeff_JDD = np.empty((resid_len_JDD), dtype=np.float64)
    Js_off_JDD = np.empty((resid_len_JDD), dtype=np.float64)

    i = 0
    for nlag in np.arange(1,n_delta_t+1,1):

        jdd_sorted = JDDs[nlag-1]
        
        Js_dDcoeff_JDD[i:i+jdd_sorted.size] = ( ((np.exp(-(jdd_sorted**2/(4.0*(off+Dcoeff * nlag*delta_t)))) * \
                             jdd_sorted**2 * nlag*delta_t)/(4.0*(off+Dcoeff * nlag*delta_t)**2)) * \
                             float(((length)/(jdd_sorted.size))) )
            
        Js_off_JDD[i:i+jdd_sorted.size] = (     ((np.exp(-(jdd_sorted**2/(4.0*(off+Dcoeff * nlag*delta_t)))) * \
                             jdd_sorted**2)/(4.0*(off+Dcoeff * nlag*delta_t)**2)) * \
                             float(((length)/(jdd_sorted.size))) )
    
        i += jdd_sorted.size


    Js_dDcoeff_MSD = ( - 4.0*(np.arange(1,MSD.size+1,1)*delta_t) * \
                     (resid_len_JDD - n_delta_t)/MSD.size  * 0.5/np.mean(MSD) )
    Js_off_MSD = np.array( ([- 4.0 * (resid_len_JDD - n_delta_t)/MSD.size * \
                             0.5/np.mean(MSD)] * MSD.size ) )
        
    Js_dDcoeff = np.empty((len(Js_dDcoeff_JDD) + len(Js_dDcoeff_MSD) ))
    Js_off = np.empty((len(Js_off_JDD) + len(Js_off_MSD) ))
    
    Js_dDcoeff[:resid_len_JDD] = Js_dDcoeff_JDD
    Js_dDcoeff[resid_len_JDD:] = Js_dDcoeff_MSD

    Js_off[:resid_len_JDD] = Js_off_JDD
    Js_off[resid_len_JDD:] = Js_off_MSD
    
    J = np.empty((len(Js_dDcoeff), len(parms)))
    J[:,0] = Js_dDcoeff
    J[:,1] = Js_off
    return J

@jit(nopython=True,nogil=False,cache=True)
def jdd_jac_2c(parms,JDDs,length,delta_t,n_delta_t):
    Dcoeff_1 = parms[0]
    Dcoeff_2 = parms[1]
    A_1 = parms[2]
    A_2 = 1.0 - parms[2]
    off = parms[3]
    
    resid_len_JDD = length*n_delta_t-sum(range(n_delta_t))
    Js_dDcoeff_1_JDD = np.empty((resid_len_JDD ), dtype=np.float64)
    Js_dDcoeff_2_JDD = np.empty((resid_len_JDD ), dtype=np.float64)
    Js_A_1_JDD = np.empty((resid_len_JDD ), dtype=np.float64)
    Js_off_JDD = np.empty((resid_len_JDD ), dtype=np.float64)

    i = 0
    for nlag in np.arange(1,n_delta_t+1,1):

        jdd_sorted = JDDs[nlag-1]
        
        Js_dDcoeff_1_JDD[i:i+jdd_sorted.size] = ( ((A_1 * np.exp(-(jdd_sorted**2/(4.0*(off+Dcoeff_1 * nlag*delta_t)))) * \
                                                    jdd_sorted**2 * nlag*delta_t)/(4.0*(off+Dcoeff_1 * nlag*delta_t)**2)) * \
                                                    float(((length)/(jdd_sorted.size))) )
            
        Js_dDcoeff_2_JDD[i:i+jdd_sorted.size] = ( ((A_2 * np.exp(-(jdd_sorted**2/(4.0*(off+Dcoeff_2 * nlag*delta_t)))) * \
                                                    jdd_sorted**2 * nlag*delta_t)/(4.0*(off+Dcoeff_2 * nlag*delta_t)**2)) * \
                                                    float(((length)/(jdd_sorted.size))) )
        
        Js_A_1_JDD[i:i+jdd_sorted.size] = ( np.exp(-(jdd_sorted**2/(4.0*(off+Dcoeff_1 * nlag*delta_t)))) - 
                                            np.exp(-(jdd_sorted**2/(4.0*(off+Dcoeff_2 * nlag*delta_t)))) ) * \
                                            float(((length)/(jdd_sorted.size)))
        
        Js_off_JDD[i:i+jdd_sorted.size] = ( ((A_1* np.exp(-(jdd_sorted**2/(4.0*(off+Dcoeff_1 * nlag*delta_t)))) * \
                                              jdd_sorted**2)/(4.0*(off+Dcoeff_1 * nlag*delta_t)**2) + 
                                             (A_2* np.exp(-(jdd_sorted**2/(4.0*(off+Dcoeff_2 * nlag*delta_t)))) * \
                                              jdd_sorted**2)/(4.0*(off+Dcoeff_2 * nlag*delta_t)**2)) * \
                                              float(((length)/(jdd_sorted.size))) )

        i += jdd_sorted.size

    
    J = np.empty((resid_len_JDD, len(parms)))
    J[:,0] = Js_dDcoeff_1_JDD
    J[:,1] = Js_dDcoeff_2_JDD
    J[:,2] = Js_A_1_JDD
    J[:,3] = Js_off_JDD
    return J


@jit(nopython=True,nogil=False,cache=True)
def msd_jdd_jac_2c(parms,JDDs,MSD,length,delta_t,n_delta_t):
    Dcoeff_1 = parms[0]
    Dcoeff_2 = parms[1]
    A_1 = parms[2]
    A_2 = 1.0 - parms[2]
    off = parms[3]
    
    resid_len_JDD = length*n_delta_t-sum(range(n_delta_t))
    Js_dDcoeff_1_JDD = np.empty((resid_len_JDD ), dtype=np.float64)
    Js_dDcoeff_2_JDD = np.empty((resid_len_JDD ), dtype=np.float64)
    Js_A_1_JDD = np.empty((resid_len_JDD ), dtype=np.float64)
    Js_off_JDD = np.empty((resid_len_JDD ), dtype=np.float64)

    i = 0
    for nlag in np.arange(1,n_delta_t+1,1):

        jdd_sorted = JDDs[nlag-1]
        
        Js_dDcoeff_1_JDD[i:i+jdd_sorted.size] = ( ((A_1 * np.exp(-(jdd_sorted**2/(4.0*(off+Dcoeff_1 * nlag*delta_t)))) * \
                                                    jdd_sorted**2 * nlag*delta_t)/(4.0*(off+Dcoeff_1 * nlag*delta_t)**2)) * \
                                                    float(((length)/(jdd_sorted.size))) )
            
        Js_dDcoeff_2_JDD[i:i+jdd_sorted.size] = ( ((A_2 * np.exp(-(jdd_sorted**2/(4.0*(off+Dcoeff_2 * nlag*delta_t)))) * \
                                                    jdd_sorted**2 * nlag*delta_t)/(4.0*(off+Dcoeff_2 * nlag*delta_t)**2)) * \
                                                    float(((length)/(jdd_sorted.size))) )
        
        Js_A_1_JDD[i:i+jdd_sorted.size] = ( np.exp(-(jdd_sorted**2/(4.0*(off+Dcoeff_1 * nlag*delta_t)))) - 
                                            np.exp(-(jdd_sorted**2/(4.0*(off+Dcoeff_2 * nlag*delta_t)))) ) * \
                                            float(((length)/(jdd_sorted.size)))
        
        Js_off_JDD[i:i+jdd_sorted.size] = ( ((A_1* np.exp(-(jdd_sorted**2/(4.0*(off+Dcoeff_1 * nlag*delta_t)))) * \
                                              jdd_sorted**2)/(4.0*(off+Dcoeff_1 * nlag*delta_t)**2) + 
                                             (A_2* np.exp(-(jdd_sorted**2/(4.0*(off+Dcoeff_2 * nlag*delta_t)))) * \
                                              jdd_sorted**2)/(4.0*(off+Dcoeff_2 * nlag*delta_t)**2)) * \
                                              float(((length)/(jdd_sorted.size))) )

        i += jdd_sorted.size


    Js_dDcoeff_1_MSD = ( - 4.0*A_1*(np.arange(1,MSD.size+1,1)*delta_t) * \
                       ( resid_len_JDD - n_delta_t)/MSD.size  * 0.5/np.mean(MSD) )
    Js_dDcoeff_2_MSD = ( - 4.0*(1.-A_1)*(np.arange(1,MSD.size+1,1)*delta_t) * \
                       ( resid_len_JDD - n_delta_t)/MSD.size  * 0.5/np.mean(MSD) )
    
    Js_A_1_MSD = ( - 4.0*(Dcoeff_1 - Dcoeff_2)*(np.arange(1,MSD.size+1,1)*delta_t) * \
                 ( resid_len_JDD - n_delta_t)/MSD.size  * 0.5/np.mean(MSD) )
    
    Js_off_MSD = np.array( ([- 4.0 * (resid_len_JDD - n_delta_t)/MSD.size * \
                             0.5/np.mean(MSD)] * MSD.size ) )

        
    Js_dDcoeff_1 = np.empty((resid_len_JDD + MSD.size))
    Js_dDcoeff_2 = np.empty((resid_len_JDD + MSD.size))
    
    Js_A_1 = np.empty((resid_len_JDD + MSD.size))
    
    Js_off = np.empty((resid_len_JDD + MSD.size))
    
    
    Js_dDcoeff_1[:resid_len_JDD] = Js_dDcoeff_1_JDD
    Js_dDcoeff_1[resid_len_JDD:] = Js_dDcoeff_1_MSD
    Js_dDcoeff_2[:resid_len_JDD] = Js_dDcoeff_2_JDD
    Js_dDcoeff_2[resid_len_JDD:] = Js_dDcoeff_2_MSD

    Js_A_1[:resid_len_JDD] = Js_A_1_JDD
    Js_A_1[resid_len_JDD:] = Js_A_1_MSD
    
    Js_off[:resid_len_JDD] = Js_off_JDD
    Js_off[resid_len_JDD:] = Js_off_MSD
    
    
    J = np.empty((Js_dDcoeff_1.size, len(parms)))
    J[:,0] = Js_dDcoeff_1
    J[:,1] = Js_dDcoeff_2
    J[:,2] = Js_A_1
    J[:,3] = Js_off
    return J