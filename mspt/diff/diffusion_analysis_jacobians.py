import numpy as np
from mspt.diff.diffusion_analysis_functions import calc_msd
from mspt.diff.diffusion_analysis_functions import calc_jd_nth


def jdd_jac(parms,x,y,delta_t,n_delta_t):
    Dcoeff = parms[0]
    off = parms[1]
    
    Js_dDcoeff = list()
    Js_off = list()

    for nlag in np.arange(1,n_delta_t+1,1):
        jdd = calc_jd_nth(x,y,n=nlag)
        jdd_sorted = np.sort(jdd)
        
        Js_dDcoeff.append(((np.exp(-(np.concatenate([[0.0],jdd_sorted])**2/(4.0*(off+Dcoeff * nlag*delta_t)))) * np.concatenate([[0.0],jdd_sorted])**2 * nlag*delta_t)/(4.0*(off+Dcoeff * nlag*delta_t)**2)) * float(((len(x)-1)/(jdd_sorted.size))) )
        Js_off.append(((np.exp(-(np.concatenate([[0.0],jdd_sorted])**2/(4.0*(off+Dcoeff * nlag*delta_t)))) * np.concatenate([[0.0],jdd_sorted])**2)/(4.0*(off+Dcoeff * nlag*delta_t)**2)) * float(((len(x)-1)/(jdd_sorted.size))) )
    
    Js_dDcoeff = np.concatenate((Js_dDcoeff))
    Js_off = np.concatenate((Js_off))
    J = np.empty((len(Js_dDcoeff), len(parms)))
    J[:,0] = Js_dDcoeff
    J[:,1] = Js_off
    return J


def msd_jdd_jac(parms,x,y,delta_t,n_delta_t):
    Dcoeff = parms[0]
    off = parms[1]
    
    Js_dDcoeff_JDD = list()
    Js_off_JDD = list()

    for nlag in np.arange(1,n_delta_t+1,1):
        jdd = calc_jd_nth(x,y,n=nlag)
        jdd_sorted = np.sort(jdd)
        
        Js_dDcoeff_JDD.append(((np.exp(-(np.concatenate([[0.0],jdd_sorted])**2/(4.0*(off+Dcoeff * nlag*delta_t)))) * np.concatenate([[0.0],jdd_sorted])**2 * nlag*delta_t)/(4.0*(off+Dcoeff * nlag*delta_t)**2)) * float(((len(x)-1)/(jdd_sorted.size))) )
        Js_off_JDD.append(((np.exp(-(np.concatenate([[0.0],jdd_sorted])**2/(4.0*(off+Dcoeff * nlag*delta_t)))) * np.concatenate([[0.0],jdd_sorted])**2)/(4.0*(off+Dcoeff * nlag*delta_t)**2)) * float(((len(x)-1)/(jdd_sorted.size))) )

    
    Js_dDcoeff_JDD = np.concatenate((Js_dDcoeff_JDD))
    Js_off_JDD = np.concatenate((Js_off_JDD))

    MSD = calc_msd(x,y)
    Js_dDcoeff_MSD = ( - 4.0*(np.arange(1,len(MSD[1:5])+1,1)*delta_t) * ((len(x)-1) * n_delta_t)/len(MSD[1:5])  * 0.5/np.mean(MSD[1:5]) )
    Js_off_MSD = np.asarray(( [- 4.0 * ((len(x)-1) * n_delta_t)/len(MSD[1:5])  * 0.5/np.mean(MSD[1:5])] * len(MSD[1:5]) ))
        
    Js_dDcoeff = np.concatenate((Js_dDcoeff_JDD, Js_dDcoeff_MSD))
    Js_off = np.concatenate((Js_off_JDD,Js_off_MSD))
    
    J = np.empty((len(Js_dDcoeff), len(parms)))
    J[:,0] = Js_dDcoeff
    J[:,1] = Js_off
    return J

def jdd_jac_2c(parms,x,y,delta_t,n_delta_t):
    Dcoeff_1 = parms[0]
    Dcoeff_2 = parms[1]
    A_1 = parms[2]
    A_2 = 1.0 - parms[2]
    off = parms[3]
    
    Js_dDcoeff_1_JDD = list()
    Js_dDcoeff_2_JDD = list()
    Js_A_1_JDD = list()
    Js_off_JDD = list()

    for nlag in np.arange(1,n_delta_t+1,1):
        jdd = calc_jd_nth(x,y,n=nlag)
        jdd_sorted = np.sort(jdd)
        
        Js_dDcoeff_1_JDD.append( ((A_1 * np.exp(-(np.concatenate([[0.0],jdd_sorted])**2/(4.0*(off+Dcoeff_1 * nlag*delta_t)))) * np.concatenate([[0.0],jdd_sorted])**2 * nlag*delta_t)/(4.0*(off+Dcoeff_1 * nlag*delta_t)**2)) * float(((len(x)-1)/(jdd_sorted.size))) )
        Js_dDcoeff_2_JDD.append( ((A_2 * np.exp(-(np.concatenate([[0.0],jdd_sorted])**2/(4.0*(off+Dcoeff_2 * nlag*delta_t)))) * np.concatenate([[0.0],jdd_sorted])**2 * nlag*delta_t)/(4.0*(off+Dcoeff_2 * nlag*delta_t)**2)) * float(((len(x)-1)/(jdd_sorted.size))) )
        
        Js_A_1_JDD.append( np.exp(-(np.concatenate([[0.0],jdd_sorted])**2/(4.0* (off + Dcoeff_1 * nlag*delta_t)))) )
        
        Js_off_JDD.append( ((A_1* np.exp(-(np.concatenate([[0.0],jdd_sorted])**2/(4.0*(off+Dcoeff_1 * nlag*delta_t)))) * np.concatenate([[0.0],jdd_sorted])**2)/(4.0*(off+Dcoeff_1 * nlag*delta_t)**2) + (A_2* np.exp(-(np.concatenate([[0.0],jdd_sorted])**2/(4.0*(off+Dcoeff_2 * nlag*delta_t)))) * np.concatenate([[0.0],jdd_sorted])**2)/(4.0*(off+Dcoeff_2 * nlag*delta_t)**2)) * float(((len(x)-1)/(jdd_sorted.size))) )

    
    Js_dDcoeff_1_JDD = np.concatenate((Js_dDcoeff_1_JDD))
    Js_dDcoeff_2_JDD = np.concatenate((Js_dDcoeff_2_JDD))
    
    Js_A_1_JDD = np.concatenate((Js_A_1_JDD))
    
    Js_off_JDD = np.concatenate((Js_off_JDD))
    
    J = np.empty((len(Js_dDcoeff_1_JDD), len(parms)))
    J[:,0] = Js_dDcoeff_1_JDD
    J[:,1] = Js_dDcoeff_2_JDD
    J[:,2] = Js_A_1_JDD
    J[:,3] = Js_off_JDD
    return J


def msd_jdd_jac_2c(parms,x,y,delta_t,n_delta_t):
    Dcoeff_1 = parms[0]
    Dcoeff_2 = parms[1]
    A_1 = parms[2]
    A_2 = 1.0 - parms[2]
    off = parms[3]
    
    Js_dDcoeff_1_JDD = list()
    Js_dDcoeff_2_JDD = list()
    Js_A_1_JDD = list()
    Js_off_JDD = list()

    for nlag in np.arange(1,n_delta_t+1,1):
        jdd = calc_jd_nth(x,y,n=nlag)
        jdd_sorted = np.sort(jdd)
        
        Js_dDcoeff_1_JDD.append( ((A_1 * np.exp(-(np.concatenate([[0.0],jdd_sorted])**2/(4.0*(off+Dcoeff_1 * nlag*delta_t)))) * np.concatenate([[0.0],jdd_sorted])**2 * nlag*delta_t)/(4.0*(off+Dcoeff_1 * nlag*delta_t)**2)) * float(((len(x)-1)/(jdd_sorted.size))) )
        Js_dDcoeff_2_JDD.append( ((A_2 * np.exp(-(np.concatenate([[0.0],jdd_sorted])**2/(4.0*(off+Dcoeff_2 * nlag*delta_t)))) * np.concatenate([[0.0],jdd_sorted])**2 * nlag*delta_t)/(4.0*(off+Dcoeff_2 * nlag*delta_t)**2)) * float(((len(x)-1)/(jdd_sorted.size))) )
        
        Js_A_1_JDD.append( np.exp(-(np.concatenate([[0.0],jdd_sorted])**2/(4.0* (off + Dcoeff_1 * nlag*delta_t)))) )
        
        Js_off_JDD.append( ((A_1* np.exp(-(np.concatenate([[0.0],jdd_sorted])**2/(4.0*(off+Dcoeff_1 * nlag*delta_t)))) * np.concatenate([[0.0],jdd_sorted])**2)/(4.0*(off+Dcoeff_1 * nlag*delta_t)**2) + (A_2* np.exp(-(np.concatenate([[0.0],jdd_sorted])**2/(4.0*(off+Dcoeff_2 * nlag*delta_t)))) * np.concatenate([[0.0],jdd_sorted])**2)/(4.0*(off+Dcoeff_2 * nlag*delta_t)**2)) * float(((len(x)-1)/(jdd_sorted.size))) )

    
    Js_dDcoeff_1_JDD = np.concatenate((Js_dDcoeff_1_JDD))
    Js_dDcoeff_2_JDD = np.concatenate((Js_dDcoeff_2_JDD))
    
    Js_A_1_JDD = np.concatenate((Js_A_1_JDD))
    
    Js_off_JDD = np.concatenate((Js_off_JDD))


    MSD = calc_msd(x,y)

    Js_dDcoeff_1_MSD = ( - 4.0*A_1*            (np.arange(1,len(MSD[1:5])+1,1)*delta_t) * ((len(x)-1) * n_delta_t)/len(MSD[1:5])  * 0.5/np.mean(MSD[1:5]) )
    Js_dDcoeff_2_MSD = ( - 4.0*(1.0-A_1)*      (np.arange(1,len(MSD[1:5])+1,1)*delta_t) * ((len(x)-1) * n_delta_t)/len(MSD[1:5])  * 0.5/np.mean(MSD[1:5]) )    
    
    Js_A_1_MSD = ( - 4.0*(Dcoeff_1 - Dcoeff_2)*(np.arange(1,len(MSD[1:5])+1,1)*delta_t) * ((len(x)-1) * n_delta_t)/len(MSD[1:5])  * 0.5/np.mean(MSD[1:5]) )
    
    Js_off_MSD = np.asarray(( [- 4.0 * ((len(x)-1) * n_delta_t)/len(MSD[1:5])  * 0.5/np.mean(MSD[1:5])] * len(MSD[1:5]) ))

        
    Js_dDcoeff_1 = np.concatenate((Js_dDcoeff_1_JDD, Js_dDcoeff_1_MSD))
    Js_dDcoeff_2 = np.concatenate((Js_dDcoeff_2_JDD, Js_dDcoeff_2_MSD))
    
    Js_A_1 = np.concatenate((Js_A_1_JDD, Js_A_1_MSD))
    
    Js_off = np.concatenate((Js_off_JDD,Js_off_MSD))
    
    J = np.empty((len(Js_dDcoeff_1), len(parms)))
    J[:,0] = Js_dDcoeff_1
    J[:,1] = Js_dDcoeff_2
    J[:,2] = Js_A_1
    J[:,3] = Js_off
    return J