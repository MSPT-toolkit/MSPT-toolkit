import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.optimize import OptimizeResult
from numba.typed import List

from mspt.diff.diffusion_analysis_functions import calc_msd, calc_jd_nth, lin_fit_msd_offset, lin_fit_msd_offset_iterative
from mspt.diff.diffusion_analysis_functions import fit_jdd_cumul_off, fit_jdd_cumul_off_2c, fit_msd_jdd_cumul_off_global, fit_msd_jdd_cumul_off_global_2c
from mspt.diff.diffusion_analysis_jacobians import jdd_jac, jdd_jac_2c, msd_jdd_jac, msd_jdd_jac_2c


def fit_JDD_MSD(trajectory_id, trajs_df, frame_rate=199.8, pixel_size=84.4, n_timelags_MSD=None, n_timelags_JDD=None):

    dict_traj = dict()
    dict_jdd_msd = dict()


    for d, i in enumerate(trajectory_id):
        traj = trajs_df[trajs_df['particle']==i]
        ti = np.asarray(traj['frame'])

        c = np.asarray(-traj['contrast'])
        x = np.asarray(traj['x']) * pixel_size / 1000.0 # in microns
        y = np.asarray(traj['y']) * pixel_size / 1000.0 # in microns
        
        dict_traj[d] = (traj['x'].values,traj['y'].values,-traj['contrast'].values)
        
        length = len(x)
        med_c = np.median(c)
        mean_c = np.mean(c)

        center_time = np.median(ti)
        
       
        ### MSD fit ##########################
        MSD = calc_msd(x,y)
        if n_timelags_MSD is None:
            res_lsq_msd = lin_fit_msd_offset_iterative(MSD[1:], # MSD[0] is 0
                                                       1./frame_rate, # time lag in seconds
                                                       max_it=10)
        else:
            slope, offset, SSR = lin_fit_msd_offset(MSD[1:], # MSD[0] is 0
                                                    1./frame_rate) # time lag in seconds
            
            res_lsq_msd = [slope/4., offset/4., SSR[0], n_timelags_MSD]
            
        # Check if first 3 MSD data points are monotonously increasing
        if np.any(np.diff(MSD)[:3]<0)==False:
            MSD_check = True
        else:
            MSD_check = False
        
        # Truncate MSD data
        MSD = MSD[1:res_lsq_msd[3]+1]
        
        # Set number of time lags included in JDD fitting
        if n_timelags_JDD is None:
            n_tau_JDD = res_lsq_msd[3]
        else:
            n_tau_JDD = n_timelags_JDD
        
        # Precalculate JDDs for different lag times for later use
        JDDs = list()
        for tau in np.arange(1,n_tau_JDD+1,1):
            jdd = calc_jd_nth(x, y, n=tau)
            jdd_sorted = np.empty((jdd.size+1,),dtype=np.float64)
            jdd_sorted[0] = 0.
            jdd_sorted[1:] = np.sort(jdd)
            JDDs.append(jdd_sorted)
            JDDs = List(JDDs)
        ######################################
        
        
        ### JDD fit: 1 component #############
        jdd_1c_flag = False

        try: 
            res_lsq_jdd = least_squares(fit_jdd_cumul_off,
                                        np.array([1.0,0.005]),
                                        jac=jdd_jac,
                                        args=(JDDs,length,1./frame_rate,n_tau_JDD),
                                        method = 'lm',
                                        x_scale='jac')
            
        except: 
            try: # Restrict the offset parameter if fit failed
                jdd_1c_flag = True # flag trajectory if fit failed with initial boundary conditions
                bounds_x0_1c = ([0.00001, -0.03],[np.inf, np.inf])
                res_lsq_jdd = least_squares(fit_jdd_cumul_off,
                                            np.array([0.5,0.005]),
                                            jac=jdd_jac,
                                            args=(JDDs,length,1./frame_rate,n_tau_JDD),
                                            bounds = bounds_x0_1c,
                                            method = 'dogbox',
                                            x_scale='jac')
            except: # Fill results dict manually if second fit failed
                res_lsq_jdd = OptimizeResult( {'x' : np.full(2, np.nan),
                                               'fun': np.array([np.nan]),
                                               'success': False} )

        ######################################
        
        ### JDD fit: 2 components ############
        jdd_2c_flag = False

        try: 
            res_lsq_jdd_2c = least_squares(fit_jdd_cumul_off_2c,
                                           np.array([0.1,1.0,0.5,0.005]),
                                           jac=jdd_jac_2c,
                                           args=(JDDs,length,1./frame_rate,n_tau_JDD),
                                           method = 'lm',
                                           x_scale='jac')
            
        except:
            try: # Restrict the offset parameter if fit failed
                jdd_2c_flag = True # flag trajectory if fit failed with initial boundary conditions
                bounds_x0_2c = ([0.00001, 0.00001, 0.0,-0.03],[np.inf, np.inf, 1.0,np.inf])           
                res_lsq_jdd_2c = least_squares(fit_jdd_cumul_off_2c,
                                               np.array([0.1,1.0,0.5,0.005]),
                                               jac=jdd_jac_2c,
                                               args=(JDDs,length,1./frame_rate,n_tau_JDD),
                                               bounds = bounds_x0_2c,
                                               method = 'dogbox',
                                               x_scale='jac')
            except: # Fill results dict manually if second fit failed
                res_lsq_jdd_2c = OptimizeResult( {'x' : np.full(4, np.nan),
                                                  'fun': np.array([np.nan]),
                                                  'success': False} )

        ######################################







        ### Global fit MSD & JDD: 1 component
        msd_jdd_1c_flag = False  

        try:
            res_lsq_msd_jdd_1c = least_squares(fit_msd_jdd_cumul_off_global,
                                               np.array([1.0,0.004]),
                                               jac=msd_jdd_jac,
                                               args=(JDDs,MSD,length,1./frame_rate,n_tau_JDD),
                                               method = 'lm',
                                               x_scale='jac')
            
        except:
            try: # Restrict the offset parameter if fit failed
                msd_jdd_1c_flag = True
                bounds_x0_1c = ([0.00001, -0.03],[np.inf, np.inf])
                res_lsq_msd_jdd_1c = least_squares(fit_msd_jdd_cumul_off_global,
                                                   np.array([1.0,0.004]),
                                                   jac=msd_jdd_jac,
                                                   args=(JDDs,MSD,length,1./frame_rate,n_tau_JDD),
                                                   bounds = bounds_x0_1c,
                                                   method = 'dogbox',
                                                   x_scale='jac')
            except: # Fill results dict manually if second fit failed
                res_lsq_msd_jdd_1c = OptimizeResult( {'x' : np.full(2, np.nan),
                                                      'fun': np.array([np.nan]),
                                                      'success': False} )

        ######################################
        
        
        
        
        
        
        ### Global fit MSD & JDD: 2 components
        msd_jdd_2c_flag = False

        try:
            res_lsq_msd_jdd_2c = least_squares(fit_msd_jdd_cumul_off_global_2c,
                                               np.array([0.1,1.0,0.5,0.004]),
                                               jac=msd_jdd_jac_2c,
                                               args=(JDDs,MSD,length,1./frame_rate,n_tau_JDD),
                                               method = 'lm',
                                               x_scale='jac')
            
        except:
            try: # Restrict the offset parameter if fit failed
                msd_jdd_2c_flag = True
                bounds_x0_2c = ([0.00001, 0.00001, 0.0,-0.03],[np.inf, np.inf, 1.0,np.inf])
                res_lsq_msd_jdd_2c = least_squares(fit_msd_jdd_cumul_off_global_2c,
                                                   np.array([0.1,1.0,0.5,0.004]),
                                                   jac=msd_jdd_jac_2c,
                                                   args=(JDDs,MSD,length,1./frame_rate,n_tau_JDD),
                                                   bounds = bounds_x0_2c,
                                                   method = 'trf',
                                                   x_scale='jac')
            except: # Fill results dict manually if second fit failed
                res_lsq_msd_jdd_2c = OptimizeResult( {'x' : np.full(4, np.nan),
                                                      'fun': np.array([np.nan]),
                                                      'success': False} )

        ######################################
        
        
        
        
        
        
        
    
        tmp_array = np.full((34),np.nan)
        
        ### Trajectory statistics ##############################################################################################################
        tmp_array[0] = length       # Trajectory length
        tmp_array[1] = center_time  # Center frame of trajectory
        tmp_array[2] = med_c        # Median contrast of trajectory
        tmp_array[3] = mean_c       # Mean contrast of trajectory
        ########################################################################################################################################
        
        ### MSD fit ############################################################################################################################
        tmp_array[4] = res_lsq_msd[0]                           # Diffusion coefficient
        tmp_array[5] = res_lsq_msd[1]                           # Localization uncertainty squared
        
        if res_lsq_msd[3] == 2:
            tmp_array[6] = 0                                    # Reduced chi squared = 0, exact solution (line through 2 datapoints)
        else:
            tmp_array[6] = res_lsq_msd[2]/(res_lsq_msd[3] - 2.) # Reduced chi squared
        tmp_array[7] = MSD_check                                # True if first 3 MSD data points are monotonously increasing
        ########################################################################################################################################
        
        ### JDD fit: 1 component ###############################################################################################################
        tmp_array[8] = res_lsq_jdd.x[0]                                        # Diffusion coefficient
        tmp_array[9] = res_lsq_jdd.x[1]                                        # Localization uncertainty squared
        
        tmp_array[10] = np.sum(res_lsq_jdd.fun**2)/(len(res_lsq_jdd.fun) - 2.) # Reduced chi squared
        tmp_array[11] = res_lsq_jdd.success                                    # True if fit successful
        tmp_array[12] = jdd_1c_flag                                            # True if fit with initial boundary conditions failed
        ########################################################################################################################################
        
        ### JDD fit: 2 components ##############################################################################################################
        tmp_array[13] = res_lsq_jdd_2c.x[0]                                          # Diffusion coefficient component 1
        tmp_array[14] = res_lsq_jdd_2c.x[1]                                          # Diffusion coefficient component 2
        tmp_array[15] = res_lsq_jdd_2c.x[2]                                          # Amplitude component 1
        tmp_array[16] = 1.0 - res_lsq_jdd_2c.x[2]                                    # Amplitude component 2
        tmp_array[17] = res_lsq_jdd_2c.x[3]                                          # Localization uncertainty squared
        
        tmp_array[18] = np.sum(res_lsq_jdd_2c.fun**2)/(len(res_lsq_jdd_2c.fun) - 4.) # Reduced chi squared
        tmp_array[19] = res_lsq_jdd_2c.success                                       # True if fit successful
        tmp_array[20] = jdd_2c_flag                                                  # True if fit with initial boundary conditions failed
        ########################################################################################################################################
        
        ### Global fit MSD & JDD: 1 component ##################################################################################################
        tmp_array[21] = res_lsq_msd_jdd_1c.x[0]                                  # Diffusion coefficient
        tmp_array[22] = res_lsq_msd_jdd_1c.x[1]                                  # Localization uncertainty squared
        
        tmp_array[23] = np.sum((res_lsq_msd_jdd_1c.fun[:])**2)/float(len(x) - 2) # Reduced chi squared
        tmp_array[24] = res_lsq_msd_jdd_1c.success                               # True if fit successful
        tmp_array[25] = msd_jdd_1c_flag                                          # True if fit with initial boundary conditions failed
        ########################################################################################################################################
        
        ### Global fit MSD & JDD: 2 components #################################################################################################
        tmp_array[26] = res_lsq_msd_jdd_2c.x[0]                                   # Diffusion coefficient component 1
        tmp_array[27] = res_lsq_msd_jdd_2c.x[1]                                   # Diffusion coefficient component 1
        tmp_array[28] = res_lsq_msd_jdd_2c.x[2]                                   # Amplitude component 1
        tmp_array[29] = 1.0 - res_lsq_msd_jdd_2c.x[2]                             # Amplitude component 2
        tmp_array[30] = res_lsq_msd_jdd_2c.x[3]                                   # Localization uncertainty squared
        
        tmp_array[31] = np.sum((res_lsq_msd_jdd_2c.fun[:])**2)/float(len(x) - 4)  # Reduced chi squared
        tmp_array[32] = res_lsq_msd_jdd_2c.success                                # True if fit successful
        tmp_array[33] = msd_jdd_2c_flag                                           # True if fit with initial boundary conditions failed
        ########################################################################################################################################

        dict_jdd_msd[d] = tmp_array

    
    df_jdd_msd = pd.DataFrame.from_dict(dict_jdd_msd,
                                        orient='index',
                                        columns=['len','center frame', 'med_c','mean_c',
                                                 'D_MSD','off_MSD', 'chi_MSD' ,'MSD_check', 
                                                 'D_JDD', 'off_JDD', 'chi_JDD', 'fit_JDD_success', 'flag_JDD_c1',
                                                 'D_1_JDD_2c', 'D_2_JDD_2c', 'A_1_JDD_2c', 'A_2_JDD_2c', 'off_JDD_2c', 'chi_JDD_2c', 'fit_JDD_2c_success', 'flag_JDD_2c',
                                                 'D_MSD_JDD','off_MSD_JDD', 'chi_MSD_JDD','fit_MSD_JDD_1c_success', 'flag_MSD_JDD_1c',
                                                 'D_1_MSD_JDD_2c','D_2_MSD_JDD_2c','A_1_MSD_JDD_2c','A_2_MSD_JDD_2c', 'off_MSD_JDD_2c', 'chi_MSD_JDD_2c' , 'fit_MSD_JDD_2c_success', 'flag_MSD_JDD_2c'])
    
    dtypes = {'len': np.uint32,
              'MSD_check': np.bool_,
              'fit_JDD_success': np.bool_,
              'flag_JDD_c1': np.bool_,
              'fit_JDD_2c_success': np.bool_,
              'flag_JDD_2c': np.bool_,
              'fit_MSD_JDD_1c_success': np.bool_,
              'flag_MSD_JDD_1c': np.bool_,
              'fit_MSD_JDD_2c_success': np.bool_,
              'flag_MSD_JDD_2c': np.bool_}
    
    df_jdd_msd = df_jdd_msd.astype(dtypes)

    # Calculate effective diffusion coefficient for 2 component JDD
    df_jdd_msd['Deff_JDD_2c'] = np.where( ( (df_jdd_msd['fit_JDD_2c_success']==True) & 
                                            (df_jdd_msd['D_1_JDD_2c']>0) & 
                                            (df_jdd_msd['D_2_JDD_2c']>0) & 
                                            (df_jdd_msd['A_1_JDD_2c'].between(0,1)) ),
                                         
                                            (df_jdd_msd['A_1_JDD_2c'] * df_jdd_msd['D_1_JDD_2c'] +
                                             df_jdd_msd['A_2_JDD_2c'] * df_jdd_msd['D_2_JDD_2c'] ),
                                            
                                             np.nan )
    
    # Select 1 or 2 component JDD fit based on reduced chi squared criteria
    # In case of non-physical fit results, choose 1 component JDD
    df_jdd_msd['Deff_JDD'] = np.where( ( (df_jdd_msd['chi_JDD_2c']<df_jdd_msd['chi_JDD']) &
                                         (~df_jdd_msd['Deff_JDD_2c'].isna()) ),
                                          df_jdd_msd['Deff_JDD_2c'],
                                          df_jdd_msd['D_JDD'])

    # Calculate effective diffusion coefficient for 2 component global MSD and JDD fit
    df_jdd_msd['Deff_MSD_JDD_2c'] = np.where( ( (df_jdd_msd['fit_MSD_JDD_2c_success']==True) & 
                                                (df_jdd_msd['D_1_MSD_JDD_2c']>0) & 
                                                (df_jdd_msd['D_2_MSD_JDD_2c']>0) & 
                                                (df_jdd_msd['A_1_MSD_JDD_2c'].between(0,1)) ),
                                             
                                                (df_jdd_msd['A_1_MSD_JDD_2c'] * df_jdd_msd['D_1_MSD_JDD_2c'] + 
                                                 df_jdd_msd['A_2_MSD_JDD_2c'] * df_jdd_msd['D_2_MSD_JDD_2c'] ),
                                                
                                                 np.nan)
    
    # Select 1 or 2 component global MSD and JDD fit based on reduced chi squared criteria
    # In case of non-physical fit results, choose 1 component JDD
    df_jdd_msd['Deff_MSD_JDD'] = np.where( ( (df_jdd_msd['chi_MSD_JDD_2c']<df_jdd_msd['chi_MSD_JDD']) &
                                             (~df_jdd_msd['Deff_MSD_JDD_2c'].isna()) ),
                                              df_jdd_msd['Deff_MSD_JDD_2c'],
                                              df_jdd_msd['D_MSD_JDD'])
        
    # Create DataFrame containing the whole trajectory information (list of x positions, y positions, and contrasts) in three columns
    traj_df_temp = pd.DataFrame.from_dict(dict_traj,
                                          orient='index',
                                          columns=['x pos','y pos','contrast'])
    # Set dtype to object as multiple values are contained in each cell
    traj_df_temp = traj_df_temp.astype(object)
    # Merge DataFrames horizontally
    df_jdd_msd = pd.concat([df_jdd_msd, traj_df_temp], axis=1)

    return df_jdd_msd