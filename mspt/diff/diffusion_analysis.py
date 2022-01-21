import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from mspt.diff.diffusion_analysis_functions import calc_msd, lin_fit_msd_offset
from mspt.diff.diffusion_analysis_functions import fit_jdd_cumul_off, fit_jdd_cumul_off_2c, fit_msd_jdd_cumul_off_global, fit_msd_jdd_cumul_off_global_2c
from mspt.diff.diffusion_analysis_jacobians import jdd_jac, jdd_jac_2c, msd_jdd_jac, msd_jdd_jac_2c


def fit_JDD_MSD(trajectory_id, trajs_df, frame_rate=199.8, pixel_size=84.4):
        
    traj_dict = dict()
    D_jdd = dict()
    bounds_msd = [[0.0001, -np.inf],[np.inf, np.inf]]

    for d, i in enumerate(trajectory_id):
        traj = trajs_df[trajs_df['particle']==i]
        ti = np.asarray(traj['frame'])

        c = np.asarray(-traj['contrast'])
        x = np.asarray(traj['x']) * pixel_size / 1000.0 # in microns
        y = np.asarray(traj['y']) * pixel_size / 1000.0 # in microns
        
        traj_dict[d] = (traj['x'].values,traj['y'].values,-traj['contrast'].values)
        
        med_c = np.median(c)
        mean_c = np.mean(c)

        centre_times = np.median(ti)
        
       
        ### MSD fit ##########################
        MSD = calc_msd(x,y)
        res_lsq_msd = least_squares(lin_fit_msd_offset,
                                    [1.0, 0.004], # [D, offset]
                                    args=(MSD[1:5],
                                    1./frame_rate), # time lag in seconds
                                    bounds = bounds_msd,
                                    method='trf',
                                    x_scale='jac',
                                    jac='cs')

        if np.any(np.diff(MSD)[:3]<0)==False:
            MSD_check = False
        else:
            MSD_check = True
        ######################################
        
        
        ### JDD fit: 1 component #############
        jdd_1c_flag = False
        bounds_x0_1c = ([0.00001, -np.inf], [np.inf, np.inf])
        try: 
            res_lsq_jdd_n4 = least_squares(fit_jdd_cumul_off,
                                           [1.0,0.005],
                                           jac=jdd_jac,
                                           args=(x,y,1./frame_rate,4),
                                           bounds = bounds_x0_1c,
                                           method = 'dogbox',
                                           x_scale='jac')
        except: 
            try: # Restrict the offset parameter if fit failed
                jdd_1c_flag = True # flag trajectory if fit failed with initial boundary conditions
                bounds_x0_1c = ([0.00001, -0.03],[np.inf, np.inf])
                res_lsq_jdd_n4 = least_squares(fit_jdd_cumul_off,
                                               [0.5,0.005],
                                               jac=jdd_jac,
                                               args=(x,y,1./frame_rate,4),
                                               bounds = bounds_x0_1c,
                                               method = 'dogbox',
                                               x_scale='jac')
            except: # Restrict the offset parameter to non-negative values if second fit failed
                jdd_1c_flag = True # flag trajectory if fit failed with initial boundary conditions
                bounds_x0_1c = ([0.0001, 0.0],[np.inf, np.inf])
                res_lsq_jdd_n4 = least_squares(fit_jdd_cumul_off,
                                               [0.5,0.005],
                                               jac=jdd_jac,
                                               args=(x,y,1./frame_rate,4),
                                               bounds = bounds_x0_1c,
                                               method = 'trf',
                                               x_scale='jac')
        ######################################
        
        ### JDD fit: 2 components ############
        jdd_2c_flag = False
        bounds_x0_2c = ([0.00001, 0.00001, 0.0,-np.inf],[np.inf, np.inf, 1.0,np.inf])
        try: 
            res_lsq_jdd_n4_2c = least_squares(fit_jdd_cumul_off_2c,
                                              [0.1,1.0,0.5,0.005],
                                              jac=jdd_jac_2c,
                                              args=(x,y,1./frame_rate,4),
                                              bounds = bounds_x0_2c,
                                              method = 'dogbox',
                                              x_scale='jac')
        except:
            try: # Restrict the offset parameter if fit failed
                jdd_2c_flag = True # flag trajectory if fit failed with initial boundary conditions
                bounds_x0_2c = ([0.00001, 0.00001, 0.0,-0.03],[np.inf, np.inf, 1.0,np.inf])           
                res_lsq_jdd_n4_2c = least_squares(fit_jdd_cumul_off_2c,
                                                  [0.1,1.0,0.5,0.005],
                                                  jac=jdd_jac_2c,
                                                  args=(x,y,1./frame_rate,4),
                                                  bounds = bounds_x0_2c,
                                                  method = 'dogbox',
                                                  x_scale='jac')
            except: # Restrict the offset parameter to non-negative values if second fit failed
                jdd_2c_flag = True # flag trajectory if fit failed with initial boundary conditions
                bounds_x0_2c = ([0.00001, 0.00001, 0.0,0.0],[np.inf, np.inf, 1.0,np.inf])
                res_lsq_jdd_n4_2c = least_squares(fit_jdd_cumul_off_2c,
                                                  [0.1,1.0,0.5,0.005],
                                                  jac=jdd_jac_2c,
                                                  args=(x,y,1./frame_rate,4),
                                                  bounds = bounds_x0_2c,
                                                  method = 'trf',
                                                  x_scale='jac')
        ######################################


        ### Global fit MSD & JDD: 1 component
        msd_jdd_1c_flag = False  
        bounds_x0_1c = ([0.0001, -np.inf],[np.inf, np.inf])
        try:
            res_lsq_msd_jdd_1c = least_squares(fit_msd_jdd_cumul_off_global,
                                               [1.0,0.004],
                                               jac=msd_jdd_jac,
                                               args=(x,y,1./frame_rate,4),
                                               bounds = bounds_x0_1c,
                                               method = 'dogbox',
                                               x_scale='jac')
        except:
            try: # Restrict the offset parameter if fit failed
                msd_jdd_1c_flag = True
                bounds_x0_1c = ([0.0001, -0.03],[np.inf, np.inf])
                res_lsq_msd_jdd_1c = least_squares(fit_msd_jdd_cumul_off_global,
                                                   [1.0,0.004],
                                                   jac=msd_jdd_jac,
                                                   args=(x,y,1./frame_rate,4),
                                                   bounds = bounds_x0_1c,
                                                   method = 'dogbox',
                                                   x_scale='jac')
            except: # Restrict the offset parameter to non-negative values if second fit failed
                msd_jdd_1c_flag = True
                bounds_x0_1c = ([0.0001, 0.0],[np.inf, np.inf])
                res_lsq_msd_jdd_1c = least_squares(fit_msd_jdd_cumul_off_global,
                                                   [1.0,0.004],
                                                   jac=msd_jdd_jac,
                                                   args=(x,y,1./frame_rate,4),
                                                   bounds = bounds_x0_1c,
                                                   method = 'trf',
                                                   x_scale='jac')
        ######################################
        
        ### Global fit MSD & JDD: 2 components
        msd_jdd_2c_flag = False
        bounds_x0_2c = ([0.00001, 0.00001, 0.0,-np.inf],[np.inf, np.inf, 1.0,np.inf])
        try:
            res_lsq_msd_jdd_2c = least_squares(fit_msd_jdd_cumul_off_global_2c,
                                               [0.1,1.0,0.5,0.004],
                                               jac=msd_jdd_jac_2c,
                                               args=(x,y,1./frame_rate,4),
                                               bounds = bounds_x0_2c,
                                               method = 'trf',
                                               x_scale='jac')
        except:
            try: # Restrict the offset parameter if fit failed
                msd_jdd_2c_flag = True
                bounds_x0_2c = ([0.00001, 0.00001, 0.0,-0.03],[np.inf, np.inf, 1.0,np.inf])
                res_lsq_msd_jdd_2c = least_squares(fit_msd_jdd_cumul_off_global_2c,
                                                   [0.1,1.0,0.5,0.004],
                                                   jac=msd_jdd_jac_2c,
                                                   args=(x,y,1./frame_rate,4),
                                                   bounds = bounds_x0_2c,
                                                   method = 'trf',
                                                   x_scale='jac')
            except: # Restrict the offset parameter to non-negative values if second fit failed
                msd_jdd_2c_flag = True
                bounds_x0_2c = ([0.00001, 0.00001, 0.0,0.0],[np.inf, np.inf, 1.0,np.inf])
                res_lsq_msd_jdd_2c = least_squares(fit_msd_jdd_cumul_off_global_2c,
                                                   [0.1,1.0,0.5,0.004],
                                                   jac=msd_jdd_jac_2c,
                                                   args=(x,y,1./frame_rate,4),
                                                   bounds = bounds_x0_2c,
                                                   method = 'trf',
                                                   x_scale='jac')
        ######################################
        
        
        tmp_array = np.full((34),np.nan)
        
        ### Trajectory statistics
        tmp_array[0] = (ti[-1] - ti[0])+1
        tmp_array[1] = centre_times
        tmp_array[2] = med_c
        tmp_array[3] = mean_c
        ###
        
        ### MSD fit
        tmp_array[4] = res_lsq_msd.x[0]
        tmp_array[5] = res_lsq_msd.x[1]
        
        tmp_array[6] = np.sum((res_lsq_msd.fun[:])**2)/(4. - 2.)
        tmp_array[7] = MSD_check
        ###
        
        ### JDD fit: 1 component
        tmp_array[8] = res_lsq_jdd_n4.x[0]
        tmp_array[9] = res_lsq_jdd_n4.x[1]
        
        tmp_array[10] = np.sum(res_lsq_jdd_n4.fun**2)/(len(res_lsq_jdd_n4.fun) - 2.)
        tmp_array[11] = res_lsq_jdd_n4.success
        tmp_array[12] = jdd_1c_flag
        ###
        
        ### JDD fit: 2 components
        tmp_array[13] = res_lsq_jdd_n4_2c.x[0]
        tmp_array[14] = res_lsq_jdd_n4_2c.x[1]
        tmp_array[15] = res_lsq_jdd_n4_2c.x[2]
        tmp_array[16] = 1.0 - res_lsq_jdd_n4_2c.x[2]
        tmp_array[17] = res_lsq_jdd_n4_2c.x[3]
        
        tmp_array[18] = np.sum(res_lsq_jdd_n4_2c.fun**2)/(len(res_lsq_jdd_n4_2c.fun) - 4.)  
        tmp_array[19] = res_lsq_jdd_n4_2c.success
        tmp_array[20] = jdd_2c_flag
        ###
        
        ### Global fit MSD & JDD: 1 component
        tmp_array[21] = res_lsq_msd_jdd_1c.x[0]
        tmp_array[22] = res_lsq_msd_jdd_1c.x[1]
        
        tmp_array[23] = np.sum((res_lsq_msd_jdd_1c.fun[:])**2)/float(len(x) - 2)
        tmp_array[24] = res_lsq_msd_jdd_1c.success
        tmp_array[25] = msd_jdd_1c_flag
        ###
        
        ### Global fit MSD & JDD: 2 components
        tmp_array[26] = res_lsq_msd_jdd_2c.x[0]
        tmp_array[27] = res_lsq_msd_jdd_2c.x[1]
        tmp_array[28] = res_lsq_msd_jdd_2c.x[2]
        tmp_array[29] = 1.0 - res_lsq_msd_jdd_2c.x[2]
        tmp_array[30] = res_lsq_msd_jdd_2c.x[3]
        
        tmp_array[31] = np.sum((res_lsq_msd_jdd_2c.fun[:])**2)/float(len(x) - 4)
        tmp_array[32] = res_lsq_msd_jdd_2c.success
        tmp_array[33] = msd_jdd_2c_flag
        ###

        D_jdd[d] = tmp_array

    
    D_jdd_df = pd.DataFrame.from_dict(D_jdd,
                                      orient='index',
                                      columns=['len','centre frame', 'med_c','mean_c',
                                               'D_MSD_n4','off_MSD_n4', 'chi_MSD_n4' ,'MSD_check', 
                                               'D_JDD_n4', 'off_JDD_n4', 'chi_JDD_n4', 'fit_JDD_success', 'flag_JDD_n4_c1',
                                               '2c_JDD_D_1_n4', '2c_JDD_D_2_n4', '2c_JDD_A_1_n4', '2c_JDD_A_2_n4', '2c_JDD_off_n4', '2c_JDD_chi_n4', 'fit_JDD_2c_success', 'flag_JDD_n4_c2',
                                               'D_MSD_JDD_n4','off_MSD_JDD_n4', 'chi_MSD_JDD_n4','fit_MSD_JDD_1c_success', 'flag_MSD_JDD_n4_c1',
                                               'D_1_MSD_JDD_n4_c2','D_2_MSD_JDD_n4_c2','A_1_MSD_JDD_n4_c2','A_2_MSD_JDD_n4_c2', 'off_MSD_JDD_n4_c2', 'chi_MSD_JDD_n4_c2' , 'fit_MSD_JDD_2c_success', 'flag_MSD_JDD_n4_c2'])
    

    traj_df_temp = pd.DataFrame.from_dict(traj_dict,
                                          orient='index',
                                          columns=['x pos','y pos','contrast'])
    
    traj_df_temp = traj_df_temp.astype(object)
    D_jdd_df = pd.concat([D_jdd_df, traj_df_temp], axis=1)

    return D_jdd_df.copy()