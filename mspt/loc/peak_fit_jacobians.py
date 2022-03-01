"""
    mspt.loc.peak_fit_jacobians.py
    
    ~~~~~~~~~~~~~~~~~~~~
    
    Supply analytical Jacobian matrices for MLE to scipy.optimize.minimize
"""
import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True, nogil=False, parallel=False, cache=True)
def fit_peak_DoG_mle_jacobian(parms, X, Y, im, s_fixed=True):
    A=parms[0]
    dx=parms[1]
    dy=parms[2]
    T=parms[3]
    s=parms[4]
    offset=parms[5]
    sx=parms[6]
    
    if s_fixed:
        sy=parms[6]
    else:
        sy=parms[7]

    Js_A = np.sum((np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                 ((-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/s)* \
                  (-im + offset + A*(np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                 ((-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/s)))
     
     
    Js_deltax = np.sum((A*(-dx + X)*(np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                ((-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/s**3)* \
                 (-im + offset + A*(np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                ((-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/s)))/sx**2)
 
                            
    Js_deltay = np.sum((A*(-dy + Y)*(np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                ((-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/s**3)* \
                 (-im + offset + A*(np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                ((-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/s)))/sy**2)
    
    
    Js_T = np.sum((A*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2))*(-im + offset +
                 A*(np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                 ((-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/s)))/s)
   

    Js_s = np.sum((A*(-1 + T)*(dy**2*sx**2 + dx**2*sy**2 - s**2*sx**2*sy**2 - 
                   2*dx*sy**2*X + sy**2*X**2 - 2*dy*sx**2*Y + sx**2*Y**2)* \
                  np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2))* \
                  (-im + offset + A*(np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                 ((-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/s)))/(s**4*sx**2*sy**2))


    Js_offset = np.sum(-im + offset + A*(np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                      ((-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/s))

    if s_fixed:
        
        Js_sx = np.sum((A*((dx - X)**2 + (dy - Y)**2)*(s**3*np.exp(-((dx - X)**2 + (dy - Y)**2)/(2.*sx**2)) + 
                       (-1 + T)*np.exp(-((dx - X)**2 + (dy - Y)**2)/(2.*s**2*sx**2)))* \
                       (-im + offset + A*(np.exp(-((dx - X)**2 + (dy - Y)**2)/(2.*sx**2)) + 
                      ((-1 + T)*np.exp(-(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)/(2.*s**2*sx**2)))/s)))/
                       (s**3*sx**3))
     
                
    else:
        Js_sx = np.sum((A*(dx - X)**2*(np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) + 
                      ((-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/s**3)* \
                       (-im + offset + A*(np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) + 
                      ((-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/s)))/sx**3 )

        
    J = np.empty((len(parms), ))
    J[0] = Js_A
    J[1] = Js_deltax
    J[2] = Js_deltay
    J[3] = Js_T
    J[4] = Js_s
    J[5] = Js_offset
    J[6] = Js_sx
    
    if not s_fixed:         
        Js_sy = np.sum((A*(dy - Y)**2*(np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) + 
                      ((-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/s**3)* \
                       (-im + offset + A*(np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) + 
                      ((-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/s)))/sy**3 )
            
        J[7] = Js_sy
        
    return J