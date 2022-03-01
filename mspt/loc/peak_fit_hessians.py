"""
    mspt.loc.peak_fit_hessians.py
    
    ~~~~~~~~~~~~~~~~~~~~
    
    Supply analytical Jacobian matrices for MLE to scipy.optimize.minimize
"""
import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True, nogil=False, parallel=False, cache=True)
def fit_peak_DoG_mle_hessian(parms, X, Y, im, s_fixed=True):
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
    

    Js_AA = np.sum((np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                  ((-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/s)**2)
    
    Js_dxdx = np.sum((A*(A*(dx - X)**2*(s**3*np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                     (-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))**2 +
                     (im*s - offset*s - A*s*np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                     (A - A*T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))* \
                     (s**5*sx**2*np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) -
                      s**5*(dx - X)**2*np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                      s**2*sx**2*(-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)) -
                      (-1 + T)*(dx - X)**2*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))))/(s**6*sx**4))
        
    Js_dydy = np.sum((A*(A*(dy - Y)**2*(s**3*np.exp(-(dx - X)**2/(2.*sx**2) -
                     (dy - Y)**2/(2.*sy**2)) + (-1 + T)*np.exp(-((dx - X)**2/sx**2 +
                     (dy - Y)**2/sy**2)/(2.*s**2)))**2 + (im*s - offset*s -
                     A*s*np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                     (A - A*T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))* \
                     (s**5*sy**2*np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) -
                      s**5*(dy - Y)**2*np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                      s**2*sy**2*(-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)) -
                      (-1 + T)*(dy - Y)**2*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))))/(s**6*sy**4))
                                                          
    Js_TT = np.sum((A**2*np.exp(-((-dx + X)**2/(s**2*sx**2)) - (-dy + Y)**2/(s**2*sy**2)))/s**2)
    
    Js_ss = np.sum((A*(1 - T)*np.exp(-(((dx - X)**2/sx**2 +
                   (dy - Y)**2/sy**2)/s**2))*(A*(1 - T)*(-(s**2*sx**2*sy**2) +
                    sy**2*(dx - X)**2 + sx**2*(dy - Y)**2)**2 + (2*s**4*sx**4*sy**4 -
                    5*s**2*sx**2*sy**2*(sy**2*(dx - X)**2 + sx**2*(dy - Y)**2) +
                    (sy**2*(dx - X)**2 + sx**2*(dy - Y)**2)**2)*(im*s - offset*s -
                    A*s*np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                    (A - A*T)*np.exp(-((dx - X)**2/sx**2 +
                    (dy - Y)**2/sy**2)/(2.*s**2)))*np.exp(((dx - X)**2/sx**2 +
                    (dy - Y)**2/sy**2)/(2.*s**2))))/(s**8*sx**4*sy**4))
                                                           
    Js_offsetoffset = 1. * im.shape[0] * im.shape[1]

    if s_fixed:
        Js_sxsx = np.sum((A*(A*((dx - X)**2 + (dy - Y)**2)**2* \
                         (s**3*np.exp(-((dx - X)**2 + (dy - Y)**2)/(2.*sx**2)) + 
                         (-1 + T)*np.exp(-((dx - X)**2 + (dy - Y)**2)/(2.*s**2*sx**2)))**2 + 
                        ((dx - X)**2 + (dy - Y)**2)*(im*s - offset*s - 
                          A*s*np.exp(-(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)/(2.*sx**2)) + 
                         (A - A*T)*np.exp(-(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)/(2.*s**2*sx**2)))* \
                         (-(s**5*((dx - X)**2 + (dy - Y)**2)*np.exp(-((dx - X)**2 + (dy - Y)**2)/(2.*sx**2))) + 
                          3*s**5*sx**2*np.exp(-(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)/(2.*sx**2)) + 
                          3*s**2*sx**2*(-1 + T)*np.exp(-(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)/
                         (2.*s**2*sx**2)) - (-1 + T)*(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)* \
                          np.exp(-(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)/(2.*s**2*sx**2)))))/(s**6*sx**6))
    
    
    Js_Adx = np.sum(((dx - X)*((-1 + T)*np.exp(((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/2.) +
                    s**3*np.exp(((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))* \
                    np.exp(-(((1 + s**2)*(dy**2*sx**2 + dx**2*sy**2 - 2*dx*sy**2*X +
                    sy**2*X**2 - 2*dy*sx**2*Y + sx**2*Y**2))/(s**2*sx**2*sy**2)))* \
                    (-2*A*((-1 + T)*np.exp(((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/2.) +
                    s*np.exp(((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2))) +
                    (im - offset)*s*np.exp(((1 + s**2)*(dy**2*sx**2 + dx**2*sy**2 -
                    2*dx*sy**2*X + sy**2*X**2 - 2*dy*sx**2*Y + sx**2*Y**2))/(2.*s**2*sx**2*sy**2))))/(s**4*sx**2))
        
    Js_Ady = np.sum(((dy - Y)*((-1 + T)*np.exp(((dx - X)**2/sx**2 +
                    (dy - Y)**2/sy**2)/2.) + s**3*np.exp(((dx - X)**2/sx**2 +
                    (dy - Y)**2/sy**2)/(2.*s**2)))*np.exp(-(((1 + s**2)* \
                    (dy**2*sx**2 + dx**2*sy**2 - 2*dx*sy**2*X + sy**2*X**2 -
                     2*dy*sx**2*Y + sx**2*Y**2))/(s**2*sx**2*sy**2)))*(-2*A*((-1 + T)* \
                    np.exp(((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/2.) +
                    s*np.exp(((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2))) +
                    (im - offset)*s*np.exp(((1 + s**2)*(dy**2*sx**2 + dx**2*sy**2 -
                    2*dx*sy**2*X + sy**2*X**2 - 2*dy*sx**2*Y + sx**2*Y**2))/(2.*s**2*sx**2*sy**2))))/(s**4*sy**2))
                                                                             
    Js_AT = np.sum(((2*A*((-1 + T)*np.exp(((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/2.) +
                     s*np.exp(((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2))) -
                     (im - offset)*s*np.exp(((1 + s**2)*(dy**2*sx**2 + dx**2*sy**2 -
                     2*dx*sy**2*X + sy**2*X**2 - 2*dy*sx**2*Y + sx**2*Y**2))/(2.*s**2*sx**2*sy**2)))* \
                     np.exp(-((2 + s**2)*(dy**2*sx**2 + dx**2*sy**2 - 2*dx*sy**2*X +
                     sy**2*X**2 - 2*dy*sx**2*Y + sx**2*Y**2))/(2.*s**2*sx**2*sy**2)))/s**2)
        
    Js_As = np.sum(((-1 + T)*(dy**2*sx**2 + dx**2*sy**2 - s**2*sx**2*sy**2 -
                     2*dx*sy**2*X + sy**2*X**2 - 2*dy*sx**2*Y + sx**2*Y**2)* \
                    (2*A*((-1 + T)*np.exp(((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/2.) +
                     s*np.exp(((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2))) -
                     (im - offset)*s*np.exp(((1 + s**2)*(dy**2*sx**2 + dx**2*sy**2 -
                     2*dx*sy**2*X + sy**2*X**2 - 2*dy*sx**2*Y + sx**2*Y**2))/(2.*s**2*sx**2*sy**2)))* \
                     np.exp(-((2 + s**2)*(dy**2*sx**2 + dx**2*sy**2 - 2*dx*sy**2*X + sy**2*X**2 -
                     2*dy*sx**2*Y + sx**2*Y**2))/(2.*s**2*sx**2*sy**2)))/(s**5*sx**2*sy**2))
    
    Js_Aoffset = np.sum(np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                       ((-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/s)
    
    Js_Asx = np.sum(-(((dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)* \
                       (-np.exp((dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)/(2.*sx**2)) +
                        T*np.exp((dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)/(2.*sx**2)) +
                        s**3*np.exp((dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)/(2.*s**2*sx**2)))* \
                        np.exp(-(((1 + s**2)*(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2))/(s**2*sx**2)))* \
                      ((im - offset)*s*np.exp(((1 + s**2)*(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2))/(2.*s**2*sx**2)) -
                        2*A*np.exp(-(((1 + s**2)*(dx*X + dy*Y))/(s**2*sx**2)))* \
                       (s*np.exp((dx**2 + dy**2 + 2*dx*s**2*X + X**2 + 2*dy*s**2*Y + Y**2)/(2.*s**2*sx**2)) +
                        (-1 + T)*np.exp((dx**2*s**2 + dy**2*s**2 + 2*dx*X + 2*dy*Y + s**2*(X**2 + Y**2))/(2.*s**2*sx**2)))))/(s**4*sx**3)))
  
    Js_dxA = Js_Adx
    Js_dxdy = np.sum((A*(dx - X)*(dy - Y)*(A*(s**3*np.exp(-(dx - X)**2/(2.*sx**2) -
                     (dy - Y)**2/(2.*sy**2)) + (-1 + T)*np.exp(-((dx - X)**2/sx**2 +
                     (dy - Y)**2/sy**2)/(2.*s**2)))**2 - (s**5*np.exp(-(dx - X)**2/(2.*sx**2) -
                     (dy - Y)**2/(2.*sy**2)) + (-1 + T)*np.exp(-((dx - X)**2/sx**2 +
                     (dy - Y)**2/sy**2)/(2.*s**2)))*(im*s - offset*s - A*s*np.exp(-(dx - X)**2/(2.*sx**2) -
                     (dy - Y)**2/(2.*sy**2)) + (A - A*T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))))/(s**6*sx**2*sy**2))
                                                                                  
    Js_dxT = np.sum(-((A*(dx - X)*(2*A*(-1 + T)*np.exp(((dx - X)**2/sx**2 +
                      (dy - Y)**2/sy**2)/2.) + A*(s + s**3)*np.exp(((dx - X)**2/sx**2 +
                      (dy - Y)**2/sy**2)/(2.*s**2)) - (im - offset)*s*np.exp(((1 + s**2)* \
                      (dy**2*sx**2 + dx**2*sy**2 - 2*dx*sy**2*X + sy**2*X**2 - 2*dy*sx**2*Y +
                       sx**2*Y**2))/(2.*s**2*sx**2*sy**2)))*np.exp(-((2 + s**2)* \
                      (dy**2*sx**2 + dx**2*sy**2 - 2*dx*sy**2*X + sy**2*X**2 - 2*dy*sx**2*Y + sx**2*Y**2))/(2.*s**2*sx**2*sy**2)))/(s**4*sx**2)))
                                                                     
    Js_dxs = np.sum((A*(1 - T)*(dx - X)*np.exp(-((dx - X)**2/sx**2 + 
                    (dy - Y)**2/sy**2)/(2.*s**2))*(-(A*(s**2*sx**2*sy**2 -
                     sy**2*(dx - X)**2 - sx**2*(dy - Y)**2)*(s**3*np.exp(-(dx - X)**2/(2.*sx**2) -
                    (dy - Y)**2/(2.*sy**2)) + (-1 + T)*np.exp(-((dx - X)**2/sx**2 +
                    (dy - Y)**2/sy**2)/(2.*s**2)))) + (3*s**2*sx**2*sy**2 - sy**2*(dx - X)**2 -
                     sx**2*(dy - Y)**2)*(im*s - offset*s - A*s*np.exp(-(dx - X)**2/(2.*sx**2) -
                    (dy - Y)**2/(2.*sy**2)) + (A - A*T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))))/(s**7*sx**4*sy**2))
   
    Js_dxoffset = np.sum((A*(-dx + X)*(np.exp(-(dx - X)**2/(2.*sx**2) -
                         (dy - Y)**2/(2.*sy**2)) + ((-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/s**3))/sx**2)
    
    Js_dxsx = np.sum((A*(dx - X)*(-(A*((dx - X)**2 + (dy - Y)**2)*(s**3* \
                      np.exp(-((dx - X)**2 + (dy - Y)**2)/(2.*sx**2)) + (-1 + T)* \
                      np.exp(-((dx - X)**2 + (dy - Y)**2)/(2.*s**2*sx**2)))**2) -
                     (im*s - offset*s - A*s*np.exp(-(dx**2 + dy**2 - 2*dx*X + X**2 -
                      2*dy*Y + Y**2)/(2.*sx**2)) + (A - A*T)* \
                      np.exp(-(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)/(2.*s**2*sx**2)))* \
                     (-(s**5*((dx - X)**2 + (dy - Y)**2)*np.exp(-((dx - X)**2 +
                     (dy - Y)**2)/(2.*sx**2))) + 2*s**5*sx**2* \
                      np.exp(-(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)/(2.*sx**2)) +
                      2*s**2*sx**2*(-1 + T)*np.exp(-(dx**2 + dy**2 - 2*dx*X + X**2 -
                      2*dy*Y + Y**2)/(2.*s**2*sx**2)) - (-1 + T)*(dx**2 + dy**2 -
                      2*dx*X + X**2 - 2*dy*Y + Y**2)*np.exp(-(dx**2 + dy**2 - 2*dx*X +
                      X**2 - 2*dy*Y + Y**2)/(2.*s**2*sx**2)))))/(s**6*sx**5))
    
    Js_dyA = Js_Ady
    Js_dydx = Js_dxdy
    Js_dyT = np.sum(-((A*(dy - Y)*(2*A*(-1 + T)*np.exp(((dx - X)**2/sx**2 +
                      (dy - Y)**2/sy**2)/2.) + A*(s + s**3)*np.exp(((dx - X)**2/sx**2 +
                      (dy - Y)**2/sy**2)/(2.*s**2)) - (im - offset)*s* \
                      np.exp(((1 + s**2)*(dy**2*sx**2 + dx**2*sy**2 - 2*dx*sy**2*X +
                      sy**2*X**2 - 2*dy*sx**2*Y + sx**2*Y**2))/(2.*s**2*sx**2*sy**2)))* \
                      np.exp(-((2 + s**2)*(dy**2*sx**2 + dx**2*sy**2 - 2*dx*sy**2*X +
                      sy**2*X**2 - 2*dy*sx**2*Y + sx**2*Y**2))/(2.*s**2*sx**2*sy**2)))/(s**4*sy**2)))
   
    Js_dys = np.sum((A*(1 - T)*(dy - Y)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2))* \
                     (-(A*(s**2*sx**2*sy**2 - sy**2*(dx - X)**2 - sx**2*(dy - Y)**2)* \
                     (s**3*np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) + 
                      (-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))) + 
                      (3*s**2*sx**2*sy**2 - sy**2*(dx - X)**2 - sx**2*(dy - Y)**2)* \
                      (im*s - offset*s - A*s*np.exp(-(dx - X)**2/(2.*sx**2) - 
                      (dy - Y)**2/(2.*sy**2)) + (A - A*T)*np.exp(-((dx - X)**2/sx**2 +
                      (dy - Y)**2/sy**2)/(2.*s**2)))))/(s**7*sx**2*sy**4))
 
    Js_dyoffset = np.sum((A*(-dy + Y)*(np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                        ((-1 + T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/s**3))/sy**2)
   
    Js_dysx = np.sum((A*(dy - Y)*(-(A*((dx - X)**2 + (dy - Y)**2)*(s**3* \
                      np.exp(-((dx - X)**2 + (dy - Y)**2)/(2.*sx**2)) + (-1 + T)* \
                      np.exp(-((dx - X)**2 + (dy - Y)**2)/(2.*s**2*sx**2)))**2) -
                     (im*s - offset*s - A*s*np.exp(-(dx**2 + dy**2 - 2*dx*X + X**2 - 
                      2*dy*Y + Y**2)/(2.*sx**2)) + (A - A*T)*np.exp(-(dx**2 + dy**2 -
                      2*dx*X + X**2 - 2*dy*Y + Y**2)/(2.*s**2*sx**2)))*(-(s**5*((dx - X)**2 +
                     (dy - Y)**2)*np.exp(-((dx - X)**2 + (dy - Y)**2)/(2.*sx**2))) +
                      2*s**5*sx**2*np.exp(-(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)/(2.*sx**2)) +
                      2*s**2*sx**2*(-1 + T)*np.exp(-(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)/(2.*s**2*sx**2)) -
                      (-1 + T)*(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)* \
                      np.exp(-(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)/(2.*s**2*sx**2)))))/(s**6*sx**5))
    
    Js_TA = Js_AT
    Js_Tdx = Js_dxT
    Js_Tdy = Js_dyT
    Js_Ts = np.sum((A*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2))* \
                   (im - offset - A*np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                  ((A - A*T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/s +
                  (A*(1 - T)*(s**2*sx**2*sy**2 - sy**2*(dx - X)**2 - sx**2*(dy - Y)**2)* \
                   np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/(s**3*sx**2*sy**2) -
                 ((sy**2*(dx - X)**2 + sx**2*(dy - Y)**2)*(im*s - offset*s -
                   A*s*np.exp(-(dx - X)**2/(2.*sx**2) - (dy - Y)**2/(2.*sy**2)) +
                   (A - A*T)*np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2))))/(s**3*sx**2*sy**2)))/s**2)
  
    Js_Toffset = np.sum((A*np.exp(-(-dx + X)**2/(2.*s**2*sx**2) - 
                                   (-dy + Y)**2/(2.*s**2*sy**2)))/s)
    
    Js_Tsx = np.sum((A*(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)* \
                     np.exp(-((2 + s**2)*(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + 
                     Y**2))/(2.*s**2*sx**2))*(-((im - offset)*s*np.exp(((1 + s**2)* \
                    (dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2))/(2.*s**2*sx**2))) +
                     A*np.exp(-(((1 + s**2)*(dx*X + dy*Y))/(s**2*sx**2)))*((s + s**3)* \
                     np.exp((dx**2 + dy**2 + 2*dx*s**2*X + X**2 + 2*dy*s**2*Y + Y**2)/(2.*s**2*sx**2)) +
                     2*(-1 + T)*np.exp((dx**2*s**2 + dy**2*s**2 + 2*dx*X + 2*dy*Y + s**2*(X**2 + Y**2))/(2.*s**2*sx**2)))))/(s**4*sx**3))
    
    Js_sA = Js_As
    Js_sdx = Js_dxs
    Js_sdy = Js_dys
    Js_sT = Js_Ts
    Js_soffset = np.sum((A*(-1 + T)*(dy**2*sx**2 + dx**2*sy**2 - s**2*sx**2*sy**2 -
                         2*dx*sy**2*X + sy**2*X**2 - 2*dy*sx**2*Y + sx**2*Y**2)* \
                         np.exp(-((dx - X)**2/sx**2 + (dy - Y)**2/sy**2)/(2.*s**2)))/(s**4*sx**2*sy**2))
        
    Js_ssx = np.sum((A*(1 - T)*np.exp(-((dx - X)**2 + (dy - Y)**2)/(2.*s**2*sx**2))* \
                    (A*(s**2*sx**2 - (dx - X)**2 - (dy - Y)**2)*((dx - X)**2 + (dy - Y)**2)* \
                    (s**3*np.exp(-((dx - X)**2 + (dy - Y)**2)/(2.*sx**2)) + (-1 + T)* \
                     np.exp(-((dx - X)**2 + (dy - Y)**2)/(2.*s**2*sx**2))) + (3*s**2*sx**2* \
                   ((dx - X)**2 + (dy - Y)**2) - ((dx - X)**2 + (dy - Y)**2)**2)* \
                  (-(im*s) + offset*s + A*s*np.exp(-(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)/(2.*sx**2)) -
                   (A - A*T)*np.exp(-(dx**2 + dy**2 - 2*dx*X + X**2 - 2*dy*Y + Y**2)/(2.*s**2*sx**2)))))/(s**7*sx**5))
    
    Js_offsetA = Js_Aoffset
    Js_offsetdx = Js_dxoffset
    Js_offsetdy = Js_dyoffset
    Js_offsetT = Js_Toffset
    Js_offsets = Js_soffset
    Js_offsetsx = np.sum((A*((dx - X)**2 + (dy - Y)**2)*(s**3* \
                          np.exp(-((dx - X)**2 + (dy - Y)**2)/(2.*sx**2)) +
                          (-1 + T)*np.exp(-((dx - X)**2 + (dy - Y)**2)/(2.*s**2*sx**2))))/(s**3*sx**3))
    
    Js_sxA = Js_Asx
    Js_sxdx = Js_dxsx
    Js_sxdy = Js_dysx
    Js_sxT = Js_Tsx
    Js_sxs = Js_ssx
    Js_sxoffset = Js_offsetsx
    
    
    J = np.empty((len(parms),len(parms)))
    J[0, :] = [Js_AA,      Js_Adx,      Js_Ady,      Js_AT,      Js_As,      Js_Aoffset,      Js_Asx]
    J[1, :] = [Js_dxA,     Js_dxdx,     Js_dxdy,     Js_dxT,     Js_dxs,     Js_dxoffset,     Js_dxsx]
    J[2, :] = [Js_dyA,     Js_dydx,     Js_dydy,     Js_dyT,     Js_dys,     Js_dyoffset,     Js_dysx]
    J[3, :] = [Js_TA,      Js_Tdx,      Js_Tdy,      Js_TT,      Js_Ts,      Js_Toffset,      Js_Tsx]
    J[4, :] = [Js_sA,      Js_sdx,      Js_sdy,      Js_sT,      Js_ss,      Js_soffset,      Js_ssx]
    J[5, :] = [Js_offsetA, Js_offsetdx, Js_offsetdy, Js_offsetT, Js_offsets, Js_offsetoffset, Js_offsetsx]
    J[6, :] = [Js_sxA,     Js_sxdx,     Js_sxdy,     Js_sxT,     Js_sxs,     Js_sxoffset,     Js_sxsx]
    # J[7, :] = np.sum(Js_sy)
    return J
