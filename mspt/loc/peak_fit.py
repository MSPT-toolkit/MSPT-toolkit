import numpy as np
import scipy.optimize

from numba import jit

from mspt.loc.peak_fit_jacobians import fit_peak_DoG_mle_jacobian
from mspt.loc.peak_fit_hessians import fit_peak_DoG_mle_hessian

import mspt.loc.radialcenter as radialcenter



@jit(nopython=True, nogil=False, parallel=False, cache=True)
def difference_of_gaussians(A, delta_x, delta_y, T, s, offset, sx, sy ):
    DoG = A*(np.exp(-((delta_x)**2/(2*(sx**2))+(delta_y)**2/(2*(sy**2))))-((1-T)/s)*np.exp(-((delta_x)**2/(2*((s*sx)**2))+(delta_y)**2/(2*((s*sy)**2)))))+offset

    return DoG

@jit(nopython=True,  nogil=False, parallel=False, cache=True)
def difference_of_gaussians_im(amp, dx, dy, T, s, offset, sx, sy, X, Y):
    return difference_of_gaussians(amp,
    delta_x=(X - dx), delta_y=(Y - dy), T=T, s=s, offset=offset, sx=sx, sy=sy)



@jit(nopython=True, nogil=False, parallel=False, cache=True)
def _norm_logpdf(x):
    return -x**2 / 2.0 - np.log(np.sqrt(2*np.pi))


@jit(nopython=True, nogil=False, parallel=False, cache=True)
def err_nll(parms, X, Y, im):
    return np.sum(- _norm_logpdf( (difference_of_gaussians_im(parms[0], # amp
                                                              parms[1], # dx
                                                              parms[2], # dy
                                                              parms[3], # T
                                                              parms[4], # s
                                                              parms[5], # offset
                                                              parms[6], # sx
                                                              parms[6], # sy
                                                              X, Y) - im)) )


##########


# @jit(nopython=True, nogil=False, parallel=False)
def err_nlls(parms, X, Y, im):
    amp = parms[0]
    dx = parms[1]
    dy = parms[2]
    T = parms[3]
    s = parms[4]
    offset = parms[5]
    sx = parms[6]
    sy = parms[6]
    return (difference_of_gaussians_im(amp, dx, dy, T, s, offset, sx, sy, X, Y) - im).ravel()



def fit_peak_DoG_mle(
    peak,
    T_guess=None,
    s_guess=None,
    sigma_guess=None,
    offset_guess=None,
    method="trust-ncg",
    full_output=False):
    '''
    Fit peak with the difference of two concentric 2D Gaussian functions.

    Parameters
    ----------
    peak : TYPE
        DESCRIPTION.
    T_guess : TYPE, optional
        DESCRIPTION. The default is None.
    s_guess : TYPE, optional
        DESCRIPTION. The default is None.
    sigma_guess : TYPE, optional
        DESCRIPTION. The default is None.
    offset_guess : TYPE, optional
        DESCRIPTION. The default is None.
    method : TYPE, optional
        DESCRIPTION. The default is "trust-ncg".
    full_output : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    side_length = peak.shape[0] # squared, uneven ROI
    half_length = side_length // 2
    
    x = np.arange(-half_length, half_length + 1)
    X, Y = np.meshgrid(x, x)


    dx_guess, dy_guess, _ = radialcenter.radialcenter(peak)


    amp_guess = peak[half_length-1:half_length+2,half_length-1:half_length+2].mean() / ( 1.-(1.-T_guess) /s_guess )
    c_guess = amp_guess * ( 1.-(1.-T_guess ) / s_guess )
    
    if not sigma_guess:
        sigma_guess = 1.2921
    if not offset_guess:
        offset_guess = -0.0001
        
    x0 = (amp_guess, dx_guess, dy_guess, T_guess, s_guess, offset_guess, sigma_guess)
    
    # bounds = ((-np.inf, np.inf), (-lh, lh), (-lh, lh), (0.0, 1.0), (1.0, 10.0), (np.inf, -np.inf), (0.1, np.inf))

    res = scipy.optimize.minimize(
        err_nll,
        x0,
        args=(X, Y, peak),
        method=method,
        # bounds=bounds,
        jac=fit_peak_DoG_mle_jacobian,
        hess=fit_peak_DoG_mle_hessian)

    
    (amp_fit, dx_fit, dy_fit, T_fit, s_fit, offset_fit, sigma_fit) = res.x
    c_fit = amp_fit * ( 1.- ( 1.-T_fit ) / s_fit )


    if res.success==True and np.sign(c_guess) == np.sign(c_fit):
        success = True
    else:
        success = False

    residual = np.sqrt(res.fun)


    if full_output:

        peak_fit = difference_of_gaussians_im(amp_fit, dx_fit, dy_fit, T_fit, s_fit, offset_fit, sigma_fit, sigma_fit, X, Y)

        return ((c_fit, dx_fit, dy_fit, T_fit, s_fit, offset_fit, sigma_fit, sigma_fit, success, residual),
            peak_fit, res)
    
    else:
        return (c_fit, dx_fit, dy_fit, T_fit, s_fit, offset_fit, sigma_fit, sigma_fit, success, residual)
    
    