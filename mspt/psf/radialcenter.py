# % Adapted from: radialcenter.m
# %
# % Copyright 2011-2012, Raghuveer Parthasarathy, The University of Oregon
# %
# % Calculates the center of a 2D intensity distribution.
# % Method: Considers lines passing through each half-pixel point with slope
# % parallel to the gradient of the intensity at that point.  Considers the
# % distance of closest approach between these lines and the coordinate
# % origin, and determines (analytically) the origin that minimizes the
# % weighted sum of these distances-squared.
# % 
# % Does not allow revision / re-weighting of center determination, as this 
# % gives negligibly small improvement in accuracy.  See radialcenter_r for
# % code.
# %
# % Inputs
# %   I  : 2D intensity distribution (i.e. a grayscale image)
# %        Size need not be an odd number of pixels along each dimension
# %
# % Outputs
# %   xc, yc : the center of radial symmetry,
# %            px, from px #1 = left/topmost pixel
# %            So a shape centered in the middle of a 2*N+1 x 2*N+1
# %            square (e.g. from make2Dgaussian.m with x0=y0=0) will return
# %            a center value at x0=y0=N+1.
# %            (i.e. same convention as gaussfit2Dnonlin.m and my other 
# %            particle finding functions)
# %            Note that y increases with increasing row number (i.e. "downward")
# %   sigma  : Rough measure of the width of the distribution (sqrt. of the 
# %            second moment of I - min(I));
# %            Not determined by the fit -- output mainly for consistency of
# %            formatting compared to my other fitting functions, and to get
# %            an estimate of the particle "width."  Can eliminate for speed.
# %   meand2 : weighted mean weighted distance^2 from the gradient line distance
# %            minimization (Feb. 2013).  
# %            Not necessary -- output to assess goodness of fit. 
# %            Can eliminate for speed.
# %
# %
# % Copyright 2011-2017, Raghuveer Parthasarathy

import numpy as np
import scipy.signal

def radialcenter(I):
    # Number of grid points
    (Ny, Nx) = I.shape
    N = Nx
    half_width = (N - 1) // 2

    # grid coordinates
    m1d = np.linspace(-(half_width - 0.5), half_width - 0.5, 2 * half_width)
    xm, ym = np.meshgrid(m1d, m1d)

    # Calculate derivatives along 45-degree shifted coordinates (u and v)
    # Note that y increases "downward" (increasing row number) -- we'll deal
    # with this when calculating "m" below.
    dIdu = I[:-1, 1:] - I[1:, :-1]
    dIdv = I[:-1, :-1] - I[1:, 1:]

    
    # Smoothing
    h = (1./9.) * np.ones((3, 3))
    fdu = scipy.signal.convolve2d(dIdu, h, mode='same')
    fdv = scipy.signal.convolve2d(dIdv, h, mode='same')
    dImag2 = fdu ** 2 + fdv ** 2

    # Slope of the gradient.
    m = -(fdv + fdu) / (fdu - fdv)

    # *Very* rarely, m might be NaN if (fdv + fdu) and (fdv - fdu) are both
    # zero.  In this case, replace with the un-smoothed gradient.
    nanpix = np.isnan(m)
    Nnanpix = nanpix.sum()
    if Nnanpix > 0:
        unsmoothm = (dIdv + dIdu) / (dIdu - dIdv)
        m[nanpix] = unsmoothm[nanpix]
    nanpix = np.isnan(m)
    Nnanpix = nanpix.sum()
    if Nnanpix > 0:
        m[nanpix] = 0

    # Almost as rarely, an element of m can be infinite if the smoothed u and v
    # derivatives are identical.  To avoid NaNs later, replace these with some
    # large number -- 10x the largest non-infinite slope.  The sign of the
    # infinity doesn't matter
    infpix = np.isinf(m)
    Ninfpix = infpix.sum()
    if Ninfpix > 0:
        if Ninfpix < m.size:
            m[infpix] = 10 * m[infpix].max()
        else:
            unsmoothm = (dIdv + dIdu) / (dIdu - dIdv)
            m = unsmoothm

    # Shorthand "b", which also happens to be the
    # y intercept of the line of slope m that goes through each grid midpoint
    b = ym - m * xm

    # Weighting: weight by square of gradient magnitude and inverse
    # distance to gradient intensity centroid.
    sdI2 = dImag2.sum()
    xcentroid = ((dImag2 * xm).sum()) / sdI2
    ycentroid = ((dImag2 * ym).sum()) / sdI2
    w = dImag2 / np.sqrt((xm - xcentroid) ** 2 + (ym - ycentroid) ** 2)

    # least-squares minimization to determine the translated coordinate
    # system origin (xc, yc) such that lines y = mx+b have
    # the minimal total distance^2 to the origin:
    (xc, yc) = lsradialcenterfit(m, b, w)

    # A rough measure of the particle width.
    # Not at all connected to center determination, but may be useful for tracking applications;
    # could eliminate for (very slightly) greater speed
    ##
    Isub = I - I.min()
    (px, py) = np.meshgrid(np.arange(Nx), np.arange(Ny))
    xoffset = px - xc
    yoffset = py - yc
    r2 = xoffset ** 2 + yoffset ** 2
    sigma = np.sqrt((Isub * r2).sum() / Isub.sum()) / 2.0

    return xc, yc, sigma

    
def lsradialcenterfit(m, b, w):
    # inputs m, b, w are defined on a grid
    # w are the weights for each point
    wm2p1 = w /(m**2 + 1);
    sw  = np.sum(np.sum(wm2p1));
    smmw = np.sum(np.sum(m**2 * wm2p1));
    smw  = np.sum(np.sum(m*wm2p1));
    smbw = np.sum(np.sum(m*b*wm2p1));
    sbw  = np.sum(np.sum(b*wm2p1));
    det = smw**2 - smmw*sw;
    xc = (smbw*sw - smw*sbw)/det;    # relative to image center
    yc = (smbw*smw - smmw*sbw)/det;  # relative to image center
    return xc, yc




