''' a load of methods we might switch out'''
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.optimize import minimize
from patsy import dmatrix

def spline(x, knots, degree=3, include_intercept=True):
    dm_formula = "bs(x, knots={}, degree={}, include_intercept={}) - 1" \
                     "".format(list(knots), degree, include_intercept)
    return sparse.csr_matrix((dmatrix(dm_formula, {"x": x})))


def get_weights(A, flux, err):
    sigma_f_inv = sparse.diags(1/(err**2))
    sigma_w_inv = A.transpose().dot(sigma_f_inv).dot(A)
    B = A.transpose().dot(sigma_f_inv).dot(flux)
    return spsolve(sigma_w_inv, B).T

def bin_down(x, y, e, res, method=np.median):
    digit = np.digitize(x, np.linspace(x.min(), x.max(), res))
    x_b = np.asarray([np.nanmean(x[digit == d]) for d in np.unique(digit)])
    y_b = np.asarray([method(y[digit == d]) for d in np.unique(digit)])
    e_b = np.asarray([np.nansum(e[digit == d]**2)**0.5 for d in np.unique(digit)])
    e_b /= np.asarray([np.nansum(digit == d) for d in np.unique(digit)])

    e_b_2 = np.asarray([np.nanstd(y[digit == d]) for d in np.unique(digit)])
    e_b_2 /= np.asarray([np.nansum(digit == d)**0.5 for d in np.unique(digit)])

    e_b = (e_b**2 + e_b_2**2)**0.5
    return x_b[:-1], y_b[:-1], e_b[:-1]


def simple_vsr(observation):
    ''' Simple variable scan rate
    '''
    sci = np.copy(observation.sci)
    err = np.copy(observation.err)
    bad = (err/sci > 0.01)
    sci[bad], err[bad] = np.nan, np.nan
    mask = observation.mask
#    mask = np.ones(sci.shape, bool)
    vsr = np.zeros(((observation.gimage).sum(), observation.ns))
    for f, frame in enumerate(sci / mask.astype(float)):
        frame[~np.isfinite(frame)] = np.nan
        frame /= np.atleast_2d(np.nanmedian(frame, axis=0))
        vsr[f, :] = np.nanmedian(frame, axis=1)
        vsr[f, :] /= np.nanmedian(vsr[f, :])
#    vsr[(vsr == 0) | ~np.isfinite(vsr)] = 1
    vsr[~np.isfinite(vsr)] = np.nan

    # All should have the same amount of flux...
    corr = np.atleast_2d(np.nansum(vsr, axis=1)).T
    corr /= np.nanmedian(corr)

    vsr /= corr
    vsr /= np.nanmedian(vsr)
    vsr = np.atleast_3d(vsr)
    vsr[~np.isfinite(vsr)] = 1
    return vsr


def simple_mask(observation, spectral_cut=0.2, spatial_cut=0.85):
    mask = np.atleast_3d(observation.sci.mean(axis=0) > np.nanpercentile(observation.sci, 50)).transpose([2, 0, 1])
    data = observation.sci/mask
    data[~np.isfinite(data)] = np.nan

    spectral = np.nanmean(data, axis=(0, 1))
    spatial = np.nanmean(data, axis=(0, 2))

    spectral = spectral > np.nanmax(spectral) * spectral_cut
    spatial = spatial > np.nanmax(spatial) * spatial_cut

    mask = (np.atleast_3d(np.atleast_2d(spectral) * np.atleast_2d(spatial).T)).transpose([2, 0, 1])
    return mask, np.atleast_3d(np.atleast_2d(spectral)).transpose([2, 0, 1]), np.atleast_3d(np.atleast_2d(spatial.T)).transpose([2, 1, 0])
