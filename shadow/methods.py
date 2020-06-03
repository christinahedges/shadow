'''a load of methods we might switch out'''
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.optimize import minimize
from patsy import dmatrix
from tqdm import tqdm
import lightkurve as lk
import matplotlib.pyplot as plt
from matplotlib import animation

from scipy import sparse

import theano.tensor as tt
import theano
import pymc3 as pm
import exoplanet as xo


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




def simple_vsr(obs, gradient=False):
    """Simple estimate of Variable Scan rate

    Parameters
    ----------
    obs : shadow.Observation
        Input observation

    Returns
    -------
    m_spat : np.ndarray
        Array of the mean variable scan rate
    g_spat : np.ndarray
        Array of the gradient of the variable scan rate
    """

    #Cut out
    dat = np.copy((obs.sci/obs.flat))[:, obs.spatial.reshape(-1)][:, :, obs.spectral.reshape(-1)]
    err = np.copy((obs.err/obs.flat))[:, obs.spatial.reshape(-1)][:, :, obs.spectral.reshape(-1)]

    # Divide through by average spectrum
    err /= np.atleast_3d(np.average(dat, weights=1/err, axis=1)).transpose([0, 2, 1])
    dat /= np.atleast_3d(np.average(dat, weights=1/err, axis=1)).transpose([0, 2, 1])

    ff = np.average(dat[~obs.in_transit], weights=(1/err)[~obs.in_transit], axis=0)
    ff = np.atleast_3d(ff).transpose([2, 0, 1]) * np.ones(obs.data.shape)
#    ff[obs.cosmic_rays] = 1
    ff[obs.error/obs.data > 0.1] = 1

    dat /= ff
    err /= ff

    x = np.linspace(0.5, -0.5, dat.shape[1])
    x_p = np.linspace(-0.5, 0.5, dat.shape[1]*10)

    # Mean spatial model
    m_spat = np.zeros((dat.shape[0], dat.shape[1]))
    if gradient:
            # Gradient of spatial model
            g_spat = np.zeros((dat.shape[0], dat.shape[1]))

    for idx in tqdm(range(dat.shape[0]), desc='Building Simple VSR'):
        l = lk.LightCurve(x, np.average(dat[idx], weights=1/err[idx], axis=1)).normalize()
        m_spat[idx, :] = l.flux
        if gradient:
            r = lk.RegressionCorrector(l)
            dm = lk.designmatrix.create_sparse_spline_matrix(x, knots=list(np.linspace(-0.5, 0.5, int(dat.shape[1]/2))[1:-1])).append_constant()
            dm_p = lk.designmatrix.create_sparse_spline_matrix(x_p, knots=list(np.linspace(-0.5, 0.5, int(dat.shape[1]/2))[1:-1])).append_constant()

            _ = r.correct(dm, sigma=1e10)
            model = dm_p.X.dot(r.coefficients)
            g_spat[idx, :] = np.gradient(model, x_p)[::10]

    m_spat = np.atleast_3d(m_spat) * np.ones(dat.shape)
    #g_spat = np.atleast_3d(g_spat) * np.ones(dat.shape)
    if gradient:
        return m_spat, g_spat
    return m_spat

def simple_spectrum(obs):
    """Simple estimate of spectrum

    Parameters
    ----------
    obs : shadow.Observation
        Input observation

    Returns
    -------
    m_spat : np.ndarray
        Array of the mean spectrum
    g_spat : np.ndarray
        Array of the gradient of the spectrum
    """

    #Cut out
    dat = np.copy((obs.sci/obs.flat))[:, obs.spatial.reshape(-1)][:, :, obs.spectral.reshape(-1)]
    err = np.copy((obs.err/obs.flat))[:, obs.spatial.reshape(-1)][:, :, obs.spectral.reshape(-1)]

    dat /= obs.vsr_mean
    err /= obs.vsr_mean

    m_spec = np.average(dat[~obs.in_transit][(~obs.in_transit).sum()//2], weights=1/err[~obs.in_transit][(~obs.in_transit).sum()//2], axis=0)
    m_spec /= np.mean(m_spec)

    x = np.linspace(0.5, -0.5, dat.shape[2])
    x_p = np.linspace(-0.5, 0.5, dat.shape[2]*10)
    l = lk.LightCurve(x, m_spec)
    l = l.normalize()
    r = lk.RegressionCorrector(l)
    dm = lk.designmatrix.create_spline_matrix(x, knots=list(np.linspace(-0.5, 0.5, int(dat.shape[2]/1.5))[1:-1])).append_constant()
    dm_p = lk.designmatrix.create_spline_matrix(x_p, knots=list(np.linspace(-0.5, 0.5, int(dat.shape[2]/1.5))[1:-1])).append_constant()
    _ = r.correct(dm, sigma=1e10)

    model = dm_p.X.dot(r.coefficients)
    g_spec = np.gradient(model, x_p)[::10]
    m_spec = (np.atleast_3d(m_spec) * np.ones(dat.shape).transpose([0, 2, 1])).transpose([0, 2, 1])
    return m_spec, g_spec
#
#
# def vsr(obs):
#     """ Build a full model of the vsr
#
#     Parameters
#     ----------
#     obs : shadow.Observation
#         Input observation
#
#     Returns
#     -------
#     vsr_model : np.ndarray
#         Array of the mean variable scan rate
#     gradients : np.ndarray
#         Array of the gradient of the variable scan rate
#     """
#     frame = np.copy(obs.data)/obs.basic_model
#     frame_err = np.copy(obs.error)/obs.basic_model
#     frame_err[obs.cosmic_rays] = 1e10
#     frame_err[obs.error/obs.data > 0.01] = 1e10
#
#     frame_err /= np.average(frame, weights=1/frame_err, axis=0)
#     frame /= np.average(frame, weights=1/frame_err, axis=0)
#
#     frame_err[frame_err < 1e-4] = 1e10
#
#     # Shared variables help us loop over the model
#     y = theano.shared(frame[0].ravel())
#     ye = theano.shared(frame_err[0].ravel())
#     g_spat_t = theano.shared(obs.vsr_grad_simple[0])
#     with pm.Model() as model:
#         X_t = tt.as_tensor(obs.X[0]).flatten()
#
#         # Stellar spectrum gradient
#         pg = pm.Normal("pixel_gradient", mu=g_spat_t.eval(),
#                                             sd=np.ones(obs.data.shape[1]),
#                                             testval=g_spat_t.eval(),
#                                             shape=obs.data.shape[1])
#
#         # Reshaping to the correct dimensions of the image
#         pg_2d = (pg + tt.zeros(obs.data[0].T.shape)).T
#         pg_flat = pg_2d.flatten()
#
#         # Design matrix with all the shifts and tilts
#         A = tt.stack([
#                 X_t,
#                  pg_flat,
#                  pg_flat * X_t,
#                  tt.ones_like(pg_flat),
#                 ]).T
#
#         # Linear algebra to find the best fitting shifts
#         sigma_w_inv = A.T.dot(A/ye[:, None]**2)
#         B = A.T.dot((y/ye**2))
#         w = pm.Deterministic('w', tt.slinalg.solve(sigma_w_inv, B))
#         y_model = A.dot(w)
#         pm.Normal("obs", mu=y_model, sd=ye, observed=y)
#
#
#         vsr_grad_model = np.zeros(obs.data.shape)
#         gradients = np.zeros((obs.data.shape[0], obs.data.shape[1]))
#         ws = np.zeros((obs.data.shape[0], 4))
#
#         for idx in tqdm(range(obs.data.shape[0]), desc='Building Gradient VSR model'):
#             with model:
#                 # Optimize to find the best fitting spectral gradient
#                 y.set_value(frame[idx].ravel())
#                 ye.set_value(frame_err[idx].ravel())
#                 g_spat_t.set_value(obs.vsr_grad_simple[idx])
#
#                 map_soln = xo.optimize(start=model.test_point, vars=[pg], verbose=False)
#                 vsr_grad_model[idx] = xo.eval_in_model(y_model, point=map_soln).reshape(obs.data.shape[1:])
#                 gradients[idx] = map_soln['pixel_gradient']
#                 ws[idx] = map_soln['w']
#
#             # fig, ax = plt.subplots(1, 2, figsize=(5, 10))
#             # ax[0].imshow(frame[idx], vmin=0.98, vmax=1.02);
#             # im = ax[1].imshow(frame[idx]/vsr_grad_model[idx], vmin=0.98, vmax=1.02)
# #            plt.colorbar(im, ax=ax[1])
#             #plt.plot(frame[idx].ravel(), ms=0.1, ls='', marker='.')
#             #plt.ylim(0.95, 1.05)
#             #plt.figure()
#             #plt.plot(gradients[idx] - obs.vsr_grad_simple[idx])
# #            break
#     return vsr_grad_model, gradients


def vsr(obs):
    """ Build a full model of the vsr

    Parameters
    ----------
    obs : shadow.Observation
        Input observation

    Returns
    -------
    vsr_grad_model : np.ndarray
        Array of the mean variable scan rate shifts
    """

    frames = obs.data / (obs.basic_model)
    frames_err = obs.error  / (obs.basic_model)
    frames_err[obs.cosmic_rays] = 1e10
    frames_err[obs.error/obs.data > 0.1] = 1e10


    ff = np.average(frames[~obs.in_transit], weights=(1/frames_err)[~obs.in_transit], axis=0)
    ff = np.atleast_3d(ff).transpose([2, 0, 1]) * np.ones(obs.data.shape)
    ff[obs.cosmic_rays] = 1
    ff[obs.error/obs.data > 0.1] = 1

    frames /= ff
    frames_err /= ff

    corr = (np.atleast_3d(np.average(frames, weights=1/frames_err, axis=(1))).transpose([0, 2, 1]) * np.ones(obs.data.shape))
    frames /= corr
    frames_err /= corr
#    plt.imshow(frames[0], vmin=0.9, vmax=1.1)

#    return None

    knots = np.linspace(-0.5, 0.5, int(obs.data.shape[1]))
    spline = lk.designmatrix.create_sparse_spline_matrix(obs.Y[0].ravel(), knots=knots)
    A = sparse.hstack([sparse.csr_matrix(np.ones(np.product(obs.data.shape[1:]))).T,
                        sparse.csr_matrix(obs.X[0].ravel()).T,
                        spline.X,
                        spline.X.T.multiply(obs.X[0].ravel()).T,
                        spline.X.T.multiply(obs.X[0].ravel()**2).T], format='csr')


    dm = lk.designmatrix.SparseDesignMatrix(A)
    vsr_grad_model = np.zeros(obs.data.shape)
    ws = []
    prior_sigma = np.ones(A.shape[1]) * 100
    prior_mu = np.zeros(A.shape[1])
    prior_mu[0] = 1
    prior_sigma[1] = 0.05
    for idx in tqdm(range(obs.data.shape[0]), desc='Building Detailed VSR'):
        y = frames[idx].ravel()
        ye = frames_err[idx].ravel()

        # Linear algebra to find the best fitting shifts
        sigma_w_inv = A.T.dot(A/ye[:, None]**2)
        sigma_w_inv += np.diag(1. / prior_sigma**2)
        B = A.T.dot((y/ye**2))
        B += (prior_mu / prior_sigma**2)
        w = np.linalg.solve(sigma_w_inv, B)
        ws.append(w)
        vsr_grad_model[idx] = A.dot(w).reshape(frames[0].shape)

    return vsr_grad_model, ws

def spectrum(obs):
    """ Build a spectrum estimate """
    dat = (obs.data/obs.basic_model)
    dat /= obs.vsr_grad_model
    err = (obs.error/obs.basic_model)
    err /= obs.vsr_grad_model

    y = (dat).ravel()
    ye = (err).ravel()
    ye[obs.cosmic_rays.ravel()] = 1e10
    ye[ye/y > 0.01] =1e10
    ye[ye < 1e-4] = 1e10


    in_transit = obs.in_transit

    xshift = obs.xshift
    xshift -= np.mean(xshift)
    xshift /= (np.max(xshift) - np.min(xshift))
    xshift = np.atleast_3d(xshift).transpose([1, 0, 2]) * np.ones(obs.data.shape)

    # This will help us mask the transit later.
    cadence_mask = (np.atleast_3d(~in_transit).transpose([1, 0, 2]) * np.ones(obs.data.shape, bool)).ravel()

    with pm.Model() as model:
        # Track each dimension in theano tensors
        X_t = tt.as_tensor(obs.X[~in_transit]).flatten()
        Y_t = tt.as_tensor(obs.Y[~in_transit]).flatten()
        T_t = tt.as_tensor(obs.T[~in_transit]).flatten()
        xshift_t = tt.as_tensor(xshift[~in_transit]).flatten()
        g_spat_t = tt.as_tensor((np.atleast_3d(obs.spec_grad_simple).transpose([0, 2, 1]) * np.ones(obs.data.shape))[~in_transit]).flatten()

        # Stellar spectrum gradient
        sg = pm.Normal("spectrum_gradient", mu=obs.spec_grad_simple,
                                            sd=np.ones(obs.data.shape[2]) * 1,
                                            testval=obs.spec_grad_simple,
                                            shape=obs.data.shape[2])

        # Reshaping to the correct dimensions of the image
        sg_3d = sg + tt.zeros(obs.data[~in_transit].shape)
        sg_flat = sg_3d.flatten()

        # Design matrix with all the shifts and tilts
        A = tt.stack([

                  # Zeropoints and trends
                  Y_t,
                  T_t,
                  Y_t * X_t,
                  Y_t * X_t * T_t,

                   # Spectrum tilts
                   sg_flat,
                   sg_flat * Y_t,
                   sg_flat * X_t * Y_t,

                  # Spectrum shifts
                   sg_flat * xshift_t,
                   sg_flat * xshift_t**2,
                   sg_flat * xshift_t * X_t,

                  # Spectrum stretches
                   tt.abs_(sg_flat) * Y_t,
                   tt.abs_(sg_flat) * xshift_t * X_t,

                 tt.ones_like(sg_flat),
                 ]).T


        # Linear algebra to find the best fitting shifts
        sigma_w_inv = A.T.dot(A/ye[cadence_mask, None]**2)
        B = A.T.dot((y[cadence_mask]/ye[cadence_mask]**2))
        w = pm.Deterministic('w', tt.slinalg.solve(sigma_w_inv, B))
        y_model = A.dot(w)

        pm.Normal("obs", mu=y_model, sd=ye[cadence_mask], observed=y[cadence_mask])
        # Optimize to find the best fitting spectral gradient
        map_soln = xo.optimize(start=model.test_point, vars=[sg])

    sg_n = (np.atleast_3d(map_soln['spectrum_gradient']) * np.ones(dat.shape).transpose([0, 2, 1])).transpose([0, 2, 1])
    A = np.vstack([
               obs.Y.ravel(),
               obs.T.ravel(),
               obs.Y.ravel() * obs.X.ravel(),
               obs.Y.ravel() * obs.X.ravel() * obs.T.ravel(),
                sg_n.ravel(),
                (sg_n * obs.Y).ravel(),
                (sg_n * obs.X * obs.Y).ravel(),
                (sg_n * xshift).ravel(),
                (sg_n * xshift**2).ravel(),
                (sg_n * xshift * obs.Y).ravel(),
                (np.abs(sg_n) * obs.Y).ravel(),
                (np.abs(sg_n) * obs.Y * xshift).ravel(),
                np.ones(np.product(dat.shape))
              ]).T
    model_n = A.dot(map_soln['w']).reshape(obs.data.shape)
    return model_n


# def vsr(obs):
#     """Build a variable scan model"""
#     # Better VSR Model
#     vsr_model = np.zeros((obs.nt, obs.nsp, obs.nwav))
#     sp_knots = np.linspace(obs.Y.min() + 1, obs.Y.max() - 1, int(obs.nsp/1.5))
#
#     xcent = obs.X.mean()
#
#     A = spline(obs.Y[0].ravel(), sp_knots)
#     A1 = A.toarray() * np.atleast_2d((obs.X[0] - xcent).ravel()).T
#     A2 = A.toarray() * np.atleast_2d(((obs.X[0] - xcent)**2).ravel()).T
#     A = sparse.hstack([A, A1, A2], format='csr')
#
#     for idx in tqdm(np.arange(0, obs.nt), desc='Building VSR'):
#
#         dat = np.copy(obs.data[idx])
#         err = np.copy(obs.error[idx])
#
#         err[err/dat > 0.1] = np.inf
#
#         dat_corr = dat/np.nanmedian(dat, axis=0)
#         err_corr = err/np.nanmedian(dat, axis=0)
#
#         vsr_model[idx] =  A.dot(get_weights(A, dat_corr.ravel(), err_corr.ravel())).reshape((obs.nsp, obs.nwav))
#     norm = np.atleast_3d(np.mean(vsr_model, axis=(1, 2))).transpose([1, 0, 2])
#     return vsr_model/norm

# def simple_vsr(observation):
#     ''' Simple variable scan rate
#     '''
#     sci = np.copy(observation.sci)
#     err = np.copy(observation.err)
#     bad = (err/sci > 0.01)
#     sci[bad], err[bad] = np.nan, np.nan
#     mask = observation.mask
# #    mask = np.ones(sci.shape, bool)
#     vsr = np.zeros(((observation.gimage).sum(), observation.ns))
#     for f, frame in enumerate(sci / mask.astype(float)):
#         frame[~np.isfinite(frame)] = np.nan
#         frame /= np.atleast_2d(np.nanmedian(frame, axis=0))
#         vsr[f, :] = np.nanmedian(frame, axis=1)
#         vsr[f, :] /= np.nanmedian(vsr[f, :])
# #    vsr[(vsr == 0) | ~np.isfinite(vsr)] = 1
#     vsr[~np.isfinite(vsr)] = np.nan
#
#     # All should have the same amount of flux...
#     corr = np.atleast_2d(np.nansum(vsr, axis=1)).T
#     corr /= np.nanmedian(corr)
#
#     vsr /= corr
#     vsr /= np.nanmedian(vsr)
#     vsr = np.atleast_3d(vsr)
#     vsr[~np.isfinite(vsr)] = 1
#     return vsr


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


def animate(data, scale='linear', output='out.mp4', **kwargs):
    '''Create an animation of all the frames in `data`.

    Parameters
    ----------
    data : np.ndarray
        3D np.ndarray
    output : str
        File to output mp4 to
    '''
    fig, ax = plt.subplots(figsize=(6, 6))
    idx = 0
    if scale is 'log':
        dat = np.log10(np.copy(data))
    else:
        dat = data
    cmap = plt.get_cmap('Greys_r')
    cmap.set_bad('black')
    if 'vmax' not in kwargs:
        kwargs['vmin'] = np.nanpercentile(dat, 70)
        kwargs['vmax'] = np.nanpercentile(dat, 99.9)
    im = ax.imshow(dat[idx], origin='bottom', cmap=cmap,
                   **kwargs)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.axis('off')

    def animate(idx):
        im.set_data(dat[idx])
        return im,

    anim = animation.FuncAnimation(fig, animate, frames=len(
        dat), interval=30, blit=True)
    anim.save(output, dpi=150)
