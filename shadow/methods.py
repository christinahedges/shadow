''' a load of methods we might switch out'''
import numpy as np


def simple_vsr(observation):
    ''' Simple variable scan rate
    '''
    sci = np.copy(observation.sci)
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


def simple_mask(observation, spectral_cut=0.6, spatial_cut=0.85):
    mask = np.atleast_3d(observation.sci.mean(axis=0) > np.nanpercentile(observation.sci, 50)).transpose([2, 0, 1])
    data = observation.sci/mask
    data[~np.isfinite(data)] = np.nan

    spectral = np.nanmean(data, axis=(0, 1))
    spatial = np.nanmean(data, axis=(0, 2))

    spectral = spectral > np.nanmax(spectral) * spectral_cut
    spatial = spatial > np.nanmax(spatial) * spatial_cut

    mask = (np.atleast_3d(np.atleast_2d(spectral) * np.atleast_2d(spatial).T)).transpose([2, 0, 1])
    return mask, np.atleast_3d(np.atleast_2d(spectral)).transpose([2, 0, 1]), np.atleast_3d(np.atleast_2d(spatial.T)).transpose([2, 1, 0])
