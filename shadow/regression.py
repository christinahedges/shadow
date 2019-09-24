"""Some sort of regression module for shadow"""
import copy
import numpy as np
import scipy
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve

__all__ = ['Regressor', 'RegressorCollection']

class Regressor(object):
    def __init__(self, x, mu, sigma, name):
        self.x = x
        self.mu = mu
        self.sigma = sigma
        self.name = name

    def __repr__(self):
        return 'Vector ({}) [{} points]'.format(self.name, len(self.x))

    def __getitem__(self, key):
        copy_self = copy.copy(self)
        copy_self.x = self.x[key]
        return copy_self

    def toscipy(self, npix):
        return scipy.sparse.vstack([scipy.sparse.diags(np.ones(npix)) * xs for xs in self.x])

    @property
    def shape(self):
        return self.x.shape



class RegressorCollection(object):
    def __init__(self, ls):
        self.regressors = ls

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.regressors[key]
        else:
            copy_self = copy.copy(self)
            copy_self.regressors = [i[key] for i in self]
            return copy_self

    def __repr__(self):
        return 'RegressorCollection:\n' + ''.join(['\t{}\n'.format(i.__repr__()) for i in self])

    @property
    def names(self):
        return [i.name for i in self]

    def toscipy(self, npix):
        return scipy.sparse.hstack([i.toscipy(npix) for i in self])

    def solve_w(self, y, y_err, spectrum, cadence_mask=None):
        if cadence_mask is None:
            cadence_mask = np.ones(len(y), dtype=bool)

        if not np.all([i.shape[0] == y.shape[0] for i in self]):
            raise ValueError('`y` must have same shape as Regressors.')

        npix = np.product(y_err.shape[1:])
        sigma = (np.hstack([np.ones(npix) * i.sigma for i in self]))
        mu = (np.hstack([np.ones(npix) * i.mu for i in self]))

        S = scipy.sparse.diags(spectrum[cadence_mask].ravel())
        Sigma_f_inv = scipy.sparse.diags(1/(y_err[cadence_mask].ravel()**2))
        SA = S.dot(self[cadence_mask].toscipy(npix=npix))

        Sigma_w_inv = SA.transpose().dot(Sigma_f_inv).dot(SA) + scipy.sparse.diags(1/(sigma**2))
        B = SA.transpose().dot(Sigma_f_inv).dot(y[cadence_mask].ravel()) + (mu/sigma**2)
        w_hat = spsolve(Sigma_w_inv, B)
        self.w_hat = np.array_split(w_hat, np.arange(1, len(self.regressors) + 1) * npix)[:-1]
        best_fit = [self[idx].toscipy(npix).dot(self.w_hat[idx]).reshape(y.shape) for idx in range(len(self.regressors))]
        self.best_fit = dict(zip(self.names, best_fit))

        S = scipy.sparse.diags(spectrum.ravel())
        SA = S.dot(self.toscipy(npix=npix))
        self.model = SA.dot(w_hat).reshape(y.shape)

    def plot_w(self, frame=0):
        fig, ax = plt.subplots(1, len(self.regressors) + 1, figsize=(len(self.regressors)*5, 4))

        for idx, key in enumerate(self.best_fit.keys()):
            im = ax[idx].imshow(self.best_fit[key][frame], vmin=self[idx].mu - self[idx].sigma/4, vmax=self[idx].mu + self[idx].sigma/4)
            ax[idx].set_title(self[idx].name)
            cbar = plt.colorbar(im, ax=ax[idx])

        idx += 1
        im = ax[idx].imshow(self.model[frame])
        ax[idx].set_title('Model')
        cbar = plt.colorbar(im, ax=ax[idx])
