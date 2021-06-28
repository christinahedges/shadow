'''Routines to extract data from HST and correct it.
'''

import numpy as np
import warnings
import os
from glob import glob
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm
import pandas as pd
from matplotlib import animation
import matplotlib.pyplot as plt
import copy

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.convolution import Box2DKernel, convolve
from astropy.wcs import WCS, FITSFixedWarning
from astropy.stats import sigma_clipped_stats, sigma_clip
from astroquery.mast import Observations


from . import PACKAGEDIR
from . import modeling
from . import methods

from starry.extensions import from_nexsci

log = logging.getLogger('shadow')

CALIPATH = '{}{}'.format(PACKAGEDIR, '/data/calibration/')


def shadow_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'


warnings.formatwarning = shadow_formatwarning


class ShadowUserInputError(Exception):
    '''Raised if the user inputs something they shouldn't'''
    pass


class ShadowValueError(Exception):
    '''Raised if there is an exception that isn't the users fault'''
    pass


class Observation(object):
    '''Holds all the data information?'''

    def _get_headers(self, f_extn=['flt', 'ima'], propid=None):
        '''Obtain and sort the headers

        Parameters
        ----------
        dir: str
            Directory containing all the input files.
        visit: None or int
            Number of visits to use. If None, all visits
            will be carried forward. If int, only specified
            visit will be used.
        f_extn: str or list of str
            String for the file extention to used. If a list of strings is passed
            while go through this list in order until files are found.
        '''

        '''Make sure flux is in the right units! (ELECTRONS/S)'''

        if not isinstance(f_extn, list):
            f_extn = [f_extn]
        log.debug('Reading FITS Headers')
        idx = 0
        for extn in f_extn:
            if len(self.fnames) != 0:
                self.f_extn = extn
                break
            idx += 1

        if len(self.fnames) == 0:
            raise FileNotFoundError(
                'No files found. Try a different file extention.\n (Currently using {})'.format(f_extn))


        log.debug('{} {} Files found'.format(len(self.fnames), f_extn[idx]))

        time, exptime, postarg1, postarg2 = np.zeros(len(self.fnames)), np.zeros(
            len(self.fnames)), np.zeros(len(self.fnames)), np.zeros(len(self.fnames))
        filters = [None] * len(self.fnames)
        start_date = np.zeros(len(self.fnames))
        propid = np.zeros(len(self.fnames))
        PI = [None] * len(self.fnames)
        target_name = [None] * len(self.fnames)
        for idx, file in enumerate(self.fnames):
            hdr = fits.open(file)[0].header
            propid[idx] = hdr['PROPOSID']
            time[idx] = Time(datetime.strptime('{}-{}'.format(hdr['DATE-OBS'],
                                                              hdr['TIME-OBS']), '%Y-%m-%d-%H:%M:%S')).jd
            start_date[idx] = Time(datetime.strptime(
                '{}'.format(hdr['DATE-OBS']), '%Y-%m-%d')).jd
            filters[idx] = hdr['FILTER']
            exptime[idx] = hdr['EXPTIME']
            postarg1[idx] = hdr['POSTARG1']
            postarg2[idx] = hdr['POSTARG2']
            PI[idx] = hdr['PR_INV_L']
            target_name[idx] = hdr['TARGNAME']
            if hdr['INSTRUME'] != 'WFC3':
                raise ShadowUserInputError(
                    'At least one of the input files is not a WFC3 file. ({})'.format(file))
        if not len(np.unique(propid)) == 1:
            raise ShadowUserInputError(
                'Passed multiple proposal ids ({})'.format(np.unique(propid)))
        self.propid = np.unique(propid)[0]
        self.name = np.unique(target_name)[0]
        self.ra = hdr['RA_TARG']
        self.dec = hdr['DEC_TARG']

        filters = np.asarray(filters)
        s = np.argsort(time)
        self.start_date = start_date[s]
        self.image_names, self.time, self.exptime = np.asarray(self.fnames)[s], time[s], exptime[s]
        self.postarg1, self.postarg2, self.filters = postarg1[s], postarg2[s], filters[s]
        self.dimage = np.asarray(['F' in filt for filt in self.filters])
        self.gimage = np.asarray(['G' in filt for filt in self.filters])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FITSFixedWarning)
            d_hdr = fits.open(self.image_names[self.dimage][0])[1].header
            g_hdr = fits.open(self.image_names[self.gimage][0])[1].header

            self.d_ref1 = d_hdr['CRPIX1'] - g_hdr['CRPIX1']
            self.d_ref2 = d_hdr['CRPIX2'] - g_hdr['CRPIX2']

            if self.f_extn == 'ima':
                self.ns = d_hdr['NAXIS1'] - 10
            else:
                self.ns = d_hdr['NAXIS1']
            log.info('Shadow assumes direct images and grism images are all taken in the same'
                     ' subarray. This target is assumed to be {0} x {0}.'.format(self.ns))
            self.wcs = WCS(d_hdr)
            self._ra_d, self._dec_d = WCS(d_hdr).wcs_pix2world(np.asarray([(np.arange(self.ns)), (np.arange(self.ns))]).T, 0).T
            self.source_x, self.source_y = WCS(d_hdr).wcs_world2pix(np.asarray([[self.ra], [self.dec]]).T, 0).T


    def _get_visits(self, visit=None):
        '''Find all the visit breaks, return some sort of mask or list for it'''
        # VISIT MASKS
        self.visits = {}
        possible_start_dates = np.unique(self.start_date)
        nv = len(possible_start_dates)
        log.debug('{} Visits Found'.format(nv))
        for v, start in enumerate(possible_start_dates):
            self.visits[v + 1] = self.start_date[self.gimage] == start

        # ORBIT MASKS

        # FORWARD AND BACKWARD MASKS

        # Reduce to only the specified visit.
        if visit is not None:
            log.debug('Only carrying forward Visit {}'.format(visit))

    def _get_ephemeris(self):
        '''Find the in and out of transit points, return some sort of mask'''
        model = self.transitmodel()
        self.in_transit = model != 1
        self.out_of_transit = model == 1

    def _get_data(self, bitmask=(512 | 8)):
        '''Get all the science images and data quality. Collapse all the reads.
        Remove sky?'''
        files = self.image_names[self.gimage]
        with warnings.catch_warnings():
            warnings.simplefilter('once', UserWarning)
            for jdx, file in enumerate(files):
                sci, err, dq = fits.open(file)[1:4]
                if sci.header['BUNIT'] not in ['ELECTRONS', 'ELECTRONS/S']:
                    raise ShadowValueError('This file has units of {}.'
                                           ''.format(sci.header['BUNIT']))

                if self.f_extn == 'ima':
                    ns_corr = 5
                else:
                    ns_corr = 0

                if 'scis' not in locals():
                    scis = np.zeros((len(files), sci.data.shape[0] - ns_corr*2, sci.data.shape[1] - ns_corr*2))
                    errs = np.zeros((len(files), sci.data.shape[0] - ns_corr*2, sci.data.shape[1] - ns_corr*2))
                    dqs = np.zeros((len(files), sci.data.shape[0] - ns_corr*2, sci.data.shape[1] - ns_corr*2))

                scis[jdx, :, :] = sci.data[ns_corr:self.ns + ns_corr, ns_corr:self.ns + ns_corr]
                errs[jdx, :, :] = err.data[ns_corr:self.ns + ns_corr, ns_corr:self.ns + ns_corr]
                dqs[jdx, :, :] = dq.data[ns_corr:self.ns + ns_corr, ns_corr:self.ns + ns_corr]

                if sci.header['BUNIT'] == 'ELECTRONS':
                    warnings.warn(UserWarning('Found units of ELECTRONS. Switching to ELECTRONS/S'))
                    scis[jdx, :, :] /= sci.header['SAMPTIME']
                    errs[jdx, :, :] /= sci.header['SAMPTIME']

        dqs = np.asarray(dqs, dtype=int)
        qmask = 1 | 2 | 4 | 8 | 16 | 32 | 256
        errs[(dqs & qmask) != 0] = 1e10
#        bad = dqs & (512 | 8) != 0
#        scis[bad] = np.nan
#        errs[bad] = np.nan
        self.sci = scis
        self.err = errs
        self.dq = dqs
        self.sun_alt = np.asarray([fits.open(f)[0].header['SUN_ALT'] for f in files])
        self.velocity_aberration = np.asarray([fits.open(f)[1].header['VAFACTOR'] for f in files])

        files = self.image_names[self.dimage]
        self.dimage_data = np.asarray([fits.open(f)[1].data for f in files])
        # Flat field the data...
    #
    # def _remove_cosmic_rays(self):
    #     '''Set the cosmic rays in the data to np.nan'''
    #     cmr_mask = np.zeros(self.sci.shape)
    #     for v in self.visits:
    #         m = np.atleast_3d(self.sci[self.visits[v]].mean(
    #             axis=0)).transpose([2, 0, 1])
    #         s = np.atleast_3d(self.sci[self.visits[v]].std(
    #             axis=0)).transpose([2, 0, 1])
    #         cmr_mask[self.visits[v]] = (
    #             self.sci[self.visits[v]] - m)/s
    #     cmr_mask[~np.isfinite(cmr_mask)] = 0
    #     cmr_mask = cmr_mask > 6
    #     for idx, c in enumerate(cmr_mask):
    #         cmr_mask[idx] = convolve(c, Box2DKernel(3)) > 0.2
    #     self.sci[cmr_mask] *= np.nan
    #     self.err[cmr_mask] *= np.nan

    def _find_sources(self):
        '''Find the sources in the direct image/s

        Returns
        -------

        sources: table
            Table of all the sources in the direct image
        '''
    #
    # def _find_edges(self):
    #     '''Find the edges of the spatial scan trace'''
    #
    # def _find_shifts(self):
    #     '''Find the edges of the spatial scan trace'''
    #
    #
    def _find_transits(self):
        ''' Find in transit masks
        THIS IS JUST TO TEST WITH CHANGE THIS
        '''
        wl = np.nansum(self.sci, axis=(1,2))
        wl /= np.nanmedian(wl)
        self.in_transit = wl < 0.99
        self.out_transit = wl >= 0.99

    def _find_mask(self):
        '''Find the variable scan rate'''
        self.mask, self.spectral, self.spatial = methods.simple_mask(self)



    #
    # def _find_cosmics(self):
    #     ''' Get some cosmic rays '''
    #
    #     data = (self.sci/self.flat)/self.mask
    #     data[~np.isfinite(data)] = np.nan
    #     wl = np.nanmean(data, axis=(1,2))
    #     wl /= np.median(wl)
    #
    #     outlier_ar = (((self.sci/self.flat)/self._model)*np.atleast_3d(wl).transpose([1, 0, 2]))
    #     mean, med, std = sigma_clipped_stats(outlier_ar, sigma=5)
    #     outliers = np.abs(outlier_ar - med) > 5 * std
    #     self.outliers = outliers


    def _get_quadrants(self, trim=None):
        """ get a mask of each quadrant of the detector """
        if trim is None:
            trim =[(0, self.ns), (0, self.ns)]

        quadrants = []
        q = np.zeros((self.ns, self.ns))

        q[:self.ns//2, :self.ns//2] = 1
        quadrants.append(np.copy(q[trim[0][0]:trim[0][1], trim[1][0]:trim[1][1]]).ravel())
        q *= 0

        q[:self.ns//2, self.ns//2:] = 1
        quadrants.append(np.copy(q[trim[0][0]:trim[0][1], trim[1][0]:trim[1][1]]).ravel())
        q *= 0

        q[self.ns//2:, :self.ns//2] = 1
        quadrants.append(np.copy(q[trim[0][0]:trim[0][1], trim[1][0]:trim[1][1]]).ravel())
        q *= 0

        q[self.ns//2:, self.ns//2:] = 1
        quadrants.append(np.copy(q[trim[0][0]:trim[0][1], trim[1][0]:trim[1][1]]).ravel())
        return quadrants

    def _find_flat(self, model=None):
        ''' Get flat field, this is data driven, watch out
        '''
        g141_flatfile = CALIPATH+'WFC3.IR.G141.flat.2.fits'
        ff_hdu = fits.open(g141_flatfile)
        a1, a2 = np.where(self.spatial[0, :, 0])[0][0] , np.where(self.spatial[0, :, 0])[0][-1] + 1
        b1, b2 = np.where(self.spectral[0, 0, :])[0][0] , np.where(self.spectral[0, 0, :])[0][-1] + 1

        d1 = self.sci[:, a1:a2, :][:, :, b1:b2]
        e1 = self.err[:, a1:a2, :][:, :, b1:b2]
        q1 = self.dq[:, a1:a2, :][:, :, b1:b2]

        # Increasing the errors for "Hot Pixels"
        bad = (q1 & 256) != 0
        e1[bad] = 1e3

        # Some handy shapes
        nt, nrow, ncol = d1.shape

        norm = np.nansum(self.sci, axis=(1, 2))
        norm /= np.median(norm)
        norm = np.atleast_3d(norm).transpose([1, 0, 2])

        if model is None:
            spec = np.average((d1/norm), weights=1/(e1**2), axis=(0, 1))
            vsr = np.average((d1/norm), weights=1/(e1**2), axis=2)
            model = np.atleast_3d(spec).transpose([0, 2, 1]) * np.atleast_3d(vsr/np.atleast_2d(np.median(vsr, axis=1)).T)

        f = np.asarray([ff_hdu[i].data for i in range(4)])
        f = f[:, 1014//2 - self.ns//2 : 1014//2 + self.ns//2, 1014//2 - self.ns//2 : 1014//2 + self.ns//2]

#        quadrants = self._get_quadrants(trim=[(a1, a2), (b1, b2)])
        # X = np.hstack([f[:, a1:a2, :][:, :, b1:b2][:, np.ones((nrow, ncol), bool)].T
        #                    * np.atleast_2d(quadrants[idx]).T for idx in range(4)])
        # X = np.hstack([X, np.asarray(quadrants).T])
        f = np.vstack([f, np.ones((1, f.shape[1], f.shape[2]))])
        X = f[:, a1:a2, :][:, :, b1:b2][:, np.ones((nrow, ncol), bool)].T

        avg = np.nanmean((d1/norm/model), axis=0)[np.ones((nrow, ncol), bool)]
        avg_err = ((1/nt)*np.nansum(((e1/norm/model))**2, axis=0)**0.5)[np.ones((nrow, ncol), bool)]

        sigma_w_inv = np.dot(X.T, X/avg_err[:, None]**2)
        B = np.dot(X.T, (avg/avg_err**2)[:, None])
        w = np.linalg.solve(sigma_w_inv, B)[:, 0]

        # quadrants = self._get_quadrants()
        # X = np.hstack([f[:, np.ones((self.ns, self.ns), bool)].T
        #                    * np.atleast_2d(quadrants[idx]).T for idx in range(4)])
        # X = np.hstack([X, np.asarray(quadrants).T])
        X = f[:, np.ones((self.ns, self.ns), bool)].T

        flat = np.dot(X, w).reshape((self.ns, self.ns))
        flat /= np.median(flat)
        flat[(self.dq[0] & 256) != 0] = 1
        flat = np.atleast_3d(flat).transpose([2, 0, 1]) * np.ones(self.sci.shape)
        self.flat = flat


    def _find_shifts(self):

        def find_shifts(data):
            X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[2]))
            cent = [np.average(X, weights=np.nan_to_num(d/np.nanmedian(d))) for d in data]
            #cent -= np.median(cent)
            return cent

        norm = np.atleast_3d(np.nansum(self.sci, axis=(1, 2))).transpose([1, 0, 2])
        self.xshift = find_shifts((self.sci/self.flat/self.vsr/norm))
        self.yshift = find_shifts((self.sci/self.flat/self.vsr/norm).transpose([0, 2, 1]))
        # plt.scatter(self.time[self.gimage], xcent, label='Data')
        # plt.scatter(self.time[self.gimage][bottom_of_transit], xcent[bottom_of_transit], label='Data')
        #plt.ylim(-1, 1)


    def _calibrate(self):
        '''Calibrate the detector'''
        g141_skyfile = CALIPATH+'WFC3.IR.G141.sky.V1.0.fits'
        pix_areafile = CALIPATH+'ir_wfc3_map.fits'
        g141_flatfile = CALIPATH+'WFC3.IR.G141.flat.2.fits'
        g141_sensfile = CALIPATH+'WFC3.IR.G141.1st.sens.2.fits'
        for f in [g141_skyfile, pix_areafile, g141_flatfile, g141_sensfile]:
            if not (os.path.isfile(f)):
                raise ShadowValueError('Missing calibration files. Please reinstall.')
        hdu = fits.open('{}/data/calibration/WFC3.IR.G141.1st.sens.2.fits'.format(PACKAGEDIR))
        sens = hdu[1].data['SENSITIVITY']
        wav = hdu[1].data['WAVELENGTH']*u.angstrom
        return(sens, wav)


    def __init__(self, input, in_transit=None, f_extn=['flt', 'ima'], load_only=False, errors=False, teff=6000):
        if isinstance(input, str):
            if input.endswith('/'):
                fnames = glob(input + "*")
        elif isinstance(input, (list, np.ndarray)):
            fnames = input
        else:
            raise ValueError('Can not parse `input`. '
                             'Please pass a directory of list of file names.')

        if (not load_only) & (in_transit is None):
            raise ValueError('If not `load_only` must specify transit mask')
        self.fnames = fnames
        self.visit = None
        self._get_headers(f_extn=f_extn)
        self._get_visits(self.visit)
#        self._get_ephemeris()
        self._get_data()
#        self._remove_cosmic_rays()
        # MASK AND COLLAPSE DATA

        # Run some procedures
        # self._find_sources()
        # self._find_edges()

        self.in_transit = in_transit
#        self._find_transits()
        self._find_mask()
        self._find_flat()
#        self._find_cosmics()
#        self._find_shifts()



        ok = self.exptime[self.gimage] == np.median(self.exptime[self.gimage])
        self.sci = self.sci[ok]
        self.err = self.err[ok]
        self.flat = self.flat[ok]
        self.dq = self.dq[ok]
#        if not load_only:
#            self.in_transit = self.in_transit[ok]

        ok = (self.exptime == np.median(self.exptime[self.gimage])) | self.dimage
        self.time = self.time[ok]
        self.filters = self.filters[ok]
        self.postarg1 = self.postarg1[ok]
        self.postarg2 = self.postarg2[ok]
        self.dimage = self.dimage[ok]
        self.gimage = self.gimage[ok]
        self.exptime = self.exptime[ok]


        self.data = ((self.sci)/self.flat)[:, self.spatial.reshape(-1)][:, :, self.spectral.reshape(-1)]
        self.data[~np.isfinite(self.data)] = np.nan
        self.error = (((self.err))/self.flat)[:, self.spatial.reshape(-1)][:, :, self.spectral.reshape(-1)]
        self.error[~np.isfinite(self.error)] = np.nan


        T = (np.atleast_3d(self.time[self.gimage]) * np.ones(self.data.shape).transpose([1, 0, 2])).transpose([1, 0, 2])
        T -= self.time[self.gimage][0]
        self.T = (T/T.max() - 0.5)

        Y, X = np.mgrid[:self.data.shape[1], :self.data.shape[2]]
        Y = Y/(self.data.shape[1] - 1) - 0.5
        X = X/(self.data.shape[2] - 1) - 0.5

        self.X = np.atleast_3d(X).transpose([2, 0, 1]) * np.ones(self.data.shape)
        self.Y = np.atleast_3d(Y).transpose([2, 0, 1]) * np.ones(self.data.shape)

        self.nt, self.nsp, self.nwav = self.data.shape

        self.forward = self.postarg2[self.gimage] > 0


        if not load_only:
            self.vsr_mean = methods.simple_vsr(self)
            self.spec_mean, self.spec_grad_simple = methods.simple_spectrum(self)


            # Basic model is the average spectral model x the average spatial scan model
            self.basic_model = self.spec_mean * self.vsr_mean
            # Normalize so each frame has the same flux as the true data
            self.basic_model *= np.atleast_3d(np.average(self.data/self.basic_model, weights=self.basic_model/self.error, axis=(1, 2))).transpose([1, 0, 2])

            self.cosmic_rays = sigma_clip(self.data - self.basic_model, sigma=8).mask

            if errors:
                self.vsr_grad_model, self.vsr_grad_model_errs, self.ws = methods.vsr(self, errors=errors)
            else:
                self.vsr_grad_model, self.ws = methods.vsr(self, errors=errors)

            self.vsr_model = self.vsr_grad_model * self.vsr_mean

            w = np.ones(self.sci.shape)
            w[:, self.mask[0]] += (self.cosmic_rays[:, np.ones(self.cosmic_rays.shape[1:], bool)]) * 1e10
            w[self.err/self.sci > 0.1] = 1e10

            larger_mask = convolve(self.mask[0], Box2DKernel(11)) > 1e-5
            _, X = np.mgrid[:self.sci.shape[1], :self.sci.shape[2]]
            X = np.atleast_3d(X).transpose([2, 0, 1]) * np.ones(self.sci.shape)
            w1 = (self.sci/w)[:, larger_mask]
            w1 = (w1.T/w1.mean(axis=1)).T
            xshift = np.mean(X[:, larger_mask] * w1, axis=1)
            self.xshift = xshift - np.median(xshift)


            if errors:
                self.spec_grad_model, self.spec_grad_model_errs = methods.spectrum(self, errors=errors)
            else:
                self.spec_grad_model = methods.spectrum(self, errors=errors)


            self.wavelength, self.sensitivity = methods.calibrate(self, teff)



            model = (self.spec_mean * self.vsr_mean * self.vsr_grad_model)
            if errors:
                model_err = (self.vsr_grad_model_errs/self.vsr_grad_model) * model
                model_err = np.hypot(model_err/model, self.spec_grad_model_errs/self.spec_grad_model) * model * self.spec_grad_model
            model *= self.spec_grad_model

            model /= np.atleast_3d(model.mean(axis=(1, 2))).transpose([1, 0, 2])

            self.model = model
            if errors:
                self.model_err = model_err

            frames = self.data / model
            frames_err = self.error / model
            if errors:
                frames_err = np.hypot(self.error / self.data, model_err/model) * frames
            frames_err[self.cosmic_rays] = 1e10
            frames_err[self.error/self.data > 0.1] = 1e10

            res = (frames.T - np.mean(frames, axis=(1, 2))).T

            chi0 = (np.sum(res[~self.in_transit]**2/frames_err[~self.in_transit]**2, axis=(0, 1)))/np.product(self.data.shape[:2])
            m1 = np.zeros(len(chi0), bool)
            m1[10:-10] = True

            chi1 = (np.sum(res[~self.in_transit]**2/frames_err[~self.in_transit]**2, axis=(0)))/np.product(self.data.shape[:1])
            m = (np.ones(self.data.shape[1:], bool) * m1) & ~sigma_clip(chi1, sigma=3).mask

            weights = 1/np.copy(frames_err)
            weights[:, ~m] = 1e-99

            self.clcs = np.average(frames, weights=weights, axis=1)
            draws = np.random.normal(frames, frames_err, size=(50, *self.data.shape))
            clc_samples = np.asarray([np.average(draws[idx], weights=weights, axis=1) for idx in range(50)])
            self.clcs_err = np.std(clc_samples, axis=0)
            self.clcs[:, ~m1] *= np.nan
            self.clcs_err[:, ~m1] *= np.nan


            bm = self.spec_mean * self.vsr_mean
            self.raw_lcs = np.average(self.data/bm, weights=bm/self.error, axis=1)
            draws = np.random.normal(self.data/bm, self.error/bm, size=(50, *self.data.shape))
            raw_lcs_samples = np.asarray([np.average(draws[idx], weights=bm/self.error, axis=1) for idx in range(50)])
            self.raw_lcs_err = np.std(raw_lcs_samples, axis=0)

            self.wl = np.average(np.nan_to_num(self.clcs), weights=np.nan_to_num(1/self.clcs_err), axis=1)
            draws = np.random.normal(np.nan_to_num(self.clcs), np.nan_to_num(self.clcs_err), size=(50, *self.clcs.shape))
            self.wl_err = np.asarray([np.average(d, weights=np.nan_to_num(1/self.clcs_err), axis=1) for d in draws]).std(axis=0)


    def copy(self):
        return copy.deepcopy(self)


    def plot_clcs(self, offset=10, lines=True, residuals=False, **kwargs):
        c = np.nanmedian(self.clcs[~self.in_transit]) *  np.ones(self.nt)
        cmap = kwargs.pop('cmap', plt.get_cmap('coolwarm'))
        fig, ax = plt.subplots(1, 2, figsize=(15, 25), sharey=True)

        if residuals:
            [ax[0].scatter(self.time[self.gimage], self.raw_lcs[:, kdx] + kdx * offset - self.wl,
                        c=np.ones(self.nt) * self.wavelength[kdx], s=1,
                        vmin=self.wavelength[0], vmax=self.wavelength[-1],
                        cmap=cmap)
                            for kdx in range(len(self.wavelength))];

            [ax[1].scatter(self.time[self.gimage], self.clcs[:, kdx] + kdx * offset - self.wl,
                        c=np.ones(self.nt) * self.wavelength[kdx], s=1,
                         vmin=self.wavelength[0], vmax=self.wavelength[-1],
                         cmap=cmap)
                            for kdx in range(len(self.wavelength))];
            if lines:
                [ax[1].plot(self.time[self.gimage], np.ones(self.nt) * kdx * offset, c='grey', ls='--', lw=0.5, zorder=-10) for kdx in range(len(self.wavelength)) if (np.nansum(self.clcs[:, kdx]) != 0)];

        else:
            [ax[0].scatter(self.time[self.gimage], self.raw_lcs[:, kdx] + kdx * offset,
                        c=np.ones(self.nt) * self.wavelength[kdx], s=1,
                        vmin=self.wavelength[0], vmax=self.wavelength[-1],
                        cmap=cmap)
                            for kdx in range(len(self.wavelength))];

            [ax[1].scatter(self.time[self.gimage], self.clcs[:, kdx] + kdx * offset,
                        c=np.ones(self.nt) * self.wavelength[kdx], s=1,
                         vmin=self.wavelength[0], vmax=self.wavelength[-1],
                         cmap=cmap)
                            for kdx in range(len(self.wavelength))];
            if lines:
                [ax[1].plot(self.time[self.gimage], c + kdx * offset, c='grey', ls='--', lw=0.5, zorder=-10) for kdx in range(len(self.wavelength)) if (np.nansum(self.clcs[:, kdx]) != 0)];
        ax[1].set(xlabel='Time', ylabel='Flux', title='Corrected', yticklabels='')
        ax[0].set(xlabel='Time', ylabel='Flux', title='Raw', yticklabels='')
        plt.subplots_adjust(wspace=0.)
        return fig


    @staticmethod
    def from_MAST(targetname, visit=None, direction=None, **kwargs):
        """Download a target from MAST
        """
        def download_target(targetname, radius='10 arcsec'):
            download_dir = os.path.join(os.path.expanduser('~'), '.shadow-cache')
            if not os.path.isdir(download_dir):
                try:
                    os.mkdir(download_dir)
                # downloads locally if OS error occurs
                except OSError:
                    log.warning('Warning: unable to create {}. '
                                'Downloading MAST files to the current '
                                'working directory instead.'.format(download_dir))
                    download_dir = '.'

            logging.getLogger('astropy').setLevel(log.getEffectiveLevel())
            obsTable = Observations.query_criteria(target_name=targetname,
                                                    obs_collection='HST',
                                                    project='HST',
                                                    radius=radius)
            obsTable = obsTable[(obsTable['instrument_name'] == 'WFC3/IR') &
                                (obsTable['dataRights'] == "PUBLIC")]

            fnames = ['{}/mastDownload/'.format(download_dir) +
                        url.replace('mast:', '').replace('product', url.split('/')[-1].split('_')[0])[:-9] +
                         '_flt.fits' for url in obsTable['dataURL']]
            log.info('Found {} files.'.format(len(obsTable)))
            paths = []
            for idx, t in enumerate(tqdm(obsTable, desc='Downloading files', total=len(obsTable))):
                if os.path.isfile(fnames[idx]):
                    paths.append(fnames[idx])
                else:
                    t1 = Observations.get_product_list(t)
                    t1 = t1[t1['productSubGroupDescription'] == 'FLT']
                    paths.append(Observations.download_products(t1, mrp_only=False, download_dir=download_dir)['Local Path'][0])
            return paths

        paths = np.asarray(download_target(targetname))

        if isinstance(visit, int):
            start_time = np.asarray([Time(datetime.strptime('{}'.format(fits.open(fname)[0].header['DATE-OBS']), '%Y-%m-%d')).jd for fname in paths])
            visits = np.asarray([np.where(np.sort(np.unique(start_time)) == t1)[0][0] + 1 for t1 in start_time])
            mask = visits == visit
            if not mask.any():
                raise ValueError('No data in visit {}'.format(visit))
            paths = paths[mask]

        if isinstance(direction, str):
            directions = np.asarray([fits.open(fname)[0].header['POSTARG2'] for fname in paths])
            if direction == 'forward':
                paths = paths[directions >= 0]
            elif direction == 'backward':
                paths = paths[directions <= 0]
            else:
                raise ValueError("Can not parse direction {}. "
                                    "Choose from `'forward'` or `'backward'`".format(direction))

        if not np.asarray([fits.open(fname)[0].header['FILTER'] == 'G141' for fname in paths]).any():
            raise ValueError('No G141 files available. Try changing `visit` and `direction` keywords.')
        return Observation(paths, **kwargs)

    def __repr__(self):
        return '{} (WFC3 Observation)'.format(self.name)

    def __str__(self):
        return '{} (WFC3 Observation)'.format(self.name)

    def __len__(self):
        return len(self.time)

    def transitmodel(self):
        '''Compute the transit model using batman and the exoplanet archive.'''
        sys = from_nexsci('{}'.format(self.name))
        return modeling.transitmodel(self)

    def plotFrame(self, frameno=0, ax=None):
        '''Plot a single grism frame.

        Parameters
        ----------
        frameno : int
            Which gimage index number to plot
        ax : None or matplotlib.pyplot.axes
            Frame to plot into.
        '''
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))
        dat = self.sci[frameno]
        cmap = plt.get_cmap('Greys_r')
        cmap.set_bad('black')
        im = ax.imshow(dat, origin='bottom', cmap=cmap,
                       vmin=np.nanpercentile(dat, 70), vmax=np.nanpercentile(dat, 99.9))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Flux [electrons/s]')
        ax.set_title('{} Frame: {}'.format(self.name, frameno))
        return ax

    def plotDimage(self, frameno=0, ax=None):
        '''Plot a single direct image frame.

        Parameters
        ----------
        frameno : int
            Which gimage index number to plot
        ax : None or matplotlib.pyplot.axes
            Frame to plot into.
        '''
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))
        dat = np.log10(self.dimage_data[frameno])
        cmap = plt.get_cmap('Greys_r')
        cmap.set_bad('black')
        im = ax.imshow(dat, origin='bottom', cmap=cmap,
                       vmin=np.nanpercentile(dat, 70), vmax=np.nanpercentile(dat, 99.9))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('log$_{10}$Flux [electrons/s]')
        ax.set_title('{} Frame: {}'.format(self.name, frameno))
        ax.scatter(self.source_x, self.source_y, marker='o', facecolor='None',
                   lw=3, s=30, edgecolor='C1')
        return ax

    def plotWhiteLight(self, ax=None):
        '''Plot up the white light transit'''

    def plotColoredLight(self, ax=None):
        '''Plot up the coloured light as a surface'''

    def animate(self, visit=1, scale='linear', output='out.mp4', **kwargs):
        '''Create an animation of all the frames in a given visit.

        Parameters
        ----------
        output : str
            File to output mp4 to
        visit : int
            Visit number. Default is 1.
        '''
        methods.animate(self.sci[self.visits[visit]], scale=scale, output=output, **kwargs)
