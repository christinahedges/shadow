'''Routines to extract data from HST and correct it.
'''

import numpy as np
import warnings
import os
from glob import glob
import numpy as np
from datetime import datetime
import logging
import pandas as pd
from matplotlib import animation
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.convolution import Box2DKernel, convolve
from astropy.wcs import WCS, FITSFixedWarning


from . import PACKAGEDIR
from . import modeling

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

    def _get_headers(self, dir, f_extn=['flt', 'ima']):
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
        image_names = []
        idx = 0
        for extn in f_extn:
            image_names = np.asarray(glob('{}*_{}.fits*'.format(dir, extn)))
            if len(image_names) != 0:
                break

        if len(image_names) == 0:
            raise FileNotFoundError(
                'No files found. Try a different file extention.\n (Currently using {})'.format(f_extn))

        image_names = np.asarray(image_names)
        log.debug('{} {} Files found'.format(len(image_names), f_extn[idx]))

        time, exptime, postarg1, postarg2 = np.zeros(len(image_names)), np.zeros(
            len(image_names)), np.zeros(len(image_names)), np.zeros(len(image_names))
        filters = [None] * len(image_names)
        start_date = np.zeros(len(image_names))
        propid = np.zeros(len(image_names))
        PI = [None] * len(image_names)
        target_name = [None] * len(image_names)
        for idx, file in enumerate(image_names):
            hdr = fits.open(file)[0].header
            time[idx] = Time(datetime.strptime('{}-{}'.format(hdr['DATE-OBS'],
                                                              hdr['TIME-OBS']), '%Y-%m-%d-%H:%M:%S')).jd
            start_date[idx] = Time(datetime.strptime(
                '{}'.format(hdr['DATE-OBS']), '%Y-%m-%d')).jd
            filters[idx] = hdr['FILTER']
            exptime[idx] = hdr['EXPTIME']
            postarg1[idx] = hdr['POSTARG1']
            postarg2[idx] = hdr['POSTARG2']
            PI[idx] = hdr['PR_INV_L']
            propid[idx] = hdr['PROPOSID']
            target_name[idx] = hdr['TARGNAME']
            if hdr['INSTRUME'] != 'WFC3':
                raise ShadowUserInputError(
                    'At least one of the input files is not a WFC3 file. ({})'.format(file))
        if not len(np.unique(propid)) == 1:
            raise ShadowUserInputError(
                'Passed multiple proposal ids ({})'.format(np.unique(propid)))
        if not len(np.unique(propid)) == 1:
            raise ShadowUserInputError(
                'Passed multiple objects ({})'.format(np.unique(target_name)))
        self.propid = np.unique(propid)[0]
        self.name = np.unique(target_name)[0]
        self.ra = hdr['RA_TARG']
        self.dec = hdr['DEC_TARG']

        filters = np.asarray(filters)
        s = np.argsort(time)
        self.start_date = start_date[s]
        self.image_names, self.time, self.exptime = image_names[s], time[s], exptime[s]
        self.postarg1, self.postarg2, self.filters = postarg1[s], postarg2[s], filters[s]
        self.dimage = np.asarray(['F' in filt for filt in self.filters])
        self.gimage = np.asarray(['G' in filt for filt in self.filters])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FITSFixedWarning)
            d_hdr = fits.open(self.image_names[self.dimage][0])[1].header
            g_hdr = fits.open(self.image_names[self.gimage][0])[1].header

            self.d_ref1 = d_hdr['CRPIX1'] - g_hdr['CRPIX1']
            self.d_ref2 = d_hdr['CRPIX2'] - g_hdr['CRPIX2']

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

                if 'scis' not in locals():
                    scis = np.zeros((len(files), sci.data.shape[0], sci.data.shape[1]))
                    errs = np.zeros((len(files), sci.data.shape[0], sci.data.shape[1]))
                    dqs = np.zeros((len(files), sci.data.shape[0], sci.data.shape[1]))
                scis[jdx, :, :] = sci.data
                errs[jdx, :, :] = err.data
                dqs[jdx, :, :] = dq.data

                if sci.header['BUNIT'] == 'ELECTRONS':
                    warnings.warn(UserWarning('Found units of ELECTRONS. Switching to ELECTRONS/S'))
                    scis[jdx, :, :] /= sci.header['SAMPTIME']
                    errs[jdx, :, :] /= sci.header['SAMPTIME']

        dqs = np.asarray(dqs, dtype=int)
        bad = dqs & (512 | 8) != 0
        scis[bad] = np.nan
        errs[bad] = np.nan
        self.sci = scis
        self.err = errs
        files = self.image_names[self.dimage]
        self.dimage_data = np.asarray([fits.open(f)[1].data for f in files])

        # Flat field the data...

    def _remove_cosmic_rays(self):
        '''Set the cosmic rays in the data to np.nan'''
        cmr_mask = np.zeros(self.sci.shape)
        for v in self.visits:
            m = np.atleast_3d(self.sci[self.visits[v]].mean(
                axis=0)).transpose([2, 0, 1])
            s = np.atleast_3d(self.sci[self.visits[v]].std(
                axis=0)).transpose([2, 0, 1])
            cmr_mask[self.visits[v]] = (
                self.sci[self.visits[v]] - m)/s
        cmr_mask[~np.isfinite(cmr_mask)] = 0
        cmr_mask = cmr_mask > 6
        for idx, c in enumerate(cmr_mask):
            cmr_mask[idx] = convolve(c, Box2DKernel(3)) > 0.2
        self.sci[cmr_mask] *= np.nan
        self.err[cmr_mask] *= np.nan

    def _find_sources(self):
        '''Find the sources in the direct image/s

        Returns
        -------

        sources: table
            Table of all the sources in the direct image
        '''

    def _find_edges(self):
        '''Find the edges of the spatial scan trace'''

    def _find_shifts(self):
        '''Find the edges of the spatial scan trace'''

    def _find_vsr(self):
        '''Find the variable scan rate'''

    def _calibrate(self):
        '''Calibrate the detector'''
        g141_skyfile = CALIPATH+'WFC3.IR.G141.sky.V1.0.fits'
        pix_areafile = CALIPATH+'ir_wfc3_map.fits'
        g141_flatfile = CALIPATH+'WFC3.IR.G141.flat.2.fits'
        g141_sensfile = CALIPATH+'WFC3.IR.G141.1st.sens.2.fits'
        for f in [g141_skyfile, pix_areafile, g141_flatfile, g141_sensfile]:
            if not (os.path.isfile(f)):
                raise ShadowValueError('Missing calibration files. Please reinstall.')
        hdu = fits.open(
            '/Users/ch/Cambs/shadow/shadow/data/calibration/WFC3.IR.G141.1st.sens.2.fits')
        sens = hdu[1].data['SENSITIVITY']
        wav = hdu[1].data['WAVELENGTH']*u.angstrom
        return(sens, wav)

    def _collapse(self):
        '''Take all the input and create a clean 2D spectrum / wavelength result'''

        '''
        self.WhiteLight =
        self.ColoredLight =
        self.WhiteLight_err =
        self.ColoredLight_err =
        '''

    def _clean(self):
        '''Remove the large fits files and other things we're not going to care about.'''

    def __init__(self, dir, visit=None):
        self.dir = dir
        self.visit = visit
        self._get_headers(self.dir)
        self._get_visits(self.visit)
#        self._get_ephemeris()
        self._get_data()
#        self._remove_cosmic_rays()
        # MASK AND COLLAPSE DATA

        # Run some procedures
        self._find_sources()
        self._find_edges()
        self._find_shifts()
        self._find_vsr()

        # Run calibration
        self._calibrate()

        # Collapse everything into a final data product
        self._collapse()

        # No need to carry around Mb of useless data.
        self._clean()

    def __repr__(self):
        return '{} (WFC3 Observation)'.format(self.name)

    def __str__(self):
        return '{} (WFC3 Observation)'.format(self.name)

    def transitmodel(self):
        '''Compute the transit model using batman and the exoplanet archive.'''
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

    def animate(self, scale='linear', output='out.mp4', visit=1, **kwargs):
        '''Create an animation of all the frames in a given visit.

        Parameters
        ----------
        output : str
            File to output mp4 to
        visit : int
            Visit number. Default is 1.
        '''
        fig, ax = plt.subplots(figsize=(6, 6))
        idx = 0
        if scale is 'log':
            dat = np.log10(self.sci[self.visits[visit]])
        else:
            dat = self.sci[self.visits[visit]]
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
