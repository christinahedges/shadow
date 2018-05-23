'''Routines to extract data from HST and correct it.
'''

import numpy as np
import os
from astropy.io import fits
from glob import glob
import numpy as np
from datetime import datetime
from astropy.time import Time

import logging
log = logging.getLogger()


class Observation(Object):
    '''Holds all the data information?'''

    def _get_headers(self):
        '''Obtain and sort the headers'''
        self.dimage =  # MASK
        self.gimage =  # MASK
        '''Sort everything into the correct order...clean out garbage?
        Make sure flux is in the right units! (ELECTRONS/S)'''

    def _get_visits(self):
        '''Find all the visit breaks, return some sort of mask or list for it'''

    def _get_ephemeris(self):
        '''Find the in and out of transit points, return some sort of mask'''

    def _get_data(self):
        '''Get all the science images and data quality. Collapse all the reads.
        Remove sky?'''

    def _get_mask(self):
        '''Build the data masks from DQ'''

    def _remove_cosmic_rays(self):
        '''Flag the cosmic rays in the data'''

    def _find_sources(self):
        '''Find the sources in the direct image/s

        Returns
        -------

        sources : table
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

    def _collapse(self):
        '''Take all the input and create a clean 2D spectrum/ wavelength result'''

        self.WhiteLight =
        self.ColoredLight =
        self.WhiteLight_err =
        self.ColoredLight_err =

    def _clean(self):
        '''Remove the large fits files and other things we're not going to care about.'''

    def __init__(self, verbose):
        self.get_headers()
        self._get_visits()
        self._get_ephemeris()
        self._get_data()
        self._get_mask()
        self._remove_cosmic_rays()
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
        print('{}'.format(self.name))

    def plot_Frame(self, framemno=0, ax=None):
        '''Plot up just one of the images'''

    def plot_WhiteLight(self, ax=None):
        '''Plot up the white light transit'''

    def plot_ColoredLight(self, ax=None):
        '''Plot up the coloured light as a surface'''
