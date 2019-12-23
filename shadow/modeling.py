'''Routines to fit stuff'''
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import pandas as pd
import batman

class ShadowConnectionError(Exception):
    '''Raised if there seems to be a connection problem'''
    pass

def _hard_code_planet_parameters(planets):
    '''Some of the information from the exoplanet archive is missing.
    This adds that information back in for HD209458
    '''
    loc = np.where(np.asarray(planets.pl_hostname) == 'HD 209458')[0]
    planets.loc[loc, 'pl_tranmid'] = 2452826.628514
    planets.loc[loc, 'pl_orbsmax'] = 0.04723
    planets.loc[loc, 'pl_orbpererr1'] = 3.8e-7
    planets.loc[loc, 'pl_orbpererr2'] = -3.8e-7

    loc = np.where(np.asarray(planets.pl_hostname) == 'HAT-P-11')[0]
    planets.loc[loc, 'pl_tranmid'] = 2454957.813556
    planets.loc[loc, 'pl_orbsmax'] = 0.053
    planets.loc[loc, 'pl_orbpererr1'] = 3.8e-7
    planets.loc[loc, 'pl_orbpererr2'] = -3.8e-7
    planets.loc[loc, 'pl_orbincl'] = 90

    loc = np.where(np.asarray(planets.pl_hostname) == 'HAT-P-3')[0]
    planets.loc[loc, 'pl_tranmid'] = 2454856.70118 + 0.05
    planets.loc[loc, 'pl_orbsmax'] = 0.03866
    return planets


def find_planet_parameters(target):
    '''Find planets for a given target and returns a dictionary with all the transit parameters.

    Parameters
    ----------
    target : shadow.observation.Observation
        HST WFC3 observation

    Returns
    -------
    planets_dictionary : dict
        Dictionary of planet parameters from exoplanet archive, including errors.
    '''

    NEXSCI_API = 'http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI'
    try:
        planets = pd.read_csv(NEXSCI_API + ('?table=planets&select=pl_hostname,pl_letter,'
        'pl_disc,pl_discmethod,ra,dec,pl_trandep,pl_tranflag,pl_orbsmax,pl_orbsmaxerr1,'
        'pl_orbsmaxerr2,pl_radj,pl_radjerr1,pl_radjerr2,pl_bmassj,pl_bmassjerr1,'
        'pl_bmassjerr2,pl_eqt,pl_orbper,pl_orbpererr1,pl_orbpererr2,pl_k2flag,'
        'pl_kepflag,pl_facility,st_rad,st_raderr1,st_raderr2,st_teff,st_optmag,'
        'st_mass,st_logg,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,pl_orbincl,'
        'pl_orbinclerr1,pl_orbinclerr2,pl_orbeccen,pl_orbeccenerr1,pl_orbeccenerr2'), comment='#')
    except:
        raise ShadowConnectionError('There appears to be a problem with your internet connection.'
                                    ' Functions such as ephemeris finding and transit fitting will not work '
                                    'as these depend on NexScI values. Re-establish connection to use these '
                                    'features.')
    c = SkyCoord(np.asarray(planets['ra'])*u.deg, np.asarray(planets['dec'])*u.deg)
    planets = _hard_code_planet_parameters(planets)
    loc = SkyCoord(target.ra*u.deg, target.dec*u.deg).separation(c) < 100 * u.arcsec
    planets = planets.loc[loc]
    planets_dictionary = {}

    for idx, p in planets.iterrows():
        planets_dictionary[idx] = {}
        planets_dictionary[idx]['letter'] = p['pl_letter']
        planets_dictionary[idx]['teff'] = p['st_teff']
        planets_dictionary[idx]['st_logg'] = p['st_logg']

        planets_dictionary[idx]['period'] = p['pl_orbper']
        if np.isfinite(p['pl_orbpererr1']) & np.isfinite(p['pl_orbpererr2']):
            planets_dictionary[idx]['period_err'] = np.asarray(
                [p['pl_orbpererr2'], p['pl_orbpererr1']])
        else:
            planets_dictionary[idx]['period_err'] = np.asarray(
                [-(20*u.minute).to(u.day).value, (20*u.minute).to(u.day).value])

        planets_dictionary[idx]['tranmid'] = p['pl_tranmid']
        if np.isfinite(p['pl_tranmiderr1']) & np.isfinite(p['pl_tranmiderr2']):
            planets_dictionary[idx]['tranmid_err'] = np.asarray(
                [p['pl_tranmiderr2'], p['pl_tranmiderr1']])
        else:
            planets_dictionary[idx]['tranmid_err'] = np.asarray(
                [-(20*u.minute).to(u.day).value, (20*u.minute).to(u.day).value])

        planets_dictionary[idx]['orbsmax'] = (p['pl_orbsmax']*u.AU).to(u.solRad).value
        if np.isfinite(p['pl_orbsmaxerr1']) & np.isfinite(p['pl_orbsmaxerr2']):
            planets_dictionary[idx]['orbsmax_err'] = np.asarray(
                [(p['pl_orbsmaxerr2']*u.AU).to(u.solRad).value, (p['pl_orbsmaxerr1']*u.AU).to(u.solRad).value])
        else:
            planets_dictionary[idx]['orbsmax_err'] = np.asarray(
                [-0.05*p['pl_orbsmax'], 0.05*p['pl_orbsmax']])

        planets_dictionary[idx]['st_rad'] = p['st_rad']
        if np.isfinite(p['st_raderr1']) & np.isfinite(p['st_raderr2']):
            planets_dictionary[idx]['st_rad_err'] = np.asarray([p['st_raderr2'], p['st_raderr1']])
        else:
            planets_dictionary[idx]['st_rad_err'] = np.asarray(
                [-0.05*p['st_rad'], 0.05*p['st_rad']])

        planets_dictionary[idx]['radj'] = (p['pl_radj']*u.jupiterRad).to(u.solRad).value
        if np.isfinite(p['pl_radjerr1']) & np.isfinite(p['pl_radjerr2']):
            planets_dictionary[idx]['radj_err'] = np.asarray(
                [(p['pl_radjerr2']*u.jupiterRad).to(u.solRad).value, (p['pl_radjerr1']*u.jupiterRad).to(u.solRad).value])
        else:
            planets_dictionary[idx]['radj_err'] = np.asarray(
                [-0.05*p['pl_radj'], 0.05*p['pl_radj']])

        planets_dictionary[idx]['incl'] = np.nanmin([p['pl_orbincl'], 90])
        if np.isfinite(p['pl_orbinclerr1']) & np.isfinite(p['pl_orbinclerr2']):
            planets_dictionary[idx]['incl_err'] = np.asarray(
                [p['pl_orbinclerr2'], p['pl_orbinclerr1']])
        else:
            planets_dictionary[idx]['incl_err'] = np.asarray([-0.5, 0.5])

        if np.isfinite(p['pl_orbeccen']):
            planets_dictionary[idx]['ecc'] = p['pl_orbeccen']
        else:
            planets_dictionary[idx]['ecc'] = 0
        if np.isfinite(p['pl_orbeccenerr1']) & np.isfinite(p['pl_orbeccenerr2']):
            planets_dictionary[idx]['ecc_err'] = np.asarray(
                [p['pl_orbeccenerr2'], p['pl_orbeccenerr1']])
        else:
            planets_dictionary[idx]['ecc_err'] = np.asarray([-0.05, 0.05])

    return planets_dictionary


def transitmodel(target, time=None):
    '''Some day soon this will be starry'''
    if time is None:
        time = target.time

    planets_dictionary = find_planet_parameters(target)
    mlc_flux = np.ones(len(time))
    for idx, planet in planets_dictionary.items():
        params = batman.TransitParams()
        params.limb_dark = "linear"  # limb darkening model
        params.u = [0.5]  # limb darkening coefficients [u1, u2]
        params.w = 90.  # longitude of periastron (in degrees)

        params.t0 = planet['tranmid']  # time of inferior conjunction
        params.per = planet['period']  # orbital period
        params.ecc = planet['ecc']  # eccentricity
        params.a = planet['orbsmax']/planet['st_rad']  # semi-major axis (in units of stellar radii)
        params.rp = planet['radj']/planet['st_rad']  # planet radius (in units of stellar radii)
        params.inc = planet['incl']  # orbital inclination (in degrees)

        m = batman.TransitModel(params, time)  # initializes model
        mlc_flux += (m.light_curve(params) - 1)
    return mlc_flux/np.median(mlc_flux)
