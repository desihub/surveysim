from __future__ import print_function, division, absolute_import
import numpy as np
import os.path
from shutil import copyfile
from datetime import datetime, timedelta
from astropy.time import Time
from astropy.table import Table, vstack
import astropy.io.fits as pyfits
from surveysim.weather import weatherModule
from desisurvey.nightcal import getCalAll
from desisurvey.afternoonplan import surveyPlan
from surveysim.nightops import obsCount, nightOps

def surveySim(sd0, ed0, seed=None, tilesubset=None, use_jpl=False):
    """
    Main driver for survey simulations.

    Args:
        sd0: tuple of three integers: startyear, startmonth, startday
        ed0: tuple of three integers: endyear, endmonth, endday

    Optional:
        seed: integer, to initialise random number generator for weather simulator
        tilesubset : array of integer tileIDs to use while ignoring others
            in the DESI footprint
        use_jpl: bool, which avoidobject to use; True if astropy+jplephem,
            False if pyephem
    """

    # Note 1900 UTC is midday at KPNO, which is in Mountain Standard Time UTC-7
    # and does not observe daylight savings.
    tz_offset = (19, 0, 0) # hours, minutes, seconds
    (startyear, startmonth, startday) = sd0
    startdate = Time(datetime(*(sd0 + tz_offset)))
    enddate = Time(datetime(*(ed0 + tz_offset)))

    # Tabulate sun and moon ephemerides for each night of the survey.
    surveycal = getCalAll(startdate, enddate, use_cache=True)

    # Build the survey plan.
    sp = surveyPlan(startdate.mjd, enddate.mjd, surveycal, tilesubset=tilesubset)

    # Initialize the survey weather conditions generator.
    w = weatherModule(startdate.datetime, seed)

    tile_file = 'tiles_observed.fits'
    if os.path.exists(tile_file):
        tilesObserved = Table.read(tile_file, format='fits')
        start_val = len(tilesObserved)+1
    else:
        print("The survey will start from scratch.")
        tilesObserved = Table(names=('TILEID', 'STATUS'), dtype=('i8', 'i4'))
        tilesObserved.meta['MJDBEGIN'] = startdate.mjd
        start_val = 0

    ocnt = obsCount(start_val)

    # Define the summer monsoon season.
    monsoon_start = (7, 13)  # month, day
    monsoon_stop = (8, 27)   # month, day
    oneday = timedelta(days=1)
    day = startdate.datetime
    survey_done = False
    iday = 0
    tiles_todo = sp.numtiles
    while (day <= enddate.datetime and survey_done == False):
        print('>> Simulating', day)
        if (day < datetime(day.year, *monsoon_start) or
            day > datetime(day.year, *monsoon_stop)):
            day_stats = surveycal[iday]
            ntodate = len(tilesObserved)
            if day_stats['MoonFrac'] < 0.85:
                w.resetDome(day)
                obsplan = sp.afternoonPlan(day_stats, tilesObserved)
                tilesObserved = nightOps(day_stats, obsplan, w, ocnt, tilesObserved, use_jpl=use_jpl)
                t = Time(day, format = 'datetime')
                ntiles_tonight = len(tilesObserved)-ntodate
                tiles_todo -= ntiles_tonight
                print ('On the night starting ', t.iso, ', we observed ', ntiles_tonight, ' tiles.')
                print ('There are ', tiles_todo, 'left to observe.')
                if (sp.numtiles - len(tilesObserved)) == 0:
                    survey_done = True
            else:
                print ('Monthly maintenance period around Full Moon. No observing.')
        day += oneday
        iday += 1

    tilesObserved.write(tile_file, format='fits', overwrite=True)
