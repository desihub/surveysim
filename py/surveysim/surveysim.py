import numpy as np
import os
from shutil import copyfile
from datetime import datetime, timedelta
from astropy.time import Time
from astropy.table import Table, vstack
import astropy.io.fits as pyfits
from surveysim.weather import weatherModule
from desisurvey.nightcal import getCal
from desisurvey.afternoonplan import surveyPlan
from desisurvey.nightops import obsCount, nightOps

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
    """

    # Note 1900 UTC is midday at KPNO
    (startyear, startmonth, startday) = sd0
    startdate = datetime(startyear, startmonth, startday, 19, 0, 0)
    (endyear, endmonth, endday) = ed0
    enddate = datetime(endyear, endmonth, endday, 19, 0, 0)

    sp = surveyPlan(tilesubset=tilesubset)
    day0 = Time(datetime(startyear, startmonth, startday, 19, 0, 0))
    mjd_start = day0.mjd
    w = weatherModule(startdate, seed)

    tile_file = 'tiles_observed.fits'
    if os.path.exists(tile_file):
        tilesObserved = Table.read(tile_file, format='fits')
        start_val = len(tilesObserved)+1
    else:
        print("The survey will start from scratch.")
        tilesObserved = Table(names=('TILEID', 'STATUS'), dtype=('i8', 'i4'))
        tilesObserved.meta['MJDBEGIN'] = mjd_start
        start_val = 0

    ocnt = obsCount(start_val)
    
    oneday = timedelta(days=1)
    day = startdate
    day_monsoon_start = 13
    month_monsoon_start = 7
    day_monsoon_end = 27
    month_monsoon_end = 8
    survey_done = False
    while (day <= enddate and survey_done == False):
        if ( not (day >= datetime(day.year, month_monsoon_start, day_monsoon_start) and
                  day <= datetime(day.year, month_monsoon_end, day_monsoon_end)) ):
            day_stats = getCal(day)
            ntodate = len(tilesObserved)
            w.resetDome(day)
            tiles_todo, obsplan = sp.afternoonPlan(day_stats, tilesObserved)
            tilesObserved = nightOps(day_stats, obsplan, w, ocnt, tilesObserved, use_jpl=use_jpl)
            t = Time(day, format = 'datetime')
            ntiles_tonight = len(tilesObserved)-ntodate
            print ('On the night starting ', t.iso, ', we observed ', ntiles_tonight, ' tiles.')
            if (tiles_todo-ntiles_tonight) == 0:
                survey_done = True
        day += oneday

    tilesObserved.write(tile_file, format='fits', overwrite=True)