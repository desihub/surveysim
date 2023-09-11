# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
=================
surveysim.weather
=================

Simulate stochastic observing weather conditions.

The simulated conditions include seeing, transparency and the dome-open fraction.
"""
from datetime import datetime

import numpy as np

import astropy.time
import astropy.table
import astropy.units as u

import desiutil.log

import desimodel.weather

import desisurvey.config
import desisurvey.ephem
import desisurvey.utils


class Weather(object):
    """Simulate weather conditions affecting observations.

    The start/stop date range is taken from the survey config.

    Seeing and transparency values are stored with 32-bit floats to save
    some memory.

    Parameters
    ----------
    seed : int
        Random number seed to use to generate stochastic conditions.
        The seed determines the same seeing and transparency realization
        independent of the value of ``replay``.
    replay : str
        Either 'random' or a comma-separated list of years whose
        historical weather should be replayed, e.g. 'Y2010,Y2012'.
        Replayed weather will be used cyclically if necessary.
        Random weather will be a boostrap sampling of all available
        years with historical weather data.  Use 'Y2015' for the
        worst-case weather scenario.
    time_step : float or :class:`astropy.units.Quantity`, optional
        Time step calculating updates. Must evenly divide 24 hours.
        If unitless float, will be interpreted as minutes.
    restore : filename or None
        Restore an existing weather simulation from the specified file name.
        All other parameters are ignored when this is provided. A relative path
        name refers to the :meth:`configuration output path
        <desisurvey.config.Configuration.get_path>`.
    extra_downtime : float
        Additionally close the dome completely on some nights.  Nights are
        chosen randomly, with the chance of the night being closed equal to
        extra_random_close_fraction.  This is intended to include margin.
    """
    def __init__(self, seed=1, replay='random', time_step=5, restore=None,
                 extra_downtime=0):
        if not isinstance(time_step, u.Quantity):
            time_step = time_step * u.min
        self.log = desiutil.log.get_logger()
        config = desisurvey.config.Configuration()
        ephem = desisurvey.ephem.get_ephem()

        if restore is not None:
            fullname = config.get_path(restore)
            self._table = astropy.table.Table.read(fullname)
            self.start_date = desisurvey.utils.get_date(
                self._table.meta['START'])
            self.stop_date = desisurvey.utils.get_date(
                self._table.meta['STOP'])
            self.num_nights = self._table.meta['NIGHTS']
            self.steps_per_day = self._table.meta['STEPS']
            self.replay = self._table.meta['REPLAY']
            self.log.info('Restored weather from {}.'.format(fullname))
            return
        else:
            self.log.info('Generating random weather with seed={} replay="{}".'
                          .format(seed, replay))

        gen = np.random.RandomState(seed)

        # Use our config to set any unspecified dates.
        start_date = config.first_day()
        stop_date = config.last_day()
        num_nights = (stop_date - start_date).days
        if num_nights <= 0:
            raise ValueError('Expected start_date < stop_date.')

        # Check that the time step evenly divides 24 hours.
        steps_per_day = int(round((1 * u.day / time_step).to(1).value))
        if not np.allclose((steps_per_day * time_step).to(u.day).value, 1.):
            raise ValueError(
                'Requested time_step does not evenly divide 24 hours: {0}.'
                .format(time_step))

        # Calculate the number of times where we will tabulate the weather.
        num_rows = num_nights * steps_per_day
        meta = dict(START=str(start_date), STOP=str(stop_date),
                    NIGHTS=num_nights, STEPS=steps_per_day, REPLAY=replay)
        self._table = astropy.table.Table(meta=meta)

        # Initialize column of MJD timestamps.
        t0 = desisurvey.utils.local_noon_on_date(start_date)
        times = t0 + (np.arange(num_rows) / float(steps_per_day)) * u.day
        self._table['mjd'] = times.mjd

        # Generate a random atmospheric seeing time series.
        dt_sec = 24 * 3600. / steps_per_day
        self._table['seeing'] = desimodel.weather.sample_seeing(
            num_rows, dt_sec=dt_sec, gen=gen).astype(np.float32)

        # Generate a random atmospheric transparency time series.
        self._table['transparency'] = desimodel.weather.sample_transp(
            num_rows, dt_sec=dt_sec, gen=gen).astype(np.float32)

        if replay == 'random':
            # Generate a bootstrap sampling of the historical weather years.
            years_to_simulate = config.last_day().year - config.first_day().year + 1
            history = ['Y{}'.format(year) for year in range(2007, 2018)]
            replay = ','.join(gen.choice(history, years_to_simulate, replace=True))

        # Lookup the dome closed fractions for each night of the survey.
        # This step is deterministic and only depends on the config weather
        # parameter, which specifies which year(s) of historical daily
        # weather to replay during the simulation.
        dome_closed_frac = desimodel.weather.dome_closed_fractions(
            start_date, stop_date, replay=replay)
        r = gen.uniform(size=num_nights)
        r2 = gen.uniform(size=num_nights)
        dome_closed_frac[r2 < extra_downtime] = 1.

        # Convert fractions of scheduled time to hours per night.
        ilo, ihi = (start_date - ephem.start_date).days, (stop_date - ephem.start_date).days
        bright_dusk = ephem._table['brightdusk'].data[ilo:ihi]
        bright_dawn = ephem._table['brightdawn'].data[ilo:ihi]
        dome_closed_time = dome_closed_frac * (bright_dawn - bright_dusk)

        # Randomly pick between three scenarios for partially closed nights:
        # 1. closed from dusk, then open the rest of the night.
        # 2. open at dusk, then closed for the rest of the night.
        # 3. open and dusk and dawn, with a closed period during the night.
        # Pick scenarios 1+2 with probability equal to the closed fraction.
        # Use a fixed number of random numbers to decouple from the seeing
        # and transparency sampling below.
        self._table['open'] = np.ones(num_rows, bool)
        for i in range(num_nights):
            sl = slice(i * steps_per_day, (i + 1) * steps_per_day)
            night_mjd = self._table['mjd'][sl]
            # Dome is always closed before dusk and after dawn.
            closed = (night_mjd < bright_dusk[i]) | (night_mjd >= bright_dawn[i])
            if dome_closed_frac[i] == 0:
                # Dome open all night.
                pass
            elif dome_closed_frac[i] == 1:
                # Dome closed all night. This occurs with probability frac / 2.
                closed[:] = True
            elif r[i] < 0.5 * dome_closed_frac[i]:
                # Dome closed during first part of the night.
                # This occurs with probability frac / 2.
                closed |= (night_mjd < bright_dusk[i] + dome_closed_time[i])
            elif r[i] < dome_closed_frac[i]:
                # Dome closed during last part of the night.
                # This occurs with probability frac / 2.
                closed |= (night_mjd > bright_dawn[i] - dome_closed_time[i])
            else:
                # Dome closed during the middle of the night.
                # This occurs with probability 1 - frac.  Use the value of r[i]
                # as the fractional time during the night when the dome reopens.
                dome_open_at = bright_dusk[i] + r[i] * (bright_dawn[i] - bright_dusk[i])
                dome_closed_at = dome_open_at - dome_closed_time[i]
                closed |= (night_mjd >= dome_closed_at) & (night_mjd < dome_open_at)
            self._table['open'][sl][closed] = False

        self.start_date = start_date
        self.stop_date = stop_date
        self.num_nights = num_nights
        self.steps_per_day = steps_per_day
        self.replay = replay

    def save(self, filename, overwrite=True):
        """Save the generated weather to a file.

        The saved file can be restored using the constructor `restore`
        parameter.

        Parameters
        ----------
        filename : str
            Name of the file where the weather should be saved. A
            relative path name refers to the :meth:`configuration output path
            <desisurvey.config.Configuration.get_path>`.
        overwrite : bool
            Silently overwrite any existing file when this is True.
        """
        config = desisurvey.config.Configuration()
        filename = config.get_path(filename)
        self._table.write(filename, overwrite=overwrite)
        self.log.info('Saved weather to {0}.'.format(filename))

    def get(self, time):
        """Get the weather conditions at the specified time(s).

        Returns the conditions at the closest tabulated time, rather than
        using interpolation.

        Parameters
        ----------
        time : astropy.time.Time
            Time(s) when the simulated weather is requested.

        Returns
        -------
        table slice
            Slice of precomputed table containing row(s) corresponding
            to the requested time(s).
        """
        offset = np.floor(
            (time.mjd - self._table['mjd'][0]) * self.steps_per_day + 0.5
            ).astype(int)
        if np.any(offset < 0) or np.any(offset > len(self._table)):
            raise ValueError('Cannot get weather beyond tabulated range.')
        return self._table[offset]
