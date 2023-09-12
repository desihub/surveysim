# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
===================
surveysim.simulator
===================

Top-level survey simulation manager.
"""

import datetime

import numpy as np

import astropy.table
import astropy.time
import astropy.units as u

import desiutil.log

import desisurvey.ephem
import desisurvey.old.schedule
import desisurvey.utils
import desisurvey.config

import surveysim.nightops
import surveysim.weather


class Simulator(object):
    """Initialize a survey simulation.

    Parameters
    ----------
    start_date : datetime.date
        Survey starts on the evening of this date.
    stop_date : datetime.date
        Survey stops on the morning of this date.
    progress : desisurvey.progress.Progress
        Progress of survey at the start of this simulation.
    weather : surveysim.weather.Weather
        Simulated weather conditions use use.
    stats : astropy.table.Table or None
        Table of per-night efficiency statistics to update.
    strategy : str
        Strategy to use for scheduling tiles during each night.
    plan : str
        Name of plan file to use.
    gen : numpy.random.RandomState or None
        Random number generator to use for reproducible samples. Will be
        initialized (un-reproducibly) if None.
    """
    def __init__(self, start_date, stop_date, progress, weather, stats,
                 strategy, plan, gen=None):
        self.log = desiutil.log.get_logger()
        self.config = desisurvey.config.Configuration()

        # Validate date range.
        if start_date >= stop_date:
            raise ValueError('Expected start_date < stop_date.')
        if (start_date < self.config.first_day() or stop_date > self.config.last_day()):
            raise ValueError('Cannot simulate beyond nominal survey dates.')
        self.start_date = start_date
        self.stop_date = stop_date
        self.last_index = (self.stop_date - self.config.first_day()).days

        # Load the cached empherides to use.
        self.ephem = desisurvey.ephem.get_ephem(use_cache=True)

        # Load the survey scheduler to use.
        self.sp = desisurvey.old.schedule.Scheduler()

        self.strategy = strategy
        self.gen = gen
        self.weather = weather

        # Load the plan to use.
        self.plan = astropy.table.Table.read(self.config.get_path(plan))
        assert np.all(self.sp.tiles['tileid'] == self.plan['tileid'])

        if (stats is not None) and (len(stats) != (self.config.last_day() - self.config.first_day()).days):
            raise ValueError('Input stats table has wrong length.')
        self.stats = stats

        self.day_index = (self.start_date - self.config.first_day()).days
        self.survey_done = False
        self.completed = progress.completed()
        self.progress = progress
        self.log.info(
            'Will simulate {0} to {1} with {2:.1f} / {3} tiles completed.'
            .format(start_date, stop_date, self.completed, progress.num_tiles))

    @property
    def date(self):
        """The current simulation date as a datetime object.
        """
        return self.config.first_day() + datetime.timedelta(days=self.day_index)

    def next_day(self, scores=None):
        """Simulate the next day of survey operations.

        A day runs from local noon to local noon. A survey ends, with
        this method returning False, when either we reach the last
        scheduled day or else we run out of tiles to observe.

        Returns
        -------
        bool
            True if there are more days to simulate.
        """
        if self.day_index >= self.last_index or self.survey_done:
            return False

        self.log.info('Simulating {0}'.format(self.date))

        if desisurvey.utils.is_monsoon(self.date):
            self.log.info('No observing during monsoon.')
        elif self.ephem.is_full_moon(self.date):
            self.log.info('No observing during full moon.')
        else:

            # Simulate tonight's observing.
            totals = surveysim.nightops.nightOpsDeprecated(
                self.date, self.ephem, self.sp, self.weather, self.progress,
                self.strategy, self.plan, scores, self.gen)

            # Update our efficiency tracker.
            if self.stats is not None:
                for mode in totals:
                    self.stats[mode][self.day_index] = (
                        totals[mode].to(u.day).value)

            # Progress report.
            completed = self.progress.completed()
            self.log.info('Completed {0:.1f} tiles tonight.'
                          .format(completed - self.completed))
            self.completed = completed

            if self.progress.num_tiles - completed < 0.1:
                self.survey_done = True

        self.day_index += 1
        if self.day_index == self.last_index:
            self.survey_done = True

        return not self.survey_done
