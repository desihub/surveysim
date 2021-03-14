import unittest
import datetime

import numpy as np

import astropy.time

import desisurvey.config
import desisurvey.ephem
import desisurvey.tiles

from desisurvey.test.base import Tester
from surveysim.stats import SurveyStatistics


class TestStats(Tester):

    def test_basic(self):
        tiles = desisurvey.tiles.get_tiles()
        gen = np.random.RandomState(seed=1)
        stats = SurveyStatistics()
        # Accumulate some stats
        num_nights = (self.stop - self.start).days
        for i in range(num_nights):
            night = self.start + datetime.timedelta(i)
            nightstats = stats.get_night(night)
            nightstats['tsched'] = gen.uniform()
            for pidx in range(len(tiles.programs)):
                nightstats['topen'][pidx] = gen.uniform()
            for program in tiles.programs:
                progidx = tiles.program_index[program]
                nightstats['tscience'][progidx] = gen.uniform()
                nightstats['nexp'][progidx] = progidx
        # Save and restore
        stats.save('stats_test.fits', comment='unit test')
        stats2 = SurveyStatistics(restore='stats_test.fits')
        # Check for consistency
        self.assertEqual(stats._data.dtype, stats2._data.dtype)
        for name in stats._data.dtype.names:
            self.assertTrue(np.array_equal(stats._data[name], stats2._data[name]))


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
