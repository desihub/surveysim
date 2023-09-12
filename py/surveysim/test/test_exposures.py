# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
import unittest
import datetime

import numpy as np

import astropy.time

import desisurvey.config
import desisurvey.ephem
import desisurvey.tiles

from desisurvey.test.base import Tester
from surveysim.exposures import ExposureList


class TestExposures(Tester):

    def test_basic(self):
        tiles = desisurvey.tiles.get_tiles()
        gen = np.random.RandomState(seed=1)
        exp = ExposureList()
        # Accumulate some exposures
        now = 0.
        for tileID in tiles.tileID[:100]:
            exp.add(now, gen.uniform(), tileID, *gen.uniform(size=6))
        # Save and restore
        exp.save('exposures_test.fits', comment='unit test')
        exp2 = ExposureList(restore='exposures_test.fits')
        # Check for consistency
        # self.assertEqual(stats._data.dtype, stats2._data.dtype)
        # for name in stats._data.dtype.names:
        #     self.assertTrue(np.array_equal(stats._data[name], stats2._data[name]))


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
