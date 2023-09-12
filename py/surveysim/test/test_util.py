# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test surveysim.util.
"""
import unittest

import numpy as np

import astropy.table

import desisurvey.tiles

import surveysim.exposures

from desisurvey.test.base import Tester
from ..util import add_calibration_exposures


class TestUtil(Tester):
    """Test surveysim.util.
    """
    def test_add_calibration_exposures(self):
        """Test adding calibration exposures to science exposures.
        """
        config = desisurvey.config.Configuration()
        tiles = desisurvey.tiles.get_tiles()
        tileID = tiles.tileID[0]
        # List some science exposures.
        exposures = surveysim.exposures.ExposureList()
        exposures.add(58849., 1., tileID, 1., 1., 1., 1.1, 0.9, 1.0)
        exposures.add(58850., 1., tileID, 1., 1., 1., 1.1, 0.9, 1.0)
        for mode in 'obj', 'recarray', 'table':
            if mode == 'obj':
                input = exposures
            elif mode == 'recarray':
                input = exposures._exposures[:exposures.nexp]
            elif mode == 'table':
                exposures.save('exposures.fits')
                input = astropy.table.Table.read(
                    config.get_path('exposures.fits'), hdu='EXPOSURES')
            # Validate the output.
            output = add_calibration_exposures(input)
            self.assertEqual(len(output), 14)
            self.assertEqual('EXPID', output.colnames[0])
            self.assertTrue(np.all(output['EXPID'] == np.arange(14, dtype=np.int32)))
            self.assertTrue(np.all(np.diff(output['MJD']) >= 0))
            self.assertTrue(np.all(np.diff(output['EXPID']) == 1))

        # List some out-of-order science exposures.
        bad_exposures = surveysim.exposures.ExposureList()
        bad_exposures.add(58851., 1., tileID, 1., 1., 1., 1.1, 0.9, 1.0)
        bad_exposures.add(58850., 1., tileID, 1., 1., 1., 1.1, 0.9, 1.0)
        with self.assertRaises(ValueError):
            output = add_calibration_exposures(bad_exposures)


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
