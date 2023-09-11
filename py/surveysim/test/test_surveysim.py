# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
import unittest
import os.path

import desisurvey.config
import desisurvey.scripts.surveyinit

import surveysim.stats
import surveysim.exposures
import surveysim.scripts.surveysim

from desisurvey.test.base import Tester


class TestSimulator(Tester):

    def test_end_to_end(self):
        cmd = 'surveyinit --max-cycles 5 --init zero'
        args = desisurvey.scripts.surveyinit.parse(cmd.split()[1:])
        desisurvey.scripts.surveyinit.main(args)
        cmd = 'surveysim --rules rules-layers.yaml --name test'
        args = surveysim.scripts.surveysim.parse(cmd.split()[1:])
        surveysim.scripts.surveysim.main(args)

        config = desisurvey.config.Configuration()
        self.assertTrue(os.path.exists(config.get_path('stats_test.fits')))
        self.assertTrue(os.path.exists(config.get_path('exposures_test.fits')))
        stats = surveysim.stats.SurveyStatistics(restore='stats_test.fits')
        exposures = surveysim.exposures.ExposureList(restore='exposures_test.fits')


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
