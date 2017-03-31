import unittest, os, shutil, uuid
import numpy as np
from astropy.table import Table
from astropy import units
from astropy.time import Time

class TestSurveySim(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.origdir = os.getcwd()
        cls.testdir = os.path.abspath('./test-{}'.format(uuid.uuid4()))
        os.mkdir(cls.testdir)
        os.chdir(cls.testdir)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.origdir)
        if os.path.exists(cls.testdir):
            shutil.rmtree(cls.testdir)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_surveysim(self):
        from surveysim.surveysim import surveySim
        surveySim((2019,9,1), (2019,9,8), seed=123456, use_jpl=False)

        #- A plan should exist for every night
        for i in range(1,9):
            self.assertTrue(os.path.exists('obsplan201909{:02d}.fits'.format(i)))

        #- Observations will exist for just some nights
        self.assertTrue(os.path.exists('obslist_all.fits'))
        self.assertTrue(os.path.exists('tiles_observed.fits'))
        
        #- Confirm that observations map to obslistYEARMMDD.fits files
        obs = Table.read('obslist_all.fits')
        nights = set()
        for dateobs in obs['DATE-OBS']:
            #- convert DATE-OBS into NIGHT of sunset
            localtime = Time(dateobs) - 7*units.hour   #- AZ = UTC-7
            sunset_date = (localtime - 12*units.hour).to_datetime().strftime('%Y%m%d')
            nights.add(sunset_date)

        for night in sorted(nights):
            obsfile = 'obslist{}.fits'.format(night)
            self.assertTrue(os.path.exists(obsfile), 'Missing {}'.format(obsfile))

if __name__ == '__main__':
    unittest.main()
