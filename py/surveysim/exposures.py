"""Record simulated exposures and collect per-tile statistics.
"""
from __future__ import print_function, division, absolute_import

import numpy as np

import astropy.io.fits

import desiutil.log

import desisurvey.config
import desisurvey.tiles


class ExposureList(object):
    """Record simulated exposures and collect per-tile statistics.

    Parameters
    ----------
    restore : str or None
        Restore internal state from the snapshot saved to this filename,
        or initialize a new object when None. Use :meth:`save` to
        save a snapshot to be restored later. Filename is relative to
        the configured output path unless an absolute path is
        provided.
    max_nexp : int
        The maximum expected number of exposures, which determines the
        memory size of this object.
    """
    def __init__(self, restore=None, max_nexp=60000):
        self.tiles = desisurvey.tiles.get_tiles()
        self._exposures = np.empty(max_nexp, dtype=[
            ('ID', np.float64),
            ('MJD', np.float64),
            ('EXPTIME', np.float32),
            ('TILEID', np.int32),
            ('SNR2FRAC', np.float32),
            ('AIRMASS', np.float32),
            ('SEEING', np.float32),
            ('TRANSP', np.float32),
            ('SKY', np.float32),
        ])
        self._tiledata = np.empty(self.tiles.ntiles, dtype=[
            ('AVAIL', np.int32),
            ('PLANNED', np.int32),
            ('EXPTIME', np.float32),
            ('SNR2FRAC', np.float32),
            ('NEXP', np.int32)
        ])
        if restore is not None:
            config = desisurvey.config.Configuration()
            fullname = config.get_path(restore)
            with astropy.io.fits.open(fullname, memmap=False) as hdus:
                header = hdus[0].header
                comment = header['COMMENT']
                self.nexp = header['NEXP']
                self._exposures[:self.nexp] = hdus['EXPOSURES'].data
                self._tiledata[:] = hdus['TILEDATA'].data
                self.initial_night = desisurvey.utils.get_date(header['INITIAL']) if header['INITIAL'] else None
            log = desiutil.log.get_logger()
            log.info('Restored stats from {}'.format(fullname))
            if comment:
                log.info('  Comment: "{}".'.format(comment))
        else:
            self.nexp = 0
            self._tiledata['AVAIL'] = -1
            self._tiledata['PLANNED'] = -1
            self._tiledata['EXPTIME'] = 0.
            self._tiledata['SNR2FRAC'] = 0.
            self._tiledata['NEXP'] = 0
            self.initial_night = None

    def update_tiles(self, night, available, planned):
        """Update tile availability and planning status.

        Parameters
        ----------
        night : datetime.date
            Night of initial observing.
        available : array
            Array of tile indices that are newly available since the last
            call to this method.
        planned : array
            Array of tile indices that are newly planned (priority > 0) since
            the last call to this method.
        """
        if self.initial_night is None:
            self.initial_night = night
            night_index = 0
        else:
            night_index = (night - self.initial_night).days
            if night_index < 0:
                raise ValueError('night must be advancing.')
        self._tiledata['AVAIL'][available] = night_index
        self._tiledata['PLANNED'][planned] = night_index

    def add(self, mjd, exptime, tileID, snr2frac, airmass, seeing, transp, sky):
        """Record metadata for a single exposure.

        Parameters
        ----------
        mjd : float
            MJD timestamp at the start of the exposure.
        tileID : int
            ID of the observed tile.
        snr2frac : float
            Fractional SNR2 accumulated on this tile during this exposure.
        airmass : float
            Average airmass during this exposure.
        seeing : float
            Average atmospheric seeing in arcseconds during this exposure.
        transp : float
            Average atmospheric transparency during this exposure.
        sky : float
            Average sky background level during this exposure.
        """
        if self.nexp >= len(self._exposures):
            raise RuntimeError(
                'Need to increase max_nexp={}'.format(len(self._exposures)))
        self._exposures[self.nexp] = (
            self.nexp, mjd, exptime, tileID, snr2frac, airmass, seeing,
            transp, sky)
        self.nexp += 1
        tileinfo = self._tiledata[self.tiles.index(tileID)]
        tileinfo['EXPTIME'] += exptime
        tileinfo['SNR2FRAC'] = snr2frac
        tileinfo['NEXP'] += 1

    def save(self, name='exposures.fits', comment='', overwrite=True):
        """Save exposures to a FITS file with two binary tables.

        The saved file size scales linearly with the number of exposures
        added so far, and is independent of the memory size of this
        object.

        Parameters
        ----------
        name : str
            File name to write. Will be located in the configuration
            output path unless it is an absolute path. Pass the same
            name to the constructor's ``restore`` argument to restore
            this snapshot.
        comment : str
            Comment to include in the saved header, for documentation
            purposes.
        overwrite : bool
            Silently overwrite any existing file when True.
        """
        hdus = astropy.io.fits.HDUList()
        header = astropy.io.fits.Header()
        header['TILES'] = self.tiles.tiles_file
        header['NEXP'] = self.nexp
        header['COMMENT'] = comment
        header['INITIAL'] = self.initial_night.isoformat() if self.initial_night else ''
        header['EXTNAME'] = 'META'
        hdus.append(astropy.io.fits.PrimaryHDU(header=header))
        hdus.append(astropy.io.fits.BinTableHDU(self._exposures[:self.nexp], name='EXPOSURES'))
        hdus.append(astropy.io.fits.BinTableHDU(self._tiledata, name='TILEDATA'))
        config = desisurvey.config.Configuration()
        name = config.get_path(name)
        hdus.writeto(name, overwrite=overwrite)
        log = desiutil.log.get_logger()
        log.info('Saved {} exposures to {}'.format(self.nexp, name))
        if comment:
            log.info('Saved with comment "{}".'.format(header['COMMENT']))


def load(name, extra_nexp=0):
    config = desisurvey.config.Configuration()
    name = config.get_path(name)
    with astropy.io.fits.open(name) as hdus:
        header = hdus[0].header
        comment = header['COMMENT']
        nexp = header['NEXP']
        max_nexp = nexp + extra_nexp
        explist = ExposureList(max_nexp=max_nexp)
        explist._exposures[:nexp] = hdus['EXPOSURES'].data
        explist._tiledata[:] = hdus['TILEDATA'].data
    log = desiutil.log.get_logger()
    log.info('Loaded {} exposures from {}'.format(nexp, name))
    if comment:
        log.info('Loaded with comment "{}".'.format(comment))
    return explist
