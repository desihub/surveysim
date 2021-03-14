"""Simulation utilities that may be used by other packages.
"""
from __future__ import print_function, division, absolute_import

import numpy as np

import astropy.table

import desiutil.log

import desisurvey.utils
import desisurvey.tiles

import surveysim.exposures


def add_calibration_exposures(exposures, flats_per_night=3, arcs_per_night=3,
                              darks_per_night=0, zeroes_per_night=0,
                              exptime=None, readout=30.0):
    """Prepare a list of science exposures for desisim.wrap-newexp.

    Insert calibration exposures at the start of each night, and add
    the following columns for all exposures: EXPID, PROGRAM, NIGHT,
    FLAVOR.

    Parameters
    ----------
    exposures : table like or :class:`surveysim.exposures.ExposureList`
        A table of science exposures including, at a minimum,
        MJD, EXPTIME and TILEID columns. The exposures must be sorted
        by increasing MJD. Could be a numpy recarray, an astropy
        table, or an ExposureList object. Columns other than
        the required ones are copied to the output.
    flats_per_night : :class:`int`, optional
        Add this many arc exposures per night (default 3).
    arcs_per_night : :class:`int`, optional
        Add this many arc exposures per night (default 3).
    darks_per_night : :class:`int`, optional
        Add this many dark exposures per night (default 0).
    zeroes_per_night : :class:`int`, optional
        Add this many zero exposures per night (default 0).
    exptime : :class:`dict`, optional
        A dictionary setting calibration exposure times for each
        calibration flavor.
    readout : :class:`float`, optional
        Set readout time for calibration exposures (default 30.0 s).

    Returns
    -------
    :class:`astropy.table.Table`
        The output table augmented with calibration exposures and
        additional columns.

    Raises
    ------
    ValueError
        If the input is not sorted by increasing MJD/timestamp.
    """
    if isinstance(exposures, surveysim.exposures.ExposureList):
        exposures = exposures._exposures[:exposures.nexp]
    nexp = len(exposures)
    MJD = exposures['MJD']
    if not np.all(np.diff(MJD) > 0):
        raise ValueError("Input is not sorted by increasing MJD!")
    if exptime is None:
        exptime = {'flat': 10.0, 'arc': 10.0, 'dark': 1000.0, 'zero': 0.0}

    # Define the start of night calibration sequence.
    calib_time = lambda x: exptime[x] + readout
    calib_sequence = (['arc']*arcs_per_night + ['flat']*flats_per_night +
                      ['dark']*darks_per_night + ['zero']*zeroes_per_night)
    calib_times = np.cumsum(np.array([calib_time(c) for c in calib_sequence]))[::-1]

    # Group exposures by night.
    MJD0 = desisurvey.utils.local_noon_on_date(desisurvey.utils.get_date(MJD[0])).mjd
    night_idx = np.floor(MJD - MJD0).astype(int)
    nights = np.unique(night_idx)
    ncalib = len(calib_sequence) * len(nights)

    # Initialize the output table.
    output = astropy.table.Table()
    nout = nexp + ncalib
    output['EXPID'] = np.arange(nout, dtype=np.int32)
    template = astropy.table.Table(dtype=exposures.dtype)
    template.remove_column('EXPID')
    for colname in template.colnames:
        col = template[colname]
        output[colname] = astropy.table.Column(dtype=col.dtype, length=nout)
    output['PROGRAM'] = astropy.table.Column(dtype=(str, len('BRIGHT')), length=nout)
    output['NIGHT'] = astropy.table.Column(dtype=(str, len('YYYYMMDD')), length=nout)
    output['FLAVOR'] = astropy.table.Column(dtype=(str, len('science')), length=nout)
    tiles = desisurvey.tiles.get_tiles()

    # Moon parameters are hardcoded for now.
    output['MOONFRAC'] = 0.5
    output['MOONALT'] = -10.
    output['MOONSEP'] = 90.

    # Loop over nights.
    out_idx = 0
    for n in nights:
        sel = (night_idx == n)
        nsel = np.count_nonzero(sel)
        first = np.where(sel)[0][0]
        MJD_first = MJD[first]
        NIGHT = desisurvey.utils.get_date(MJD_first).isoformat().replace('-', '')
        # Append the calibration sequence.
        for j, c in enumerate(calib_sequence):
            output['MJD'][out_idx] = MJD_first - calib_times[j]/86400.0
            output['EXPTIME'][out_idx] = exptime[c]
            output['TILEID'][out_idx] = -1
            output['PROGRAM'][out_idx] = 'CALIB'
            output['NIGHT'][out_idx] = NIGHT
            output['FLAVOR'][out_idx] = c
            out_idx += 1
        # Append the night's science exposures.
        outslice = slice(out_idx, out_idx + nsel)
        for colname in template.colnames:
            output[colname][outslice] = exposures[colname][sel]
        TILEIDs = exposures['TILEID'][sel]
        output['PROGRAM'][outslice] = tiles.tileprogram[tiles.index(TILEIDs)]
        output['NIGHT'][outslice] = NIGHT
        output['FLAVOR'][outslice] = 'science'
        out_idx += nsel
    assert out_idx == nout

    log = desiutil.log.get_logger()
    log.info('Added {} nightly calibration sequences of {} exposures each to {} science exposures.'
             .format(len(nights), len(calib_sequence), nexp))
    return output
