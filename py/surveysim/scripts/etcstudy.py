"""Script to run simulations for exposure-time studies.

Use the BGS_ETC_Study notebook to generate the input files we need.

This should eventually be combined with the corresponding ELG study.

Instructions for running at NERSC are messy because several package must
be run out of dev branches for now::

    module unload desimodel
    module load desimodel/master
    module unload specsim
    export PYTHONPATH=$SCRATCH/desi/code/surveysim/py:$PYTHONPATH
    export PATH=$SCRATCH/desi/code/surveysim/bin:$PATH
    module unload surveysim
    export PYTHONPATH=$SCRATCH/desi/code/specsim:$PYTHONPATH
    module unload desisurvey
    export PYTHONPATH=$SCRATCH/desi/code/desisurvey/py:$PYTHONPATH
    export DESISURVEY_OUTPUT=/global/projecta/projectdirs/desi/datachallenge/surveysim2017/depth_0m
    cd ~/jupyter
    etcstudy --verbose --seed 1 --texp 300 --simpar bgs_simargs.fits --nominal bgs_nominal --progress progress.fits --ncond 10 --output bgs_1.fits
"""
from __future__ import print_function, division, absolute_import

import argparse
import sys

import numpy as np

import astropy.table
import astropy.units as u

import desisurvey.utils
import desisurvey.progress

import specsim.simulator

import desiutil.log


def parse(options=None):
    """Parse command-line options.

    Note that the nominal seeing (1.1") is implicitly fixed by the fiberloss
    fractions we load and cannot be varied in this study.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action='store_true',
        help='display log messages with severity >= info')
    parser.add_argument('--debug', action='store_true',
        help='display log messages with severity >= debug (implies verbose)')
    parser.add_argument(
        '--seed', type=int, default=123, metavar='N',
        help='random number seed for reproducible results')
    parser.add_argument(
        '--texp', type=float, default=1000., metavar='TEXP',
        help='fixed exposure time in seconds to use for simulations')
    parser.add_argument(
        '--simpar', default=None, metavar='FILE',
        help='name of file containing tabulated simulation parameters to load')
    parser.add_argument(
        '--nominal', default=None, metavar='FILE',
        help='name of file with simulated nominal conditions')
    parser.add_argument(
        '--progress', default=None, metavar='FILE',
        help='name of progress file to use for sampling observing conditions')
    parser.add_argument(
        '--output', default='etcstudy.fits', metavar='FILE',
        help='name of output file to write with results table')
    parser.add_argument(
        '--nspec', type=int, default=50, metavar='NS',
        help='number of spectra to simulate for each observing condition')
    parser.add_argument(
        '--ncond', type=int, default=250, metavar='NC',
        help='number of observing conditions to simulate')
    parser.add_argument(
        '--interval', type=int, default=50, metavar='NP',
        help='log progress messages with this interval')
    parser.add_argument(
        '--nomoon', action='store_true',
        help='only simulate conditions with no scattered moon')
    parser.add_argument(
        '--nosun', action='store_true',
        help='only simulate conditions with no scattered twilight sun')

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def BGS_SNR(cameras):
    # Calculate the median SNR / 0.5A pixel in the red camera (5621.7-7744.3A).
    num_source_electrons = cameras[1]['num_source_electrons']
    variance_electrons = cameras[1]['variance_electrons']
    # Ignore dark current so that variance is proportional to exposure time.
    variance_electrons -= cameras[1]['read_noise_electrons'] ** 2
    return np.median(num_source_electrons / np.sqrt(variance_electrons), axis=0)


def main(args):
    """Command-line driver for running survey simulations.
    """
    if args.simpar is None:
        print('Missing required --simpar file name.')
        sys.exit(-1)
    if args.progress is None:
        print('Missing required --progress file name.')
        sys.exit(-1)

    desisurvey.utils.freeze_iers()

    # Set up the logger
    if args.debug:
        log = desiutil.log.get_logger(desiutil.log.DEBUG)
        args.verbose = True
    elif args.verbose:
        log = desiutil.log.get_logger(desiutil.log.INFO)
    else:
        log = desiutil.log.get_logger(desiutil.log.WARNING)

    # Initialize random numbers.
    gen = np.random.RandomState(args.seed)

    # Load the progress file to use.
    progress = desisurvey.progress.Progress(restore=args.progress)
    conditions = progress.get_exposures(
        tile_fields='', exp_fields=
        'airmass,seeing,transparency,moonfrac,moonalt,moonsep')
    moon_up = np.where(conditions['MOONALT'] > 0)[0]
    nmoon = len(moon_up)
    log.info('Loaded conditions for {} ({} moon up) exposures.'
             .format(len(conditions), nmoon))

    # Shuffle moon-up conditions to simulate below.
    moon_conditions = conditions[moon_up][gen.permutation(nmoon)]

    # Load the tabulated simulation parameters to use.
    simpar = astropy.table.Table.read(args.simpar)
    z_array = simpar['z'].copy()
    source_fluxes=np.array(simpar['source_fluxes'])
    fiber_acceptance_fraction=np.array(simpar['fiber_acceptance_fraction'])
    log.info('Loaded {} targets with <z> = {:.2f} from {}'
             .format(len(z_array), np.median(z_array), args.simpar))
    if args.nspec > len(z_array):
        log.warn('Requested nspec > number of spectra available.')
        args.nspec = len(z_array)
    flux_unit = u.Unit("erg / (Angstrom cm2 s)")

    # Initialize the simulator.
    desi = specsim.simulator.Simulator('desi', num_fibers=args.nspec)
    desi.observation.exposure_time = args.texp * u.s
    moon = desi.atmosphere.moon
    twilight = desi.atmosphere.twilight
    log.info('Initialized specsim for {} fibers and texp={:.1f}.'
             .format(args.nspec, desi.observation.exposure_time))

    # Load the simulated spectra under nominal conditions.
    nominal = []
    for i, band in enumerate('rrr'): #enumerate(desi.camera_names):
        nominal.append(astropy.table.Table.read(
            '{}_{}.fits'.format(args.nominal, band)))

    # Calculate the SNR under nominal conditions and clean up.
    snr0 = BGS_SNR(nominal)
    log.info('Median SNR under nominal conditions is {:.3f}'
             .format(np.median(snr0)))
    del nominal

    # Initialize the table of output results.
    results = astropy.table.Table(
        meta=dict(seed=args.seed, texp=args.texp, nspec=args.nspec))
    results['airmass'] = astropy.table.Column(
        length=args.ncond, format='%.1f',
        description='Airmass of observation')
    results['moonfrac'] = astropy.table.Column(
        length=args.ncond, format='%.3f',
        description='Moon illuminated fraction (0-1)')
    results['moonalt'] = astropy.table.Column(
        length=args.ncond, format='%.1f',
        description='Moon altitude angle in degrees', unit='deg')
    results['moonsep'] = astropy.table.Column(
        length=args.ncond, format='%.1f',
        description='Moon-tile separation angle in degrees', unit='deg')
    results['moonv'] = astropy.table.Column(
        length=args.ncond, format='%.1f',
        description='V-band magnitude of scattered moonlight')
    results['sunalt'] = astropy.table.Column(
        length=args.ncond, format='%.1f',
        description='Sun altitude angle in degrees, clipped below -20',
        unit='deg')
    results['sundaz'] = astropy.table.Column(
        length=args.ncond, format='%.1f',
        description='Sun-tile relative azimuth angle in degrees',
        unit='deg')
    results['sunr'] = astropy.table.Column(
        length=args.ncond, format='%.1f',
        description='r-band magnitude of twilight scattered sun')
    results['tratio'] = astropy.table.Column(
        length=args.ncond, shape=(3,), format='%.3f',
        description='(5, 50, 95) percentile exposure time ratios')

    # Default fixed moon conditions.
    moonfrac, moonalt, moonsep = 0., -30., 90.

    # Default fixed twilight conditions for now.
    sunalt, sundaz = -25., 180.

    # Loop over simulated observing conditions.
    for i in range(args.ncond):

        # Always set airmass from the moon-up conditions.
        cond = moon_conditions[i % nmoon]
        airmass = cond['AIRMASS']
        desi.atmosphere.airmass = airmass

        # Set moon observing conditions.
        if not args.nomoon:
            moonfrac, moonalt, moonsep = (
                cond['MOONFRAC'], cond['MOONALT'], cond['MOONSEP'])

        moon.moon_phase = np.arccos(2 * moonfrac - 1) / np.pi
        moon.moon_zenith = (90 - moonalt) * u.deg
        moon.separation_angle = moonsep * u.deg

        # Set twilight observing conditions.
        if not args.nosun:
            # Generate uniform twilight conditions.
            sunalt = gen.uniform(-12., -18.)
            sundaz = gen.uniform(-180., 180.)

        twilight.sun_altitude = sunalt * u.deg
        twilight.sun_relative_azimuth = sundaz * u.deg

        # Pick a random subset of spectra to simulate.
        idx = gen.permutation(len(z_array))[:args.nspec]

        # Run the simulation.
        desi.simulate(
            source_fluxes=source_fluxes[idx] * flux_unit,
            fiber_acceptance_fraction=fiber_acceptance_fraction[idx])

        # Calculate the SNR of each simulated spectrum.
        snr = BGS_SNR(desi.camera_output)

        # Calculate the exposure-time ratio for each simulated spectrum.
        # We ignore the small violation of snr ~ sqrt(t) scaling due to
        # to readout noise for now.
        tratio = (snr0[idx] / snr) ** 2

        # Look up scattered magnitudes.
        moonv = moon.scattered_V
        sunr = twilight.scattered_r

        # Save the results.
        row = results[i]
        row['airmass'] = airmass
        row['moonfrac'] = moonfrac
        row['moonsep'] = moonsep
        row['moonalt'] = moonalt
        row['moonv'] = moonv.value if moonv else -np.inf
        row['sunalt'] = sunalt
        row['sundaz'] = sundaz
        row['sunr'] =  sunr.value if sunr else -np.inf
        row['tratio'] = np.percentile(tratio, (5, 50, 95))

        if i % args.interval == 0:
            log.info('Completed {} / {} simulations.'
                     .format(i + 1, args.ncond))

    results.write(args.output, overwrite=True)
