"""Script wrapper for running survey simulations.

Simulate a sequence of observations until either the nominal
survey end date is reached, or all tiles have been observed.
See doc/tutorial.rst for details.

To run this script from the command line, use the ``surveysim``
entry point that is created when this package is installed, and
should be in your shell command search path.
"""
from __future__ import print_function, division, absolute_import

import argparse
import datetime
import os
import warnings
import sys

import numpy as np

import desiutil.log

import desisurvey.config
import desisurvey.rules
import desisurvey.plan
import desisurvey.scheduler

import surveysim.weather
import surveysim.stats
import surveysim.exposures
import surveysim.nightops


def parse(options=None):
    """Parse command-line options for running survey simulations.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action='store_true',
        help='display log messages with severity >= info')
    parser.add_argument('--debug', action='store_true',
        help='display log messages with severity >= debug (implies verbose)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
        help='nightly interval for logging periodic info messages')
    parser.add_argument(
        '--start', type=str, default=None, metavar='DATE',
        help='survey starts on the evening of this day, formatted as YYYY-MM-DD')
    parser.add_argument(
        '--stop', type=str, default=None, metavar='DATE',
        help='survey stops on the morning of this day, formatted as YYYY-MM-DD')
    parser.add_argument(
        '--name', type=str, default='surveysim', metavar='NAME',
        help='name to use for saving simulated stats and exposures')
    parser.add_argument(
        '--comment', type=str, default='', metavar='COMMENT',
        help='comment to save with simulated stats and exposures')
    parser.add_argument(
        '--rules', type=str, default='rules.yaml', metavar='YAML',
        help='name of YAML file with survey strategy rules to use')
    parser.add_argument('--twilight', action='store_true',
        help='include twilight in the scheduled time')
    parser.add_argument('--save-restore', action='store_true',
        help='save/restore the planner and scheduler state after each night')
    parser.add_argument(
        '--seed', type=int, default=1, metavar='N',
        help='random number seed for generating random observing conditions')
    parser.add_argument(
        '--replay', type=str, default='random', metavar='REPLAY',
        help='Replay specific weather years, e.g., "Y2015,Y2011" or "random"')
    parser.add_argument(
        '--output-path', default=None, metavar='PATH',
        help='output path to use instead of config.output_path')
    parser.add_argument(
        '--tiles-file', default=None, metavar='TILES',
        help='name of tiles file to use instead of config.tiles_file')
    parser.add_argument(
        '--config-file', default='config.yaml', metavar='CONFIG',
        help='input configuration file')

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    # Validate start/stop date args and convert to datetime objects.
    # Unspecified values are taken from our config.
    config = desisurvey.config.Configuration(file_name=args.config_file)
    if args.start is None:
        args.start = config.first_day()
    else:
        try:
            args.start = desisurvey.utils.get_date(args.start)
        except ValueError as e:
            raise ValueError('Invalid start: {0}'.format(e))
    if args.stop is None:
        args.stop = config.last_day()
    else:
        try:
            args.stop = desisurvey.utils.get_date(args.stop)
        except ValueError as e:
            raise ValueError('Invalid stop: {0}'.format(e))
    if args.start >= args.stop:
        raise ValueError('Expected start < stop.')

    return args


def main(args):
    """Command-line driver for running survey simulations.
    """
    # Set up the logger
    if args.debug:
        os.environ['DESI_LOGLEVEL'] = 'DEBUG'
        args.verbose = True
    elif args.verbose:
        os.environ['DESI_LOGLEVEL'] = 'INFO'
    else:
        os.environ['DESI_LOGLEVEL'] = 'WARNING'
    log = desiutil.log.get_logger()

    # Set the output path if requested.
    config = desisurvey.config.Configuration()
    if args.output_path is not None:
        config.set_output_path(args.output_path)
    if args.tiles_file is not None:
        config.tiles_file.set_value(args.tiles_file)

    # Initialize simulation progress tracking.
    stats = surveysim.stats.SurveyStatistics(args.start, args.stop)
    explist = surveysim.exposures.ExposureList()

    # Initialize the survey strategy rules.
    rules = desisurvey.rules.Rules(args.rules)
    
    # Initialize afternoon planning.
    planner = desisurvey.plan.Planner(rules)

    # Initialize next tile selection.
    scheduler = desisurvey.scheduler.Scheduler()

    # Generate random weather conditions.
    weather = surveysim.weather.Weather(seed=args.seed, replay=args.replay)

    # Loop over nights.
    num_simulated = 0
    num_nights = (args.stop - args.start).days
    for num_simulated in range(num_nights):
        night = args.start + datetime.timedelta(num_simulated)

        if args.save_restore and num_simulated > 0:
            # Restore the planner and scheduler saved after the previous night.
            planner = desisurvey.plan.Planner(rules, restore='desi-status_{}.fits'.format(last_night))
            scheduler = desisurvey.scheduler.Scheduler(restore='desi-status_{}.fits'.format(last_night))
            scheduler.update_tiles(planner.tile_available, planner.tile_priority)

        # Perform afternoon planning.
        explist.update_tiles(night, *scheduler.update_tiles(*planner.afternoon_plan(night, scheduler.completed)))

        if not desisurvey.utils.is_monsoon(night) and not scheduler.ephem.is_full_moon(night):
            # Simulate one night of observing.
            surveysim.nightops.simulate_night(
                night, scheduler, stats, explist, weather=weather, use_twilight=args.twilight)
            if scheduler.survey_completed():
                log.info('Survey complete on {}.'.format(night))
                break

        if args.save_restore:
            last_night = night.isoformat()
            planner.set_donefrac(scheduler.tiles.tileID, scheduler.snr2frac,
                                 scheduler.lastexpid)
            planner.save('desi-status_{}.fits'.format(last_night))

        if num_simulated % args.log_interval == args.log_interval - 1:
            log.info('Completed {} / {} tiles after {} / {} nights.'.format(
                scheduler.completed_by_pass.sum(), scheduler.tiles.ntiles,
                num_simulated + 1, num_nights))

    explist.save('exposures_{}.fits'.format(args.name), comment=args.comment)
    stats.save('stats_{}.fits'.format(args.name), comment=args.comment)
    if args.verbose:
        stats.summarize()
