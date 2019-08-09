"""Simulate one night of observing.
"""
from __future__ import print_function, division, absolute_import

import numpy as np

import desiutil.log

import desisurvey.utils
import desisurvey.etc
import desisurvey.plots


def simulate_night(night, scheduler, stats, explist, weather,
                   use_twilight=False, update_interval=10.,
                   plot=False, verbose=False):
    """Simulate one night of observing.

    Uses the online tile scheduler and exposure time calculator.

    Parameters
    ----------
    night : datetime.date
        Date that the simulated night starts.
    scheduler : :class:`desisurvey.scheduler.Scheduler`
        Next tile scheduler to use.
    stats : :class:`surveysim.stats.SurveyStatistics`
        Object for accumulating simulated survey statistics.
    explist : :class:`surveysim.exposures.ExposureList`
        Object for recording simulated exposures.
    weather : :class:`surveysim.weather.Weather`
        Simulated weather conditions to use.
    use_twlight : bool
        Observe during twilight when True.
    update_interval : float
        Interval in seconds for simulated ETC updates.
    plot : bool
        Generate a plot summarizing the simulated night when True.
    verbose : bool
        Produce verbose output on simulation progress when True.
    """
    log = desiutil.log.get_logger()
    update_interval_days = update_interval / 86400.
    night = desisurvey.utils.get_date(night)
    nightstats = stats.get_night(night)
    label = str(night)
    # Lookup this night's sunset and sunrise MJD values.
    night_ephem = scheduler.ephem.get_night(night)
    if use_twilight:
        begin = night_ephem['brightdusk']
        end = night_ephem['brightdawn']
    else:
        begin = night_ephem['dusk']
        end = night_ephem['dawn']
    nightstats['tsched'] = end - begin
    log.debug('Simulating observing on {} from MJD {:.5f} - {:.5f}.'
              .format(night, begin, end))

    # Find weather time steps that cover this night.
    weather_mjd = weather._table['mjd'].data
    ilo = np.searchsorted(weather_mjd, begin, side='left') - 1
    ihi = np.searchsorted(weather_mjd, end, side='right') + 2
    assert weather_mjd[ilo] < begin and weather_mjd[ihi] > end
    weather_mjd = weather_mjd[ilo:ihi]
    seeing = weather._table['seeing'].data[ilo:ihi]
    transp = weather._table['transparency'].data[ilo:ihi]
    # Fix this in the weather generator instead?
    transp = np.maximum(0.1, transp)
    dome = weather._table['open'].data[ilo:ihi]

    if not np.any(dome):
        log.debug('Dome closed all night.')
        return

    scheduler.init_night(night, use_twilight=use_twilight)
    ETC = desisurvey.etc.ExposureTimeCalculator(save_history=plot)
    nexp_last = explist.nexp

    # Build linear interpolators for observing conditions.
    # This implementation is faster than scipy.interpolate.interp1d()
    # when mjd values are gradually increasing.
    weather_idx = 0
    dmjd_weather = weather_mjd[1] - weather_mjd[0]
    def get_weather(mjd):
        nonlocal weather_idx
        while mjd >= weather_mjd[weather_idx + 1]:
            weather_idx += 1
        s = (mjd - weather_mjd[weather_idx]) / dmjd_weather
        return (
            seeing[weather_idx] * (1 - s) + seeing[weather_idx + 1] * s,
            transp[weather_idx] * (1 - s) + transp[weather_idx + 1] * s)

    # Define time intervals to use in units of days (move to config?)
    NO_TILE_AVAIL_DELAY = 30. / 86400.

    # Step through the night.
    dome_is_open = False
    mjd_now = weather_mjd[0]
    completed_last = scheduler.completed_by_pass.copy()
    sky_now = 1.0  # placeholder
    while mjd_now < end:
        if not dome_is_open:
            # Advance to the next dome opening, if any.
            idx_now = np.searchsorted(weather_mjd, mjd_now, side='left')
            if not np.any(dome[idx_now:]):
                # Dome is closed for the rest of the night.
                mjd_now = end
                break
            idx_open = idx_now + np.argmax(dome[idx_now:])
            assert dome[idx_open] == True and (idx_open == 0 or dome[idx_open - 1] == False)
            mjd_now = weather_mjd[idx_open]
            if mjd_now >= end:
                # The next dome opening is after the end of the night.
                # This can happen if we are not using twilight.
                break
            # Find the next closing.
            if np.all(dome[idx_open:]):
                next_dome_closing = end
            else:
                idx_close = idx_open + np.argmin(dome[idx_open:])
                assert dome[idx_close] == False and dome[idx_close - 1] == True
                next_dome_closing = min(end, weather_mjd[idx_close])
            dome_is_open = True
            weather_idx = idx_open

        # == NEXT TILE ===========================================================        
        # Dome is open from mjd_now to next_dome_closing.
        mjd_last = mjd_now
        tdead = 0.
        # Get the current observing conditions.
        seeing_now, transp_now = get_weather(mjd_now)
        # Get the next tile to observe from the scheduler.
        tileid, passnum, snr2frac_start, exposure_factor, airmass, sched_program, mjd_program_end = \
            scheduler.next_tile(mjd_now, ETC, seeing_now, transp_now, sky_now)
        if tileid is None:
            # Deadtime while we delay and try again.
            mjd_now += NO_TILE_AVAIL_DELAY
            if mjd_now >= next_dome_closing:
                # Dome closed during deadtime.
                mjd_now = next_dome_closing
                dome_is_open = False
            tdead += mjd_now - mjd_last
        else:
            # Setup for a new field.
            mjd_now += ETC.NEW_FIELD_SETUP
            if mjd_now >= next_dome_closing:
                # Setup interrupted by dome closing.
                mjd_now = next_dome_closing
                dome_is_open = False
                # Record an aborted setup.
                nightstats['nsetup_abort'][passnum] += 1
            else:
                # Record a completed setup.
                nightstats['nsetup'][passnum] += 1
            # Charge this as setup time whether or not it was aborted.
            nightstats['tsetup'][passnum] += mjd_now - mjd_last

            if dome_is_open:            
                # Lookup the program of the next tile, which might be
                # different from the scheduled program in ``sched_program``.
                tile_program = scheduler.tiles.pass_program[passnum]
                # Loop over repeated exposures of the same tile.
                continue_this_tile = True
                while continue_this_tile:
                    # -- NEXT EXPOSURE ---------------------------------------------------
                    # Get the current observing conditions.
                    seeing_now, transp_now = get_weather(mjd_now)
                    sky_now = 1.
                    # Use the ETC to control the shutter.
                    mjd_open_shutter = mjd_now
                    ETC.start(mjd_now, tileid, tile_program, snr2frac_start, exposure_factor,
                              seeing_now, transp_now, sky_now)
                    integrating = True
                    while integrating:
                        mjd_now += update_interval_days
                        if mjd_now >= next_dome_closing:
                            # Current exposure is interrupted by dome closing.
                            mjd_now = next_dome_closing
                            dome_is_open = False
                            integrating = False
                            continue_this_tile = False
                        elif mjd_now >= mjd_program_end:
                            # Current exposure is interrupted by a program change.
                            mjd_now = mjd_program_end
                            integrating = False
                            continue_this_tile = False
                        # Get the current observing conditions.
                        seeing_now, transp_now = get_weather(mjd_now)
                        sky_now = 1.
                        # Update the SNR.
                        if not ETC.update(mjd_now, seeing_now, transp_now, sky_now):
                            # Current exposure reached its target SNR according to the ETC.
                            integrating= False
                    # stop() will return False if this is a cosmic split and
                    # more integration is still required.
                    if ETC.stop(mjd_now):
                        continue_this_tile = False

                    # Record this exposure
                    assert np.allclose(ETC.exptime, mjd_now - mjd_open_shutter)
                    nightstats['tscience'][passnum] += ETC.exptime
                    nightstats['nexp'][passnum] += 1
                    explist.add(
                        mjd_now - ETC.exptime, 86400 * ETC.exptime, tileid, ETC.snr2frac,
                        airmass, seeing_now, transp_now, sky_now)
                    scheduler.update_snr(tileid, ETC.snr2frac,
                                         explist.nexp)

                    if continue_this_tile:
                        # Prepare for the next exposure of the same tile.
                        snr2frac_start = ETC.snr2frac
                        mjd_split_start = mjd_now
                        mjd_now += ETC.SAME_FIELD_SETUP
                        if mjd_now >= next_dome_closing:
                            # Setup for next exposure of same tile interrupted by dome closing.
                            mjd_now = next_dome_closing
                            dome_is_open = False
                            continue_this_tile = False
                            # Record an aborted split.
                            nightstats['nsplit_abort'][passnum] += 1
                        else:
                            # Record a completed split.
                            nightstats['nsplit'][passnum] += 1
                        # Charge this as split time, whether or not is was aborted.
                        nightstats['tsplit'][passnum] += mjd_now - mjd_split_start
                    # --------------------------------------------------------------------

        # Update statistics for the scheduled program (which might be different from
        # the program of the tile we just observed).
        pidx = scheduler.tiles.PROGRAM_INDEX[sched_program]
        nightstats['tdead'][pidx] += tdead
        nightstats['topen'][pidx] += mjd_now - mjd_last

        # All done if we have observed all tiles.
        if scheduler.survey_completed():
            break
        # ========================================================================

    # Save the number of tiles completed per pass in the nightly statistics.
    nightstats['completed'][:] = scheduler.completed_by_pass - completed_last

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        ax = axes[0]
        ax.plot(weather_mjd, seeing, 'r-', label='Seeing')
        ax.plot([], [], 'b-', label='Transparency')
        ax.legend(ncol=2, loc='lower center')
        ax.set_ylabel('Seeing FWHM [arcsec]')
        rhs = ax.twinx()
        rhs.plot(weather_mjd, transp, 'b-')
        rhs.set_ylabel('Transparency')
        ax = axes[1]
        changes = np.where(np.abs(np.diff(dome)) == 1)[0]
        for idx in changes:
            ax.axvline(weather_mjd[idx + 1], ls='-', c='r')
        mjd_history = np.array(ETC.history['mjd'])
        snr2frac_history = np.array(ETC.history['snr2frac'])
        for expinfo in explist._exposures[nexp_last: explist.nexp]:
            passnum = scheduler.tiles.passnum[scheduler.tiles.index(expinfo['TILEID'])]
            program = scheduler.tiles.pass_program[passnum]
            color = desisurvey.plots.program_color[program]
            t1 = expinfo['MJD']
            t2 = t1 + expinfo['EXPTIME'] / 86400
            sel = (mjd_history >= t1) & (mjd_history <= t2)
            ax.fill_between(mjd_history[sel], snr2frac_history[sel], color=color, alpha=0.5, lw=0)
        for t in scheduler.night_changes:
            ax.axvline(t, c='b', ls=':')
        ax.set_xlim(weather_mjd[0], weather_mjd[-1])
        ax.set_xlabel('MJD During {}'.format(label))
        ax.set_ylim(0, 1)
        ax.set_ylabel('Integrated SNR2 Fraction')
