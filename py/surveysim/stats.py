"""Record simulated nightly statistics by pass.
"""
from __future__ import print_function, division, absolute_import

import numpy as np

import astropy.io.fits

import desiutil.log

import desisurvey.config
import desisurvey.utils
import desisurvey.tiles
import desisurvey.plots


class SurveyStatistics(object):
    """Collect nightly statistics by pass.

    Parameters
    ----------
    start_date : datetime.date or None
        Record statistics for a survey that starts on the evening of this date.
        Uses the configured nominal start date when None.
    stop_date : datetime.date
        Record statistics for a survey that stops on the morning of this date.
        Uses the configured nominal stop date when None.
    restore : str or None
        Restore internal state from the snapshot saved to this filename,
        or initialize a new object when None. Use :meth:`save` to
        save a snapshot to be restored later. Filename is relative to
        the configured output path unless an absolute path is
        provided.
    """
    def __init__(self, start_date=None, stop_date=None, restore=None,
            tiles_file=None, bgs_footprint=None):
        if tiles_file is None: 
            self.tiles = desisurvey.tiles.Tiles(bgs_footprint=bgs_footprint)
        else: 
            self.tiles = desisurvey.tiles.Tiles(tiles_file=tiles_file, bgs_footprint=bgs_footprint)
        config = desisurvey.config.Configuration()
        if start_date is None:
            self.start_date = config.first_day()
        else:
            self.start_date = desisurvey.utils.get_date(start_date)
        if stop_date is None:
            self.stop_date = config.last_day()
        else:
            self.stop_date = desisurvey.utils.get_date(stop_date)
        self.num_nights = (self.stop_date - self.start_date).days
        if self.num_nights <= 0:
            raise ValueError('Expected start_date < stop_date.')
        # Build our internal array.
        dtype = []
        for name in 'MJD', 'tsched',:
            dtype.append((name, np.float))
        nprograms = len(self.tiles.PROGRAMS)
        for name in 'topen', 'tdead',:
            dtype.append((name, np.float, (nprograms,)))
        for name in 'tscience', 'tsetup', 'tsplit',:
            dtype.append((name, np.float, (self.tiles.npasses,)))
        for name in 'completed', 'nexp', 'nsetup', 'nsplit', 'nsetup_abort', 'nsplit_abort',:
            dtype.append((name, np.int32, (self.tiles.npasses,)))
        self._data = np.zeros(self.num_nights, dtype)
        if restore is not None:
            # Restore array contents from a FITS file.
            fullname = config.get_path(restore)
            with astropy.io.fits.open(fullname, memmap=None) as hdus:
                header = hdus[1].header
                comment = header['COMMENT']
                if header['TILES'] != self.tiles.tiles_file:
                    raise ValueError('Header mismatch for TILES.')
                if header['START'] != self.start_date.isoformat():
                    raise ValueError('Header mismatch for START.')
                if header['STOP'] != self.stop_date.isoformat():
                    raise ValueError('Header mismatch for STOP.')
                self._data[:] = hdus['STATS'].data
            log = desiutil.log.get_logger()
            log.info('Restored stats from {}'.format(fullname))
            if comment:
                log.info('  Comment: "{}".'.format(comment))
        else:
            # Initialize local-noon MJD timestamp for each night.
            first_noon = desisurvey.utils.local_noon_on_date(self.start_date).mjd
            self._data['MJD'] = first_noon + np.arange(self.num_nights)

    def save(self, name='stats.fits', comment='', overwrite=True):
        """Save a snapshot of these statistics as a binary FITS table.

        The saved file size is ~800 Kb.

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
        header['START'] = self.start_date.isoformat()
        header['STOP'] = self.stop_date.isoformat()
        header['COMMENT'] = comment
        header['EXTNAME'] = 'STATS'    
        hdus.append(astropy.io.fits.PrimaryHDU())
        hdus.append(astropy.io.fits.BinTableHDU(self._data, header=header, name='STATS'))
        config = desisurvey.config.Configuration()
        fullname = config.get_path(name)
        hdus.writeto(fullname, overwrite=overwrite)
        log = desiutil.log.get_logger()
        log.info('Saved stats to {}'.format(fullname))
        if comment:
            log.info('Saved with comment "{}".'.format(header['COMMENT']))

    @property
    def nexp(self):
        return self._data['nexp'].sum()

    def get_night(self, night):
        night = desisurvey.utils.get_date(night)
        assert night < self.stop_date
        idx = (night - self.start_date).days
        return self._data[idx]

    def validate(self):
        D = self._data
        # Every exposure must be preceded by a setup or split.
        if not np.all(D['nexp'] == D['nsplit'] + D['nsetup']):
            return False
        # Sum live time per program over nights.
        tlive = (D['topen'] - D['tdead']).sum(axis=1)
        # Sum time spent in each state per pass over nights.
        ttotal = (D['tsetup'] + D['tscience'] + D['tsplit']).sum(axis=1)
        return np.allclose(tlive, ttotal)

    def summarize(self):
        """Print a tabular summary of the accumulated statistics to stdout.
        """
        assert self.validate()
        D = self._data
        tsched = 24 * D['tsched'].sum()
        topen = 24 * D['topen'].sum()
        tscience = 24 * D['tscience'].sum()
        print('Scheduled {:.3f} hr Open {:.3f}% Live {:.3f}%'.format(
            tsched, 100 * topen / max(1e-6, tsched), 100 * tscience / max(1e-6, topen)))
        print('=' * 82)
        print('PROG PASS    TILES  NEXP SETUP ABT SPLIT ABT    TEXP TSETUP TSPLIT   TOPEN  TDEAD')
        print('=' * 82)
        # Summarize by pass.
        for progidx, program in enumerate(self.tiles.PROGRAMS):
            ntiles_p, ndone_p, nexp_p, nsetup_p, nsplit_p, nsetup_abort_p, nsplit_abort_p = [0] * 7
            tscience_p, tsetup_p, tsplit_p = [0.] * 3
            passes = []
            ntiles_all = 0
            for passnum in list(self.tiles.program_passes[program]) + [' ']:
                if passnum == ' ':
                    sel = passes
                    ntiles = ntiles_all
                else:
                    ntiles = self.tiles.pass_ntiles[passnum]
                    ntiles_all += ntiles
                    passidx = self.tiles.pass_index[passnum]
                    sel = passidx
                    passes.append(passidx)
                ndone = D['completed'][:, sel].sum()
                nexp = D['nexp'][:, sel].sum()
                nsetup = D['nsetup'][:, sel].sum()
                nsplit = D['nsplit'][:, sel].sum()
                nsetup_abort = D['nsetup_abort'][:, sel].sum()
                nsplit_abort = D['nsplit_abort'][:, sel].sum()
                tscience = 86400 * D['tscience'][:, sel].sum() / max(1, ndone)
                tsetup = 86400 * D['tsetup'][:, sel].sum() / max(1, ndone)
                tsplit = 86400 * D['tsplit'][:, sel].sum() / max(1, ndone)
                line = '{:6s} {} {:4d}/{:4d} {:5d} {:5d} {:3d} {:5d} {:3d} {:6.1f}s {:5.1f}s {:5.1f}s'.format(
                    program, passnum, ndone, ntiles, nexp, nsetup, nsetup_abort, nsplit, nsplit_abort, tscience, tsetup, tsplit)
                if passnum == ' ':
                    # Open and deadtime are accumulated by program, not pass.
                    topen = 86400 * D['topen'][:, progidx].sum() / max(1, ndone)
                    tdead = 86400 * D['tdead'][:, progidx].sum() / max(1, ndone)
                    line += ' {:6.1f}s {:5.1f}s\n{}'.format(topen, tdead, '-' * 82)
                print(line)

    def plot(self, forecast=None):
        """Plot a summary of the survey statistics.

        Requires that matplotlib is installed.
        """
        import matplotlib.pyplot as plt
        assert self.validate()
        D = self._data
        nprograms = len(self.tiles.PROGRAMS)
        npasses = self.tiles.npasses
        # Find the last day of the survey.
        last = np.argmax(np.cumsum(D['completed'].sum(axis=1))) + 1
        # Combine passes into programs.
        tsetup = np.zeros((last, nprograms))
        tsplit = np.zeros((last, nprograms))
        ntiles = np.zeros(nprograms, int)
        for passnum in range(npasses):
            progidx = self.tiles.PROGRAM_INDEX[self.tiles.pass_program[passnum]]
            passidx = self.tiles.pass_index[passnum]
            tsetup[:, progidx] += D['tsetup'][:last, passidx]
            tsplit[:, progidx] += D['tsplit'][:last, passidx]
            ntiles[progidx] += self.tiles.pass_ntiles[passnum]
        actual = np.cumsum(D['completed'], axis=0)

        dt = 1 + np.arange(len(D))
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 10))

        ax = axes[0]
        npasses = D['completed'].shape[-1]
        for program in self.tiles.PROGRAMS:
            color = desisurvey.plots.program_color[program]
            for i, passnum in enumerate(self.tiles.program_passes[program]):
                npass = self.tiles.pass_ntiles[passnum]
                passidx = self.tiles.pass_index[passnum]
                if forecast:
                    ax.plot(dt, 100 * forecast.pass_progress[passidx] / npass, ':', c=color, lw=1)
                ax.plot(dt[:last], 100 * actual[:last, passidx] / npass,
                        lw=3, alpha=0.5, c=color, label=program if i == 0 else None)
        if forecast:
            ax.plot([], [], 'b:', lw=1, label='forecast')
        ax.legend(ncol=1)
        ax.axvline(dt[last], ls='-', c='r')
        ax.set_ylim(0, 100)
        ax.set_ylabel('Completed [%]')
        yaxis = ax.yaxis
        yaxis.tick_right()
        yaxis.set_label_position('right')

        ax = axes[1]
        # Plot overheads by program.
        for progidx, program in enumerate(self.tiles.PROGRAMS):
            c = desisurvey.plots.program_color[program]
            scale = 86400 / ntiles[progidx] # secs / tile
            ax.plot(dt[:last], scale * np.cumsum(tsetup[:, progidx]), '-', c=c)
            ax.plot(dt[:last], scale * np.cumsum(tsplit[:, progidx]), '--', c=c)
            ax.plot(dt[:last], scale * np.cumsum(D['tdead'][:last, progidx]), ':', c=c)
            if forecast:
                row = forecast.df.iloc[self.tiles.PROGRAM_INDEX[program]]
                ax.scatter([dt[-1], dt[-1], dt[-1]], [
                    row['Setup overhead / tile (s)'],
                    row['Cosmic split overhead / tile (s)'],
                    row['Operations overhead / tile (s)']], s=50, lw=0, c=c)
        ax.plot([], [], 'b-', label='setup')
        ax.plot([], [], 'b--', label='split')
        ax.plot([], [], 'b:', label='dead')
        for program in self.tiles.PROGRAMS:
            ax.plot([], [], '-', c=desisurvey.plots.program_color[program], label=program)
        ax.legend(ncol=2)
        ax.axvline(dt[last], ls='-', c='r')
        ax.set_xlabel('Elapsed Days')
        ax.set_ylabel('Overhead / Tile [s]')
        ax.set_xlim(0, dt[-1] + 1)
        ax.set_ylim(0, None)
        yaxis = ax.yaxis
        yaxis.set_minor_locator(plt.MultipleLocator(10))
        yaxis.tick_right()
        yaxis.set_label_position('right')
        plt.subplots_adjust(hspace=0.05)
        return fig, axes
