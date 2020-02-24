=================================================
Tutorial Guide to Running DESI Survey Simulations
=================================================

Introduction
------------

The instructions below simulate the DESI survey as a sequence of tile exposures
and record metadata for each exposure and accumulate some summary statistics.
Simulations are stochastic since the scheduling algorithms respond to randomly
generated weather conditions.

These instructions do not perform fiber assignment or simulate any spectra,
but instead provide the necessary scheduling inputs for these tasks.

Please `create an issue <https://github.com/desihub/surveysim/issues/new>`__
with any corrections or suggestions for improvement to this tutorial.

Quick Start
-----------

Login to cori, then::

    source /project/projectdirs/desi/software/desi_environment.sh 18.12
    mkdir -p $SCRATCH/desi/output
    export DESISURVEY_OUTPUT=$SCRATCH/desi/output
    surveyinit
    surveysim

The results are then saved as ``stats_surveysim.fits``
and ``exposures_surveysim.fits`` in ``$DESISURVEY_OUTPUT``.
For a tutorial on interpreting
these outputs `start here
<https://github.com/desihub/tutorials/blob/master/survey-simulations.ipynb>`__.

For more details and variations on these steps, read on.

Install Software
----------------

Requirements
~~~~~~~~~~~~

If this is your first exposure to DESI software,
`start here <https://desi.lbl.gov/trac/wiki/Pipeline/GettingStarted>`__.
We use `git for source control <https://desi.lbl.gov/trac/wiki/Computing/UsingGit>`__
and you will need to install the base DESI packages
`on your laptop <https://desi.lbl.gov/trac/wiki/Pipeline/GettingStarted/Laptop>`__
or else `work at NERSC <https://desi.lbl.gov/trac/wiki/Pipeline/GettingStarted/NERSC>`__.

The following DESI packages must be installed to run this tutorial:

- `specsim <https://github.com/desihub/specsim>`_
- `desiutil <https://github.com/desihub/desiutil>`_
- `desimodel <https://github.com/desihub/desimodel>`_
- `desisurvey <https://github.com/desihub/desisurvey>`_
- `surveysim <https://github.com/desihub/surveysim>`_

In addition, the following non-DESI packages must be installed via pip
since they are not included with the anaconda distribution:

- fitsio
- speclite
- ephem
- healpy

Note that these packages are already included in the custom DESI
anaconda distribution installed at NERSC, so only need to be installed
when running on your laptop or if you need to use non-default versions.

NERSC Installation
~~~~~~~~~~~~~~~~~~

Setup the standard DESI conda environment using, for example::

    source /project/projectdirs/desi/software/desi_environment.sh 18.12

Replace ``18.12`` with ``master`` for the latest and greatest (which might not work),
or leave it out for the current default.

If you want to use new features of ``desisurvey`` or ``surveysim`` that are
not yet included in a numbered DESI conda environment, swap them in using::

    module swap desisurvey/master
    module swap surveysim/master

For even more bleeding-edge features that are only available on a development
branch, use, for example::

    module unload desisurvey
    pip install --user git+https://github.com/desihub/desisurvey@refactor
    export PATH=$HOME/.local/bin:$PATH

where ``refactor`` is the branch name in this example.

Laptop Installation
~~~~~~~~~~~~~~~~~~~

The following instructions assume that you have installed the
`anaconda scientific python distribution <https://docs.continuum.io/anaconda/install>`__
and will create a new python environment for your DESI work.
Start from the directory you wish to install software into, then::

    conda create --name desi pip ipython jupyter numpy scipy astropy pyyaml requests h5py scikit-learn matplotlib basemap
    source activate desi
    pip install fitsio speclite ephem healpy
    for package in specsim desiutil desimodel desisurvey surveysim; do
        pip install git+https://github.com/desihub/$package
    done
    export DESIMODEL=$PWD/desimodel
    install_desimodel_data -d $DESIMODEL

Notes for experts:

- The instructions above assume that you are using the bash shell, and need to
  be modified slightly for (t)csh.
- The matplotlib and basemap packages are not required to follow the
  instructions below but are useful for plotting the outputs.

Setup Environment
-----------------

In General
~~~~~~~~~~

Create an output directory to hold all survey planning and simulation outputs
and create an environment variable pointing to it.

Ensure that your ``$DESIMODEL`` environment variable points to a valid data directory::

    ls $DESIMODEL/data/weather

Also check that the relevant command-line scripts are in your path::

    surveyinit --help
    surveysim --help

Note that all output from these commands goes into ``$DESISURVEY_OUTPUT`` so they
can be run from any directory and will not write anything to your current working
directory.

NERSC Environment
~~~~~~~~~~~~~~~~~

Save the output to the ``$SCRATCH`` volume, for example::

    mkdir -p $SCRATCH/desi/output
    export DESISURVEY_OUTPUT=$SCRATCH/desi/output

Note that we use ``$SCRATCH`` for faster I/O but files are periodically
removed. See `NERSC best practices
<https://www.nersc.gov/users/data-analytics/data-analytics-2/python/best-practices>`__
for details.

Laptop Environment
~~~~~~~~~~~~~~~~~~

Enter the parent directory where you will save outputs, then::

    mkdir output
    export DESISURVEY_OUTPUT=$PWD/output

If you followed the installation recipe above then make sure you have activated your
``desi`` environment with::

    conda activate desi

(Older versions of ``conda`` might require ``source activate desi`` instead.)

Configuration
-------------

Parameters for planning and scheduling the DESI survey are stored in a 
`configuration file
<https://github.com/desihub/desisurvey/blob/master/py/desisurvey/data/config.yaml>`__
which is well commented and provides a good overview of the assumptions being made.
You do not normally need to change these parameters, but are welcome to experiment by copying and editing
this file then passing your custom version to the ``surveyinit`` and ``surveysim`` scripts described below
using their ``config-file`` option.

Initialize Survey Planning
--------------------------

Before starting the survey, we precompute some tabulated planning data using::

    surveyinit --verbose

This step takes about 50 minutes (on cori) and writes the following files into ``$DESISURVEY_OUTPUT``:

- ``ephem_2019-01-01_2025-12-31.fits``: tabulated ephemerides during 2019-25.
- ``surveyinit.fits``: estimated average weather and optimized initial hour angle (HA) assignments for each tile.

These files take some time to generate, but are cached and not regenerated
after the first time you run this command. If you want to force these files
to be recalculated, add the ``--recalc`` option.  To avoid generating these
files yourself, you can also copy them into your ``$DESISURVEY_OUTPUT`` from
this NERSC directory::

    $DESI_ROOT/datachallenge/surveysim2018/shared/

To ensure they have been copied correctly, you should still run ``surveyinit --verbose``,
which should now exit immediately.

Simulate Observations
---------------------

To simulate the nomimal 5-year survey, use::

    surveysim

This should complete in about 2 minutes (on cori) and writes two FITS files to
``$DESISURVEY_OUTPUT``:

 - ``stats.fits``: tables of per-tile and per-night summary statistics.
 - ``exposures.fits``: table listing all simulated exposures in time order.

For a tutorial on interpreting these outputs `start here
<https://github.com/desihub/tutorials/blob/master/survey-simulations.ipynb>`__.

By default, simulations are run entirely in memory for speed. However, during
operations the internal states of the afternoon planner and tile scheduler are
written to disk daily and then restored the next day. Use the ``--save-restore``
option to ``surveysim`` to run in this mode, and write daily files:

 - ``planner_YYYY-MM-DD.fits``: internal state of the planner after afternoon planning for YYYY-MM-DD.
 - ``scheduler_YYYY-MM-DD.fits``: internal state of the tile scheduler after observing on the night of YYYY-MM-DD.

This mode gives identical results but is slower (about 3 minutes) and writes many files (about 3.6K files
totalling amost 1Gb), so is mainly intended as a technical check of this mode and for developing tools
that read these intermediate files.

Variations
~~~~~~~~~~

There are many options you can experiment with to simulate a different survey weather,
strategy, or schedule, for example. For a full list, refer to::

    surveyinit --help
    surveysim --help

You can also vary parameters in the survey configuration file.

In order to keep the outputs from different runs separate, use a separate output
directory each time a change to the ``surveyinit`` outputs is required.
For example, when changing the tiles file or nominal survey start/stop dates.
To run with a different output directory you can either update
``$DESISURVEY_OUTPUT`` or else use the ``--output-path`` option
of ``surveyinit`` and ``surveysim``.

For different runs with the same ``surveyinit`` outputs, use the ``--name``
and ``--comment`` options to ``surveysim`` to distinguish each run.
For example::

    surveysim --name twilight --comment 'Include twilight in schedule' --twilight

Will run with twilight included in the schedule and save
`stats_twilight.fits` and `exposures_twilight.fits` to ``$DESISURVEY_OUTPUT``.

To study how survey progress depends on the random weather realization (including
seeing and transparency), change the default seed (1), for example::

    surveysim --name weather1 --comment 'Random weather realization #1' --seed 1
    surveysim --name weather2 --comment 'Random weather realization #2' --seed 2
    surveysim --name weather3 --comment 'Random weather realization #3' --seed 3

To simulate with an estimate of the worst-case weather, replay the historical
dome-open fractions from 2015 during each year of the simulation with::

    surveysim --name worstcase --comment 'Worst-case dome-open fractions' --replay Y2015

Custom Simulation
-----------------

Instead of running ``surveysim``, you can incorporate and customize the following
top-level simulation driver directly into your own script or jupyter notebook::

    import datetime

    import desisurvey.config
    import desisurvey.rules
    import desisurvey.plan
    import desisurvey.scheduler

    import surveysim.weather
    import surveysim.stats
    import surveysim.exposures
    import surveysim.nightops

    def simulate_survey(rules, weather, use_twilight=False):

        # Simulate the nominal survey dates.
        config = desisurvey.config.Configuration()
        start, stop = config.first_day(), config.last_day()
        num_nights = (stop - start).days

        # Initialize simulation progress tracking.
        stats = surveysim.stats.SurveyStatistics()
        explist = surveysim.exposures.ExposureList()
        
        # Initialize afternoon planning.
        planner = desisurvey.plan.Planner(rules, simulate=True)

        # Initialize next tile selection.
        scheduler = desisurvey.scheduler.Scheduler()
        
        # Loop over nights.
        num_simulated = 0
        for num_simulated in range(num_nights):
            night = start + datetime.timedelta(num_simulated)

            # Perform afternoon planning.
            explist.update_tiles(night, *scheduler.update_tiles(*planner.afternoon_plan(night)))

            if not desisurvey.utils.is_monsoon(night) and not scheduler.ephem.is_full_moon(night):
                # Simulate one night of observing.
                surveysim.nightops.simulate_night(
                    night, scheduler, stats, explist, weather=weather, use_twilight=use_twilight)
		planner.set_donefrac(scheduler.tiles.tileID, scheduler.snr2frac,
	                             scheduler.lastexpid)

                if scheduler.survey_completed():
                    break

        return stats, explist

To run a simulation, define the survey strategy rules, e.g.::

    rules = desisurvey.rules.Rules('rules-depth.yaml')

and the random weather realization to use, e.g.::

    weather = surveysim.weather.Weather(seed=1, replay='random')

then call the function defined above::

    stats, exposures = simulate_survey(rules, weather)

Visualization
-------------

The ``surveymovie`` script reads simulation outputs and generates a movie with one
frame per exposure to visualize the scheduler algorithm and survey progress::

    surveymovie --verbose

`An example is available <https://www.youtube.com/watch?v=vO1QZD_aCIo>`__.
A key describing the information displayed in each frame is
`here <https://github.com/desihub/desisurvey/blob/master/doc/img/surveymovie-key.png>`__.
To generate a PNG of a single frame, use::

    surveymovie --expid 123 --save exposure123

to create ``exposure123.png``.

To generate a smaller summary movie with one frame per night, use the `--nightly` option, e.g.::

    surveymovie --nightly --save summary

The ``surveymovie`` script uses the external ``ffmpeg`` program to generate movies, so
this must be `installed <https://www.ffmpeg.org/download.html>`__. At NERSC, use::

    module add ffmpeg
