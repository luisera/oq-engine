# Copyright (c) 2010-2013, GEM Foundation.
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.

"""
Core functionality for the classical PSHA hazard calculator.
"""
import numpy

import openquake.hazardlib
import openquake.hazardlib.calc
import openquake.hazardlib.imt
from openquake.hazardlib.tom import PoissonTOM

from openquake.engine import logs
from openquake.engine.calculators.hazard import general as haz_general
from openquake.engine.calculators.hazard.classical import (
    post_processing as post_proc)
from openquake.engine.db import models
from openquake.engine.utils import tasks as utils_tasks
from openquake.engine.utils.general import block_splitter
from openquake.engine.performance import EnginePerformanceMonitor
from openquake.engine.input import logictree

# FIXME: the following import must go after the openquake.engine.db import
# so that the variable DJANGO_SETTINGS_MODULE is properly set
from django.db import transaction

RUPTURE_BLOCK_SIZE = 1000


@utils_tasks.oqtask
def compute_hazard_curves(job_id, src_ids, lt_rlz_id, ltp):
    """
    Celery task for hazard curve calculator.

    Samples logic trees, gathers site parameters, and calls the hazard curve
    calculator.

    Once hazard curve data is computed, result progress updated (within a
    transaction, to prevent race conditions) in the
    `htemp.hazard_curve_progress` table.

    :param int job_id:
        ID of the currently running job.
    :param src_ids:
        List of ids of parsed source models to take into account.
    :param lt_rlz_id:
        Id of logic tree realization model to calculate for.
    :param ltp:
        a :class:`openquake.engine.input.LogicTreeProcessor` instance
    """
    hc = models.HazardCalculation.objects.get(oqjob=job_id)

    lt_rlz = models.LtRealization.objects.get(id=lt_rlz_id)

    apply_uncertainties = ltp.parse_source_model_logictree_path(
        lt_rlz.sm_lt_path)
    gsims = ltp.parse_gmpe_logictree_path(lt_rlz.gsim_lt_path)

    parsed_sources = models.ParsedSource.objects.filter(pk__in=src_ids)

    imts = haz_general.im_dict_to_hazardlib(
        hc.intensity_measure_types_and_levels)

    # Prepare args for the calculator.
    calc_kwargs = {'gsims': gsims,
                   'truncation_level': hc.truncation_level,
                   'time_span': hc.investigation_time,
                   'sources': [apply_uncertainties(s.nrml)
                               for s in parsed_sources],
                   'imts': imts,
                   'sites': hc.site_collection}

    if hc.maximum_distance:
        dist = hc.maximum_distance
        # NB: a better approach could be to filter the sources by distance
        # at the beginning and to store into the database only the relevant
        # sources, as we do in the event based calculator: I am not doing that
        # for the classical calculator because I wonder about the performance
        # impact in in SHARE-like calculations. So at the moment we store
        # everything in the database and we filter on the workers. This
        # will probably change in the future (MS).
        calc_kwargs['source_site_filter'] = (
            openquake.hazardlib.calc.filters.source_site_distance_filter(dist))
        calc_kwargs['rupture_site_filter'] = (
            openquake.hazardlib.calc.filters.rupture_site_distance_filter(
                dist))

    # mapping "imt" to 2d array of hazard curves: first dimension -- sites,
    # second -- IMLs
    with EnginePerformanceMonitor(
            'computing hazard curves', job_id,
            compute_hazard_curves, tracing=True):
        matrices = openquake.hazardlib.calc.hazard_curve.\
            hazard_curves_poissonian(**calc_kwargs)

    with EnginePerformanceMonitor(
            'saving hazard curves', job_id,
            compute_hazard_curves, tracing=True):
        _update_curves(hc, matrices, lt_rlz)


def _update_curves(hc, matrices, lt_rlz):
    """
    Helper function for updating source, hazard curve, and realization progress
    records in the database.

    This is intended to be used by :func:`compute_hazard_curves`.

    :param hc:
        :class:`openquake.engine.db.models.HazardCalculation` instance.
    :param lt_rlz:
        :class:`openquake.engine.db.models.LtRealization` record for the
        current realization.
    """
    with logs.tracing('_update_curves for all IMTs'):
        for imt in hc.intensity_measure_types_and_levels:
            hazardlib_imt = haz_general.imt_to_hazardlib(imt)
            matrix = matrices[hazardlib_imt]
            if (matrix == 0.0).all():
                # The matrix for this IMT is all zeros; there's no reason to
                # update `hazard_curve_progress` records.
                logs.LOG.debug('* No hazard contribution for IMT=%s' % imt)
                continue
            else:
                # The is some contribution here to the hazard; we need to
                # update.
                with transaction.commit_on_success():
                    logs.LOG.debug('> updating hazard for IMT=%s' % imt)
                    query = """
                    SELECT * FROM htemp.hazard_curve_progress
                    WHERE lt_realization_id = %s
                    AND imt = %s
                    FOR UPDATE"""
                    [hc_progress] = models.HazardCurveProgress.objects.raw(
                        query, [lt_rlz.id, imt])

                    hc_progress.result_matrix = update_result_matrix(
                        hc_progress.result_matrix, matrix)
                    hc_progress.save()

                    logs.LOG.debug('< done updating hazard for IMT=%s' % imt)

'''
class ClassicalHazardCalculatorOld(haz_general.BaseHazardCalculator):
    """
    Classical PSHA hazard calculator. Computes hazard curves for a given set of
    points.

    For each realization of the calculation, we randomly sample source models
    and GMPEs (Ground Motion Prediction Equations) from logic trees.
    """

    core_calc_task = compute_hazard_curves

    def pre_execute(self):
        """
        Do pre-execution work. At the moment, this work entails:
        parsing and initializing sources, parsing and initializing the
        site model (if there is one), parsing vulnerability and
        exposure files and generating logic tree realizations. (The
        latter piece basically defines the work to be done in the
        `execute` phase.).
        """

        # Parse vulnerability and exposure model
        self.parse_risk_models()

        # Deal with the site model and compute site data for the calculation
        # (if a site model was specified, that is).
        self.initialize_site_model()

        # Parse logic trees and create source Inputs.
        self.initialize_sources()

        # Now bootstrap the logic tree realizations and related data.
        # This defines for us the "work" that needs to be done when we reach
        # the `execute` phase.
        # This will also stub out hazard curve result records. Workers will
        # update these periodically with partial results (partial meaning,
        # result curves for just a subset of the overall sources) when some
        # work is complete.
        self.initialize_realizations(
            rlz_callbacks=[self.initialize_hazard_curve_progress])

    def post_execute(self):
        """
        Post-execution actions. At the moment, all we do is finalize the hazard
        curve results. See
        :meth:`openquake.engine.calculators.hazard.general.\
BaseHazardCalculator.finalize_hazard_curves`
        for more info.
        """
        self.finalize_hazard_curves()

    def clean_up(self):
        """
        Delete temporary database records.
        These records represent intermediate copies of final calculation
        results and are no longer needed.

        In this case, this includes all of the data for this calculation in the
        tables found in the `htemp` schema space.
        """
        self.sources_per_rlz.clear()
        logs.LOG.debug('> cleaning up temporary DB data')
        models.HazardCurveProgress.objects.filter(
            lt_realization__hazard_calculation=self.hc.id).delete()
        logs.LOG.debug('< done cleaning up temporary DB data')

    def post_process(self):
        logs.LOG.debug('> starting post processing')

        # means/quantiles:
        if self.hc.mean_hazard_curves or self.hc.quantile_hazard_curves:
            self.do_aggregate_post_proc()

        # hazard maps:
        # required for computing UHS
        # if `hazard_maps` is false but `uniform_hazard_spectra` is true,
        # just don't export the maps
        if self.hc.hazard_maps or self.hc.uniform_hazard_spectra:
            self.parallelize(
                post_proc.hazard_curves_to_hazard_map_task,
                post_proc.hazard_curves_to_hazard_map_task_arg_gen(self.job))

        if self.hc.uniform_hazard_spectra:
            post_proc.do_uhs_post_proc(self.job)

        logs.LOG.debug('< done with post processing')
'''


def update_result_matrix(current, new):
    """
    Use the following formula to combine multiple iterations of results:

    `result = 1 - (1 - current) * (1 - new)`

    This is used to incrementally update hazard curve results by combining an
    initial value with some new results. (Each set of new results is computed
    over only a subset of seismic sources defined in the calculation model.)

    Parameters are expected to be multi-dimensional numpy arrays, but the
    formula will also work with scalars.

    :param current:
        Numpy array representing the current result matrix value.
    :param new:
        Numpy array representing the new results which need to be combined with
        the current value. This should be the same shape as `current`.
    """
    return 1 - (1 - current) * (1 - new)


########################### NEW IMPLEMENTATION ############################

@utils_tasks.oqtask
def generate_ruptures(job_id, src_ids, lt_rlz, ltp):
    """
    Celery task for hazard curve calculator.

    Samples logic trees, gathers site parameters, and calls the hazard curve
    calculator.

    :param int job_id:
        ID of the currently running job.
    :param src_ids:
        List of ids of parsed source models to take into account.
    :param lt_rlz:
        Logic tree realization model to calculate for.
    :param ltp:
        a :class:`openquake.engine.input.LogicTreeProcessor` instance
    """
    hc = models.HazardCalculation.objects.get(oqjob=job_id)

    # there is a single ses and ses_collection for each realization
    ses = models.SES.objects.get(ses_collection__lt_realization=lt_rlz)

    apply_uncertainties = ltp.parse_source_model_logictree_path(
        lt_rlz.sm_lt_path)

    sources = [apply_uncertainties(src.nrml)
               for src in models.ParsedSource.objects.filter(pk__in=src_ids)]

    ruptures = []

    f = openquake.hazardlib.calc.filters
    source_site_filter = f.source_site_distance_filter(hc.maximum_distance) \
        if hc.maximum_distance else f.source_site_noop_filter

    tom = PoissonTOM(hc.investigation_time)
    sources_sites = ((source, hc.site_collection) for source in sources)

    with EnginePerformanceMonitor(
            'generating ruptures', job_id, generate_ruptures):
        for src, _sites in source_site_filter(sources_sites):
            for i, r in enumerate(src.iter_ruptures(tom)):
                rup = models.SESRupture(
                    ses=ses,
                    rupture=r,
                    tag='rlz=%02d|ses=0|src=%s|i=%03d' % (
                        lt_rlz.ordinal, src.source_id, i),
                    hypocenter=r.hypocenter.wkt2d,
                    magnitude=r.mag,
                )
                ruptures.append(rup)

    if not ruptures:
        return

    with EnginePerformanceMonitor(
            'saving ruptures', job_id, generate_ruptures):
        with transaction.commit_on_success(using='job_init'):
            for r in ruptures:
                r.save()


@utils_tasks.oqtask
def generate_hazard_curves(job_id, rupture_ids, imts, gsims, lt_rlz):
    """
    """
    hc = models.HazardCalculation.objects.get(oqjob=job_id)
    total_sites = len(hc.site_collection)
    f = openquake.hazardlib.calc.filters
    rupture_site_filter = (
        f.rupture_site_distance_filter(hc.maximum_distance)
        if hc.maximum_distance else f.rupture_site_noop_filter)

    with EnginePerformanceMonitor(
            'reading ruptures', job_id, generate_hazard_curves):
        rupture_sites = [(r.rupture, hc.site_collection) for r in
                         models.SESRupture.objects.filter(pk__in=rupture_ids)]

    curves = dict((imt, numpy.ones([total_sites, len(imts[imt])]))
                  for imt in imts)
    with EnginePerformanceMonitor(
            'generating hazard curves', job_id, generate_hazard_curves):
        for rupture, r_sites in rupture_site_filter(rupture_sites):
            prob = rupture.get_probability_one_or_more_occurrences()
            gsim = gsims[rupture.tectonic_region_type]
            sctx, rctx, dctx = gsim.make_contexts(r_sites, rupture)
            for imt in imts:
                poes = gsim.get_poes(sctx, rctx, dctx, imt, imts[imt],
                                     hc.truncation_level)
                curves[imt] *= r_sites.expand(
                    (1 - prob) ** poes, total_sites, placeholder=1)
        for imt in imts:
            curves[imt] = 1 - curves[imt]

    with EnginePerformanceMonitor(
            'saving hazard curves', job_id, generate_hazard_curves):
        _update_curves(hc, curves, lt_rlz)


class ClassicalHazardCalculator(haz_general.BaseHazardCalculator):
    """
    Classical PSHA hazard calculator. Computes hazard curves for a given set of
    points.

    For each realization of the calculation, we randomly sample source models
    and GMPEs (Ground Motion Prediction Equations) from logic trees.
    """

    core_calc_task = generate_ruptures

    def initialize_ses(self, lt_rlz):
        output = models.Output.objects.create(
            oq_job=self.job,
            display_name='SES Collection rlz-%s' % lt_rlz.id,
            output_type='ses')

        ses_coll = models.SESCollection.objects.create(
            output=output, lt_realization=lt_rlz)

        models.SES.objects.create(
            ses_collection=ses_coll,
            investigation_time=self.hc.investigation_time,
            ordinal=1)

    def pre_execute(self):
        """
        Do pre-execution work. At the moment, this work entails:
        parsing and initializing sources, parsing and initializing the
        site model (if there is one), parsing vulnerability and
        exposure files and generating logic tree realizations. (The
        latter piece basically defines the work to be done in the
        `execute` phase.).
        """
        self.parse_risk_models()
        self.initialize_site_model()
        self.initialize_sources()
        self.initialize_realizations(
            rlz_callbacks=[self.initialize_hazard_curve_progress,
                           self.initialize_ses])

    def post_execute(self):
        """
        Generate the hazard curves from the ruptures
        """
        self.parallelize(generate_hazard_curves,
                         self.generate_curves_arg_gen())
        self.finalize_hazard_curves()

    def generate_curves_arg_gen(self):
        ltp = logictree.LogicTreeProcessor.from_hc(self.hc)
        imts = haz_general.im_dict_to_hazardlib(
            self.hc.intensity_measure_types_and_levels)
        for lt_rlz in self._get_realizations():
            gsims = ltp.parse_gmpe_logictree_path(lt_rlz.gsim_lt_path)
            rupture_ids = models.SESRupture.objects.filter(
                ses__ses_collection__lt_realization=lt_rlz
            ).values_list('id', flat=True)
            if not rupture_ids:
                continue
            for rids in block_splitter(rupture_ids, RUPTURE_BLOCK_SIZE):
                yield self.job.id, rids, imts, gsims, lt_rlz

    def clean_up(self):
        """
        Delete temporary database records.
        These records represent intermediate copies of final calculation
        results and are no longer needed.

        In this case, this includes all of the data for this calculation in the
        tables found in the `htemp` schema space.
        """
        self.sources_per_rlz.clear()
        logs.LOG.debug('> cleaning up temporary DB data')
        models.HazardCurveProgress.objects.filter(
            lt_realization__hazard_calculation=self.hc.id).delete()
        logs.LOG.debug('< done cleaning up temporary DB data')

    def post_process(self):
        # means/quantiles:
        if self.hc.mean_hazard_curves or self.hc.quantile_hazard_curves:
            self.do_aggregate_post_proc()

        # hazard maps:
        # required for computing UHS
        # if `hazard_maps` is false but `uniform_hazard_spectra` is true,
        # just don't export the maps
        if self.hc.hazard_maps or self.hc.uniform_hazard_spectra:
            self.parallelize(
                post_proc.hazard_curves_to_hazard_map_task,
                post_proc.hazard_curves_to_hazard_map_task_arg_gen(self.job))

        if self.hc.uniform_hazard_spectra:
            post_proc.do_uhs_post_proc(self.job)
