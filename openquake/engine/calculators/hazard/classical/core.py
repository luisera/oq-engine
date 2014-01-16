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

from openquake.hazardlib.imt import from_string
from openquake.hazardlib.tom import PoissonTOM

from openquake.engine import logs, writer
from openquake.engine.calculators.hazard import general
from openquake.engine.calculators.hazard.classical import (
    post_processing as post_proc)
from openquake.engine.db import models
from openquake.engine.utils import tasks
from openquake.engine.performance import EnginePerformanceMonitor


@tasks.oqtask
def compute_extra_curves(job_id, source_ruptures, gsims, ordinal):
    """
    """
    hc = models.HazardCalculation.objects.get(oqjob=job_id)
    total_sites = len(hc.site_collection)
    imts = general.im_dict_to_hazardlib(
        hc.intensity_measure_types_and_levels)
    curves = dict((imt, numpy.ones([total_sites, len(imts[imt])]))
                  for imt in imts)
    for source, ruptures in source_ruptures:
        s_sites = source.filter_sites_by_distance_to_source(
            hc.maximum_distance, hc.site_collection
        ) if hc.maximum_distance else hc.site_collection
        if s_sites is None:
            return curves
        for rupture in ruptures:
            r_sites = rupture.source_typology.\
                filter_sites_by_distance_to_rupture(
                    rupture, hc.maximum_distance, s_sites
                    ) if hc.maximum_distance else s_sites
            if r_sites is None:
                continue
            prob = rupture.get_probability_one_or_more_occurrences()
            gsim = gsims[rupture.tectonic_region_type]
            sctx, rctx, dctx = gsim.make_contexts(r_sites, rupture)
            for imt in imts:
                poes = gsim.get_poes(sctx, rctx, dctx, imt, imts[imt],
                                     hc.truncation_level)
                curves[imt] *= r_sites.expand(
                    (1. - prob) ** poes, total_sites, placeholder=1)
        logs.LOG.warn(
            'Generated %d ruptures for source %s', len(ruptures),
            source.source_id)

    # shortcut for filtered sources giving no contribution;
    # this is essential for performance, we want to avoid
    # returning big arrays of zeros (MS)
    return [None if (curves[imt] == 1.0).all()
            else 1. - curves[imt] for imt in sorted(imts)], ordinal


@tasks.oqtask
def compute_hazard_curves(job_id, sources, lt_rlz, ltp):
    """
    Celery task for hazard curve calculator.

    Samples logic trees, gathers site parameters, and calls the hazard curve
    calculator.

    Once hazard curve data is computed, result progress updated (within a
    transaction, to prevent race conditions) in the
    `htemp.hazard_curve_progress` table.

    :param int job_id:
        ID of the currently running job.
    :param sources:
        List of :class:`openquake.hazardlib.source.base.SeismicSource` objects
    :param lt_rlz:
        a :class:`openquake.engine.db.models.LtRealization` instance
    :param ltp:
        a :class:`openquake.engine.input.LogicTreeProcessor` instance
    """
    hc = models.HazardCalculation.objects.get(oqjob=job_id)
    gsims = ltp.parse_gmpe_logictree_path(lt_rlz.gsim_lt_path)
    tom = PoissonTOM(hc.investigation_time)

    extra_args = []
    for i, source in enumerate(sources):
        ruptures = list(source.iter_ruptures(tom))
        first_ruptures = ruptures[:500]
        other_ruptures = ruptures[500:]
        if other_ruptures:
            extra_args.append(
                (job_id, [(source, other_ruptures)], gsims, lt_rlz.ordinal))
        cs, ordinal = compute_extra_curves.task_func(
            job_id, [(source, first_ruptures)], gsims, lt_rlz.ordinal)
        if i == 0:  # first time
            curves = cs
        else:
            update(curves, cs)

    return curves, ordinal, extra_args


def update(curves, newcurves):
    """
    """
    for i, curve in enumerate(newcurves):
        if curve is not None:
            curves[i] = 1. - (1. - curves[i]) * (1. - curve)


def make_zeros(realizations, sites, imtls):
    """
    Returns a list of R lists containing I numpy arrays of S * L zeros, where
    R is the number of realizations, I is the number of intensity measure
    types, S the number of sites and L the number of intensity measure levels.

    :params sites: the site collection
    :param imtls: a dictionary of intensity measure types and levels
    """
    return [[numpy.zeros((len(sites), len(imtls[imt])))
             for imt in sorted(imtls)] for _ in range(len(realizations))]


class ClassicalHazardCalculator(general.BaseHazardCalculator):
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
        super(ClassicalHazardCalculator, self).pre_execute()
        imtls = self.hc.intensity_measure_types_and_levels
        self.curves_by_rlz = make_zeros(
            self._get_realizations(), self.hc.site_collection, imtls)
        self.extra_args = []
        n_rlz = len(self._get_realizations())
        n_levels = sum(len(lvls) for lvls in imtls.itervalues()
                       ) / float(len(imtls))
        n_sites = len(self.hc.site_collection)
        total = n_rlz * len(imtls) * n_levels * n_sites
        logs.LOG.info('Considering %d realization(s), %d IMT(s), %d level(s) '
                      'and %d sites, total %d', n_rlz, len(imtls), n_levels,
                      n_sites, total)

    @EnginePerformanceMonitor.monitor
    def post_execute(self):
        self.parallelize(compute_extra_curves, self.extra_args,
                         self.task_completed)
        self.save_hazard_curves()

    @EnginePerformanceMonitor.monitor
    def task_completed(self, task_result):
        """
        This is used to incrementally update hazard curve results by combining
        an initial value with some new results. (Each set of new results is
        computed over only a subset of seismic sources defined in the
        calculation model.)

        :param task_result:
            A triple (curves_by_imt, ordinal, extra_args) where curves_by_imt
            is a list of 2-D numpy arrays representing the new results which
            needs to be combined with the current value. These should be the
            same shape as self.curves_by_rlz[i][j] where i is the realization
            ordinal and j the IMT ordinal.
        """
        if len(task_result) == 3:  # coming from compute_hazard_curves
            curves_by_imt, i, extras = task_result
            self.extra_args.extend(extras)
        else:  # coming from compute_extra_curves
            curves_by_imt, i = task_result
        update(self.curves_by_rlz[i], curves_by_imt)
        self.log_percent(task_result)

    # this could be parallelized in the future, however in all the cases
    # I have seen until now, the serialized approach is fast enough (MS)
    @EnginePerformanceMonitor.monitor
    def save_hazard_curves(self):
        """
        Post-execution actions. At the moment, all we do is finalize the hazard
        curve results.
        """
        imtls = self.hc.intensity_measure_types_and_levels
        for i, curves_imts in enumerate(self.curves_by_rlz):
            rlz = models.LtRealization.objects.get(
                hazard_calculation=self.hc, ordinal=i)

            # create a new `HazardCurve` 'container' record for each
            # realization (virtual container for multiple imts)
            models.HazardCurve.objects.create(
                output=models.Output.objects.create_output(
                    self.job, "hc-multi-imt-rlz-%s" % rlz.id,
                    "hazard_curve_multi"),
                lt_realization=rlz,
                imt=None,
                investigation_time=self.hc.investigation_time)

            # create a new `HazardCurve` 'container' record for each
            # realization for each intensity measure type
            for imt, curves_by_imt in zip(sorted(imtls), curves_imts):
                hc_im_type, sa_period, sa_damping = from_string(imt)

                # save output
                hco = models.Output.objects.create(
                    oq_job=self.job,
                    display_name="Hazard Curve rlz-%s" % rlz.id,
                    output_type='hazard_curve',
                )

                # save hazard_curve
                haz_curve = models.HazardCurve.objects.create(
                    output=hco,
                    lt_realization=rlz,
                    investigation_time=self.hc.investigation_time,
                    imt=hc_im_type,
                    imls=imtls[imt],
                    sa_period=sa_period,
                    sa_damping=sa_damping,
                )

                # save hazard_curve_data
                points = self.hc.points_to_compute()
                logs.LOG.info('saving %d hazard curves for %s, imt=%s',
                              len(points), hco, imt)
                writer.CacheInserter.saveall([
                    models.HazardCurveData(
                        hazard_curve=haz_curve,
                        poes=list(poes),
                        location='POINT(%s %s)' % (p.longitude, p.latitude),
                        weight=rlz.weight)
                    for p, poes in zip(points, curves_by_imt)])
        del self.curves_by_rlz  # save memory for the post_processing phase

    def post_process(self):
        """
        Optionally generates aggregate curves, hazard maps and
        uniform_hazard_spectra.
        """
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
                post_proc.hazard_curves_to_hazard_map_task_arg_gen(self.job),
                self.log_percent)

        if self.hc.uniform_hazard_spectra:
            post_proc.do_uhs_post_proc(self.job)

        logs.LOG.debug('< done with post processing')
