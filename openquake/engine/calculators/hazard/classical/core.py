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
import operator

import numpy

from openquake.hazardlib.imt import from_string
from openquake.hazardlib.tom import PoissonTOM

from openquake.engine.input import logictree
from openquake.engine import logs, writer
from openquake.engine.calculators.hazard import general
from openquake.engine.calculators.hazard.classical import (
    post_processing as post_proc)
from openquake.engine.db import models
from openquake.engine.utils import tasks
from openquake.engine.performance import EnginePerformanceMonitor
from openquake.engine.utils.general import block_splitter

ctm = tasks.CeleryTaskManager()


@tasks.oqtask
def compute_curves(job_id, source_ruptures, gsim_dicts):
    """
    """
    hc = models.HazardCalculation.objects.get(oqjob=job_id)
    total_sites = len(hc.site_collection)
    imts = general.im_dict_to_hazardlib(
        hc.intensity_measure_types_and_levels)
    curves = [dict((imt, numpy.ones([total_sites, len(imts[imt])]))
                   for imt in imts) for _ in gsim_dicts]
    for source, ruptures in source_ruptures:
        s_sites = source.filter_sites_by_distance_to_source(
            hc.maximum_distance, hc.site_collection
        ) if hc.maximum_distance else hc.site_collection
        if s_sites is None:
            continue
        for rupture in ruptures:
            r_sites = rupture.source_typology.\
                filter_sites_by_distance_to_rupture(
                    rupture, hc.maximum_distance, s_sites
                    ) if hc.maximum_distance else s_sites
            if r_sites is None:
                continue
            prob = rupture.get_probability_one_or_more_occurrences()
            for curv, gsim_dict in zip(curves, gsim_dicts):
                gsim = gsim_dict[rupture.tectonic_region_type]
                sctx, rctx, dctx = gsim.make_contexts(r_sites, rupture)
                for imt in imts:
                    poes = gsim.get_poes(sctx, rctx, dctx, imt, imts[imt],
                                         hc.truncation_level)
                    curv[imt] *= r_sites.expand(
                        (1. - prob) ** poes, total_sites, placeholder=1)
        #logs.LOG.warn(
        #    'Generated %d ruptures for source %s', len(ruptures),
        #    source.source_id)

    # shortcut for filtered sources giving no contribution;
    # this is essential for performance, we want to avoid
    # returning big arrays of zeros (MS)
    return [[0 if (curv[imt] == 1.0).all()
            else 1. - curv[imt] for imt in sorted(imts)]
            for curv in curves]


@tasks.oqtask
def compute_ruptures(job_id, sources, gsim_dicts):
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
    :param gsim_dicts:
        a list of dictionaries containing GSIM per tectonic region type
    """
    hc = models.HazardCalculation.objects.get(oqjob=job_id)
    tom = PoissonTOM(hc.investigation_time)
    source_rupts_pairs = []
    n_ruptures = 0
    for source in sources:
        ruptures = list(source.iter_ruptures(tom))
        n_ruptures += len(ruptures)
        for rupts in block_splitter(ruptures, 200):
            source_rupts_pairs.append((source, rupts))
    logs.LOG.warn('Generated %d ruptures', n_ruptures)
    man = tasks.CeleryTaskManager(concurrent_tasks=max(n_ruptures // 200, 1))
    return man.spawn(compute_curves, job_id, source_rupts_pairs, gsim_dicts)


def update(curves, newcurves):
    """
    """
    return [[1. - (1. - c) * (1. - nc) for c, nc in zip(curv, newcurv)]
            for curv, newcurv in zip(curves, newcurves)]


class ClassicalHazardCalculator(general.BaseHazardCalculator):
    """
    Classical PSHA hazard calculator. Computes hazard curves for a given set of
    points.

    For each realization of the calculation, we randomly sample source models
    and GMPEs (Ground Motion Prediction Equations) from logic trees.
    """

    core_calc_task = compute_ruptures

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
        self.imtls = self.hc.intensity_measure_types_and_levels
        self.extra_args = []
        n_rlz = len(self._get_realizations())
        n_levels = sum(len(lvls) for lvls in self.imtls.itervalues()
                       ) / float(len(self.imtls))
        n_sites = len(self.hc.site_collection)
        total = n_rlz * len(self.imtls) * n_levels * n_sites
        logs.LOG.info('Considering %d realization(s), %d IMT(s), %d level(s) '
                      'and %d sites, total %d', n_rlz, len(self.imtls),
                      n_levels, n_sites, total)

    def execute(self):
        ltp = logictree.LogicTreeProcessor.from_hc(self.hc)
        for ltpath, rlzs in self.rlzs_per_ltpath.iteritems():
            sources = self.sources_per_ltpath[ltpath]
            gsim_dicts = [ltp.parse_gmpe_logictree_path(rlz.gsim_lt_path)
                          for rlz in rlzs]
            results = ctm.map_reduce(
                operator.add, compute_ruptures,
                self.job.id, sources, gsim_dicts)
            curves = ctm.reduce(results, update)
            self.save_hazard_curves(curves, rlzs)

        # logs.LOG.info('Spawned %d subtasks', len(task_results))

    # this could be parallelized in the future, however in all the cases
    # I have seen until now, the serialized approach is fast enough (MS)
    @EnginePerformanceMonitor.monitor
    def save_hazard_curves(self, curves, rlzs):
        """
        Post-execution actions. At the moment, all we do is finalize the hazard
        curve results.
        """
        imtls = self.hc.intensity_measure_types_and_levels
        for curves_imts, rlz in zip(curves, rlzs):

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
