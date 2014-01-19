# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4

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


"""Utility functions related to splitting work into tasks."""

import math
from functools import wraps

from celery.task.sets import TaskSet
from celery.result import ResultSet, EagerResult
from celery.task import task

from openquake.engine import logs, no_distribute
from openquake.engine.db import models
from openquake.engine.utils import config, general
from openquake.engine.writer import CacheInserter
from openquake.engine.performance import EnginePerformanceMonitor


class CeleryTaskManager(object):
    MAX_BLOCK_SIZE = 1000

    def __init__(self, concurrent_tasks=None, max_block_size=None):
        self.concurrent_tasks = concurrent_tasks or \
            int(config.get('hazard', 'concurrent_tasks'))
        self.max_block_size = max_block_size or self.MAX_BLOCK_SIZE

    def split(self, iterable):
        items = list(iterable)
        assert len(items) > 0, 'No items in %s' % items
        bs_float = float(len(items)) / self.concurrent_tasks
        bs = min(int(math.ceil(bs_float)), self.max_block_size)
        logs.LOG.debug('Using block size=%d', bs)
        return general.block_splitter(items, bs)

    def spawn(self, task, job_id, sequence, *extra):
        self.job_id = job_id
        arglist = list(self.split(sequence))
        if no_distribute():
            rs = [EagerResult(str(i), task.task_func(job_id, args, *extra),
                              'SUCCESS') for i, args in enumerate(arglist, 1)]
        else:
            rs = [task.delay(job_id, args, *extra) for args in arglist]
        return rs

    def initialize_progress(self, task, arglist):
        self.taskname = task.task_func.__name__
        self.num_tasks = len(arglist)
        self.tasksdone = 0
        self.percent = 0.0
        logs.LOG.progress(
            'spawning %d tasks of kind %s', self.num_tasks, self.taskname)

    def reduce(self, results, agg, acc=None):
        """
        """
        for result in ResultSet(results):
            acc = result if acc is None else agg(acc, result)
            self.log_percent(result)
        return acc

    def map_reduce(self, agg, task, job_id, sequence, *extra):
        """
        """
        return self.reduce(self.spawn(task, job_id, sequence, *extra), agg)

    def log_percent(self, result):
        """
        Log the progress percentage, if changed.
        It is called at each task completion.

        :param task_result: the result of the task (often None)
        """
        self.tasksdone += 1
        percent = int(float(self.tasksdone) / self.num_tasks * 100)
        if percent > self.percent:
            logs.LOG.progress('> %s %3d%% complete', self.taskname, percent)
            self.percent = percent
            # fix the progress handler on the engine server


def map_reduce(task, task_args, agg, acc):
    """
    Given a task and an iterable of positional arguments, apply the
    task function to the arguments in parallel and return an aggregate
    result depending on the initial value of the accumulator
    and on the aggregation function. To save memory, the order is
    not preserved and there is no list with the intermediated results:
    the accumulator is incremented as soon as a task result comes.

    :param task: a `celery` task callable.
    :param task_args: an iterable over positional arguments
    :param agg: the aggregation function, (acc, val) -> new acc
    :param acc: the initial value of the accumulator
    :returns: the final value of the accumulator

    NB: if the environment variable OQ_NO_DISTRIBUTE is set the
    tasks are run sequentially in the current process and then
    map_reduce(task, task_args, agg, acc) is the same as
    reduce(agg, itertools.starmap(task, task_args), acc).
    Users of map_reduce should be aware of the fact that when
    thousands of tasks are spawned and large arguments are passed
    or large results are returned they may incur in memory issue:
    this is way the calculators limit the queue with the
    `concurrent_task` concept.
    """
    if no_distribute():
        for the_args in task_args:
            acc = agg(acc, task.task_func(*the_args))
    else:
        taskset = TaskSet(tasks=map(task.subtask, task_args))
        for result in taskset.apply_async():
            acc = agg(acc, result)
    return acc


# used to implement BaseCalculator.parallelize, which takes in account
# the `concurrent_task` concept to avoid filling the Celery queue
def parallelize(task, task_args, side_effect=lambda val: None):
    """
    Given a celery task and an iterable of positional arguments, apply the
    callable to the arguments in parallel. It is possible to pass a
    function side_effect(val) which takes the return value of the
    callable and does something with it (such as saving or printing
    it). Notice that the order is not preserved. parallelize returns None.

    :param task: a celery task
    :param task_args: an iterable over positional arguments
    :param side_effect: a function val -> None

    NB: if the environment variable OQ_NO_DISTRIBUTE is set the
    tasks are run sequentially in the current process.
    """
    map_reduce(task, task_args, lambda acc, val: side_effect(val), None)


def oqtask(task_func):
    """
    Task function decorator which sets up logging and catches (and logs) any
    errors which occur inside the task. Also checks to make sure the job is
    actually still running. If it is not running, the task doesn't get
    executed, so we don't do useless computation.
    """

    @wraps(task_func)
    def wrapped(*args):
        """
        Initialize logs, make sure the job is still running, and run the task
        code surrounded by a try-except. If any error occurs, log it as a
        critical failure.
        """
        # job_id is always assumed to be the first argument
        job_id = args[0]
        job = models.OqJob.objects.get(id=job_id)
        if job.is_running is False:
            # the job was killed, it is useless to run the task
            return

        # it is important to save the task id soon, so that
        # the revoke functionality can work
        EnginePerformanceMonitor.store_task_id(job_id, tsk)

        with EnginePerformanceMonitor(
                'total ' + task_func.__name__, job_id, tsk, flush=True):

            with EnginePerformanceMonitor(
                    'loading calculation object', job_id, tsk, flush=True):
                calculation = job.calculation

            # tasks write on the celery log file
            logs.init_logs(
                level=job.log_level,
                calc_domain='hazard' if isinstance(
                    calculation, models.HazardCalculation) else'risk',
                calc_id=calculation.id)
            try:
                return task_func(*args)
            finally:
                CacheInserter.flushall()
                # the task finished, we can remove from the performance
                # table the associated row 'storing task id'
                models.Performance.objects.filter(
                    oq_job=job,
                    operation='storing task id',
                    task_id=tsk.request.id).delete()
    celery_queue = config.get('amqp', 'celery_queue')
    tsk = task(wrapped, queue=celery_queue)
    tsk.task_func = task_func
    return tsk
