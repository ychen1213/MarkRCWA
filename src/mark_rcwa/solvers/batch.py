"""Ordered CPU job batches with one bounded solver cache per worker process."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from numbers import Integral
from threadpoolctl import threadpool_limits
from .meent_adapter import MeentSolver


_worker_solver = None
_worker_thread_limit = None


def _initialize_worker(cache_enabled, cache_bytes):
    global _worker_solver, _worker_thread_limit
    _worker_thread_limit = threadpool_limits(limits=1)
    _worker_solver = MeentSolver(cache_enabled=cache_enabled, cache_bytes=cache_bytes)


def _solve_worker(job):
    return _worker_solver.solve(*job)


def batch_solve(jobs, workers=1, cache_enabled=True, cache_bytes=64*1024**2):
    """Solve (stack, materials, simulation) jobs and retain input ordering.

    Multi-process calls require the ordinary __main__ guard on Windows and
    other spawn-based platforms. Each process has its own cache byte budget.
    """
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
        raise ValueError('workers must be a positive integer')
    jobs = list(jobs)
    if any(not isinstance(job, (tuple, list)) or len(job) != 3 for job in jobs):
        raise ValueError('batch jobs must be (stack, materials, simulation) triples')
    # Validate options even for empty batches or before starting worker processes.
    solver = MeentSolver(cache_enabled=cache_enabled, cache_bytes=cache_bytes)
    if not jobs:
        return []
    if workers == 1:
        return [solver.solve(*job) for job in jobs]
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=multiprocessing.get_context('spawn'),
                             initializer=_initialize_worker, initargs=(cache_enabled, cache_bytes)) as pool:
        return list(pool.map(_solve_worker, jobs))
