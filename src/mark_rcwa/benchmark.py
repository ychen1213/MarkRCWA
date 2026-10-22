"""CPU benchmarks with independent worker RSS and numerical equality checks."""
from dataclasses import replace
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import psutil
import numpy as np
from mark_rcwa.geometry import rectangle, trapezoid, supercell
from mark_rcwa.materials import defaults
from mark_rcwa.solvers import MeentSolver, Simulation


def _problem(scenario):
    materials=defaults()
    kind=scenario.get('kind','rectangle')
    if kind=='trapezoid':
        stack=trapezoid(depth_um=.16,left_angle_deg=60,right_angle_deg=85,slices=scenario.get('slices',8))
    elif kind=='supercell':
        stack=supercell([rectangle(duty=.5),rectangle(duty=.44,depth_um=.15),rectangle(duty=.54,depth_um=.09)])
    else:
        stack=rectangle()
    return stack,materials,Simulation(fourier_order=scenario.get('N',21))


def compare_results(first,second):
    complex_error=float(max(np.max(abs(first.amplitude-second.amplitude)),
                            np.max(abs(first.transmission_amplitude-second.transmission_amplitude))))
    eta_error=float(max(np.max(abs(first.efficiency-second.efficiency)),
                        np.max(abs(first.transmission_efficiency-second.transmission_efficiency))))
    mask=(first.efficiency>1e-8)&(second.efficiency>1e-8)&first.propagating&second.propagating
    phase_error=float(max(abs(np.degrees(np.angle(first.amplitude[mask]/second.amplitude[mask]))),default=0))
    if complex_error>1e-11 or eta_error>1e-12 or phase_error>1e-9:
        raise ValueError(f'benchmark cached/reference numerical mismatch: {complex_error}, {eta_error}, {phase_error}')
    return {'complex_amplitude':complex_error,'efficiency':eta_error,'phase_deg':phase_error}


def _worker(scenario):
    stack,materials,spec=_problem(scenario)
    solver=MeentSolver(cache_enabled=scenario.get('cache_enabled',True))
    process=psutil.Process()
    baseline_rss=process.memory_info().rss
    started=time.perf_counter()
    first=solver.solve(stack,materials,spec)
    cold=time.perf_counter()-started
    warm_times=[]
    errors=[]
    for _ in range(3):
        started=time.perf_counter()
        result=solver.solve(stack,materials,spec)
        warm_times.append(time.perf_counter()-started)
        errors.append(compare_results(first,result))
    # Capture the timing window before allocating the uncached validation oracle.
    timing_info=process.memory_info()
    peak=getattr(timing_info,'peak_wset',timing_info.rss)
    # A separate fresh uncached solver checks optimization against its baseline.
    uncached_times=[]
    for _ in range(3):
        started=time.perf_counter()
        reference=MeentSolver(cache_enabled=False).solve(stack,materials,spec)
        uncached_times.append(time.perf_counter()-started)
        baseline_error=compare_results(first,reference)
    return {**scenario,'cold_solve_s':cold,'warm_solve_s':float(np.mean(warm_times)),
        'steady_uncached_s':float(np.mean(uncached_times)), 'uncached_samples_s':uncached_times,
        'warm_samples_s':warm_times,'baseline_rss_MiB':baseline_rss/2**20,
        'peak_rss_MiB':peak/2**20,'peak_measurement':'Windows lifetime peak working set before uncached verification' if hasattr(timing_info,'peak_wset') else 'RSS at end of timing; not a certified peak',
        'cache_info':solver.cache_info(),'cached_vs_uncached_error':baseline_error,
        'repeat_max_complex_error':max(row['complex_amplitude'] for row in errors)}


def _scan_benchmark():
    materials=defaults()
    base=rectangle()
    spec=Simulation(fourier_order=31)
    # Warm shared imports and BLAS discovery before either optimization arm.
    # Each measured arm still starts with its own empty solver cache.
    MeentSolver(cache_enabled=False).solve(base,materials,spec)
    records=[]
    for parameter,values in [('wavelength_um',[.60,.62,.633,.65,.68]),('depth_um',[.08,.1,.12,.14,.16])]:
        for enabled in (False,True):
            solver=MeentSolver(cache_enabled=enabled)
            jobs=[(base,materials,replace(spec,wavelength_um=value)) if parameter=='wavelength_um'
                  else (rectangle(depth_um=value),materials,spec) for value in values]
            started=time.perf_counter()
            results=[solver.solve(*job) for job in jobs]
            elapsed=time.perf_counter()-started
            comparisons=[compare_results(result,MeentSolver(cache_enabled=False).solve(*job)) for result,job in zip(results,jobs)]
            records.append({'parameter':parameter,'values':values,'cache_enabled':enabled,
                'total_s':elapsed,'samples':len(values),'cache_info':solver.cache_info(),
                'max_complex_error':max(row['complex_amplitude'] for row in comparisons)})
    return records


def _batch_benchmark():
    from mark_rcwa.solvers.batch import batch_solve
    materials=defaults()
    jobs=[(rectangle(duty=duty),materials,Simulation(fourier_order=31)) for duty in [.4,.42,.44,.46,.48,.52]]
    started=time.perf_counter()
    serial=batch_solve(jobs,workers=1)
    serial_time=time.perf_counter()-started
    started=time.perf_counter()
    parallel=batch_solve(jobs,workers=2)
    parallel_time=time.perf_counter()-started
    errors=[compare_results(a,b) for a,b in zip(serial,parallel)]
    return {'samples':len(jobs),'N':31,'serial_s':serial_time,'parallel_2_workers_s':parallel_time,
        'speedup':serial_time/parallel_time,
        'max_complex_error':max(row['complex_amplitude'] for row in errors),
        'interpretation':'Includes process startup. Small batches can be slower in parallel; serial is the default.'}


def run_benchmark(output='reports/benchmark'):
    directory=Path(output)
    directory.mkdir(parents=True,exist_ok=True)
    scenarios=[{'category':'Fourier','N':n} for n in [3,5,7,11,15,21,31,41,55]]
    scenarios += [{'category':'slices','kind':'trapezoid','N':31,'slices':s} for s in [1,4,8,16,32]]
    scenarios += [{'category':'periodicity','kind':'rectangle','N':21},
                  {'category':'periodicity','kind':'supercell','N':63},
                  {'category':'uncached','N':55,'cache_enabled':False}]
    rows=[]
    for scenario in scenarios:
        command=[sys.executable,'-m','mark_rcwa.benchmark','--worker',json.dumps(scenario)]
        process=subprocess.run(command,capture_output=True,text=True,timeout=180)
        if process.returncode!=0:
            raise ValueError(f'benchmark worker failed: {process.stderr}')
        rows.append(json.loads(process.stdout))
    payload={'platform':platform.platform(),'python':platform.python_version(),
        'logical_cpus':psutil.cpu_count(),'solver_precision':'complex128','BLAS_threads_per_solve':1,
        'memory_note':'Worker lifetime RSS before verification includes NumPy/SciPy imports; measured by psutil, not Python tracemalloc.',
        'timing_note':'Cold solve includes first-use library and BLAS discovery overhead. Steady uncached vs warm cached solves are the optimization comparison; scan arms prewarm shared initialization equally.',
        'scenarios':rows,'parameter_scans':_scan_benchmark(),'batch':_batch_benchmark(),
        'GPU':'Not measured; CPU baseline only. JAX/PyTorch/GPU paths are outside current validation.'}
    (directory/'benchmark.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    import csv
    fields=['category','kind','N','slices','cold_solve_s','steady_uncached_s','warm_solve_s','baseline_rss_MiB','peak_rss_MiB']
    with (directory/'benchmark.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    plot_benchmark(payload,directory/'benchmark.png')
    return payload


def plot_benchmark(payload,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    rows=[r for r in payload['scenarios'] if r['category']=='Fourier']
    axs[0,0].plot([r['N'] for r in rows],[r['cold_solve_s'] for r in rows],'o-',label='Cold')
    axs[0,0].plot([r['N'] for r in rows],[r['warm_solve_s'] for r in rows],'o-',label='Warm exact caches')
    axs[0,0].plot([r['N'] for r in rows],[r['steady_uncached_s'] for r in rows],'o-',label='Steady uncached')
    axs[0,0].set(xlabel='Fourier N',ylabel='Solve time (s)')
    axs[0,0].legend()
    axs[0,1].plot([r['N'] for r in rows],[r['peak_rss_MiB'] for r in rows],'o-')
    axs[0,1].set(xlabel='Fourier N',ylabel='Worker peak RSS (MiB)')
    slices=[r for r in payload['scenarios'] if r['category']=='slices']
    axs[1,0].plot([r['slices'] for r in slices],[r['cold_solve_s'] for r in slices],'o-')
    axs[1,0].set(xlabel='Profile slices at N=31',ylabel='Cold solve time (s)')
    scan=payload['parameter_scans']
    for parameter in ('wavelength_um','depth_um'):
        items=[r for r in scan if r['parameter']==parameter]
        axs[1,1].plot(['Uncached','Cached'],[r['total_s'] for r in items],'o-',label=parameter)
    axs[1,1].set(ylabel='Five-sample scan time (s)')
    axs[1,1].legend()
    for ax in axs.flat:
        ax.grid(alpha=.2)
    fig.suptitle('CPU performance; numerical equality checked against fresh uncached solves')
    fig.savefig(path,dpi=160)
    plt.close(fig)


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        print(json.dumps(_worker(json.loads(sys.argv[2])),allow_nan=False))
    else:
        raise SystemExit('Use python -m mark_rcwa benchmark')
