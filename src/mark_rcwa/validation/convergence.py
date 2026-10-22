"""Phase-aware paired Fourier convergence, with two successive passing steps."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from mark_rcwa.solvers import MeentSolver
from mark_rcwa.alignment import pair_apd


def convergence_scan(stack, reference_stack, materials, simulation, settings=None):
    settings = {} if settings is None else settings
    allowed={'orders','monitored_orders','efficiency_tolerance','phase_tolerance_deg',
        'apd_tolerance_nm','min_efficiency','extend_orders','auto_extend'}
    if not isinstance(settings,dict):
        raise ValueError('convergence settings must be a mapping')
    unknown=set(settings)-allowed
    if unknown:
        raise ValueError(f'unknown convergence settings: {sorted(unknown)}')
    if not isinstance(settings.get('auto_extend',True),bool):
        raise ValueError('auto_extend must be a boolean')
    requested = settings.get('orders', [3,5,7,11,15,21,31,41,55])
    if not requested or any(isinstance(n,bool) or not isinstance(n,int) or n < 0 or n > 255 for n in requested) or list(requested) != sorted(set(requested)):
        raise ValueError('convergence orders must be a strictly increasing integer sequence within [0,255]')
    monitored = settings.get('monitored_orders', [1,3,5])
    if not monitored or any(isinstance(m,bool) or not isinstance(m,int) or m <= 0 for m in monitored) or len(set(monitored)) != len(monitored):
        raise ValueError('monitored_orders must be unique positive integers')
    etol = settings.get('efficiency_tolerance', 1e-3)
    ptol = settings.get('phase_tolerance_deg', .1)
    atol = settings.get('apd_tolerance_nm', 1.)
    floor = settings.get('min_efficiency', 1e-8)
    if not np.isfinite([etol,ptol,atol,floor]).all() or min(etol,ptol,atol) <= 0 or floor < 0:
        raise ValueError('convergence tolerances must be positive and floor nonnegative')
    additional = settings.get('extend_orders', [71,95,127,155,191,223,255]) if settings.get('auto_extend', True) else []
    if any(isinstance(n,bool) or not isinstance(n,int) or not 0 <= n <= 255 for n in additional):
        raise ValueError('extend_orders must be valid Fourier orders')
    truncations = sorted(set([*requested, *[n for n in additional if n > requested[-1]]]))
    solver = MeentSolver()
    previous = None
    stable_count = 0
    history = []
    unwrapped = {}
    for n in truncations:
        spec = replace(simulation, fourier_order=n)
        result = solver.solve(stack, materials, spec)
        reference = solver.solve(reference_stack, materials, spec)
        pairs = []
        step_passed = previous is not None
        reliable_count = 0
        for m in monitored:
            item = {'order': m, 'reliable': False, 'apd_nm': None,
                    'delta_phase_rad': None, 'efficiency_delta': None,
                    'complex_delta': None, 'phase_delta_deg': None, 'pair_phase_delta_deg': None,
                    'apd_delta_nm': None, 'passed': False, 'reason': None}
            if m > n:
                item['reason'] = 'pair outside retained basis'
                step_passed = False
                pairs.append(item)
                continue
            try:
                apd = pair_apd(reference,result,m,floor)
            except ValueError as exc:
                item['reason'] = str(exc)
                step_passed = False
                pairs.append(item)
                continue
            idx = [int(np.flatnonzero(result.orders==q)[0]) for q in (m,-m)]
            item.update(test_efficiencies=result.efficiency[idx].tolist(),
                reference_efficiencies=reference.efficiency[idx].tolist(),
                test_amplitudes=[[float(a.real),float(a.imag)] for a in result.amplitude[idx]],
                reference_amplitudes=[[float(a.real),float(a.imag)] for a in reference.amplitude[idx]],
                reliable=apd.reliable, delta_phase_rad=apd.delta_phase_rad, reason=apd.reason)
            if not apd.reliable:
                # Explicitly excluded from phase accuracy, not silently passed.
                pairs.append(item)
                continue
            reliable_count += 1
            delta = apd.delta_phase_rad
            if m in unwrapped:
                delta = unwrapped[m] + np.angle(np.exp(1j*(delta-unwrapped[m])))
            unwrapped[m] = float(delta)
            item['apd_nm'] = float(stack.pitch_um*1000*delta/(4*np.pi*m))
            item['test_pair_phase_deg'] = float(np.degrees(np.angle(result.amplitude[idx[0]]/result.amplitude[idx[1]])))
            if previous is not None and m <= previous[0].orders.max():
                old_result, old_reference, old_pairs = previous
                old = next(x for x in old_pairs if x['order'] == m)
                if old['reliable']:
                    old_idx = [int(np.flatnonzero(old_result.orders==q)[0]) for q in (m,-m)]
                    current_amp = np.r_[result.amplitude[idx], reference.amplitude[idx]]
                    old_amp = np.r_[old_result.amplitude[old_idx], old_reference.amplitude[old_idx]]
                    current_eta = np.r_[result.efficiency[idx], reference.efficiency[idx]]
                    old_eta = np.r_[old_result.efficiency[old_idx], old_reference.efficiency[old_idx]]
                    phase = np.angle(current_amp/old_amp)
                    pair_change = np.array([np.angle(np.exp(1j*(phase[0]-phase[1]))), np.angle(np.exp(1j*(phase[2]-phase[3])))])
                    item.update(efficiency_delta=float(max(abs(current_eta-old_eta))),
                        complex_delta=float(max(abs(current_amp-old_amp))),
                        phase_delta_deg=float(max(abs(np.degrees(phase)))),
                        pair_phase_delta_deg=float(max(abs(np.degrees(pair_change)))),
                        apd_delta_nm=float(abs(item['apd_nm']-old['apd_nm'])))
                    item['passed'] = bool(item['efficiency_delta'] < etol and
                        max(item['phase_delta_deg'],item['pair_phase_delta_deg']) < ptol and item['apd_delta_nm'] < atol)
            if not item['passed']:
                step_passed = False
            pairs.append(item)
        step_passed = bool(step_passed and reliable_count > 0)
        stable_count = stable_count+1 if step_passed else 0
        history.append({'N': n, 'R': result.reflectance, 'T': result.transmittance,
            'absorption_flux_residual': result.absorption,
            'lossless_q_branch_corrections': result.branch_corrections,
            'pairs': pairs, 'step_passed': step_passed, 'successive_passing_steps': stable_count})
        previous = result, reference, pairs
        # Always execute the requested range, then extend only as needed.
        if n >= requested[-1] and stable_count >= 2:
            break
    return {'converged': stable_count >= 2, 'status': 'PASS' if stable_count >= 2 else 'UNCONVERGED',
        'requested_orders': requested, 'executed_orders': [row['N'] for row in history],
        'final_N': history[-1]['N'], 'monitored_orders': monitored,
        'phase_unreliable_orders': [p['order'] for p in history[-1]['pairs'] if not p['reliable']],
        'tolerances': {'efficiency': etol, 'phase_deg': ptol, 'apd_nm': atol, 'min_efficiency': floor},
        'rule': 'two successive changes pass for reliable pairs; low power explicitly excluded; this is numerical evidence, not a global error bound',
        'history': history}


def write_convergence(report, directory):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'convergence.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    records = [{'N': row['N'], **item} for row in report['history'] for item in row['pairs']]
    fields = ['N','order','reliable','apd_nm','delta_phase_rad','efficiency_delta','complex_delta',
              'phase_delta_deg','pair_phase_delta_deg','apd_delta_nm','passed','reason']
    with (directory/'convergence.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for m in report['monitored_orders']:
        items=[(row['N'],next(p for p in row['pairs'] if p['order']==m)) for row in report['history']]
        valid=[(n,p) for n,p in items if p['reliable']]
        change=[(n,p) for n,p in items if p['phase_delta_deg'] is not None]
        axs[0,0].plot([n for n,p in valid],[p['test_efficiencies'][0] for n,p in valid],'o-',label=f'm={m}')
        axs[0,1].plot([n for n,p in valid],[p['test_pair_phase_deg'] for n,p in valid],'o-',label=f'm={m}')
        axs[1,0].plot([n for n,p in valid],[p['apd_nm'] for n,p in valid],'o-',label=f'm={m}')
        axs[1,1].semilogy([n for n,p in change],[max(p['phase_delta_deg'],p['pair_phase_delta_deg'],1e-15) for n,p in change],'o-',label=f'm={m}')
    labels=['Test +m efficiency','Test pair phase (deg)','Model APD (nm)','Successive phase change (deg)']
    for ax,label in zip(axs.flat,labels):
        ax.set(xlabel='Fourier truncation N',ylabel=label)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    axs[1,1].axhline(report['tolerances']['phase_deg'],color='gray',ls='--')
    fig.suptitle(f'Fourier convergence: {report["status"]}, final N={report["final_N"]}')
    fig.savefig(directory/'convergence.png',dpi=160)
    plt.close(fig)
