"""True continuous-profile z refinement at a fixed Fourier truncation.

These checks measure staircase sensitivity at the stated N. They do not
establish a joint Fourier/staircase error bound or calibrated position accuracy.
"""
import numpy as np
from mark_rcwa.geometry import refine_slices, trapezoid
from mark_rcwa.materials import defaults
from mark_rcwa.solvers import MeentSolver, Simulation
from mark_rcwa.alignment import pair_apd


def staircase_scan(target_stack, reference_stack, materials, simulation,
                   refinement_factors=(1,2,4), monitored_orders=(1,3,5),
                   phase_tolerance_deg=.1, apd_tolerance_nm=1.,
                   efficiency_tolerance=1e-3, min_efficiency=1e-8):
    """Regenerate continuous profiles, solve each, and compare successive fields.

    Factors always refer to the original continuous recipe's requested slices;
    a factor of four therefore regenerates intersections at four times the
    initial slice count. It does not duplicate layers of a previously solved
    staircase. Every refinement uses the same simulation.fourier_order.
    """
    factors=tuple(refinement_factors)
    if len(factors)<3 or factors[0]!=1 or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) or x<1 for x in factors) or list(factors)!=sorted(set(factors)):
        raise ValueError('staircase refinement requires at least three strictly increasing integer factors starting at 1')
    monitored=tuple(monitored_orders)
    if not monitored or any(isinstance(m,(bool,np.bool_)) or not isinstance(m,(int,np.integer)) or m<=0 for m in monitored) or len(monitored)!=len(set(monitored)):
        raise ValueError('staircase monitored orders must be unique positive integers')
    if not np.isfinite([phase_tolerance_deg,apd_tolerance_nm,efficiency_tolerance,min_efficiency]).all() or min(phase_tolerance_deg,apd_tolerance_nm,efficiency_tolerance)<=0 or min_efficiency<0:
        raise ValueError('staircase tolerances must be positive and reliability floor nonnegative')
    for stack in (target_stack,reference_stack):
        if 'profile_recipe' not in stack.metadata:
            raise ValueError('staircase validation requires a continuous profile_recipe for both structures')
    solver=MeentSolver()
    previous=None
    stable_count=0
    history=[]
    for factor in factors:
        target_geometry=target_stack if factor==1 else refine_slices(target_stack,int(factor))
        reference_geometry=reference_stack if factor==1 else refine_slices(reference_stack,int(factor))
        target=solver.solve(target_geometry,materials,simulation)
        reference=solver.solve(reference_geometry,materials,simulation)
        pairs=[]
        step_passed=previous is not None
        for m in monitored:
            row={'order':int(m),'reliable':False,'apd_nm':None,'delta_pair_phase_rad':None,
                 'efficiency_delta':None,'individual_phase_delta_deg':None,'pair_phase_delta_deg':None,
                 'apd_delta_nm':None,'complex_delta':None,'passed':None,'reason':None}
            try:
                apd=pair_apd(reference,target,int(m),min_efficiency)
            except ValueError as exc:
                row.update(passed=False,reason=str(exc))
                step_passed=False
                pairs.append(row)
                continue
            indices=[int(np.flatnonzero(target.orders==q)[0]) for q in (m,-m)]
            target_amp=target.amplitude[indices]
            reference_amp=reference.amplitude[indices]
            row.update(reliable=apd.reliable,apd_nm=apd.apd_nm,delta_pair_phase_rad=apd.delta_phase_rad,
                target_amplitudes=[[float(z.real),float(z.imag)] for z in target_amp],
                reference_amplitudes=[[float(z.real),float(z.imag)] for z in reference_amp],
                target_efficiencies=target.efficiency[indices].tolist(),
                reference_efficiencies=reference.efficiency[indices].tolist(),
                target_individual_phase_deg=np.degrees(np.angle(target_amp)).tolist(),
                reference_individual_phase_deg=np.degrees(np.angle(reference_amp)).tolist(),reason=apd.reason)
            if previous is not None:
                old_target,old_reference,old_pairs=previous
                old_row=next(p for p in old_pairs if p['order']==m)
                if apd.reliable and old_row['reliable']:
                    current_amp=np.r_[target_amp,reference_amp]
                    old_amp=np.r_[old_target.amplitude[indices],old_reference.amplitude[indices]]
                    individual_phase=np.angle(current_amp/old_amp)
                    pair_phase=[np.angle(np.exp(1j*(individual_phase[0]-individual_phase[1]))),
                                np.angle(np.exp(1j*(individual_phase[2]-individual_phase[3])))]
                    current_eta=np.r_[target.efficiency[indices],reference.efficiency[indices]]
                    old_eta=np.r_[old_target.efficiency[indices],old_reference.efficiency[indices]]
                    delta_apd_phase=np.angle(np.exp(1j*(apd.delta_phase_rad-old_row['delta_pair_phase_rad'])))
                    row.update(efficiency_delta=float(max(abs(current_eta-old_eta))),
                        individual_phase_delta_deg=float(max(abs(np.degrees(individual_phase)))),
                        pair_phase_delta_deg=float(max(abs(np.degrees(pair_phase)))),
                        individual_phase_changes_deg=np.degrees(individual_phase).tolist(),
                        pair_phase_changes_deg=np.degrees(pair_phase).tolist(),
                        apd_delta_nm=float(abs(target.pitch_um*1000*delta_apd_phase/(4*np.pi*m))),
                        complex_delta=float(max(abs(current_amp-old_amp))))
                    row['passed']=bool(row['efficiency_delta']<efficiency_tolerance and
                        max(row['individual_phase_delta_deg'],row['pair_phase_delta_deg'])<phase_tolerance_deg and
                        row['apd_delta_nm']<apd_tolerance_nm)
                    if not row['passed']:
                        row['reason']='one or more staircase thresholds exceeded'
                else:
                    row.update(passed=False,reason='current or previous pair below reliability floor')
                step_passed=bool(step_passed and row['passed'])
            pairs.append(row)
        stable_count=stable_count+1 if step_passed else 0
        history.append({'refinement_factor':int(factor),
            'target_requested_slices':int(target_geometry.metadata['profile_recipe']['slices']),
            'reference_requested_slices':int(reference_geometry.metadata['profile_recipe']['slices']),
            'target_actual_slices':len(target_geometry.layers),'reference_actual_slices':len(reference_geometry.layers),
            'target_max_dz_um':float(max(layer.thickness_um for layer in target_geometry.layers)),
            'reference_max_dz_um':float(max(layer.thickness_um for layer in reference_geometry.layers)),
            'target_R':target.reflectance,'target_T':target.transmittance,
            'reference_R':reference.reflectance,'reference_T':reference.transmittance,
            'max_energy_residual':float(max(abs(target.absorption),abs(reference.absorption))),
            'pairs':pairs,'step_passed':bool(step_passed),'successive_passing_steps':stable_count})
        previous=target,reference,pairs
    return {'polarization':simulation.polarization,'fixed_fourier_order':simulation.fourier_order,
        'refinement_factors':[int(x) for x in factors], 'monitored_orders':[int(m) for m in monitored],
        'tolerances':{'individual_phase_deg':float(phase_tolerance_deg),'pair_phase_deg':float(phase_tolerance_deg),
            'apd_nm':float(apd_tolerance_nm),'efficiency':float(efficiency_tolerance),'min_efficiency':float(min_efficiency)},
        'history':history,'converged':stable_count>=2,'status':'PASS' if stable_count>=2 else 'FAIL',
        'rule':'Two successive true continuous-profile refinements pass every threshold for every monitored pair.',
        'scope':'Fixed Fourier N only; joint Fourier/staircase error bound has not been established.'}


def _validation_record(name,target_stack,reference_stack,simulation,report):
    last_pairs=[p for row in report['history'][-2:] for p in row['pairs']]
    error={key:max((p[source] for p in last_pairs if p[source] is not None),default=None)
        for key,source in [('individual_phase_deg','individual_phase_delta_deg'),
                           ('pair_phase_deg','pair_phase_delta_deg'),('apd_nm','apd_delta_nm'),
                           ('efficiency','efficiency_delta')]}
    error['unreliable_pair_steps']=sum(not p['reliable'] or p['individual_phase_delta_deg'] is None for p in last_pairs)
    tolerances={key:report['tolerances'][key] for key in ('individual_phase_deg','pair_phase_deg','apd_nm','efficiency')}
    tolerances['unreliable_pair_steps']=0
    return {'name':name,'parameters':{'target_profile':target_stack.metadata['profile_recipe'],
        'reference_profile':reference_stack.metadata['profile_recipe'],
        'wavelength_um':simulation.wavelength_um,'theta_deg':simulation.theta_deg,
        'polarization':simulation.polarization,'fixed_fourier_order':simulation.fourier_order,
        'refinement_factors':report['refinement_factors'],'monitored_orders':report['monitored_orders']},
        'solver':'Meent 0.13.2, pinned adapter','expected':{'two_successive_passing_steps':True},
        'actual':report,'error':error,'tolerances':tolerances,'status':report['status'],
        'reason':'Every geometry is regenerated from its continuous profile. Errors summarize the final two refinement steps; '
            'all earlier steps remain in actual.history. Fixed Fourier N only: a joint Fourier/staircase error bound is not established.'}


def run_staircase_validation():
    """Run demonstration TE N127 and TM N191 studies; return JSON-safe records."""
    materials=defaults()
    records=[]
    for polarization,truncation,initial_slices in [('TE',127,16),('TM',191,8)]:
        common=dict(pitch_um=8.,depth_um=.16,duty=.5,slices=initial_slices,
                    ridge='Si3N4',groove='air',top='air',bottom='glass_demo')
        target=trapezoid(left_angle_deg=60.,right_angle_deg=85.,**common)
        reference=trapezoid(left_angle_deg=85.,right_angle_deg=85.,**common)
        simulation=Simulation(wavelength_um=.633,theta_deg=0.,polarization=polarization,fourier_order=truncation,na=.8)
        factors=(1,2,4,8) if polarization=='TM' else (1,2,4)
        report=staircase_scan(target,reference,materials,simulation,refinement_factors=factors)
        records.append(_validation_record(f'staircase_asymmetric_{polarization}',target,reference,simulation,report))
    return records
