"""General 1D Fourier convergence, independent of alignment-mark observables.

Two adjacent passing changes provide numerical convergence evidence. They do
not bound the error against an exact solution, or test geometrical slicing.
"""
from dataclasses import replace
from numbers import Integral, Real

import numpy as np

from mark_rcwa.solvers import MeentSolver
from mark_rcwa.solvers.meent_adapter import outgoing_sqrt


def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError('RCWA convergence cancelled')


def _lossless_basis_covered(stack, materials, simulation, n):
    """Require every propagating order of each lossless half-space in [-N,N].

    An absorbing half-space has no finite propagating/evanescent partition:
    its positive-flux tail is instead checked by the adjacent-step comparison.
    """
    nt = materials.index(stack.top, simulation.wavelength_um).real
    shift = nt*np.sin(np.radians(simulation.theta_deg))
    scale = stack.pitch_um/simulation.wavelength_um
    for medium in (stack.top, stack.bottom):
        index = materials.index(medium, simulation.wavelength_um)
        if index.imag != 0:
            continue
        first = int(np.floor((-index.real-shift)*scale))+1
        last = int(np.ceil((index.real-shift)*scale))-1
        if first <= last and (first < -n or last > n):
            return False
    return True


def _channels(result, stack, materials, simulation):
    """Order-indexed outgoing channels; include power flow into lossy media."""
    nb = materials.index(stack.bottom, simulation.wavelength_um)
    k0 = 2*np.pi/simulation.wavelength_um
    kzbot = outgoing_sqrt((nb*k0)**2-result.kx**2)
    # The positive incident normalization cannot change the flux sign.
    transmitted_flux = (kzbot.real if simulation.polarization == 'TE'
                        else np.real(kzbot/(nb**2)))
    channels = {}
    for i, order in enumerate(result.orders):
        if result.propagating[i]:
            channels[('R', int(order))] = (
                float(result.efficiency[i]), complex(result.amplitude[i]))
        if transmitted_flux[i] > 0:
            channels[('T', int(order))] = (
                float(result.transmission_efficiency[i]),
                complex(result.transmission_amplitude[i]))
    return channels


def general_convergence(stack, materials, simulation,
                        orders=(7, 11, 15, 21, 31, 41, 55),
                        efficiency_tolerance=1e-3, phase_tolerance_deg=.1,
                        min_efficiency=1e-8, cancel_event=None, progress=None,
                        reflection_complex_relative_tolerance=.01,
                        near_zero_amplitude=1e-10):
    """Check R/T and all outgoing diffraction channels at increasing radii N.

    Efficiency differences are absolute power fractions. Phase differences
    are wrapped to [-180,180] degrees and only checked when both efficiencies
    meet ``min_efficiency`` and their amplitudes are nonzero. New channels are
    compared to zero power; a newly reliable phase requires another step.
    This prevents new propagation channels from silently escaping the test.

    Every propagating reflected complex amplitude is independently checked
    by abs(current-previous)/abs(current), without the power/phase threshold.
    When both amplitudes are at or below ``near_zero_amplitude``, the channel
    is listed as numerical near zero and receives no relative assessment.
    Crossing that floor, or a missing/new reflection channel, blocks the step.

    One cached solver is reused. ``progress`` receives one history-row dict
    after each solve. Cancellation is cooperative between solves and raises
    the standard ``InterruptedError``; a running eigensolve is not interrupted.
    """
    _check_cancel(cancel_event)
    try:
        truncations = tuple(orders)
    except TypeError as exc:
        raise ValueError('orders must be a strictly increasing integer sequence within [0,255]') from exc
    if (not truncations or
        any(isinstance(n, bool) or not isinstance(n, Integral) or not 0 <= n <= 255
            for n in truncations) or
        any(b <= a for a, b in zip(truncations, truncations[1:]))):
        raise ValueError('orders must be a strictly increasing integer sequence within [0,255]')
    truncations = tuple(int(n) for n in truncations)
    for name, value, allow_zero in (
        ('efficiency_tolerance', efficiency_tolerance, False),
        ('phase_tolerance_deg', phase_tolerance_deg, False),
        ('min_efficiency', min_efficiency, True),
        ('reflection_complex_relative_tolerance', reflection_complex_relative_tolerance, False),
        ('near_zero_amplitude', near_zero_amplitude, True),
    ):
        if (isinstance(value, bool) or not isinstance(value, Real) or
            not np.isfinite(value) or (value < 0 if allow_zero else value <= 0)):
            raise ValueError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}')
    if progress is not None and not callable(progress):
        raise ValueError('progress must be callable')

    solver = MeentSolver()
    previous_result = previous_channels = None
    history = []
    stable_count = 0
    for n in truncations:
        _check_cancel(cancel_event)
        current_simulation = replace(simulation, fourier_order=n)
        result = solver.solve(stack, materials, current_simulation)
        _check_cancel(cancel_event)
        channels = _channels(result, stack, materials, current_simulation)
        basis_covered = _lossless_basis_covered(stack, materials, current_simulation, n)
        efficiency_delta = phase_delta = None
        new_channels = []
        phases_checked = 0
        unmatched_reliable_phase = False
        reflection_relative_delta = None
        reflection_relative_checked = reflection_relative_exceeded_count = 0
        reflection_near_zero_channels = []
        reflection_new_or_missing_channels = []
        reflection_zero_transition_channels = []
        if previous_result is not None:
            efficiency_delta = max(abs(result.reflectance-previous_result.reflectance),
                                   abs(result.transmittance-previous_result.transmittance))
            phase_delta = 0.
            # Use the union, not just the common retained basis: a newly
            # admitted diffraction order must contribute to the decision.
            for key in sorted(set(channels) | set(previous_channels)):
                current_eta, current_a = channels.get(key, (0., 0j))
                previous_eta, previous_a = previous_channels.get(key, (0., 0j))
                efficiency_delta = max(efficiency_delta, abs(current_eta-previous_eta))
                if key[0] == 'R':
                    label = f'{key[0]}:{key[1]}'
                    if key not in channels or key not in previous_channels:
                        reflection_new_or_missing_channels.append(label)
                    elif max(abs(current_a), abs(previous_a)) <= near_zero_amplitude:
                        reflection_near_zero_channels.append(label)
                    elif min(abs(current_a), abs(previous_a)) <= near_zero_amplitude:
                        reflection_zero_transition_channels.append(label)
                    else:
                        relative_change = float(abs(current_a-previous_a)/abs(current_a))
                        reflection_relative_delta = max(reflection_relative_delta or 0., relative_change)
                        reflection_relative_checked += 1
                        reflection_relative_exceeded_count += relative_change > reflection_complex_relative_tolerance
                current_valid = current_eta > 0 and current_eta >= min_efficiency and abs(current_a) > 0
                previous_valid = previous_eta > 0 and previous_eta >= min_efficiency and abs(previous_a) > 0
                if key not in previous_channels:
                    new_channels.append(f'{key[0]}:{key[1]}')
                if current_valid and previous_valid:
                    change = abs(float(np.degrees(np.angle(current_a*np.conj(previous_a)))))
                    phase_delta = max(phase_delta, change)
                    phases_checked += 1
                elif current_valid or previous_valid:
                    unmatched_reliable_phase = True
        step_passed = bool(previous_result is not None and basis_covered and
            efficiency_delta <= efficiency_tolerance and phase_delta <= phase_tolerance_deg and
            not unmatched_reliable_phase and not reflection_relative_exceeded_count and
            not reflection_new_or_missing_channels and not reflection_zero_transition_channels)
        reflection_relative_status = (
            'not_compared' if previous_result is None else
            'new_or_missing_channels' if reflection_new_or_missing_channels else
            'near_zero_transition' if reflection_zero_transition_channels else
            'relative_change_exceeded' if reflection_relative_exceeded_count else
            'within_tolerance' if reflection_relative_checked else
            'near_zero_only_not_assessed' if reflection_near_zero_channels else
            'no_reflected_propagating_channels')
        stable_count = stable_count+1 if step_passed else 0
        row = {'N': n, 'R': result.reflectance, 'T': result.transmittance,
               'absorption_flux_residual': result.absorption,
               'efficiency_delta': efficiency_delta, 'phase_delta_deg': phase_delta,
               'step_passed': step_passed, 'successive_passing_steps': stable_count,
               'lossless_propagating_basis_covered': basis_covered,
               'reflected_channels': sum(key[0] == 'R' for key in channels),
               'transmitted_channels': sum(key[0] == 'T' for key in channels),
               'new_channels': new_channels, 'phases_checked': phases_checked,
               'unmatched_reliable_phase': unmatched_reliable_phase,
               'reflection_relative_delta': reflection_relative_delta,
               'reflection_relative_checked': reflection_relative_checked,
               'reflection_relative_exceeded_count': reflection_relative_exceeded_count,
               'reflection_near_zero_count': len(reflection_near_zero_channels),
               'reflection_near_zero_channels': reflection_near_zero_channels,
               'reflection_new_or_missing_channels': reflection_new_or_missing_channels,
               'reflection_zero_transition_channels': reflection_zero_transition_channels,
               'reflection_relative_status': reflection_relative_status}
        history.append(row)
        previous_result, previous_channels = result, channels
        if progress is not None:
            progress({key: list(value) if isinstance(value, list) else value for key, value in row.items()})
        _check_cancel(cancel_event)
        if stable_count >= 2:
            break

    converged = stable_count >= 2
    return {'converged': converged, 'status': 'PASS' if converged else 'UNCONVERGED',
            'final_N': history[-1]['N'], 'requested_orders': list(truncations),
            'executed_orders': [row['N'] for row in history],
            'tolerances': {'efficiency': float(efficiency_tolerance),
                           'phase_deg': float(phase_tolerance_deg),
                           'min_efficiency': float(min_efficiency),
                           'reflection_complex_relative': float(reflection_complex_relative_tolerance),
                           'near_zero_amplitude': float(near_zero_amplitude)},
            'reflection_relative_definition': 'abs(current-previous)/abs(current) for every propagating reflected amplitude above the independent amplitude floor in both resolutions; no power threshold',
            'near_zero_policy': 'both amplitudes at or below the floor are separately listed and not relatively assessed; a floor crossing or new/missing reflected channel blocks the step',
            'rule': 'two successive adjacent changes pass for total R/T, every outgoing channel efficiency, reliable phases, and reflected complex relative amplitudes; numerical near-zero channels are listed without relative assessment; numerical evidence only, not a true error bound; geometric slicing is not tested',
            'history': history}
