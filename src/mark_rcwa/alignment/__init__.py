"""Pair phase position deviation with explicit reliability and branch limits."""
from dataclasses import dataclass
import numpy as np
from mark_rcwa.detector import positive_pair_order, validate_orders


@dataclass(frozen=True)
class APDResult:
    order: int
    reliable: bool
    delta_phase_rad: float | None
    apd_nm: float | None
    recovered_geometry_shift_nm: float | None
    ambiguity_period_nm: float
    branch_limited: bool = True
    reason: str | None = None


@dataclass
class APDSeries:
    order: int
    delta_phase_rad: np.ndarray
    apd_nm: np.ndarray
    recovered_geometry_shift_nm: np.ndarray
    reliable: np.ndarray
    ambiguity_period_nm: float
    branch_limited: bool
    unwrapped: bool
    metadata: dict


def _compatible(reference, test):
    for name in ('pitch_um', 'wavelength_um', 'theta_deg', 'reference_plane_um', 'n_top'):
        a, b = getattr(reference, name), getattr(test, name)
        if not np.isfinite(a) or not np.isfinite(b) or not np.isclose(a, b, atol=1e-12, rtol=1e-12):
            raise ValueError(f'APD comparison requires matching {name}')
    if reference.polarization != test.polarization:
        raise ValueError('APD comparison requires matching polarization')


def _pair_unit_phase(result, order, min_efficiency):
    _, indices = validate_orders(result, (order, -order))
    amplitudes = result.amplitude[indices]
    if np.any(result.efficiency[indices] <= min_efficiency) or np.any(abs(amplitudes) == 0):
        return None
    # Normalize separately to avoid overflow/underflow in the amplitude product.
    return (amplitudes[0]/abs(amplitudes[0]))*(amplitudes[1]/abs(amplitudes[1])).conjugate()


def pair_apd(reference, test, order=1, min_efficiency=1e-8):
    """Positive APD shifts the ideal pair maximum to positive stage x.

    A positive geometric displacement has negative APD. Principal phases are
    defined only modulo pitch/(2*m), even when a sample is marked reliable.
    """
    order = positive_pair_order(order)
    if not np.isfinite(min_efficiency) or min_efficiency < 0:
        raise ValueError('min_efficiency must be finite and nonnegative')
    _compatible(reference, test)
    ambiguity_period_nm = reference.pitch_um*1000/(2*order)
    reference_phase = _pair_unit_phase(reference, order, min_efficiency)
    test_phase = _pair_unit_phase(test, order, min_efficiency)
    if reference_phase is None or test_phase is None:
        return APDResult(order, False, None, None, None, ambiguity_period_nm,
            reason='one or more reference/test pair efficiencies are at or below the phase reliability floor')
    delta = float(np.angle(test_phase/reference_phase))
    apd = float(reference.pitch_um*1000*delta/(4*np.pi*order))
    return APDResult(order, True, delta, apd, -apd, ambiguity_period_nm)


def apd_series(reference, tests, order=1, min_efficiency=1e-8, unwrap=True):
    """Unwrap contiguous reliable samples only; restart each segment's branch.

    Successive true phase changes must be less than pi for valid tracking.
    No unwrap can determine absolute cycle count from an initial sample.
    """
    tests = list(tests)
    if not tests:
        raise ValueError('APD series requires at least one test result')
    if not isinstance(unwrap, (bool, np.bool_)):
        raise ValueError('unwrap must be a boolean')
    values = [pair_apd(reference, test, order, min_efficiency) for test in tests]
    phase = np.array([np.nan if value.delta_phase_rad is None else value.delta_phase_rad for value in values])
    reliable = np.array([value.reliable for value in values], dtype=bool)
    segments = []
    start = 0
    while start < len(phase):
        if not reliable[start]:
            start += 1
            continue
        end = start+1
        while end < len(phase) and reliable[end]:
            end += 1
        if unwrap:
            phase[start:end] = np.unwrap(phase[start:end])
        segments.append({'start_index': start, 'end_index_exclusive': end,
                         'absolute_cycle_count_known': False})
        start = end
    order = values[0].order
    apd = reference.pitch_um*1000*phase/(4*np.pi*order)
    return APDSeries(order, phase, apd, -apd, reliable, values[0].ambiguity_period_nm,
        True, bool(unwrap), {'model': 'idealized_pair_phase_position_deviation',
        'segments': segments, 'gap_policy': 'restart principal branch after every unreliable sample',
        'unwrap_assumption': 'successive true pair phase changes below pi; no absolute cycle count',
        'reliability_floor': float(min_efficiency),
        'reasons': [value.reason for value in values]})
