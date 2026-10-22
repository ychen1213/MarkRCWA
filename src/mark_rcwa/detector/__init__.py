"""Explicit idealized detector modes with no implicit overlap between channels."""
from dataclasses import dataclass
from collections.abc import Mapping
from numbers import Integral
import math
import numpy as np


def validate_orders(result, orders):
    """Return indices after strict order/basis/propagation/pupil validation."""
    try:
        orders = tuple(orders)
    except TypeError as exc:
        raise ValueError('detector orders must be a nonempty integer sequence') from exc
    if not orders or any(isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral) for m in orders):
        raise ValueError('detector orders must be a nonempty integer sequence')
    orders = tuple(int(m) for m in orders)
    if len(set(orders)) != len(orders):
        raise ValueError('detector orders must be unique within each coherent channel')
    indices = []
    for m in orders:
        locations = np.flatnonzero(result.orders == m)
        if len(locations) != 1:
            raise ValueError(f'order {m} is outside the retained Fourier basis')
        i = int(locations[0])
        if not result.propagating[i]:
            raise ValueError(f'order {m} is evanescent and cannot be a far-field detector channel')
        if not result.collected[i]:
            raise ValueError(f'order {m} is outside the NA collection pupil')
        if not np.isfinite(result.amplitude[i]) or not np.isfinite(result.efficiency[i]):
            raise ValueError('detector inputs must contain finite amplitudes and efficiencies')
        indices.append(i)
    return orders, np.array(indices, dtype=int)


def positive_pair_order(order):
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, Integral) or order <= 0:
        raise ValueError('pair order must be a positive integer')
    return int(order)


def _weights(orders, weights):
    if weights is None:
        return np.ones(len(orders), dtype=complex)
    if isinstance(weights, Mapping):
        keys = tuple(weights)
        if any(isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral) for m in keys) or set(keys) != set(orders):
            raise ValueError('weight mapping must contain exactly the selected integer orders')
        weights = [weights[m] for m in orders]
    try:
        coefficients = np.asarray(weights, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError('optical transfer weights must be complex scalars') from exc
    if coefficients.shape != (len(orders),) or not np.isfinite(coefficients).all():
        raise ValueError('optical transfer weights must be finite and match the order sequence')
    return coefficients


@dataclass(frozen=True)
class DetectorChannel:
    orders: tuple[int, ...]
    weights: object = None


@dataclass
class ScanResult:
    stage_um: np.ndarray
    intensity: np.ndarray
    channel_intensity: np.ndarray
    metadata: dict
    period_um: float | None = None
    i0: float | None = None
    i1: float | None = None
    phase_rad: float | None = None


def _channel(result, spec, stage_um):
    if isinstance(spec, Mapping):
        if set(spec)-{'orders', 'weights'} or 'orders' not in spec:
            raise ValueError('a detector channel requires orders and optional weights')
        spec = DetectorChannel(**spec)
    if not isinstance(spec, DetectorChannel):
        raise ValueError('channels must contain DetectorChannel instances or channel mappings')
    orders, indices = validate_orders(result, spec.orders)
    coefficients = _weights(orders, spec.weights)*result.amplitude[indices]
    g = 2*np.pi/result.pitch_um
    field = coefficients@np.exp(-1j*np.asarray(orders)[:, None]*g*stage_um[None, :])
    metadata = {'orders': list(orders), 'weights': [
        {'real': float(h.real), 'imag': float(h.imag)} for h in _weights(orders, spec.weights)],
        'assumed_nonzero_mode_overlap': len(orders) > 1,
        'transfer_source': 'user-specified' if spec.weights is not None else 'ideal unit transfer'}
    return abs(field)**2, coefficients, orders, metadata


def _period(pitch_um, orders):
    if len(orders) <= 1:
        return None
    differences = [abs(m-orders[0]) for m in orders[1:]]
    return pitch_um/math.gcd(*differences)


def scan(result, stage_um, mode='pair', order=1, orders=None, weights=None, channels=None):
    """Calculate one explicit detector topology from complex reflected fields.

    Pair phase is defined by I=I0+I1*cos(4*pi*m*x/d+phase_rad).
    Independent channel intensities are added with no cross-channel terms.
    """
    try:
        stage_um = np.asarray(stage_um, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('stage_um must be a finite one-dimensional coordinate sequence') from exc
    if stage_um.ndim != 1 or not len(stage_um) or not np.isfinite(stage_um).all():
        raise ValueError('stage_um must be a finite one-dimensional coordinate sequence')
    if mode not in ('single', 'pair', 'coherent', 'independent'):
        raise ValueError('detector mode must be single, pair, coherent or independent')
    if mode != 'independent' and channels is not None:
        raise ValueError('channels are only used by independent detector mode')
    if mode in ('single', 'pair') and orders is not None:
        raise ValueError('single/pair modes use order; coherent mode uses orders')
    if mode == 'independent' and (orders is not None or weights is not None):
        raise ValueError('independent mode uses each channel own orders and weights')
    metadata = {'mode': mode, 'model': 'idealized_interference_model',
        'calibrated_equipment_prediction': False, 'stage_convention': 'positive moves material to +x',
        'inter_channel_mode_overlap': False, 'channels': []}
    period = i0 = i1 = phase = None
    try:
        with np.errstate(over='raise', invalid='raise'):
            if mode == 'single':
                selected, indices = validate_orders(result, [order])
                h = _weights(selected, weights)
                power = float(result.efficiency[indices[0]]*abs(h[0])**2)
                channel_values = np.full((1, len(stage_um)), power)
                i0, i1 = power, 0.
                metadata.update(intensity_normalization='incident-normalized single-order power',
                                assumed_nonzero_mode_overlap=False)
                metadata['channels'] = [{'orders': list(selected), 'transfer_source':
                    'user-specified' if weights is not None else 'ideal unit transfer'}]
            elif mode == 'independent':
                if channels is None:
                    raise ValueError('independent mode requires explicit detector channels')
                channels = list(channels)
                if not channels:
                    raise ValueError('independent mode requires at least one detector channel')
                values = []
                for channel in channels:
                    intensity, _, _, info = _channel(result, channel, stage_um)
                    values.append(intensity)
                    metadata['channels'].append(info)
                channel_values = np.vstack(values)
                metadata.update(intensity_normalization='sum of separate detector-mode electric intensity',
                                assumed_nonzero_mode_overlap=any(c['assumed_nonzero_mode_overlap'] for c in metadata['channels']))
            else:
                if mode == 'pair':
                    m = positive_pair_order(order)
                    orders = (m, -m)
                elif orders is None:
                    raise ValueError('coherent mode requires explicit orders')
                intensity, coefficients, selected, info = _channel(result, DetectorChannel(orders, weights), stage_um)
                channel_values = intensity[None, :]
                metadata.update(intensity_normalization='electric intensity in one ideal detector mode',
                                assumed_nonzero_mode_overlap=len(selected) > 1)
                metadata['channels'] = [info]
                i0 = float(np.sum(abs(coefficients)**2))
                period = _period(result.pitch_um, selected)
                if mode == 'pair':
                    q = coefficients[0]*coefficients[1].conjugate()
                    i1 = float(2*abs(q))
                    phase = float(-np.angle(q)) if abs(q) > 0 else None
            intensity = channel_values.sum(axis=0)
            if not np.isfinite(intensity).all():
                raise ValueError('detector calculation produced non-finite intensity')
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError('detector calculation overflow; use finite reasonable transfer weights and stage coordinates') from exc
    return ScanResult(stage_um.copy(), intensity, channel_values, metadata, period, i0, i1, phase)
