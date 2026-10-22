"""Thin adapter around pinned Meent CFS/eigensolvers and stable TMM recursion.

Orchestration is adapted from MIT-licensed Meent (see THIRD_PARTY_NOTICES.md).
Fixes only 1D TM lossy-substrate admittance, and converts native conventions.
"""
from dataclasses import dataclass, replace
from collections import OrderedDict
from numbers import Integral
import sys
import threading
import numpy as np
from threadpoolctl import threadpool_limits
from meent import call_mee
from meent.on_numpy.emsolver.convolution_matrix import to_conv_mat_vector
from meent.on_numpy.emsolver.transfer_method import (
    transfer_1d_1, transfer_1d_2, transfer_1d_3, transfer_1d_4)
from mark_rcwa.model import Stack, positive
from mark_rcwa.materials import Materials

UPSTREAM_SHA = 'bae28024476d4af2826b8412cb8288d79a0630ac'


class _CutoffLimitRequired(Exception):
    """An exact zero longitudinal mode made the modal transfer basis singular."""


def _cutoff_mask(index, normalized_kx):
    """Identify a zero longitudinal wave number up to arithmetic roundoff."""
    squared = index**2 - normalized_kx**2
    scale = np.maximum(1., np.maximum(abs(index)**2, abs(normalized_kx)**2))
    return abs(squared) <= 32*np.finfo(float).eps*scale


def _cache_size(value):
    """Conservative charged size, including keys and Python container overhead."""
    if isinstance(value, np.ndarray):
        # Charge data explicitly even for views; owning arrays may be overcharged.
        return sys.getsizeof(value)+value.nbytes
    if isinstance(value, tuple):
        return sys.getsizeof(value)+sum(_cache_size(item) for item in value)
    return sys.getsizeof(value)


def _freeze(value):
    if isinstance(value, np.ndarray):
        owned = np.array(value, copy=True, order='C')
        owned.setflags(write=False)
        return owned
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


class _BoundedCache:
    """One byte-bounded LRU shared by material, Fourier and eigen entries."""
    def __init__(self, enabled, max_bytes):
        self.enabled, self.max_bytes = enabled, max_bytes
        self.entries = OrderedDict()
        self.bytes_used = 0
        self.reset_statistics()

    def reset_statistics(self):
        self.hits = self.misses = self.evictions = self.oversized = 0
        self.by_kind = {kind: {'hits': 0, 'misses': 0} for kind in ('material', 'convolution', 'eigen')}

    def get(self, kind, key):
        compound = (kind, key)
        if self.enabled and compound in self.entries:
            self.hits += 1
            self.by_kind[kind]['hits'] += 1
            self.entries.move_to_end(compound)
            return self.entries[compound][0]
        self.misses += 1
        self.by_kind[kind]['misses'] += 1
        return None

    def put(self, kind, key, value):
        if not self.enabled or self.max_bytes == 0:
            return value
        compound = (kind, key)
        # 1024 bytes extra per entry conservatively charges LRU bookkeeping.
        size = _cache_size(compound)+_cache_size(value)+1024
        if size > self.max_bytes:
            self.oversized += 1
            return value
        stored = _freeze(value)
        # Recompute after making owning arrays, so the resident charge is bounded.
        size = _cache_size(compound)+_cache_size(stored)+1024
        if size > self.max_bytes:
            self.oversized += 1
            return value
        if compound in self.entries:
            _, previous_size = self.entries.pop(compound)
            self.bytes_used -= previous_size
        while self.entries and self.bytes_used+size > self.max_bytes:
            _, (_, previous_size) = self.entries.popitem(last=False)
            self.bytes_used -= previous_size
            self.evictions += 1
        self.entries[compound] = (stored, size)
        self.bytes_used += size
        return stored

    def clear(self):
        self.entries.clear()
        self.bytes_used = 0
        self.reset_statistics()

    def info(self):
        return {'enabled': self.enabled, 'max_bytes': self.max_bytes,
                'bytes_used': self.bytes_used, 'entries': len(self.entries),
                'hits': self.hits, 'misses': self.misses, 'evictions': self.evictions,
                'oversized_entries_skipped': self.oversized,
                'by_kind': {kind: dict(values) for kind, values in self.by_kind.items()}}


def outgoing_sqrt(value):
    root = np.sqrt(np.asarray(value, dtype=np.complex128))
    return np.where((root.imag < 0) | ((root.imag == 0) & (root.real < 0)), -root, root)


@dataclass(frozen=True)
class Simulation:
    wavelength_um: float = .633
    theta_deg: float = 0.
    polarization: str = 'TE'
    fourier_order: int = 21
    na: float = .8

    def __post_init__(self):
        positive(self.wavelength_um, 'wavelength_um')
        if not np.isfinite(self.theta_deg) or abs(self.theta_deg) >= 89.9:
            raise ValueError('theta_deg must be finite in (-89.9, 89.9)')
        if self.polarization not in ('TE', 'TM'):
            raise ValueError('polarization must be TE or TM; conical incidence unsupported')
        if isinstance(self.fourier_order, bool) or not isinstance(self.fourier_order, int) or not 0 <= self.fourier_order <= 255:
            raise ValueError('fourier_order must be an integer between 0 and 255')
        positive(self.na, 'na', allow_zero=True)


@dataclass
class DiffractionResult:
    orders: np.ndarray
    amplitude: np.ndarray  # reflected electric field, fixed laboratory basis
    transmission_amplitude: np.ndarray
    efficiency: np.ndarray
    transmission_efficiency: np.ndarray
    kx: np.ndarray
    kz: np.ndarray  # positive outgoing magnitude, upward distance L
    propagating: np.ndarray
    collected: np.ndarray
    pitch_um: float
    wavelength_um: float
    theta_deg: float
    polarization: str
    n_top: float
    reference_plane_um: float = 0.
    branch_corrections: int = 0
    cutoff_metadata: dict | None = None

    @property
    def reflectance(self):
        return float(self.efficiency.sum())

    @property
    def transmittance(self):
        return float(self.transmission_efficiency.sum())

    @property
    def absorption(self):
        return 1 - self.reflectance - self.transmittance

    def translated(self, displacement_um):
        if not np.isfinite(displacement_um):
            raise ValueError('displacement_um must be finite')
        phase = np.exp(-2j*np.pi*self.orders*displacement_um/self.pitch_um)
        return replace(self, amplitude=self.amplitude*phase,
                       transmission_amplitude=self.transmission_amplitude*phase)

    def rows(self, min_efficiency=1e-10):
        rows = []
        for i, m in enumerate(self.orders):
            a = self.amplitude[i]
            t = self.transmission_amplitude[i]
            reliable = bool(self.propagating[i] and self.efficiency[i] >= min_efficiency)
            # Transmission can carry power when this order is evanescent above
            # the grating (high-index or absorbing substrate). Its efficiency
            # already includes the substrate admittance and outgoing flux.
            transmitted_flux_positive = bool(self.transmission_efficiency[i] > 0)
            transmitted_reliable = bool(transmitted_flux_positive and
                self.transmission_efficiency[i] >= min_efficiency and abs(t) > 0)
            angle = np.degrees(np.arcsin(np.clip(self.kx[i].real/(self.n_top*2*np.pi/self.wavelength_um), -1, 1)))
            rows.append(dict(order=int(m), kx_per_um=float(self.kx[i].real),
                kz_real_per_um=float(self.kz[i].real), kz_imag_per_um=float(self.kz[i].imag),
                propagating=bool(self.propagating[i]), collected=bool(self.collected[i]),
                reflected_efficiency=float(self.efficiency[i]), transmitted_efficiency=float(self.transmission_efficiency[i]),
                amplitude_real=float(a.real), amplitude_imag=float(a.imag), amplitude_abs=float(abs(a)),
                phase_rad=float(np.angle(a)) if reliable else None,
                phase_deg=float(np.degrees(np.angle(a))) if reliable else None,
                raw_phase_rad=float(np.angle(a)) if abs(a)>0 else None,
                raw_phase_deg=float(np.degrees(np.angle(a))) if abs(a)>0 else None,
                phase_reliable=reliable, angle_deg=float(angle) if self.propagating[i] else None,
                transmitted_amplitude_real=float(t.real), transmitted_amplitude_imag=float(t.imag),
                transmitted_amplitude_abs=float(abs(t)),
                transmitted_phase_rad=float(np.angle(t)) if transmitted_reliable else None,
                transmitted_phase_deg=float(np.degrees(np.angle(t))) if transmitted_reliable else None,
                transmitted_raw_phase_rad=float(np.angle(t)) if abs(t)>0 else None,
                transmitted_raw_phase_deg=float(np.degrees(np.angle(t))) if abs(t)>0 else None,
                transmitted_phase_reliable=transmitted_reliable,
                transmitted_flux_positive=transmitted_flux_positive,
                polarization=self.polarization))
        return rows


class MeentSolver:
    def __init__(self, cache_enabled=True, cache_bytes=64*1024**2):
        if not isinstance(cache_enabled, bool):
            raise ValueError('cache_enabled must be boolean')
        if isinstance(cache_bytes, bool) or not isinstance(cache_bytes, Integral) or cache_bytes < 0:
            raise ValueError('cache_bytes must be a nonnegative integer byte budget')
        self._cache = _BoundedCache(cache_enabled, int(cache_bytes))
        self._lock = threading.RLock()

    def cache_info(self):
        with self._lock:
            return self._cache.info()

    def clear_cache(self):
        with self._lock:
            self._cache.clear()

    def _material_index(self, materials, name, wavelength_um):
        if name not in materials:
            raise ValueError(f'unknown material identifier: {name}')
        material = materials[name]
        key = (name, material.n, tuple(material.wavelength_um), tuple(material.n_table), wavelength_um)
        value = self._cache.get('material', key)
        if value is None:
            value = materials.index(name, wavelength_um)
            value = self._cache.put('material', key, value)
        return value

    def solve(self, stack: Stack, materials: Materials, sim: Simulation):
        with self._lock, threadpool_limits(limits=1), np.errstate(over='raise', invalid='raise', divide='raise'):
            try:
                try:
                    return self._solve(stack, materials, sim)
                except _CutoffLimitRequired as exc:
                    return self._solve_cutoff_limit(stack, materials, sim, str(exc))
            except (FloatingPointError, np.linalg.LinAlgError, OverflowError, MemoryError) as exc:
                raise ValueError(f'RCWA numerical failure: {exc}; refine truncation/geometry') from exc

    def _solve_cutoff_limit(self, stack, materials, sim, reason):
        """Evaluate a singular modal representation by its two-sided limit.

        Around a Rayleigh threshold the outgoing square root introduces a
        leading sqrt(delta) term.  Two symmetric wavelength pairs at delta
        and delta/4 remove that term with 2*average(delta/4)-average(delta).
        The requested wavelength and its exact wavevectors/flux are retained;
        all displaced solves and their spread are disclosed in metadata.
        This is a numerical limit, not a change to the requested experiment.
        """
        nt = self._material_index(materials, stack.top, sim.wavelength_um).real
        nb = self._material_index(materials, stack.bottom, sim.wavelength_um)
        orders = np.arange(-sim.fourier_order, sim.fourier_order+1)
        k0 = 2*np.pi/sim.wavelength_um
        normalized_kx = nt*np.sin(np.radians(sim.theta_deg)) + orders*sim.wavelength_um/stack.pitch_um
        cutoff_top = _cutoff_mask(nt, normalized_kx)
        cutoff_bottom = _cutoff_mask(nb, normalized_kx)
        kz = outgoing_sqrt(nt**2-normalized_kx**2)*k0
        kzbot = outgoing_sqrt(nb**2-normalized_kx**2)*k0
        kz[cutoff_top] = 0
        kzbot[cutoff_bottom] = 0
        kzinc = nt*k0*np.cos(np.radians(sim.theta_deg))
        prop = (abs(kz.imag) < 1e-10*k0) & (kz.real > 1e-10*k0)
        samples = []
        # The second attempt is only needed when extrapolation fails the same
        # passive-flux bound imposed on ordinary direct solves.
        for delta in (1e-8, 1e-10):
            pairs = []
            for relative in (delta, delta/4):
                pair = []
                for sign in (-1, 1):
                    shifted = replace(sim, wavelength_um=sim.wavelength_um*(1+sign*relative))
                    try:
                        value = self._solve(stack, materials, shifted)
                    except _CutoffLimitRequired as exc:
                        raise ValueError('Rayleigh limit samples remain singular; no cutoff result was returned') from exc
                    pair.append(value)
                    samples.append({'wavelength_um': shifted.wavelength_um,
                                    'relative_offset': sign*relative,
                                    'branch_corrections': value.branch_corrections})
                pairs.append(pair)
            averages = [(sum(v.amplitude for v in pair)/2,
                         sum(v.transmission_amplitude for v in pair)/2) for pair in pairs]
            r, te = (2*averages[1][i]-averages[0][i] for i in (0, 1))
            efficiency = abs(r)**2*np.maximum(kz.real, 0)/kzinc
            efficiency[~prop] = 0
            th = te if sim.polarization == 'TE' else te*nb/nt
            flux = kzbot.real/kzinc if sim.polarization == 'TE' else np.real(kzbot/(nb**2))/(kzinc/nt**2)
            transmission = abs(th)**2*flux
            finite = all(np.isfinite(v).all() for v in (r, te, efficiency, transmission))
            passive = finite and np.min(transmission) >= -1e-10 and efficiency.sum()+transmission.sum() <= 1+1e-7
            if passive:
                metadata = {'method': 'two_sided_wavelength_limit', 'reason': reason,
                            'requested_wavelength_um': sim.wavelength_um,
                            'reflection_orders': orders[cutoff_top].tolist(),
                            'transmission_orders': orders[cutoff_bottom].tolist(),
                            'samples': samples, 'final_relative_delta': delta,
                            'extrapolation': '2*average(delta/4)-average(delta); leading square-root term removed',
                            'max_reflection_extrapolation_correction': float(np.max(abs(averages[1][0]-averages[0][0]))),
                            'max_transmission_extrapolation_correction': float(np.max(abs(averages[1][1]-averages[0][1]))),
                            'max_reflection_two_sided_spread': float(np.max(abs(pairs[1][0].amplitude-pairs[1][1].amplitude))),
                            'max_transmission_two_sided_spread': float(np.max(abs(pairs[1][0].transmission_amplitude-pairs[1][1].transmission_amplitude))),
                            'accuracy_note': 'Sample spread and extrapolation correction are diagnostics, not rigorous error bounds or Fourier convergence.'}
                return replace(pairs[1][0], amplitude=r, transmission_amplitude=te,
                               efficiency=efficiency, transmission_efficiency=transmission,
                               kx=(normalized_kx*k0).astype(complex), kz=kz, propagating=prop,
                               collected=prop & (abs(normalized_kx) <= sim.na+1e-12),
                               wavelength_um=sim.wavelength_um, n_top=nt,
                               branch_corrections=sum(item['branch_corrections'] for item in samples),
                               cutoff_metadata=metadata)
        raise ValueError('Rayleigh limit did not satisfy passive flux; refine truncation or inspect the threshold separately')

    def _solve(self, stack, materials, sim):
        nt = self._material_index(materials, stack.top, sim.wavelength_um)
        nb = self._material_index(materials, stack.bottom, sim.wavelength_um)
        if nt.imag != 0:
            raise ValueError('incident/collection half-space must be lossless')
        nt = nt.real
        if sim.na > nt:
            raise ValueError('NA cannot exceed incident medium refractive index')
        pol = int(sim.polarization == 'TM')
        n = sim.fourier_order
        matrix_bytes = 3*len(stack.layers)*(2*n+1)**2*np.dtype(np.complex128).itemsize
        if matrix_bytes > 512*1024**2:
            raise ValueError('Fourier matrices exceed the 512 MiB allocation guard; reduce slices/order or partition the study')
        orders = np.arange(-n, n+1)
        k0 = 2*np.pi/sim.wavelength_um
        kx = nt*k0*np.sin(np.radians(sim.theta_deg)) + orders*2*np.pi/stack.pitch_um
        kz = outgoing_sqrt((nt*k0)**2-kx**2)
        kzbot = outgoing_sqrt((nb*k0)**2-kx**2)
        cutoff_top = _cutoff_mask(nt, kx/k0)
        cutoff_bottom = _cutoff_mask(nb, kx/k0)
        kz[cutoff_top] = 0
        kzbot[cutoff_bottom] = 0
        has_cutoff = bool(np.any(cutoff_top) or np.any(cutoff_bottom))
        matrices, layer_keys, lossless = [], [], []
        for s in stack.layers:
            actual_indices = tuple(self._material_index(materials, m, sim.wavelength_um) for m in s.materials)
            # Physical values invalidate reuse after material-registry edits or dispersion changes.
            key = (stack.pitch_um, tuple(s.edges_um), actual_indices, n, 'complex128')
            layer_keys.append(key)
            lossless.append(all(value.imag == 0 for value in actual_indices))
            entry = self._cache.get('convolution', key)
            if entry is None:
                indices = np.asarray(actual_indices, dtype=np.complex128).conjugate()
                data = [[indices.reshape(1, -1), np.array(s.edges_um[1:]).reshape(-1, 1), np.array([[stack.pitch_um]])]]
                # Native layer n-ik; CFS retains exact edges without an x raster.
                ex, ey, ezi = to_conv_mat_vector(data, n, 0)
                entry = self._cache.put('convolution', key, (ex[0], ey[0], ezi[0]))
            matrices.append(entry)
        native = call_mee(backend=0, n_top=nt, n_bot=nb, pol=pol,
            theta=np.radians(sim.theta_deg), phi=None, fto=[n, 0],
            wavelength=sim.wavelength_um, period=[stack.pitch_um, stack.pitch_um],
            ucell=np.ones((max(1, len(stack.layers)), 1, 1)),
            thickness=[s.thickness_um for s in stack.layers], type_complex=0)
        kxn, _ = native.get_kx_ky_vector(sim.wavelength_um)
        kzt, kzb, f, g, t = transfer_1d_1(pol, kxn, nt, nb)
        # A grazing outgoing channel carries exactly zero longitudinal flux.
        # It does not by itself make the complete boundary system singular.
        kzt[cutoff_top] = 0
        kzb[cutoff_bottom] = 0
        if pol == 0:
            g = 1j*np.diag(kzb)
        if pol == 1:
            # #100: native kz is conjugated, therefore epsilon must be too.
            g = 1j*np.diag(kzb/(nb.conjugate()**2))
        branch_corrections = 0
        for j in range(len(stack.layers)-1, -1, -1):
            key = (layer_keys[j], pol, kxn.dtype.str, kxn.tobytes())
            entry = self._cache.get('eigen', key)
            if entry is None:
                ex, ey, ezi = matrices[j]
                w, v, q = transfer_1d_2(pol, kxn, ex, ey, ezi)
                corrections = 0
                if lossless[j]:
                    # Native +iwt convention needs q=+i|kz|. Correct only roundoff
                    # branch errors in nearly imaginary modes of exactly lossless layers.
                    bad = (q.imag < 0) & (abs(q.real) <= 1e-12*np.maximum(1., abs(q)))
                    if np.any(bad):
                        previous = q.copy()
                        q = q.copy()
                        q[bad] = 1j*abs(q[bad].imag)
                        ratio = np.ones_like(q)
                        ratio[bad] = q[bad]/previous[bad]
                        v = v*ratio[None, :]
                        corrections = int(bad.sum())
                entry = self._cache.put('eigen', key, (w, v, q, corrections))
            w, v, q, corrections = entry
            branch_corrections += corrections
            modal_zero_tolerance = 32*np.finfo(float).eps*max(1., float(np.max(abs(kxn)**2)), float(np.max(abs(q)**2)))
            if np.any(abs(q)**2 <= modal_zero_tolerance):
                raise _CutoffLimitRequired(f'zero longitudinal layer mode in layer {j}; modal V inverse singular')
            try:
                _, f, g, t, _, _ = transfer_1d_3(k0, w, v, q, stack.layers[j].thickness_um, f, g, t)
            except np.linalg.LinAlgError as exc:
                if has_cutoff:
                    raise _CutoffLimitRequired(f'modal transfer singular at external Rayleigh cutoff in layer {j}') from exc
                raise
        try:
            out, _ = transfer_1d_4(pol, 2*n+1, f, g, t, kzt, kzb,
                                 native.theta, nt, nb.conjugate())
        except np.linalg.LinAlgError as exc:
            if has_cutoff:
                raise _CutoffLimitRequired('boundary matching singular at external Rayleigh cutoff') from exc
            raise
        r = out['res']['R_s' if pol == 0 else 'R_p'].ravel().conjugate()
        th = out['res']['T_s' if pol == 0 else 'T_p'].ravel().conjugate()
        if pol == 1:
            r = -r  # reflected laboratory p_E basis has H_y=-n_top E_p/Z0
        te = th if pol == 0 else th*nt/nb
        kzinc = nt*k0*np.cos(np.radians(sim.theta_deg))
        prop = (abs(kz.imag) < 1e-10*k0) & (kz.real > 1e-10*k0)
        efficiency = abs(r)**2*np.maximum(kz.real, 0)/kzinc
        efficiency[~prop] = 0
        flux = kzbot.real/kzinc if pol == 0 else np.real(kzbot/(nb**2))/(kzinc/nt**2)
        transmission = abs(th)**2*flux
        collect = prop & (abs(kx)/k0 <= sim.na+1e-12)
        for array in (r, te, efficiency, transmission):
            if not np.isfinite(array).all():
                raise ValueError('RCWA returned non-finite values')
        if np.min(transmission) < -1e-10 or efficiency.sum()+transmission.sum() > 1+1e-7:
            raise ValueError('passive structure produced nonphysical power gain or negative outgoing flux')
        return DiffractionResult(orders, r, te, efficiency, transmission, kx.astype(complex),
            kz, prop, collect, stack.pitch_um, sim.wavelength_um, sim.theta_deg,
            sim.polarization, nt, branch_corrections=branch_corrections,
            cutoff_metadata=({'method': 'direct_zero_flux',
                              'requested_wavelength_um': sim.wavelength_um,
                              'reflection_orders': orders[cutoff_top].tolist(),
                              'transmission_orders': orders[cutoff_bottom].tolist(),
                              'samples': []} if has_cutoff else None))
