from dataclasses import replace
from numbers import Integral
import numpy as np
from mark_rcwa.model import positive
from mark_rcwa.solvers.meent_adapter import outgoing_sqrt


def propagate(result, distance_um):
    """Transfer reflected fields outward by L=-z. Efficiency is boundary flux."""
    positive(distance_um, 'outward distance_um', allow_zero=True)
    return replace(result, amplitude=result.amplitude*np.exp(1j*result.kz*distance_um),
                   reference_plane_um=result.reference_plane_um-distance_um)


def pupil_transfer(result, na=None, weights=None):
    """Complex pupil coefficients; evanescent or uncollected fields are zero."""
    mask = result.collected.copy()
    if na is not None:
        positive(na, 'NA', allow_zero=True)
        if na > result.n_top:
            raise ValueError('NA exceeds collection medium refractive index')
        mask = result.propagating & (abs(result.kx)*result.wavelength_um/(2*np.pi) <= na+1e-12)
    transfer = mask.astype(complex)
    if weights is not None:
        transfer[:] = 0
        for m, value in weights.items():
            if isinstance(m,(bool,np.bool_)) or not isinstance(m,Integral):
                raise ValueError('pupil diffraction orders must be integers')
            index = np.flatnonzero(result.orders == int(m))
            if len(index) != 1:
                raise ValueError(f'pupil order {m} outside retained Fourier basis')
            if not mask[index[0]]:
                raise ValueError(f'pupil order {m} is evanescent or outside NA')
            if not np.isfinite(value):
                raise ValueError('pupil coefficients must be finite')
            transfer[index[0]] = complex(value)
    return transfer


def angular_spectrum_1d(field, dx_um, wavelength_um, distance_um, n=1.):
    """Scalar homogeneous-medium interface, FFT-periodic computational window.

    This propagates an already known field; it does not solve a finite mark.
    """
    positive(dx_um, 'dx_um')
    positive(wavelength_um, 'wavelength_um')
    positive(distance_um, 'distance_um', allow_zero=True)
    positive(n, 'medium index')
    field = np.asarray(field, dtype=complex)
    if field.ndim != 1 or len(field) < 2 or not np.isfinite(field).all():
        raise ValueError('field must be a finite 1D array with at least two samples')
    kx = 2*np.pi*np.fft.fftfreq(len(field), dx_um)
    kz = outgoing_sqrt((n*2*np.pi/wavelength_um)**2-kx**2)
    return np.fft.ifft(np.fft.fft(field)*np.exp(1j*kz*distance_um))
