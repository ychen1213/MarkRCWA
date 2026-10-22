"""Full Jones matrices for a one-dimensional grating under conical incidence.

The depth coordinate points from the incident half-space into the substrate.
All coefficients use exp(+i k.r-i omega t), local s=(ky,-kx,0)/kt and p=khat
cross s, matching the explicit polarization geometry of the Lumerical fixtures.
Matrices have axes [diffraction order, output polarization, input polarization].
Reference planes are the actual upper/lower stack interfaces, x=0 at the
left endpoint of Slice edges; callers must apply any coordinate translation.
"""
from dataclasses import dataclass
import numpy as np
from threadpoolctl import threadpool_limits
from meent.on_numpy.emsolver.convolution_matrix import to_conv_mat_vector
from meent.on_numpy.emsolver.transfer_method import (
    transfer_1d_conical_2, transfer_1d_conical_3, transfer_1d_conical_4,
)
from mark_rcwa.solvers.meent_adapter import outgoing_sqrt
from mark_rcwa.model import positive


@dataclass
class ConicalResult:
    orders: np.ndarray
    R: np.ndarray
    T: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    kz_top: np.ndarray
    kz_bottom: np.ndarray
    reflected_efficiency: np.ndarray
    transmitted_efficiency: np.ndarray
    propagating_top: np.ndarray
    propagating_bottom: np.ndarray
    incident_basis_s: np.ndarray
    incident_basis_p: np.ndarray
    reflected_basis_s: np.ndarray
    reflected_basis_p: np.ndarray
    transmitted_basis_s: np.ndarray
    transmitted_basis_p: np.ndarray
    branch_corrections: int


def _basis(kx, ky, kz, index, incident_s):
    transverse = np.hypot(kx.real, ky.real)
    active = transverse > 1e-14
    s = np.zeros((len(kx), 3), dtype=complex)
    s[active, 0] = ky[active] / transverse[active]
    s[active, 1] = -kx[active] / transverse[active]
    s[~active] = incident_s
    khat = np.column_stack((kx, ky, kz)) / index
    return s, np.cross(khat, s)


def solve_conical(stack, materials, wavelength_um, theta_deg, phi_deg, fourier_order):
    """Solve both incident polarizations without perturbing material constants.

    Angles are degrees in a right-handed depth-down frame; phi is measured
    from +x towards +y. R/T are electric-field amplitudes, not sqrt(power).
    Evanescent amplitudes use the analytic continuation of the declared basis;
    only propagating orders have ordinary unit-vector electric amplitudes.
    """
    positive(wavelength_um, 'wavelength_um')
    if not np.isfinite(theta_deg) or not 0 <= theta_deg < 89.9:
        raise ValueError('conical theta_deg must be finite in [0, 89.9)')
    if not np.isfinite(phi_deg):
        raise ValueError('phi_deg must be finite')
    if isinstance(fourier_order, bool) or not isinstance(fourier_order, int) or not 0 <= fourier_order <= 255:
        raise ValueError('fourier_order must be an integer between 0 and 255')
    with threadpool_limits(limits=1), np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            return _solve(stack, materials, wavelength_um, theta_deg, phi_deg, fourier_order)
        except (FloatingPointError, np.linalg.LinAlgError, OverflowError, MemoryError) as exc:
            raise ValueError(f'Conical RCWA numerical failure: {exc}; inspect truncation and cutoffs') from exc


def _solve(stack, materials, wavelength, theta_deg, phi_deg, n):
    nt = materials.index(stack.top, wavelength)
    nb = materials.index(stack.bottom, wavelength)
    if nt.imag != 0:
        raise ValueError('incident half-space must be lossless')
    nt = nt.real
    theta, phi = np.radians([theta_deg, phi_deg])
    k0 = 2*np.pi/wavelength
    orders = np.arange(-n, n+1)
    count = len(orders)
    if 12*max(1, len(stack.layers))*count**2*16 > 1024*1024**2:
        raise ValueError('Conical Fourier matrices exceed 1 GiB allocation guard')
    kx = (nt*np.sin(theta)*np.cos(phi) + orders*wavelength/stack.pitch_um).astype(complex)
    ky = np.full(count, nt*np.sin(theta)*np.sin(phi), dtype=complex)
    qtop = outgoing_sqrt(nt**2-kx**2-ky**2)
    qbot = outgoing_sqrt(nb**2-kx**2-ky**2)
    if np.any(abs(qtop) < 1e-10) or np.any(abs(qbot) < 1e-10):
        raise ValueError('exact Rayleigh cutoff produces singular modal normalization')
    # Meent uses exp(+i omega t) and downward radiation kz conjugates.
    kzt, kzb = qtop.conjugate(), qbot.conjugate()
    # arctan2 removes the upstream arctan(ky/kx) quadrant ambiguity, and
    # handles kx=0 exactly without an artificial angle perturbation.
    varphi = np.arctan2(ky.real, kx.real)
    varphi[np.hypot(kx.real, ky.real) <= 1e-14] = phi
    eye = np.eye(count, dtype=complex)
    zero = np.zeros_like(eye)
    f = np.block([[eye, zero], [zero, 1j*np.diag(kzb)/(nb.conjugate()**2)]])
    g = np.block([[1j*np.diag(kzb), zero], [zero, eye]])
    t = np.eye(2*count, dtype=complex)
    corrections = 0
    for layer in reversed(stack.layers):
        indices = np.array([materials.index(m, wavelength) for m in layer.materials], complex)
        data = [[indices.conjugate().reshape(1, -1), np.array(layer.edges_um[1:]).reshape(-1, 1),
                 np.array([[stack.pitch_um]])]]
        ex, ey, ezi = to_conv_mat_vector(data, n, 0)
        w, v, q = transfer_1d_conical_2(kx, ky[:1], ex[0], ey[0], ezi[0])
        if np.all(indices.imag == 0):
            bad = (q.imag < 0) & (abs(q.real) <= 1e-12*np.maximum(1., abs(q)))
            if np.any(bad):
                previous = q.copy()
                q[bad] = 1j*abs(q[bad].imag)
                ratio = np.ones_like(q)
                ratio[bad] = q[bad]/previous[bad]
                # Only these diagonal blocks contain Q1/Q2. The cross
                # blocks V12/V21 are independent of the eigenvalue roots.
                v[:count, :count] *= ratio[:count][None, :]
                v[count:, count:] *= ratio[count:][None, :]
                corrections += int(bad.sum())
        _, f, g, t, _, _ = transfer_1d_conical_3(k0, w, v, q, layer.thickness_um, varphi, f, g, t)
    result, _ = transfer_1d_conical_4(count, 1, f, g, t, kzt, kzb, np.pi/2, theta, nt, nb.conjugate())
    r = np.empty((count, 2, 2), complex)
    trans = np.empty_like(r)
    for column, key in enumerate(('res_te_inc', 'res_tm_inc')):
        raw = result[key]
        r[:, 0, column] = raw['R_s'].ravel().conjugate()
        r[:, 1, column] = 1j*raw['R_p'].ravel().conjugate()/nt
        trans[:, 0, column] = raw['T_s'].ravel().conjugate()
        trans[:, 1, column] = 1j*raw['T_p'].ravel().conjugate()/nb
    # The conical boundary equations use E_s and -i*H_s amplitudes. Their
    # i*H/n -> E conversion gives the declared p=khat cross s basis.
    qinc = nt*np.cos(theta)
    reflected = abs(r)**2*np.maximum(qtop.real, 0)[:, None, None]/qinc
    transmitted = abs(trans)**2*np.maximum(qbot.real, 0)[:, None, None]/qinc
    # For lossy exit media, p flux is Re(q/n^2)*|n|^2 for this E basis.
    transmitted[:, 1] = abs(trans[:, 1])**2*(np.real(qbot/(nb**2))*abs(nb)**2/qinc)[:, None]
    prop_top = (abs(qtop.imag) < 1e-10) & (qtop.real > 1e-10)
    prop_bot = (abs(qbot.imag) < 1e-10) & (qbot.real > 1e-10)
    reflected[~prop_top] = 0
    if not all(np.isfinite(a).all() for a in (r, trans, reflected, transmitted)):
        raise ValueError('Conical RCWA returned nonfinite values')
    if np.any(transmitted < -1e-10) or np.any(reflected.sum(axis=(0, 1))+transmitted.sum(axis=(0, 1)) > 1+1e-7):
        raise ValueError('passive conical structure produced nonphysical outgoing power')
    incident_s = np.array([np.sin(phi), -np.cos(phi), 0.])
    incident_k = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    rs, rp = _basis(kx, ky, -qtop, nt, incident_s)
    ts, tp = _basis(kx, ky, qbot, nb, incident_s)
    return ConicalResult(orders, r, trans, kx*k0, ky*k0, qtop*k0, qbot*k0,
                         reflected, transmitted, prop_top, prop_bot, incident_s,
                         np.cross(incident_k, incident_s), rs, rp, ts, tp, corrections)
