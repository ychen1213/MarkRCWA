"""Independent planar Maxwell references and reciprocal grating checks.

The planar reference uses 2x2 tangential-electric characteristic matrices,
not Meent eigenvectors, convolution matrices, roots, or transfer operators.
Finite-film absorption is also evaluated from a volume dissipation integral.
"""
from dataclasses import replace
import numpy as np
from mark_rcwa.materials import Material, Materials
from mark_rcwa.model import Stack, Slice
from mark_rcwa.solvers import MeentSolver, Simulation


def _root(value):
    root = complex(np.sqrt(complex(value)))
    if root.imag < 0 or (root.imag == 0 and root.real < 0):
        root = -root
    return root


def _json(value):
    if isinstance(value, complex) or np.iscomplexobj(value) and np.ndim(value) == 0:
        return [float(np.real(value)), float(np.imag(value))]
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def characteristic_reference(n_top, n_bottom, films, wavelength_um, theta_deg, polarization):
    """Return analytic r,t_E,R,T and volume absorption for isotropic films.

    films are (complex refractive index, thickness_um) pairs, ordered top down.
    Fields use exp(-i omega t), z into stack, and a reflected p basis with +x.
    The transmitted p basis has bilinear unit norm in an absorbing half-space;
    power uses its physical tangential Poynting flux, not |t_E|^2 alone.
    """
    nt, nb = complex(n_top), complex(n_bottom)
    if nt.imag != 0 or nt.real <= 0:
        raise ValueError('analytic incident index must be positive and lossless')
    if polarization not in ('TE', 'TM'):
        raise ValueError('analytic polarization must be TE or TM')
    k0 = 2*np.pi/wavelength_um
    cosine = np.cos(np.radians(theta_deg))
    transverse = nt.real*np.sin(np.radians(theta_deg))
    qt, qb = _root(nt*nt-transverse**2), _root(nb*nb-transverse**2)
    yt = qt if polarization == 'TE' else nt*nt/qt
    yb = qb if polarization == 'TE' else nb*nb/qb
    matrix = np.eye(2, dtype=complex)
    modes = []
    for n_layer, thickness in films:
        n_layer = complex(n_layer)
        q = _root(n_layer*n_layer-transverse**2)
        y = q if polarization == 'TE' else n_layer*n_layer/q
        delta = k0*q*thickness
        film = np.array([[np.cos(delta), -1j*np.sin(delta)/y],
                         [-1j*y*np.sin(delta), np.cos(delta)]])
        matrix = matrix@film
        modes.append((n_layer, thickness, q, y))
    a, b = matrix@np.array([1, yb])
    r = (yt*a-b)/(yt*a+b)
    tangent_t = 2*yt/(yt*a+b)
    te = tangent_t if polarization == 'TE' else tangent_t*(qt/nt)/(qb/nb)
    reflected_power = abs(r)**2
    transmitted_power = abs(tangent_t)**2*yb.real/yt.real
    # Dissipation density: k0 Im(epsilon)|E|^2/(n_top cos(theta)).
    # Recover layer fields from the independently computed boundary fields.
    incident_tangent = 1 if polarization == 'TE' else cosine
    e, h = incident_tangent*(1+r), yt*incident_tangent*(1-r)
    nodes, weights = np.polynomial.legendre.leggauss(96)
    volume_absorption = 0.
    layer_absorption = []
    for n_layer, thickness, q, y in modes:
        forward, backward = .5*(e+h/y), .5*(e-h/y)
        z = (nodes+1)*thickness/2
        ef = forward*np.exp(1j*k0*q*z)
        eb = backward*np.exp(-1j*k0*q*z)
        electric_squared = abs(ef+eb)**2
        if polarization == 'TM':
            electric_squared += abs(transverse/q*(-ef+eb))**2
        integral = thickness/2*float(weights@electric_squared)
        absorption = float(k0*(n_layer*n_layer).imag*integral/(nt.real*cosine))
        volume_absorption += absorption
        layer_absorption.append(absorption)
        delta = k0*q*thickness
        e, h = (np.cos(delta)*e+1j*np.sin(delta)/y*h,
                1j*y*np.sin(delta)*e+np.cos(delta)*h)
    return {'r_electric':r, 't_electric':te, 'R':float(reflected_power),
            'T':float(transmitted_power), 'A_volume':float(volume_absorption),
            'layer_absorption':layer_absorption, 'off_order_max':0.}


def analytic_cases():
    """80 independently specified planar cases; no demonstration-data lookup."""
    profiles = [
        ('fresnel_glass',1.,1.5,()),
        ('fresnel_absorbing',1.33,1.65+.043j,()),
        ('fresnel_metal',1.,.25+4.1j,()),
        ('lossless_twofilm',1.,1.5,((2.,.089),(1.46,.137))),
        ('absorbing_twofilm',1.33,1.65+.043j,((1.65+.043j,.129),(1.46,.237))),
        ('metal_backfilm',1.,.25+4.1j,((1.46,.217),(1.65+.043j,.052))),
        ('total_internal_reflection',1.8,1.,()),
        ('total_internal_reflection_film',1.8,1.,((1.46,.134),)),
    ]
    return [dict(name=f'{name}_{pol}_{theta:g}deg',n_top=nt,n_bottom=nb,films=films,
                 wavelength_um=.633,theta_deg=theta,polarization=pol,fourier_order=3,pitch_um=8.)
            for name,nt,nb,films in profiles
            for theta in [0.,17.3,47.2,-31.9,68.]
            for pol in ['TE','TM']]


def _record(name, parameters, expected, actual, error, tolerances, reason, solver='Meent 0.13.2, pinned adapter'):
    passed = all(np.isfinite(value) and value <= tolerances[key] for key,value in error.items())
    return _json(dict(name=name,parameters=parameters,solver=solver,expected=expected,actual=actual,
        error=error,tolerances=tolerances,status='PASS' if passed else 'FAIL',reason=reason))


def validate_analytic_case(case):
    """Run one planar case and return the common JSON-safe validation schema."""
    materials = Materials({'top':Material('top',case['n_top']),
                           'bottom':Material('bottom',case['n_bottom'])})
    layers = []
    for i,(index,thickness) in enumerate(case['films']):
        name=f'film_{i}'
        materials[name]=Material(name,index)
        layers.append(Slice(thickness,(0.,case['pitch_um']),(name,)))
    spec = Simulation(wavelength_um=case['wavelength_um'],theta_deg=case['theta_deg'],
        polarization=case['polarization'],fourier_order=case['fourier_order'],na=.8)
    result = MeentSolver().solve(Stack(case['pitch_um'],tuple(layers),'top','bottom'),materials,spec)
    expected = characteristic_reference(case['n_top'],case['n_bottom'],case['films'],
        case['wavelength_um'],case['theta_deg'],case['polarization'])
    central = int(np.flatnonzero(result.orders == 0)[0])
    off = max(float(np.max(abs(np.delete(result.amplitude,central)),initial=0)),
              float(np.max(abs(np.delete(result.transmission_amplitude,central)),initial=0)))
    actual = {'r_electric':result.amplitude[central], 't_electric':result.transmission_amplitude[central],
              'R':result.reflectance,'T':result.transmittance,'A_volume':result.absorption,'off_order_max':off}
    keys = ['r_electric','t_electric','R','T','A_volume','off_order_max']
    error = {key:float(abs(actual[key]-expected[key])) for key in keys}
    tolerances = {key:(1e-12 if key == 'off_order_max' else 5e-11) for key in keys}
    return _record(case['name'],case,expected,actual,error,tolerances,
        'Independent 2x2 tangential-E matrices; A checked against integral of Im(epsilon)|E|^2 over finite films. '
        'T is boundary flux into the half-space; half-space dissipation is excluded.')


def _grating_materials():
    return Materials({'top':Material('top',1.),'bottom':Material('bottom',1.5),
                      'ridge':Material('ridge',1.7),'groove':Material('groove',1.)})


def grating_identity_records():
    """Actual independent solves for mirror symmetry and reciprocal reflection."""
    from mark_rcwa.geometry import rectangle
    stack = rectangle(pitch_um=2.4,depth_um=.12,duty=.37,ridge='ridge',groove='groove',top='top',bottom='bottom')
    materials = _grating_materials()
    rows=[]
    for pol in ['TE','TM']:
        for theta in [0.,17.3,41.]:
            spec=Simulation(theta_deg=theta,polarization=pol,fourier_order=31)
            base=MeentSolver().solve(stack,materials,spec)
            mirror=MeentSolver().solve(stack,materials,replace(spec,theta_deg=-theta))
            errors={'reflected_amplitude_max':float(max(abs(base.amplitude-mirror.amplitude[::-1]))),
                    'transmitted_amplitude_max':float(max(abs(base.transmission_amplitude-mirror.transmission_amplitude[::-1]))),
                    'reflected_efficiency_max':float(max(abs(base.efficiency-mirror.efficiency[::-1])))}
            rows.append(_record(f'mirror_symmetry_{pol}_{theta:g}deg',
                {'pitch_um':stack.pitch_um,'duty':.37,'depth_um':.12,'theta_deg':theta,'polarization':pol,'fourier_order':31},
                {'orders':base.orders,'reflected_electric':mirror.amplitude[::-1],
                 'transmitted_electric':mirror.transmission_amplitude[::-1],'reflected_efficiency':mirror.efficiency[::-1]},
                {'orders':base.orders,'reflected_electric':base.amplitude,
                 'transmitted_electric':base.transmission_amplitude,'reflected_efficiency':base.efficiency},
                errors,{key:2e-10 for key in errors},'Centered binary grating: solve +theta and -theta separately and reverse diffraction order.'))
        # Time-reverse a reflected outgoing port, then illuminate it as a new
        # incident plane wave. The reciprocal amplitude includes sqrt(kz_out/kz_in).
        # A finite Fourier window shifts between ports; report a measured
        # truncation error from N=91 and N=131 in addition to absolute equality.
        for order in [-1,1,2]:
            values=[]
            for truncation in [91,131]:
                theta=12.3
                spec=Simulation(theta_deg=theta,polarization=pol,fourier_order=truncation)
                base=MeentSolver().solve(stack,materials,spec)
                index=int(np.flatnonzero(base.orders==order)[0])
                reverse_sine=-base.kx[index].real/(2*np.pi/spec.wavelength_um)
                reverse_theta=float(np.degrees(np.arcsin(reverse_sine)))
                reverse=MeentSolver().solve(stack,materials,replace(spec,theta_deg=reverse_theta))
                rev_index=int(np.flatnonzero(reverse.orders==order)[0])
                kz_inc=2*np.pi/spec.wavelength_um*np.cos(np.radians(theta))
                kz_out=base.kz[index].real
                forward=base.amplitude[index]*np.sqrt(kz_out/kz_inc)
                backward=reverse.amplitude[rev_index]*np.sqrt(kz_inc/kz_out)
                values.append((forward,backward,reverse_theta))
            low,high=values
            error={'flux_normalized_complex_error':float(abs(high[0]-high[1])),
                   'truncation_amplitude_change':float(max(abs(high[0]-low[0]),abs(high[1]-low[1])))}
            rows.append(_record(f'reciprocity_{pol}_order_{order}',
                {'pitch_um':stack.pitch_um,'depth_um':.12,'duty':.37,'theta_deg':12.3,'reverse_theta_deg':high[2],
                 'order':order,'polarization':pol,'fourier_orders':[91,131]},
                {'reciprocal_port_electric':high[1]}, {'forward_port_electric':high[0]},error,
                {'flux_normalized_complex_error':2e-7,'truncation_amplitude_change':2e-6},
                'Lorentz reciprocity of actual reflected ports: reverse incident kx=-outgoing kx; compare E*sqrt(kz_out/kz_in). '
                'Different retained Fourier windows require the reported truncation check.'))
    return rows


def reference_plane_records():
    """Propagate actual reflected films and compare to an analytic plane-wave field."""
    from mark_rcwa.propagation import propagate
    rows=[]
    for pol in ['TE','TM']:
        case = next(c for c in analytic_cases() if c['name']==f'absorbing_twofilm_{pol}_17.3deg')
        mats=Materials({'top':Material('top',case['n_top']),'bottom':Material('bottom',case['n_bottom'])})
        layers=[]
        for i,(n,d) in enumerate(case['films']):
            name=f'f{i}'
            mats[name]=Material(name,n)
            layers.append(Slice(d,(0.,case['pitch_um']),(name,)))
        spec=Simulation(theta_deg=case['theta_deg'],polarization=pol,fourier_order=3)
        result=MeentSolver().solve(Stack(case['pitch_um'],tuple(layers),'top','bottom'),mats,spec)
        expected_r=characteristic_reference(case['n_top'],case['n_bottom'],case['films'],
            case['wavelength_um'],case['theta_deg'],pol)['r_electric']
        for distance in [0.,.23,1.17]:
            shifted=propagate(result,distance)
            kz=case['n_top']*2*np.pi/spec.wavelength_um*np.cos(np.radians(spec.theta_deg))
            expected=expected_r*np.exp(1j*kz*distance)
            actual=shifted.amplitude[3]
            error={'complex_amplitude_error':float(abs(actual-expected)),
                   'plane_coordinate_error_um':float(abs(shifted.reference_plane_um+distance))}
            rows.append(_record(f'reference_plane_{pol}_{distance:g}um',
                {**case,'outward_distance_um':distance}, {'r_electric':expected,'z_um':-distance},
                {'r_electric':actual,'z_um':shifted.reference_plane_um},error,
                {'complex_amplitude_error':5e-11,'plane_coordinate_error_um':1e-14},
                'Compare propagated Meent film with independent characteristic-matrix r times analytic exp(+i kz L).'))
    return rows


def run_analytic_validation():
    """Return 98 records, each JSON-safe and suitable for a validation report."""
    return [validate_analytic_case(case) for case in analytic_cases()]+grating_identity_records()+reference_plane_records()
