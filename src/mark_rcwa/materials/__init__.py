"""Explicit passive n+ik data under exp(-i omega t)."""
from dataclasses import dataclass
import csv
import numpy as np
from mark_rcwa.model import positive


@dataclass(frozen=True)
class Material:
    name: str
    n: complex | None = None
    wavelength_um: tuple[float, ...] = ()
    n_table: tuple[complex, ...] = ()
    source: str = 'user-specified synthetic optical constant'

    def __post_init__(self):
        vals = [self.n] if self.n is not None else list(self.n_table)
        if not vals or any(not np.isfinite(v) or v.real <= 0 or v.imag < 0 for v in vals):
            raise ValueError('passive materials require finite n>0 and k>=0')
        if self.n is None:
            x = np.asarray(self.wavelength_um)
            if len(x) != len(vals) or len(x) < 2 or not np.isfinite(x).all() or np.any(x <= 0) or np.any(np.diff(x) <= 0):
                raise ValueError('wavelength table must be positive, strictly increasing and match n_table')

    def index(self, wavelength_um):
        positive(wavelength_um, 'wavelength_um')
        if self.n is not None:
            return complex(self.n)
        if not self.wavelength_um[0] <= wavelength_um <= self.wavelength_um[-1]:
            raise ValueError(f'{self.name}: wavelength outside measured table; extrapolation disabled')
        vals = np.asarray(self.n_table)
        return complex(np.interp(wavelength_um, self.wavelength_um, vals.real),
                       np.interp(wavelength_um, self.wavelength_um, vals.imag))


class Materials(dict):
    def index(self, name, wavelength_um):
        if name not in self:
            raise ValueError(f'unknown material identifier: {name}')
        return self[name].index(wavelength_um)


def from_csv(name, path, source=None):
    """Read explicit wavelength_um,n,k columns; no guessed wavelength units."""
    with open(path, newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    try:
        return Material(name, wavelength_um=tuple(float(r['wavelength_um']) for r in rows),
            n_table=tuple(complex(float(r['n']), float(r['k'])) for r in rows),
            source=source or f'user optical table: {path}; linear n,k interpolation; um')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'invalid optical CSV {path}: expected wavelength_um,n,k; {exc}') from exc


def from_config(specs):
    if not isinstance(specs,dict):
        raise ValueError('materials must be a mapping of material names to specifications')
    registry = defaults()
    for name, spec in specs.items():
        if not isinstance(spec,dict):
            raise ValueError(f'material {name} must be a mapping')
        allowed={'table_csv','source'} if 'table_csv' in spec else {'n','k','wavelength_um','source'}
        unknown=set(spec)-allowed
        if unknown:
            raise ValueError(f'unknown material {name} keys: {sorted(unknown)}')
        if 'table_csv' in spec:
            registry[name] = from_csv(name, spec['table_csv'], spec.get('source'))
        elif 'wavelength_um' in spec:
            wavelengths = tuple(spec['wavelength_um'])
            ns, ks = spec['n'], spec.get('k', [0]*len(wavelengths))
            if len(ns) != len(ks):
                raise ValueError('n and k table lengths must match')
            registry[name] = Material(name, wavelength_um=wavelengths,
                n_table=tuple(complex(n, k) for n, k in zip(ns, ks)),
                source=spec.get('source', 'user wavelength table; um; linear n,k interpolation'))
        else:
            registry[name] = Material(name, complex(spec['n'], spec.get('k', 0)),
                source=spec.get('source', 'user synthetic constant'))
    return registry


def defaults():
    # These constants are intentionally hypothetical, not a metrology database.
    return Materials({name: Material(name, value, source='synthetic demonstration value; not calibrated material data')
        for name, value in {'air': 1, 'glass_demo': 1.5, 'Si': 3.8+0.02j,
        'SiO2': 1.46, 'Si3N4': 2.0, 'metal_demo': 0.25+4j,
        'absorber_demo': 2+0.2j}.items()})
