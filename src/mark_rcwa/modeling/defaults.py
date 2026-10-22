"""Explicitly synthetic examples for the standalone offline desktop."""
import math


def default_recipe(period_um=8, duty=.5, height_um=.12):
    period, duty, height = map(float, (period_um, duty, height_um))
    if not all(map(math.isfinite, (period, duty, height))) or period <= 0 or height <= 0 or not 0 < duty < 1:
        raise ValueError('Require finite period>0, height>0 and 0<duty<1')
    half, ridge = period / 2, period * duty / 2
    return {
        'schema_version': '1.0', 'synthetic_demo': True,
        'description': 'Synthetic demonstration grating; example materials are not calibrated optical data',
        'period_x_um': period, 'invariant_y': True,
        'cell_bounds_um': [[-half, half], [-.5, .5], [-.5, max(.5, height + .2)]],
        'background_material': 'air', 'substrate_material': 'glass_demo',
        'substrate_extend_bottom': True,
        'steps': [
            {'id': 'substrate', 'name': '示例衬底', 'operation': 'create_substrate',
             'material': 'glass_demo', 'enabled': True, 'z_min_um': -.5, 'z_max_um': 0},
            {'id': 'ridge', 'name': '示例条纹', 'operation': 'add',
             'material': 'ridge_demo', 'enabled': True, 'z_min_um': 0, 'z_max_um': height,
             'sidewall_angle_deg': 90,
             'mask': {'polygons_xy_um': [[[-ridge, -.5], [ridge, -.5], [ridge, .5], [-ridge, .5]]]}}
        ],
    }


def default_library():
    """Tables in the existing TXT library format; all entries are examples."""
    constants = {'air': (1., 0.), 'glass_demo': (1.5, 0.),
                 'ridge_demo': (3.8, .02), 'coat_demo': (1.46, 0.),
                 'metal_demo': (.25, 4.)}
    return {
        'schema_version': '1.0', 'model_version': 'txt-linear-rcwa-1.0',
        'generation_method': 'synthetic_demonstration_constants',
        'synthetic_demo': True,
        'description': 'SYNTHETIC DEMONSTRATION ONLY; not calibrated material data',
        'materials': {name: {
            'wavelength_nm': [200., 2000.], 'n': [n, n], 'k': [k, k],
            'k_status': 'synthetic_demo', 'assume_lossless': False,
            'valid_range_nm': [200., 2000.], 'synthetic_demo': True,
            'source': 'SYNTHETIC DEMONSTRATION ONLY; not calibrated material data',
        } for name, (n, k) in constants.items()},
    }
