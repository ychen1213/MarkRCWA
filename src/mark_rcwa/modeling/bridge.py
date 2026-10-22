"""Convert saved process geometry into exact, top-to-bottom RCWA layers."""
from __future__ import annotations

import math

from mark_rcwa.model import Slice, Stack

from .geometry import validate_geometry
from .solver_geometry import canonical_rectangles


def _verified_y_rectangles(geometry):
    """Prove that each solid is a full-y rectangle, not just equal in area.

    In particular a narrow notch or a partial-y mask cannot pass an area-only
    tolerance check. Collinear vertices on the four rectangle sides are valid.
    """
    if geometry.get('invariant_y') is not True:
        raise ValueError('One-dimensional RCWA requires declared and verified y invariance')
    expected_y = geometry['cell_bounds_um'][1]
    for solid in geometry['solids']:
        p = list(solid['polygon_xy_um'])
        xlo, xhi = min(v[0] for v in p), max(v[0] for v in p)
        ylo, yhi = min(v[1] for v in p), max(v[1] for v in p)
        # Only floating point roundoff is accepted here, not a geometric gap.
        tolerance = 8 * math.ulp(max(abs(ylo), abs(yhi), abs(expected_y[0]), abs(expected_y[1]), 1.0))
        if abs(ylo - expected_y[0]) > tolerance or abs(yhi - expected_y[1]) > tolerance:
            raise ValueError(f"{solid['name']}: geometry varies along y; solid must span the complete y cell")
        for a, b in zip(p, p[1:] + p[:1]):
            horizontal = a[1] == b[1] and a[1] in (ylo, yhi)
            vertical = a[0] == b[0] and a[0] in (xlo, xhi)
            if not (horizontal or vertical):
                raise ValueError(f"{solid['name']}: geometry varies along y; only full-y rectangles are supported")


def _slab(rectangles, zlo, zhi, xmin, xmax, background):
    active = [r for r in rectangles if r['z_min_um'] <= zlo and r['z_max_um'] >= zhi]
    edges = sorted({xmin, xmax, *(r[k] for r in active for k in ('x_min_um', 'x_max_um'))})
    ids = []
    kept = [edges[0] - xmin]
    for lo, hi in zip(edges, edges[1:]):
        hits = {r['material'] for r in active if r['x_min_um'] <= lo and r['x_max_um'] >= hi}
        if len(hits) > 1:
            raise ValueError('Overlapping materials in RCWA layer')
        material = next(iter(hits), background)
        if ids and ids[-1] == material:
            kept[-1] = hi - xmin
        else:
            ids.append(material)
            kept.append(hi - xmin)
    # Use exactly the same pitch as Stack, including nonzero source origins.
    kept[0], kept[-1] = 0.0, xmax - xmin
    return Slice(zhi - zlo, tuple(kept), tuple(ids))


def geometry_to_stack(geometry, top=None, bottom=None):
    """Preserve all finite material and gaps, with explicit substrate semantics.

    The upper reference plane is the highest non-background solid. The lower
    reference plane is the declared geometry floor unless a uniform substrate
    reaches it and ``substrate_extend_bottom`` is true. Only that consecutive
    bottom substrate is then represented by a half space; patterned substrate
    above it remains finite. No absorbing boundary padding is introduced.
    """
    validate_geometry(geometry)
    _verified_y_rectangles(geometry)
    xmin, xmax = map(float, geometry['cell_bounds_um'][0])
    pitch = float(geometry['period_x_um'])
    if not math.isclose(xmax - xmin, pitch, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('period_x_um must equal the complete x cell extent')
    background = geometry['background_material']
    rectangles = [r for r in canonical_rectangles(geometry) if r['material'] != background]
    floor = float(geometry['cell_bounds_um'][2][0])
    top_z = max((r['z_max_um'] for r in rectangles), default=floor)
    levels = sorted({floor, top_z, *(r[k] for r in rectangles for k in ('z_min_um', 'z_max_um'))})
    slabs = [(lo, hi, _slab(rectangles, lo, hi, xmin, xmax, background))
             for lo, hi in zip(levels, levels[1:]) if hi > lo]
    substrate = geometry.get('substrate_material')
    removed = []
    if geometry.get('substrate_extend_bottom', True) is True and substrate:
        while slabs and slabs[0][2].materials == (substrate,):
            removed.append(slabs.pop(0))
    continued = bool(removed)
    if continued and bottom is not None and bottom != substrate:
        raise ValueError('bottom conflicts with the explicitly continued substrate; disable substrate_extend_bottom to keep it finite')
    bottom_id = bottom if bottom is not None else (substrate if continued else background)
    top_id = top if top is not None else background
    if any(not isinstance(v, str) or not v for v in (top_id, bottom_id)):
        raise ValueError('Top and bottom half spaces require nonempty material identifiers')
    layers = tuple(s for _, _, s in reversed(slabs))
    return Stack(xmax - xmin, layers, top=top_id, bottom=bottom_id, metadata={
        'source': 'MarkSim-compatible process geometry; exact rectangular layering',
        'geometry_input_hash': geometry.get('input_hash'),
        'coordinate_convention': 'layers ordered from highest physical z down; x shifted by x_origin_um',
        'x_origin_um': xmin,
        'top_z_um': top_z,
        'bottom_z_um': removed[-1][1] if continued else floor,
        'geometry_floor_um': floor,
        'substrate_continued': continued,
        'substrate_removed_thickness_um': sum(s.thickness_um for _, _, s in removed),
        'synthetic_demo': bool(geometry.get('provenance', {}).get('recipe', {}).get('synthetic_demo', False)),
    })
