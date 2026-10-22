"""Continuous profiles converted to exact lateral intervals in micrometres."""
from dataclasses import replace
import numpy as np
from mark_rcwa.model import Slice, Stack, positive


def _identifier(value):
    if not isinstance(value, str) or not value:
        raise ValueError('material identifiers must be nonempty strings')


def _slice_count(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4096:
        raise ValueError('slices must be an integer between 1 and 4096')
    return value


def _merge_slice(layer):
    edges, ids = [layer.edges_um[0]], []
    for right, material in zip(layer.edges_um[1:], layer.materials):
        if ids and ids[-1] == material:
            edges[-1] = right
        else:
            ids.append(material)
            edges.append(right)
    return Slice(layer.thickness_um, tuple(edges), tuple(ids))


def _interval_slice(pitch_um, thickness_um, intervals, ridge, groove):
    edges, ids, cursor = [0.], [], 0.
    for left, right in sorted(intervals):
        if not 0 <= left < right <= pitch_um or left < cursor:
            raise ValueError('profile intervals must be disjoint and inside the pitch')
        if left > cursor:
            edges.append(float(left))
            ids.append(groove)
        edges.append(float(right))
        ids.append(ridge)
        cursor = right
    if cursor < pitch_um:
        edges.append(float(pitch_um))
        ids.append(groove)
    return _merge_slice(Slice(float(thickness_um), tuple(edges), tuple(ids)))


def rectangle(pitch_um=8., depth_um=.12, duty=.5, ridge='Si3N4',
              groove='air', top='air', bottom='glass_demo'):
    pitch_um, depth_um = positive(pitch_um, 'pitch_um'), positive(depth_um, 'depth_um')
    if not np.isfinite(duty) or not 0 < duty < 1:
        raise ValueError('duty must lie strictly between 0 and 1')
    for name in (ridge, groove, top, bottom):
        _identifier(name)
    layer = _interval_slice(pitch_um, depth_um,
                           [(pitch_um*(1-duty)/2, pitch_um*(1+duty)/2)], ridge, groove)
    return Stack(pitch_um, (layer,), top, bottom, metadata={
        'kind': 'rectangle', 'duty': float(duty), 'top_cd_um': float(duty*pitch_um),
        'exact_lateral_edges': True, 'staircase': {'actual_slices': 1, 'max_dz_um': depth_um,
        'max_midpoint_z_error_um': 0., 'max_edge_error_bound_um': 0., 'area_error_um2': 0.}})


def _cross(a, b, c):
    u, v = b-a, c-a
    return float(u[0]*v[1]-u[1]*v[0])


def _segments_intersect(a, b, c, d, tolerance):
    crosses = (_cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b))
    if crosses[0]*crosses[1] < -tolerance**2 and crosses[2]*crosses[3] < -tolerance**2:
        return True
    for value, point, start, end in ((crosses[0], c, a, b), (crosses[1], d, a, b),
                                   (crosses[2], a, c, d), (crosses[3], b, c, d)):
        if abs(value) <= tolerance and np.all(point >= np.minimum(start, end)-tolerance) and np.all(point <= np.maximum(start, end)+tolerance):
            return True
    return False


def _validate_polygon(points, pitch_um):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError('polygon must contain finite closed (x,z) coordinates')
    if not np.array_equal(points[0], points[-1]):
        raise ValueError('polygon must be explicitly closed: repeat its first point')
    vertices = points[:-1]
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) == 0):
        raise ValueError('polygon has a zero-length edge')
    if np.any(vertices[:, 0] < 0) or np.any(vertices[:, 0] > pitch_um) or np.min(vertices[:, 1]) != 0 or np.any(vertices[:, 1] < 0):
        raise ValueError('polygon x must be in [0,pitch_um], with minimum z exactly 0')
    depth_um = positive(np.max(vertices[:, 1]), 'polygon depth_um')
    tolerance = 1e-12*max(pitch_um, depth_um, 1.)**2
    n = len(vertices)
    for i in range(n):
        for j in range(i+1, n):
            if j == i+1 or (i == 0 and j == n-1):
                continue
            if _segments_intersect(vertices[i], vertices[(i+1) % n], vertices[j], vertices[(j+1) % n], tolerance):
                raise ValueError('polygon is self-intersecting or has nonadjacent touching edges')
    area = .5*np.sum(vertices[:, 0]*np.roll(vertices[:, 1], -1)-vertices[:, 1]*np.roll(vertices[:, 0], -1))
    if abs(area) <= tolerance:
        raise ValueError('polygon has zero area')
    return vertices, depth_um, abs(float(area))


def _profile_intervals(vertices, z):
    crossings = []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        if min(a[1], b[1]) <= z < max(a[1], b[1]):
            crossings.append(float(a[0]+(z-a[1])*(b[0]-a[0])/(b[1]-a[1])))
    crossings.sort()
    if len(crossings) % 2:
        raise ValueError('polygon has an inconsistent horizontal cross-section')
    return [(left, right) for left, right in zip(crossings[::2], crossings[1::2]) if right > left]


def from_polygon(points, pitch_um=8., slices=12, ridge='Si3N4', groove='air',
                 top='air', bottom='glass_demo', materials=None):
    """Midpoint slices of a simple explicitly closed (x,z) ridge polygon.

    Uniform z planes are augmented by vertex heights to retain actual ledges.
    """
    pitch_um, slices = positive(pitch_um, 'pitch_um'), _slice_count(slices)
    for name in (ridge, groove, top, bottom):
        _identifier(name)
        if materials is not None and name not in materials:
            raise ValueError(f'unknown material identifier: {name}')
    vertices, depth_um, area = _validate_polygon(points, pitch_um)
    z_edges = np.unique(np.concatenate((np.linspace(0., depth_um, slices+1), vertices[:, 1])))
    z_edges = z_edges[np.r_[True, np.diff(z_edges) > 1e-14*max(depth_um, 1.)]]
    z_edges[-1] = depth_um
    layers = tuple(_interval_slice(pitch_um, upper-lower,
                    _profile_intervals(vertices, (lower+upper)/2), ridge, groove)
                   for lower, upper in zip(z_edges[:-1], z_edges[1:]))
    dz = float(np.diff(z_edges).max())
    slopes = [abs((b[0]-a[0])/(b[1]-a[1])) for a, b in zip(vertices, np.roll(vertices, -1, axis=0)) if a[1] != b[1]]
    sliced_area = sum(layer.thickness_um*sum(right-left for left, right, name in
                        zip(layer.edges_um[:-1], layer.edges_um[1:], layer.materials) if name == ridge)
                      for layer in layers)
    recipe = {'points': np.vstack((vertices, vertices[0])).tolist(), 'pitch_um': pitch_um,
              'slices': slices, 'ridge': ridge, 'groove': groove, 'top': top, 'bottom': bottom}
    return Stack(pitch_um, layers, top, bottom, metadata={
        'kind': 'polygon', 'continuous_polygon_um': recipe['points'], 'profile_recipe': recipe,
        'exact_lateral_edges': True, 'staircase': {
        'requested_slices': slices, 'actual_slices': len(layers), 'max_dz_um': dz,
        'max_midpoint_z_error_um': dz/2, 'max_edge_error_bound_um': min(pitch_um, max(slopes, default=0.)*dz/2),
        'continuous_area_um2': area, 'sliced_area_um2': sliced_area, 'area_error_um2': sliced_area-area,
        'method': 'midpoint z; exact x intersections; vertex heights retained'}})


def _fillet_polygon(vertices, radii, arc_segments):
    if isinstance(arc_segments, bool) or not isinstance(arc_segments, int) or not 2 <= arc_segments <= 512:
        raise ValueError('arc_segments must be an integer between 2 and 512')
    tangent_lengths, corner_data = [], []
    for i, (vertex, radius) in enumerate(zip(vertices, radii)):
        incoming, outgoing = vertices[(i-1) % len(vertices)]-vertex, vertices[(i+1) % len(vertices)]-vertex
        incoming /= np.linalg.norm(incoming)
        outgoing /= np.linalg.norm(outgoing)
        interior = np.arccos(np.clip(incoming@outgoing, -1., 1.))
        tangent = radius/np.tan(interior/2)
        tangent_lengths.append(tangent)
        bisector = incoming+outgoing
        center = vertex+bisector/np.linalg.norm(bisector)*radius/np.sin(interior/2)
        corner_data.append((vertex+incoming*tangent, vertex+outgoing*tangent, center, np.pi-interior))
    for i in range(len(vertices)):
        following = (i+1) % len(vertices)
        if tangent_lengths[i]+tangent_lengths[following] >= np.linalg.norm(vertices[following]-vertices[i]) and (radii[i] or radii[following]):
            raise ValueError('corner radii overlap along an edge; reduce top/bottom radius')
    points, chord_error = [], 0.
    for vertex, radius, (start, end, center, turn) in zip(vertices, radii, corner_data):
        if radius == 0:
            points.append(vertex.copy())
            continue
        angle = np.arctan2((start-center)[1], (start-center)[0])
        samples = np.linspace(angle, angle+turn, arc_segments+1)
        arc = center+radius*np.column_stack((np.cos(samples), np.sin(samples)))
        arc[0], arc[-1] = start, end
        points.extend(arc)
        chord_error = max(chord_error, radius*(1-np.cos(turn/(2*arc_segments))))
    points = np.asarray(points)
    points[np.abs(points[:, 1]) < 1e-14*max(np.max(vertices[:, 1]), 1.), 1] = 0.
    return np.vstack((points, points[0])), float(chord_error)


def trapezoid(pitch_um=8., depth_um=.12, duty=.5, left_angle_deg=80.,
              right_angle_deg=80., slices=12, ridge='Si3N4', groove='air',
              top='air', bottom='glass_demo', top_radius_um=0., bottom_radius_um=0.,
              arc_segments=16):
    pitch_um, depth_um = positive(pitch_um, 'pitch_um'), positive(depth_um, 'depth_um')
    if not np.isfinite(duty) or not 0 < duty < 1:
        raise ValueError('duty must lie strictly between 0 and 1')
    for name, angle in (('left_angle_deg', left_angle_deg), ('right_angle_deg', right_angle_deg)):
        if not np.isfinite(angle) or not 0 < angle <= 90:
            raise ValueError(f'{name} must lie in (0,90], measured from the horizontal')
    top_radius_um = positive(top_radius_um, 'top_radius_um', allow_zero=True)
    bottom_radius_um = positive(bottom_radius_um, 'bottom_radius_um', allow_zero=True)
    left, right = pitch_um*(1-duty)/2, pitch_um*(1+duty)/2
    left_bottom = left-depth_um/np.tan(np.radians(left_angle_deg))
    right_bottom = right+depth_um/np.tan(np.radians(right_angle_deg))
    if left_bottom < 0 or right_bottom > pitch_um:
        raise ValueError('trapezoid bottom crosses the cell boundary; reduce depth/duty or increase angle')
    vertices = np.array([[left, 0.], [right, 0.], [right_bottom, depth_um], [left_bottom, depth_um]])
    chord_error = 0.
    if top_radius_um or bottom_radius_um:
        points, chord_error = _fillet_polygon(vertices, [top_radius_um, top_radius_um, bottom_radius_um, bottom_radius_um], arc_segments)
    else:
        points = np.vstack((vertices, vertices[0]))
    stack = from_polygon(points, pitch_um, slices, ridge, groove, top, bottom)
    return replace(stack, metadata={**stack.metadata,
        'kind': 'rounded_trapezoid' if top_radius_um or bottom_radius_um else 'trapezoid',
        'duty': float(duty), 'top_cd_um': float(duty*pitch_um),
        'top_cd_definition': 'unfilleted continuous trapezoid width at z=0',
        'left_angle_deg': float(left_angle_deg), 'right_angle_deg': float(right_angle_deg),
        'top_radius_um': top_radius_um, 'bottom_radius_um': bottom_radius_um,
        'arc_segments': arc_segments, 'arc_chord_error_bound_um': chord_error,
        'rounding_method': 'circular tangent fillets polygonized into chords, then midpoint z slicing'})


def rounded_trapezoid(*args, top_radius_um=.015, bottom_radius_um=.015, **kwargs):
    return trapezoid(*args, top_radius_um=top_radius_um, bottom_radius_um=bottom_radius_um, **kwargs)


def refine_slices(stack, factor=2):
    """Regenerate continuous intersections, never duplicate coarse layers."""
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 2:
        raise ValueError('refinement factor must be an integer >=2')
    if 'profile_recipe' not in stack.metadata:
        raise ValueError('no continuous profile recipe available; regenerate composed geometry explicitly')
    recipe = {**stack.metadata['profile_recipe']}
    recipe['slices'] = _slice_count(recipe['slices']*factor)
    refined = from_polygon(**recipe)
    info = {key: value for key, value in stack.metadata.items() if key not in ('staircase', 'profile_recipe', 'translation_um')}
    refined = replace(refined, metadata={**refined.metadata, **info})
    delta = stack.metadata.get('translation_um', 0.)
    if delta:
        from .operations import translate
        refined = translate(refined, delta)
    return refined


def staircase_report(stack):
    return dict(stack.metadata.get('staircase', {'actual_slices': len(stack.layers),
                     'max_dz_um': max((layer.thickness_um for layer in stack.layers), default=0.),
                     'max_midpoint_z_error_um': None,
                     'note': 'no continuous profile for an approximation error estimate'}))
