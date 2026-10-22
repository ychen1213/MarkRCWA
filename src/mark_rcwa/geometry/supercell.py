"""Periodically repeated multi-cell structures with exact common z slicing."""
from numbers import Integral
import numpy as np
from mark_rcwa.model import Slice, Stack
from .profiles import _merge_slice


def supercell(cells):
    """Join original-period cells at shared top z=0 and a common deepest bottom.

    A shorter cell is continued with its substrate material. The resulting
    structure is periodic with the entire supercell, not an isolated mark.
    """
    try:
        cells = tuple(cells)
    except TypeError as exc:
        raise ValueError('supercell cells must be a nonempty sequence of Stack objects') from exc
    if not cells or any(not isinstance(cell, Stack) for cell in cells):
        raise ValueError('supercell cells must be a nonempty sequence of Stack objects')
    original = cells[0]
    for cell in cells:
        if cell.original_pitch_um is not None and cell.original_pitch_um != cell.pitch_um:
            raise ValueError('nested supercells are unsupported; supply individual original-period cells')
        if cell.pitch_um != original.pitch_um:
            raise ValueError('all supercell cells must have the same original pitch_um')
        if cell.top != original.top or cell.bottom != original.bottom:
            raise ValueError('all supercell cells must share top and bottom materials')
    count = len(cells)
    pitch_um = count*original.pitch_um
    cell_z_edges = [np.cumsum([0., *[layer.thickness_um for layer in cell.layers]]) for cell in cells]
    # Do not quantize geometry depths or replace small layers with a fixed z raster.
    z_edges = np.unique(np.concatenate(cell_z_edges))
    layers = []
    for lower, upper in zip(z_edges[:-1], z_edges[1:]):
        edges, ids = [0.], []
        for cell_index, (cell, z_cell) in enumerate(zip(cells, cell_z_edges)):
            # Each interval lies wholly inside an original slice (or substrate).
            # Search at its lower boundary avoids midpoint roundoff in very thin intervals.
            layer_index = int(np.searchsorted(z_cell, lower, side='right')-1)
            offset = cell_index*original.pitch_um
            if layer_index >= len(cell.layers):
                source = Slice(float(upper-lower), (0., original.pitch_um), (cell.bottom,))
            else:
                source = cell.layers[layer_index]
            edges.extend(float(offset+right) for right in source.edges_um[1:])
            ids.extend(source.materials)
        edges[-1] = pitch_um
        layers.append(_merge_slice(Slice(float(upper-lower), tuple(edges), tuple(ids))))
    metadata = {
        'kind': 'supercell', 'cell_count': count, 'original_pitch_um': original.pitch_um,
        'computational_pitch_um': pitch_um, 'exact_lateral_edges': True,
        'periodic_repetition_assumption': 'the complete supercell repeats infinitely; not an isolated finite mark',
        'depth_alignment': 'all cells share top z=0; below a shorter cell use its common substrate material',
        'cell_depths_um': [float(z[-1]) for z in cell_z_edges],
        'z_edges_um': z_edges.tolist(),
        'source_cells': [{'kind': cell.metadata.get('kind', 'explicit_layers'),
                          'pitch_um': cell.pitch_um, 'layer_count': len(cell.layers),
                          'metadata': dict(cell.metadata)} for cell in cells],
        'staircase': {'actual_slices': len(layers),
                      'max_dz_um': float(np.diff(z_edges).max()) if len(z_edges) > 1 else 0.,
                      'max_midpoint_z_error_um': None,
                      'note': 'source cells already sliced; z union adds no approximation to those slices; continuous-profile error is inherited'},
    }
    return Stack(pitch_um, tuple(layers), original.top, original.bottom,
                 original_pitch_um=original.pitch_um, metadata=metadata)


def order_mapping(stack, orders):
    """Describe how supercell orders relate to integer original-period orders.

    kx_grating_per_um is the order-dependent kx increment; add incident kx for
    its physical direction. Fractions are not additional original-period orders.
    """
    if not isinstance(stack, Stack) or stack.original_pitch_um is None:
        raise ValueError('order_mapping requires a supercell Stack with original_pitch_um')
    try:
        orders = tuple(orders)
    except TypeError as exc:
        raise ValueError('orders must be an integer sequence') from exc
    if any(isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral) for m in orders):
        raise ValueError('orders must be integers')
    count = round(stack.pitch_um/stack.original_pitch_um)
    return [{'supercell_order': int(m), 'original_order_fraction': float(m/count),
             'original_order': int(m//count) if m % count == 0 else None,
             'kx_grating_per_um': float(2*np.pi*m/stack.pitch_um)} for m in orders]
