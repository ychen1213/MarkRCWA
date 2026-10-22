"""Periodic exact-edge geometry operations and independent raster export."""
from dataclasses import replace
import numpy as np
from mark_rcwa.model import Slice, positive
from .profiles import _identifier, _merge_slice


def _translate_slice(layer, pitch_um, displacement_um):
    displacement_um = float(displacement_um % pitch_um)
    if displacement_um == 0:
        return layer
    intervals = []
    for left, right, name in zip(layer.edges_um[:-1], layer.edges_um[1:], layer.materials):
        width = right-left
        start = float((left+displacement_um) % pitch_um)
        end = start+width
        if end <= pitch_um:
            intervals.append((start, end, name))
        else:
            intervals.extend(((start, pitch_um, name), (0., end-pitch_um, name)))
    boundaries = np.unique(np.array([0., pitch_um]+[edge for left, right, _ in intervals for edge in (left, right)]))
    boundaries = boundaries[np.r_[True, np.diff(boundaries) > 1e-13*max(pitch_um, 1.)]]
    boundaries[0], boundaries[-1] = 0., pitch_um
    ids = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        mid = (left+right)/2
        candidates = [name for begin, end, name in intervals if begin <= mid < end]
        if len(candidates) != 1:
            raise ValueError('periodic translation generated inconsistent intervals')
        ids.append(candidates[0])
    return _merge_slice(Slice(layer.thickness_um, tuple(boundaries), tuple(ids)))


def translate(stack, displacement_um):
    if not np.isfinite(displacement_um):
        raise ValueError('displacement_um must be finite')
    metadata = {**stack.metadata, 'translation_um': float(stack.metadata.get('translation_um', 0.)+displacement_um)}
    return replace(stack, layers=tuple(_translate_slice(layer, stack.pitch_um, displacement_um) for layer in stack.layers), metadata=metadata)


def _composed_metadata(stack, operation):
    metadata = {**stack.metadata, 'operations': [*stack.metadata.get('operations', []), operation]}
    metadata.pop('profile_recipe', None)
    metadata['refinement_note'] = 'regenerate the generator and composition operations for z refinement'
    return metadata


def shift_layer(stack, layer_index, displacement_um):
    if isinstance(layer_index, bool) or not isinstance(layer_index, int) or not 0 <= layer_index < len(stack.layers):
        raise ValueError('layer_index must index an existing slice')
    if not np.isfinite(displacement_um):
        raise ValueError('displacement_um must be finite')
    layers = list(stack.layers)
    layers[layer_index] = _translate_slice(layers[layer_index], stack.pitch_um, displacement_um)
    return replace(stack, layers=tuple(layers), metadata=_composed_metadata(stack,
                   {'kind': 'shift_layer', 'layer_index': layer_index, 'displacement_um': float(displacement_um)}))


def add_film(stack, thickness_um, material='SiO2', position='top'):
    thickness_um = positive(thickness_um, 'film thickness_um')
    _identifier(material)
    if position not in ('top', 'bottom'):
        raise ValueError('film position must be top or bottom')
    film = Slice(thickness_um, (0., stack.pitch_um), (material,))
    layers = (film,)+stack.layers if position == 'top' else stack.layers+(film,)
    return replace(stack, layers=layers, metadata=_composed_metadata(stack,
                   {'kind': 'film', 'thickness_um': thickness_um, 'material': material, 'position': position}))


def asymmetric_cover(stack, width_um=.15, material='SiO2', side='left',
                     layer_indices=None, ridge='Si3N4'):
    """Replace a strip inside each selected ridge side; not a deposition model."""
    width_um = positive(width_um, 'cover width_um')
    for name in (material, ridge):
        _identifier(name)
    if side not in ('left', 'right'):
        raise ValueError('cover side must be left or right')
    selected = list(range(len(stack.layers))) if layer_indices is None else list(layer_indices)
    if not selected or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(stack.layers) for i in selected):
        raise ValueError('layer_indices must contain existing slice indices')
    layers, replacements = list(stack.layers), 0
    for index in selected:
        layer = layers[index]
        edges, ids = [0.], []
        for left, right, name in zip(layer.edges_um[:-1], layer.edges_um[1:], layer.materials):
            if name != ridge:
                edges.append(right)
                ids.append(name)
                continue
            if width_um >= right-left:
                raise ValueError('cover width must be less than every selected ridge width')
            split = left+width_um if side == 'left' else right-width_um
            edges.extend((split, right))
            ids.extend((material, ridge) if side == 'left' else (ridge, material))
            replacements += 1
        layers[index] = _merge_slice(Slice(layer.thickness_um, tuple(edges), tuple(ids)))
    if replacements == 0:
        raise ValueError('cover found no selected ridge material')
    return replace(stack, layers=tuple(layers), metadata=_composed_metadata(stack,
                   {'kind': 'asymmetric_cover', 'width_um': width_um, 'material': material,
                    'ridge': ridge, 'side': side, 'layer_indices': selected,
                    'model': 'internal lateral material strip; not conformal deposition'}))


def sample_grid(stack, materials, wavelength_um, nx=1024):
    """Return complex epsilon shape (layers,nx) at x=j*pitch/nx, including 0."""
    if isinstance(nx, bool) or not isinstance(nx, int) or nx < 2:
        raise ValueError('nx must be an integer >=2')
    positive(wavelength_um, 'wavelength_um')
    x = np.arange(nx)*stack.pitch_um/nx
    result = np.empty((len(stack.layers), nx), dtype=np.complex128)
    for row, layer in enumerate(stack.layers):
        indices = np.searchsorted(layer.edges_um, x, side='right')-1
        for index, name in enumerate(layer.materials):
            result[row, indices == index] = materials.index(name, wavelength_um)**2
    return result
