"""Read-only GDS selection helpers for the desktop import wizard.

A layout bounding box is only a starting value for the cell controls.  Neither
these helpers nor a GDS file can establish the physical repetition period or
justify reducing a two-dimensional pattern to a one-dimensional cross section.
The final geometry must still pass ``geometry_to_stack``'s y-invariance check.
"""
from __future__ import annotations

import hashlib
import math
from uuid import uuid4

import numpy as np

from .geometry import _gds, read_gds_polygons
from .paths import resolve_input_path


def _bounds(bounds, name="bounds_um"):
    try:
        result = np.asarray(bounds, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain two finite increasing intervals") from None
    if (result.shape != (2, 2) or not np.isfinite(result).all()
            or np.any(result[:, 1] <= result[:, 0])):
        raise ValueError(f"{name} must contain two finite increasing intervals")
    return result.tolist()


def _finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def inspect_layout(path) -> dict:
    """List selectable cell/layer/datatype combinations in micrometres.

    References, reference arrays, repetitions, and path outlines are included
    recursively for every cell; the source file is never flattened or written.
    """
    path = resolve_input_path(path).resolve()
    library = _gds().read_gds(path)
    factor = library.unit / 1e-6
    cells = []
    for cell in library.cells:
        groups = {}
        polygons = cell.get_polygons(apply_repetitions=True, include_paths=True, depth=None)
        for polygon in polygons:
            key = (int(polygon.layer), int(polygon.datatype))
            group = groups.setdefault(key, [])
            group.append(polygon)
        layers = []
        for (layer, datatype), polygons in sorted(groups.items()):
            boxes = np.asarray([polygon.bounding_box() for polygon in polygons], dtype=float) * factor
            lower, upper = boxes[:, 0, :].min(axis=0), boxes[:, 1, :].max(axis=0)
            layers.append({"layer": layer, "datatype": datatype,
                           "polygon_count": len(polygons),
                           "bounds_um": [[float(lower[0]), float(upper[0])],
                                         [float(lower[1]), float(upper[1])]]})
        cells.append({"name": cell.name, "layers": layers})
    return {"path": str(path), "unit_m": library.unit, "precision_m": library.precision,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "top_cells": [cell.name for cell in library.top_level()], "cells": cells}


def propose_cell(bounds_um) -> dict:
    """Suggest editable cell bounds; the caller must confirm the true period."""
    bounds = _bounds(bounds_um)
    return {"period_um": bounds[0][1] - bounds[0][0],
            "cell_bounds_um": [bounds[0], bounds[1], [-.5, .5]]}


def make_gds_step(path, *, cell, layer, datatype, name="GDS图案", material,
                  z_min_um=0, thickness_um=.12, roi_um=None,
                  delta_x_um=0, delta_y_um=0, operation="add",
                  target_materials=None) -> dict:
    """Validate a selection and retain its GDS reference as a process mask.

    ROI is expressed in the original layout coordinates, before translating
    the selected polygons.  Coordinates and the original GDS are preserved;
    the normal geometry builder subsequently clips/wraps into the chosen cell.
    For etch, ``z_min_um`` is the etch start surface and ``thickness_um`` is
    the downward etch depth.  For add/delete they specify bottom and height.
    """
    if operation not in ("add", "etch", "delete"):
        raise ValueError("GDS operation must be add, etch or delete")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("GDS step name must be nonempty")
    if not isinstance(material, str) or not material.strip():
        raise ValueError("GDS material must be a nonempty identifier")
    z_min = _finite(z_min_um, "z_min_um")
    thickness = _finite(thickness_um, "thickness_um")
    if thickness <= 0:
        raise ValueError("thickness_um must be positive")
    z_start = z_min
    if operation == "etch":
        z_max = z_min
        z_min = _finite(z_min - thickness, "z_min_um")
    else:
        z_max = _finite(z_min + thickness, "z_max_um")
    dx = _finite(delta_x_um, "delta_x_um")
    dy = _finite(delta_y_um, "delta_y_um")
    roi = None if roi_um is None else _bounds(roi_um, "roi_um")
    targets = None
    if operation in ("etch", "delete"):
        if (not isinstance(target_materials, (list, tuple)) or not target_materials
                or any(not isinstance(item, str) or not item.strip() for item in target_materials)):
            raise ValueError("Etch/delete requires an explicit nonempty target_materials list")
        targets = list(dict.fromkeys(target_materials))
    polygons, source = read_gds_polygons(path, cell=cell, layer=layer, datatype=datatype,
                                       roi_um=roi)
    if not polygons or not any(polygon.area() > 0 for polygon in polygons):
        raise ValueError("Selected GDS cell/layer/datatype/ROI contains no polygons")
    step = {"id": "gds_" + uuid4().hex[:12], "name": name, "operation": operation,
            "material": material, "enabled": True,
            "z_min_um": z_min, "z_max_um": z_max, "thickness_um": thickness,
            "sidewall_angle_deg": 90, "cd_bias_um": 0., "rotation_deg": 0.,
            "delta_x_um": dx, "delta_y_um": dy, "mask_polarity": "positive",
            "mask": {"gds_path": source["path"], "cell": cell, "layer": layer,
                     "datatype": datatype, "roi_um": roi}}
    if targets is not None:
        step["target_materials"] = targets
    if operation == "etch":
        step.update(start_z_um=z_start, etch_depth_um=thickness)
    return step
