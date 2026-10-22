"""Offline material/process modeling and a one-dimensional RCWA bridge.

The geometry exchange format and deterministic process rules are derived from
MarkSim. No MarkSim installation, external service or solver is required.
"""
from .geometry import (build_geometry, preview_geometry, inspect_gds,
                       read_gds_polygons, save_geometry, load_geometry,
                       validate_geometry, geometry_volume_by_material)
from .materials import (import_material_txt, import_material_folder,
                        materials_from_library, evaluate_library,
                        save_library, load_library)
from .bridge import geometry_to_stack
from .defaults import default_recipe, default_library

__all__ = [
    'build_geometry', 'preview_geometry', 'inspect_gds', 'read_gds_polygons',
    'save_geometry', 'load_geometry', 'validate_geometry',
    'geometry_volume_by_material', 'import_material_txt', 'import_material_folder',
    'materials_from_library', 'evaluate_library', 'save_library', 'load_library',
    'geometry_to_stack', 'default_recipe', 'default_library',
]
