"""One-dimensional exact-edge structures, with explicit staircase approximation."""
from .profiles import (rectangle, trapezoid, rounded_trapezoid, from_polygon,
                       refine_slices, staircase_report)
from .operations import translate, shift_layer, add_film, asymmetric_cover, sample_grid
from .supercell import supercell, order_mapping
from .chevron import ChevronPlan
