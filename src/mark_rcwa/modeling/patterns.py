"""One-dimensional mask patterns in a *mother-period* simulation cell.

AH geometry follows MarkSim's public ``ah-public-stripes-1.0`` templates:
AH32 has two stripes separated by P/3; AH53 three at P/5; AH74 four at
P/7.  P is the simulation period, never the internal stripe spacing.
These are geometric presets, not proprietary machine recipes.

Coordinates use micrometres. ``center_um`` shifts the complete pattern;
custom line centres are relative to that shift. Returned line centres are
folded into [-P/2, P/2). A stripe crossing that boundary is split into two
polygons, retaining its complete width. Overlap with another stripe or its
periodic copies is rejected. Touching stripes are allowed and form one mask.
"""
from __future__ import annotations

import math


_AH = {"AH32": (3, 2), "AH53": (5, 3), "AH74": (7, 4)}
_TEMPLATE_VERSION = "ah-public-stripes-1.0"
_BASE = {"kind", "period_um", "cd_um", "height_um", "center_um", "material"}
_DERIVED = {"subpitch_um", "active_width_um", "stripe_count", "target_order", "template_version"}
_MIN_FEATURE_UM = 1e-6  # Above the process builder's 1e-8 um Boolean grid.


def _kind(value):
    if not isinstance(value, str):
        raise ValueError("Pattern kind must be grating, AH32, AH53, AH74 or custom")
    normalized = value.strip().upper()
    if normalized in _AH:
        return normalized
    if normalized in ("GRATING", "CUSTOM"):
        return normalized.lower()
    raise ValueError(f"Unsupported Mark pattern: {value!r}")


def _finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive(value, name):
    result = _finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    if result < _MIN_FEATURE_UM:
        raise ValueError(f"{name} must be at least {_MIN_FEATURE_UM:g} um for reliable process geometry")
    return result


def _fold(center, period):
    # Reduce before addition, avoiding catastrophic subtraction for large shifts.
    value = center % period
    if value >= period / 2:
        value -= period
    return 0.0 if value == 0 else value


def pattern_defaults(kind="grating"):
    """Return independent editable parameters, matching MarkSim AH defaults."""
    kind = _kind(kind)
    result = {"kind": kind, "period_um": 1.0, "cd_um": .5,
              "height_um": .3, "center_um": 0.0, "material": "ridge_demo"}
    if kind in _AH:
        harmonic, _ = _AH[kind]
        result.update(period_um=16.0, cd_um=8.0 / harmonic)
    elif kind == "custom":
        result["lines"] = [{"center_um": 0.0, "width_um": .5}]
    return result


def _segments(center, width, period):
    """Clip one complete periodic stripe into the canonical cell."""
    left, right = center - width / 2, center + width / 2
    lower, upper = -period / 2, period / 2
    if left < lower:
        return [(lower, right), (left + period, upper)]
    if right > upper:
        return [(left, upper), (lower, right - period)]
    return [(left, right)]


def _validate_segments(segments, period):
    ordered = sorted(segments)
    # Allow only roundoff at an intended shared edge. Real tiny gaps cannot be
    # represented reliably by the geometry builder and are rejected explicitly.
    tolerance = 16 * math.ulp(max(period, 1.0))
    gaps = [right[0] - left[1] for left, right in zip(ordered, ordered[1:])]
    gaps.append(ordered[0][0] + period - ordered[-1][1])
    for gap in gaps:
        if gap < -tolerance:
            raise ValueError("Mark stripes overlap, including across the periodic boundary")
        if tolerance < gap < _MIN_FEATURE_UM:
            raise ValueError("Mark gap must be zero or at least 1e-6 um for reliable process geometry")


def make_pattern(spec, y_bounds=(-.5, .5)):
    """Validate a pattern and return polygons ready for a process step mask.

    The return dictionary contains ``period_um``, actual periodic ``lines``,
    ``polygons_xy_um``, normalized editable ``spec``, and a human ``summary``.
    AH derived fields may be passed back unchanged, but stale values fail.
    Custom ``lines`` require center_um and width_um; their centres are relative
    to the group shift. All stripes use the common material and height.
    """
    if not isinstance(spec, dict):
        raise ValueError("Pattern specification must be an object")
    kind = _kind(spec.get("kind", "grating"))
    allowed = _BASE | (_DERIVED if kind in _AH else set()) | ({"lines"} if kind == "custom" else set())
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"Unknown pattern parameters: {sorted(unknown)}")
    cfg = {**pattern_defaults(kind), **spec, "kind": kind}
    for name in ("period_um", "cd_um", "height_um"):
        cfg[name] = _positive(cfg[name], name)
    period = cfg["period_um"]
    if period > 1e6:
        raise ValueError("period_um exceeds the supported geometry range (1e6 um)")
    cfg["center_um"] = _fold(_finite(cfg["center_um"], "center_um"), period)
    if not isinstance(cfg["material"], str) or not cfg["material"].strip():
        raise ValueError("material must be a nonempty material identifier")
    cfg["material"] = cfg["material"].strip()
    try:
        if len(y_bounds) != 2:
            raise ValueError
        ylo, yhi = (_finite(value, "y_bounds") for value in y_bounds)
    except (TypeError, ValueError):
        raise ValueError("y_bounds must contain two finite increasing values") from None
    if yhi <= ylo or not math.isfinite(yhi - ylo) or yhi - ylo < _MIN_FEATURE_UM:
        raise ValueError("y_bounds must be increasing with at least 1e-6 um extent")

    if kind in _AH:
        harmonic, count = _AH[kind]
        pitch = period / harmonic
        if cfg["cd_um"] >= pitch:
            raise ValueError("AH CD must be smaller than the internal stripe spacing P/h, not just the mother period")
        if pitch - cfg["cd_um"] < _MIN_FEATURE_UM:
            raise ValueError("AH stripe gap must be at least 1e-6 um")
        derived = {"subpitch_um": pitch, "active_width_um": (count - 1) * pitch + cfg["cd_um"],
                   "stripe_count": count, "target_order": harmonic, "template_version": _TEMPLATE_VERSION}
        for key in _DERIVED & spec.keys():
            expected = derived[key]
            if isinstance(expected, str):
                valid = spec[key] == expected
            else:
                value = _finite(spec[key], key)
                valid = math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-10)
            if not valid:
                raise ValueError(f"Derived AH parameter {key} disagrees with mother period, CD or kind")
        cfg.update(derived)
        relative_lines = [{"center_um": (i - (count - 1) / 2) * pitch, "width_um": cfg["cd_um"]}
                          for i in range(count)]
    elif kind == "custom":
        raw = cfg["lines"]
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError("custom lines must contain at least one stripe")
        if len(raw) > 4096:
            raise ValueError("custom lines exceeds 4096 stripes")
        relative_lines = []
        for i, line in enumerate(raw):
            if not isinstance(line, dict) or set(line) != {"center_um", "width_um"}:
                raise ValueError(f"custom line {i + 1} requires center_um and width_um")
            relative_lines.append({"center_um": _fold(_finite(line["center_um"], f"line {i + 1} center_um"), period),
                                   "width_um": _positive(line["width_um"], f"line {i + 1} width_um")})
        cfg["lines"] = relative_lines
    else:
        relative_lines = [{"center_um": 0.0, "width_um": cfg["cd_um"]}]

    actual_lines, segments = [], []
    for line in relative_lines:
        width = line["width_um"]
        if width >= period:
            raise ValueError("Stripe width must be smaller than the mother period")
        center = _fold(line["center_um"] + cfg["center_um"], period)
        actual_lines.append({"center_um": center, "width_um": width})
        segments.extend(_segments(center, width, period))
    _validate_segments(segments, period)
    actual_lines.sort(key=lambda line: line["center_um"])
    polygons = [[[lo, ylo], [hi, ylo], [hi, yhi], [lo, yhi]]
                for lo, hi in sorted(segments) if hi > lo]
    detail = f"；内部条纹间距 P/{cfg['target_order']}={cfg['subpitch_um']:.6g} μm" if kind in _AH else ""
    summary = (f"{kind}：母周期 {period:.6g} μm{detail}；{len(actual_lines)} 条纹；"
               f"高度 {cfg['height_um']:.6g} μm；材料 {cfg['material']}")
    return {"period_um": period, "lines": actual_lines, "polygons_xy_um": polygons,
            "spec": cfg, "summary": summary}
