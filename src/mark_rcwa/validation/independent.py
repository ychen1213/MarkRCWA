"""Executed independent complex-field validation against external S4.

All tolerances and the phase reliability floor are set before running cases.
Matching two implementations at a fixed truncation is separate from proving
Fourier or geometrical staircase convergence.
"""
from __future__ import annotations

from dataclasses import replace
import re

import numpy as np

from mark_rcwa.geometry import rectangle, trapezoid, rounded_trapezoid, add_film, supercell
from mark_rcwa.materials import Material, Materials, defaults
from mark_rcwa.model import Stack
from mark_rcwa.solvers.meent_adapter import MeentSolver, Simulation
from .s4_reference import S4ReferenceSolver, S4_VERSION

PHASE_EFFICIENCY_FLOOR = 1e-8
REGULARIZATION_K = 1e-12
REGULARIZATION_LIMIT_K = 1e-11
TOLERANCES = {
    "max_complex_amplitude": 1e-9,
    "max_phase_deg": 0.1,
    "max_efficiency": 1e-6,
    "max_pair_apd_nm": 1.0,
    "regularization_complex_difference": 1e-8,
    "effective_vs_physical_complex_difference": 1e-8,
    "passive_energy_excess": 1e-7,
}


def _record(name, parameters, expected=None):
    return {
        "name": name, "parameters": parameters,
        "solver": f"Meent 0.13.2 adapter versus S4 {S4_VERSION}",
        "expected": expected or {"complex_fields_agree": True, "phase_floor_efficiency": PHASE_EFFICIENCY_FLOOR,
                                 "reference_plane_um": 0.0, "phase_convergence_proven": False},
        "actual": {}, "error": {}, "tolerances": dict(TOLERANCES),
        "status": "FAIL", "reason": "not yet executed",
    }


def _effective_problem(stack, registry, metadata):
    effective_registry = Materials(registry)
    names = {}
    for index, (name, values) in enumerate(metadata["effective_internal_indices"].items()):
        effective_name = f"s4_effective_{index}_{name}"
        effective_registry[effective_name] = Material(effective_name, complex(*values))
        names[name] = effective_name
    effective_stack = replace(stack, layers=tuple(
        replace(layer, materials=tuple(names[name] for name in layer.materials)) for layer in stack.layers))
    return effective_stack, effective_registry


def _complex_rows(result):
    return [{"order": int(m), "amplitude_real": float(a.real), "amplitude_imag": float(a.imag),
             "efficiency": float(efficiency)}
            for m, a, efficiency in zip(result.orders, result.amplitude, result.efficiency)]


def _compare(name, stack, registry, sim, reference):
    parameters = {
        "pitch_um": stack.pitch_um, "original_pitch_um": stack.original_pitch_um,
        "wavelength_um": sim.wavelength_um, "theta_deg": sim.theta_deg,
        "polarization": sim.polarization, "fourier_order": sim.fourier_order,
        "na": sim.na, "regularization_k": REGULARIZATION_K,
        "regularization_limit_k": REGULARIZATION_LIMIT_K,
        "phase_efficiency_floor": PHASE_EFFICIENCY_FLOOR,
        "top": stack.top, "bottom": stack.bottom,
        "layers": [{"thickness_um": layer.thickness_um, "edges_um": list(layer.edges_um),
                    "materials": list(layer.materials)} for layer in stack.layers],
        "geometry_metadata": stack.metadata,
    }
    record = _record(name, parameters)
    try:
        first = reference.solve(stack, registry, sim, regularization_k=REGULARIZATION_K)
        second = reference.solve(stack, registry, sim, regularization_k=REGULARIZATION_LIMIT_K)
        physical = MeentSolver().solve(stack, registry, sim)
        effective_stack, effective_registry = _effective_problem(stack, registry, first.metadata)
        actual = MeentSolver().solve(effective_stack, effective_registry, sim)
        np.testing.assert_array_equal(actual.orders, first.orders)
        bright = (actual.propagating & first.propagating &
                  (actual.efficiency >= PHASE_EFFICIENCY_FLOOR) &
                  (first.efficiency >= PHASE_EFFICIENCY_FLOOR))
        phase_errors = np.abs(np.degrees(np.angle(first.amplitude[bright]*actual.amplitude[bright].conj())))
        pair_errors = []
        for order in actual.orders[actual.orders > 0]:
            ip = np.flatnonzero(actual.orders == order)[0]
            im = np.flatnonzero(actual.orders == -order)[0]
            if bright[ip] and bright[im] and actual.collected[ip] and actual.collected[im] and first.collected[ip] and first.collected[im]:
                pair_ref = first.amplitude[ip]*first.amplitude[im].conjugate()
                pair_actual = actual.amplitude[ip]*actual.amplitude[im].conjugate()
                difference = abs(np.angle(pair_ref*pair_actual.conjugate()))
                pair_errors.append({"order": int(order), "error_nm": float(difference*stack.pitch_um/(4*np.pi*order)*1000)})
        errors = {
            "max_complex_amplitude": float(np.max(abs(first.amplitude-actual.amplitude))),
            "max_phase_deg": float(np.max(phase_errors)) if len(phase_errors) else 0.0,
            "max_efficiency": float(np.max(abs(first.efficiency-actual.efficiency))),
            "max_pair_apd_nm": max((pair["error_nm"] for pair in pair_errors), default=0.0),
            "regularization_complex_difference": float(np.max(abs(first.amplitude-second.amplitude))),
            "effective_vs_physical_complex_difference": float(np.max(abs(actual.amplitude-physical.amplitude))),
            "passive_energy_excess": float(max(0.0, first.reflectance+first.transmittance-1.0)),
        }
        record["parameters"]["physical_internal_indices"] = first.metadata["physical_internal_indices"]
        record["parameters"]["effective_internal_indices"] = first.metadata["effective_internal_indices"]
        record["actual"] = {
            "meent_physical_R": physical.reflectance, "meent_physical_T": physical.transmittance,
            "meent_effective_R": actual.reflectance, "meent_effective_T": actual.transmittance,
            "s4_effective_R": first.reflectance, "s4_effective_T": first.transmittance,
            "compared_phase_orders": actual.orders[bright].astype(int).tolist(),
            "phase_excluded_orders": actual.orders[~bright].astype(int).tolist(),
            "pair_apd_errors": pair_errors, "s4_first_metadata": first.metadata,
            "s4_limit_metadata": second.metadata,
            "s4_reflected_channels": _complex_rows(first),
            "meent_effective_reflected_channels": _complex_rows(actual),
            "comparison_scope": "fixed N, exact lateral intervals, specified z staircase; convergence not inferred",
        }
        record["error"] = errors
        failures = [f"{key}={value:.6g} exceeds {TOLERANCES[key]:.6g}" for key, value in errors.items()
                    if value > TOLERANCES[key]]
        if not bright.any():
            failures.append("no reliable propagating phase channel exists for this fixture")
        if not pair_errors:
            failures.append("no reliable collected +/- pair exists for APD comparison")
        record["status"] = "FAIL" if failures else "PASS"
        record["reason"] = "; ".join(failures) if failures else "executed complex-field, phase, efficiency, APD and explicit regularization-limit comparisons"
    except (ValueError, OSError, AssertionError, FloatingPointError) as exc:
        record["reason"] = str(exc)
    return record


def _fixtures():
    registry = Materials({name: Material(name, n) for name, n in {
        "air": 1., "glass": 1.5, "ridge": 1.7, "film": 1.46,
        "absorber": 1.7+.08j, "substrate_loss": 3.8+.02j,
    }.items()})
    base = rectangle(depth_um=.15, duty=.37, ridge="ridge", bottom="glass")
    shapes = {
        "rectangle": base,
        "multilayer": add_film(add_film(base, .04, "film"), .07, "ridge", position="bottom"),
        "symmetric_trapezoid": trapezoid(depth_um=.15, duty=.37, left_angle_deg=72., right_angle_deg=72.,
                                         slices=8, ridge="ridge", bottom="glass"),
        "asymmetric_trapezoid": trapezoid(depth_um=.15, duty=.37, left_angle_deg=60., right_angle_deg=85.,
                                          slices=8, ridge="ridge", bottom="glass"),
        "absorbing_grating": rectangle(depth_um=.15, duty=.37, ridge="absorber", bottom="glass"),
        "absorbing_substrate": replace(base, bottom="substrate_loss"),
        "rounded_trapezoid": rounded_trapezoid(depth_um=.15, duty=.37, left_angle_deg=70., right_angle_deg=85.,
            top_radius_um=.01, bottom_radius_um=.01, slices=8, ridge="ridge", bottom="glass"),
    }
    fixtures = []
    for name, stack in shapes.items():
        for theta in (0., 20.):
            for polarization in ("TE", "TM"):
                fixtures.append((f"S4 {name} {polarization} theta={theta:g}", stack, registry,
                                 Simulation(polarization=polarization, theta_deg=theta, fourier_order=21)))
    # Both are the actual Phase6 branch regressions, not an easier substitute.
    demo = supercell([rectangle(depth_um=.12, duty=.5), rectangle(depth_um=.15, duty=.44),
                     rectangle(depth_um=.09, duty=.54)])
    regression = supercell([rectangle(depth_um=.10, duty=.45), rectangle(depth_um=.16, duty=.56),
                           rectangle(depth_um=.12, duty=.51)])
    for name, stack, order in [("demo_supercell", demo, 63), ("CD_depth_branch_regression", regression, 24)]:
        for polarization in ("TE", "TM"):
            fixtures.append((f"S4 {name} {polarization}", stack, defaults(),
                             Simulation(polarization=polarization, fourier_order=order)))
    return fixtures


def _fresnel_cases(reference):
    records = []
    for bottom in (1.5, 1.5+.2j):
        registry = Materials({"air": Material("air", 1.), "bottom": Material("bottom", bottom)})
        for theta in (0., 30.):
            for pol in ("TE", "TM"):
                sim = Simulation(polarization=pol, theta_deg=theta, fourier_order=3)
                name = f"S4 Fresnel {pol} theta={theta:g} substrate={bottom}"
                record = _record(name, {"n_top": [1., 0.], "n_bottom": [complex(bottom).real, complex(bottom).imag],
                                       "theta_deg": theta, "polarization": pol, "regularization_k": 0.0})
                try:
                    result = reference.solve(Stack(8., (), "air", "bottom"), registry, sim)
                    ct = np.cos(np.radians(theta))
                    cb = np.sqrt(1-(np.sin(np.radians(theta))/bottom)**2)
                    expected = (ct-bottom*cb)/(ct+bottom*cb) if pol == "TE" else (cb-bottom*ct)/(cb+bottom*ct)
                    actual = result.amplitude[result.orders == 0][0]
                    error = abs(actual-expected)
                    energy_error = abs(result.reflectance+result.transmittance-1)
                    record["expected"] = {"fresnel_amplitude": [float(expected.real), float(expected.imag)],
                        "boundary_energy": 1.0, "transmission_meaning": "power crossing substrate boundary, not through infinite absorbing half-space"}
                    record["actual"] = {"amplitude": [float(actual.real), float(actual.imag)],
                        "R": result.reflectance, "T_into_substrate": result.transmittance, "s4_metadata": result.metadata}
                    record["error"] = {"complex_amplitude": float(error), "boundary_energy": float(energy_error)}
                    record["tolerances"] = {"complex_amplitude": 1e-11, "boundary_energy": 1e-11}
                    record["status"] = "PASS" if error < 1e-11 and energy_error < 1e-11 else "FAIL"
                    record["reason"] = "executed analytic complex Fresnel comparison" if record["status"] == "PASS" else "analytic Fresnel tolerance exceeded"
                except (ValueError, OSError) as exc:
                    record["reason"] = str(exc)
                records.append(record)
    return records


def _known_legacy_issue(reference):
    registry = Materials({"air": Material("air", 1.), "glass": Material("glass", 1.5),
                          "ridge": Material("ridge", 1.7)})
    # A globally translated rectangle has identical power in exact theory.
    # This archived executable fails for this original exactly lossless input.
    from mark_rcwa.model import Slice
    stack = Stack(8., (Slice(.15, (0., 2.96, 8.), ("ridge", "air")),), "air", "glass")
    record = _record("S4 legacy unregularized translated-lossless limitation",
                     {"pitch_um": 8., "width_um": 2.96, "center_um": 1.48,
                      "ridge_index": [1.7, 0.], "regularization_k": 0.0, "fourier_order": 7})
    record["kind"] = "KNOWN_UPSTREAM_ISSUE"
    record["expected"] = {"scope": "diagnostic of an unsupported unregularized legacy-S4 input",
                          "physical_lossless_R_plus_T": 1.0}
    try:
        result = reference.solve(stack, registry, Simulation(fourier_order=7), regularization_k=0.0)
        record["actual"] = {"diagnostic_R_plus_T": result.reflectance+result.transmittance,
                            "s4_metadata": result.metadata, "failure_reproduced": False}
        record["reason"] = "KNOWN_UPSTREAM_ISSUE: archived lossless instability did not reproduce on this run; fixture remains separately diagnosed"
    except ValueError as exc:
        message = str(exc)
        match = re.search(r"R\+T=([^;]+)", message)
        measured = float(match.group(1)) if match else None
        record["actual"] = {"diagnostic_exception": message, "failure_reproduced": True,
                            "diagnostic_R_plus_T": measured}
        record["error"] = {"diagnostic_energy_error": abs(measured-1) if measured is not None else None}
        record["reason"] = ("KNOWN_UPSTREAM_ISSUE: executed unregularized diagnostic failed physical passivity; "
                            "this unsupported raw mode is excluded from acceptance. Explicit regularized inputs, "
                            "original/effective constants and measured perturbation limits are validated separately.")
    record["status"] = "SKIP"
    return record


def run_independent_validation(s4_executable=None):
    """Return JSON-safe evidence records; absence yields explicit SKIP.

    The known raw legacy issue is a diagnostic SKIP with its actual failure
    recorded, not an acceptance pass. All supported comparisons are executed.
    """
    try:
        reference = S4ReferenceSolver(s4_executable)
    except FileNotFoundError as exc:
        record = _record("S4 independent reference availability", {})
        record.update(status="SKIP", reason=str(exc))
        return [record]
    except (ValueError, OSError) as exc:
        record = _record("S4 independent reference integrity", {})
        record.update(status="FAIL", reason=str(exc))
        return [record]
    records = _fresnel_cases(reference)
    records.extend(_compare(name, stack, registry, sim, reference)
                   for name, stack, registry, sim in _fixtures())
    records.append(_known_legacy_issue(reference))
    return records
