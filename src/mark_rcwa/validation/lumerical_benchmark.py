"""Reproduce the supplied Lumerical cases with PH; never run or edit Lumerical.

Compare complex power-normalized matrices in the reference's stored local
bases at its declared planes and the physical nominal x=0 origin. No fitted
phase, conjugation, order permutation, or parameter fitting is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from mark_rcwa.materials import Material, Materials
from mark_rcwa.model import Slice, Stack
from mark_rcwa.solvers import MeentSolver, Simulation
from mark_rcwa.desktop_project import reference_result

_PACKAGE = Path(__file__).parents[1]
# Capture provenance once per worker, alongside its imported implementation.
# Files must remain unchanged while a benchmark process is running.
SOURCE_HASHES = {str(path.relative_to(_PACKAGE)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (Path(__file__), _PACKAGE/'solvers'/'meent_adapter.py',
                              _PACKAGE/'solvers'/'conical.py', _PACKAGE/'model.py',
                              _PACKAGE/'materials'/'__init__.py', _PACKAGE/'desktop_project.py')}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def stack_from_geometry(geometry):
    """Exact disjoint benchmark boxes, no raster/shape approximation."""
    pitch = float(geometry["period_x_nm"]) / 1000
    xmin = -pitch / 2
    materials = Materials()
    def material(n, k=0):
        value = complex(n, k)
        name = f"n_{value.real:g}_k_{value.imag:g}"
        materials.setdefault(name, Material(name, value, source="explicit Lumerical benchmark constant"))
        return name
    top = material(geometry["superstrate_n"])
    bottom = material(geometry["substrate"]["n"])
    boxes = geometry["boxes"]
    zedges = sorted({0., *(float(b[k])/1000 for b in boxes for k in ("z_min_nm", "z_max_nm"))}, reverse=True)
    layers = []
    for high, low in zip(zedges, zedges[1:]):
        midz = (high + low)/2
        active = [b for b in boxes if b["z_min_nm"]/1000 < midz < b["z_max_nm"]/1000]
        xedges = sorted({xmin, -xmin, *(float(b[k])/1000 for b in active for k in ("x_min_nm", "x_max_nm"))})
        names = []
        for left, right in zip(xedges, xedges[1:]):
            midx = (left + right)/2
            found = [b for b in active if b["x_min_nm"]/1000 < midx < b["x_max_nm"]/1000]
            if len(found) > 1:
                raise ValueError("Overlapping benchmark boxes require an explicit priority")
            names.append(material(found[0]["n"], found[0].get("k", 0)) if found else top)
        edges = [0.]
        merged = []
        for j, name in enumerate(names):
            right = pitch if j == len(names)-1 else xedges[j+1]-xmin
            if merged and name == merged[-1]:
                edges[-1] = right
            else:
                merged.append(name)
                edges.append(right)
        layers.append(Slice(high-low, tuple(edges), tuple(merged)))
    stack = Stack(pitch, tuple(layers), top, bottom,
                  metadata={"top_z_um": max(zedges), "x_origin_um": xmin, "bottom_z_um": 0.,
                            "source": "geometry_top_incidence.json; exact boxes"})
    return stack, materials


def scalar_matrix(solver, stack, materials, wavelength, theta, order, reference_z, transmission_z):
    """PH fixed lab basis -> Lumerical per-order local basis (depth +z)."""
    count = 2*order+1
    reflection, transmission = np.zeros((count, 2, 2), complex), np.zeros((count, 2, 2), complex)
    power_r, power_t, diagnostics = [], [], []
    for col, pol in enumerate(("TE", "TM")):
        result = solver.solve(stack, materials, Simulation(wavelength_um=wavelength/1000,
                              theta_deg=theta, polarization=pol, fourier_order=order, na=0))
        diagnostics.append({key: getattr(result, key, None) for key in ("branch_corrections", "cutoff_metadata")})
        result = reference_result(result, stack, reference_z).translated(stack.metadata["x_origin_um"])
        sign = np.where(result.kx.real < -1e-12, -1., 1.)
        reflection[:, col, col] = result.amplitude * sign * (1 if col == 0 else -1)
        nt = materials.index(stack.top, wavelength/1000).real
        nb = materials.index(stack.bottom, wavelength/1000)
        k0 = 2*np.pi/(wavelength/1000)
        from mark_rcwa.solvers.meent_adapter import outgoing_sqrt
        kz_bottom = outgoing_sqrt((nb*k0)**2-result.kx**2)
        transmission[:, col, col] = result.transmission_amplitude * sign * np.exp(1j*kz_bottom*(-transmission_z))
        power_r.append(result.reflectance)
        power_t.append(result.transmittance)
    return {"R": reflection, "T": transmission, "power_R": power_r, "power_T": power_t,
            "diagnostics": diagnostics}


def conical_matrix(stack, materials, wavelength, theta, phi, order, reference_z, transmission_z):
    from mark_rcwa.solvers.conical import solve_conical
    result = solve_conical(stack, materials, wavelength/1000, theta, phi, order)
    k0 = 2*np.pi/(wavelength/1000)
    incoming_kz = materials.index(stack.top, wavelength/1000).real*k0*np.cos(np.radians(theta))
    distance = reference_z-stack.metadata["top_z_um"]
    origin = np.exp(-2j*np.pi*result.orders*stack.metadata["x_origin_um"]/stack.pitch_um)
    r = result.R*(origin*np.exp(1j*(result.kz_top+incoming_kz)*distance))[:,None,None]
    t = result.T*(origin*np.exp(1j*incoming_kz*distance+1j*result.kz_bottom*(-transmission_z)))[:,None,None]
    return {"R": r, "T": t,
            "power_R": result.reflected_efficiency.sum(axis=(0,1)).tolist(),
            "power_T": result.transmitted_efficiency.sum(axis=(0,1)).tolist(),
            "diagnostics": [{"branch_corrections": result.branch_corrections}]}


def metrics(prediction, reference, mask):
    delta = abs(prediction-reference)
    selected = delta[mask]
    if not np.isfinite(selected).all():
        raise ValueError("Nonfinite prediction on a reference propagating order")
    threshold = .002 + .01*abs(reference[mask])
    return {"max_complex_error": float(np.max(selected)),
            "rms_complex_error": float(np.sqrt(np.mean(selected**2))),
            "within_absolute_0p002": bool(np.all(selected <= .002)),
            "passed": bool(np.all(selected <= threshold)),
            "max_power_error": float(np.max(abs(abs(prediction[mask])**2-abs(reference[mask])**2))),
            "channels": {f"{out}_out_{inc}_in": float(np.max(delta[..., row, col][mask]))
                         for row, out in enumerate("sp") for col, inc in enumerate("sp")}}


def run_case(task):
    base, output, case, rays, wavelengths, force = task
    base, output = Path(base), Path(output)
    cid = case["case_id"]
    geometry = read_json(base/"inputs"/"cases"/cid/"geometry_top_incidence.json")
    stack, materials = stack_from_geometry(geometry)
    inputs = read_json(base/case["run_input_file"])
    source_hashes = {Path(name).name: digest for name, digest in SOURCE_HASHES.items()}
    reference_paths = [base/'inputs'/'cases'/cid/'geometry_top_incidence.json',
                       base/case['run_input_file'], base/case['reflection_matrix_file'],
                       (base/case['reflection_matrix_file']).with_name('transmission_matrix.npz'),
                       base/case['point_quality_file']]
    reference_hashes = {str(path.relative_to(base)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in reference_paths}
    signature = hashlib.sha256(json.dumps({'source': SOURCE_HASHES, 'reference': reference_hashes},
                                          sort_keys=True).encode()).hexdigest()
    folder = output/cid
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder/"geometry_used.json", geometry)
    with (base/case["point_quality_file"]).open(encoding="utf-8-sig", newline="") as stream:
        quality = list(csv.DictReader(stream))
    solver = MeentSolver(cache_bytes=128*1024**2)
    with np.load(base/case["reflection_matrix_file"]) as rfile, np.load((base/case["reflection_matrix_file"]).with_name("transmission_matrix.npz")) as tfile:
        ref = {key: rfile[key] for key in rfile.files}
        tref = {key: tfile[key] for key in tfile.files}
    order = int(max(abs(ref["order_n"])))
    completed = []
    for wi, wl in enumerate(ref["wavelength_nm"]):
        if wavelengths and not any(abs(wl-w)<1e-6 for w in wavelengths):
            continue
        for ai in rays:
            theta, phi = float(ref["theta_deg"][ai]), float(ref["phi_deg"][ai])
            stem = f"{wl:.6f}nm_ray{ai}"
            target = folder/(stem+".json")
            if target.is_file() and not force:
                previous = read_json(target)
                if (previous.get("status") != "error" and previous.get('run_signature') == signature
                        and target.with_suffix('.npz').is_file()):
                    completed.append(previous)
                    continue
            q = next(q for q in quality if abs(float(q["wavelength_nm"])-wl)<1e-6 and abs(float(q["theta"])-theta)<1e-8 and abs(float(q["phi"])-phi)<1e-8)
            row = {"case_id": cid, "wavelength_nm": float(wl), "theta_deg": theta, "phi_deg": phi,
                   "ray_index": ai, "fourier_order": order, "reference_level": case["selected_level"],
                   "reference_ready": q["R_x0_reference_ready"].lower()=="true",
                   "reference_R_change": float(q["max_R_x0_complex_change"]),
                   "reference_energy_error": max(float(q["energy_error_s"]), float(q["energy_error_p"])),
                   "solver_source_sha256": source_hashes, 'reference_source_sha256': reference_hashes,
                   'run_signature': signature}
            start = time.monotonic()
            try:
                reference_z = -inputs["reflection_reference_z_nm"]/1000
                transmission_z = -inputs["transmission_reference_z_nm"]/1000
                if abs(phi)>1e-12:
                    solved = conical_matrix(stack, materials, float(wl), theta, phi, order, reference_z, transmission_z)
                else:
                    solved = scalar_matrix(solver, stack, materials, float(wl), theta, order, reference_z, transmission_z)
                E, T = solved["R"], solved["T"]
                mask = ref["propagating_mask"][wi, ai, :, 0]
                tmask = tref["propagating_mask"][wi, ai, :, 0]
                S = E*ref["power_normalization_factor"][wi, ai, :, 0, None, None]
                TS = T*tref["power_normalization_factor"][wi, ai, :, 0, None, None]
                rmetrics = metrics(S, ref["S_x0"][wi, ai, :, 0], mask)
                emetrics = metrics(E, ref["E_x0"][wi, ai, :, 0], mask)
                tmetrics = metrics(TS, tref["S_x0"][wi, ai, :, 0], tmask)
                # Independent raw-grid TE reproduction establishes that the
                # supplied S_x0/E_x0 retain a half-period origin offset. Keep
                # the literal comparison alongside the fixed conversion.
                # See scripts/verify_lumerical_origin.py; never fit phase.
                origin_factor = np.exp(2j*np.pi*ref["order_n"]*stack.metadata["x_origin_um"]/stack.pitch_um)[:,None,None]
                cell_r = metrics(S*origin_factor, ref["S_x0"][wi, ai, :, 0], mask)
                cell_e = metrics(E*origin_factor, ref["E_x0"][wi, ai, :, 0], mask)
                cell_t = metrics(TS*origin_factor, tref["S_x0"][wi, ai, :, 0], tmask)
                reference_power_r = np.sum(abs(ref["S_x0"][wi, ai, mask, 0])**2, axis=(0, 1))
                reference_power_t = np.sum(abs(tref["S_x0"][wi, ai, tmask, 0])**2, axis=(0, 1))
                energy = max(abs(1-np.array(solved["power_R"])-np.array(solved["power_T"])))
                status = ("reference_limited" if not row["reference_ready"] else
                          "pass" if cell_r["within_absolute_0p002"] else "mismatch")
                row.update(status=status,
                           R_S=rmetrics, R_E=emetrics, T_S=tmetrics,
                           cell_origin_R_S=cell_r, cell_origin_R_E=cell_e, cell_origin_T_S=cell_t,
                           reference_origin_offset_nm=stack.metadata["x_origin_um"]*1000,
                           reference_origin_conversion='physical_reference = supplied_x0 * (-1)**order_n',
                           phase_fit=False,
                           max_total_R_error=float(max(abs(reference_power_r-solved["power_R"]))),
                           max_total_T_error=float(max(abs(reference_power_t-solved["power_T"]))),
                           energy_error=float(energy), diagnostics=solved["diagnostics"])
                np.savez_compressed(folder/(stem+".npz"), R_E=E, R_S=S, T_E=T, T_S=TS,
                                    cell_origin_R_E=E*origin_factor, cell_origin_R_S=S*origin_factor,
                                    cell_origin_T_E=T*origin_factor, cell_origin_T_S=TS*origin_factor,
                                    orders=ref["order_n"], wavelength_nm=wl, theta_deg=theta, phi_deg=phi)
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            row["elapsed_seconds"] = time.monotonic()-start
            write_json(target, row)
            print(f"{cid} {wl:g}nm ({theta:g},{phi:g}) {row['status']} " + (f"physical-origin R error={row['cell_origin_R_S']['max_complex_error']:.3g}" if "R_S" in row else row.get("error", "")), flush=True)
            completed.append(row)
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--rays", nargs="+", type=int, default=[0,1,2])
    parser.add_argument("--wavelengths", nargs="*", type=float)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    index = read_json(args.reference/"benchmark_index.json")
    cases = [c for c in index["cases"] if not args.cases or c["case_id"] in args.cases]
    tasks = [(str(args.reference.resolve()), str(args.output.resolve()), c, args.rays, args.wavelengths, args.force) for c in cases]
    results = []
    if args.jobs == 1:
        for task in tasks:
            results.extend(run_case(task))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for future in as_completed([pool.submit(run_case, task) for task in tasks]):
                results.extend(future.result())
    summary = {"points": len(results), "counts": {status: sum(r["status"]==status for r in results) for status in ("pass", "mismatch", "reference_limited", "error")},
               "atol": .002, "rtol": 0., "reference_quantity": "S_x0 * (-1)**order_n", "phase_fit": False,
               "results": sorted(results, key=lambda r:(r["case_id"], r["wavelength_nm"], r["ray_index"]))}
    write_json(args.output/"run_summary.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="results"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
