"""Portable desktop projects and local RCWA jobs; no optical readout required."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import threading
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_VERSION = "mark-rcwa-desktop-1"


def write_json(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint(project):
    return hashlib.sha256(json.dumps(project, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def default_project():
    from .modeling import default_library, default_recipe
    return {"schema_version": PROJECT_VERSION, "name": "一维光栅示例", "model_kind": "process",
            "process": default_recipe(), "materials": default_library(),
            "simulation": {"wavelength_nm": 633., "theta_deg": 0., "polarizations": ["TE", "TM"], "fourier_order": 31, "convergence_max_order":127, "reference_z_um":None},
            "note": "合成示例；材料常数用于软件验证，请导入真实材料后用于工程分析。"}


def load_project(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("schema_version") != PROJECT_VERSION:
        raise ValueError("不是支持的 MarkRCWA 工程；MarkSim 工程请使用‘导入 MarkSim’。")
    if data.get("model_kind", "process") not in ("process", "profile"):
        raise ValueError("不支持的结构建模方式")
    simulation_points(data["simulation"])
    return data


def _copy_asset(source, destination):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target = destination / "assets" / (digest[:16] + source.suffix.lower())
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve() and not target.exists():
        shutil.copy2(source, target)
    return target.relative_to(destination).as_posix()


def save_project(project, path, *, base_dir):
    """Copy referenced inputs, making Save As independent of the source folder."""
    target = Path(path).resolve()
    data = copy.deepcopy(project)
    base = Path(base_dir)
    for step in data.get("process", {}).get("steps", []):
        mask = step.get("mask", {})
        if "gds_path" in mask:
            source = Path(mask["gds_path"])
            source = source if source.is_absolute() else base / source
            if not source.is_file():
                raise ValueError(f"找不到 GDS 输入：{source}")
            mask["gds_path"] = _copy_asset(source, target.parent)
    for spec in data.get("materials", {}).get("materials", {}).values():
        source = Path(spec.get("source", ""))
        source = source if source.is_absolute() else base / source
        if source.is_file():
            spec["source"] = _copy_asset(source, target.parent)
    if data.get("model_kind") == "profile":
        for spec in data.get("profile_materials", {}).values():
            if "table_csv" in spec:
                source = Path(spec["table_csv"])
                source = source if source.is_absolute() else base / source
                spec["table_csv"] = _copy_asset(source, target.parent)
    write_json(target, data)
    return data


def import_marksim(folder):
    """Read material/process data only; never launch Meep or touch originals."""
    folder = Path(folder).resolve()
    if folder.is_file():
        folder = folder.parent
    source = json.loads((folder / "project.yaml").read_text(encoding="utf-8-sig"))
    def read_reference(key):
        path = Path(source[key])
        return json.loads((path if path.is_absolute() else folder / path).read_text(encoding="utf-8-sig"))
    data = default_project()
    data.update(name=source.get("name", folder.name), process=read_reference("process"), materials=read_reference("materials"))
    data["note"] = "从 MarkSim 只读导入材料与工艺。RCWA 入射和计算精度请在本软件中设置。"
    return data, folder


def import_ph_config(path):
    from .cli import load_config
    path = Path(path).resolve()
    cfg = load_config(path)
    data = default_project()
    data.update(name=path.stem, model_kind="profile", profile=cfg.get("geometry", {}),
                profile_materials=cfg.get("materials", {}))
    sim = cfg.get("simulation", {})
    data["simulation"].update(wavelength_nm=float(sim.get("wavelength_um", .633))*1000,
                              theta_deg=sim.get("theta_deg", 0), fourier_order=sim.get("fourier_order", 31),
                              polarizations=[sim.get("polarization", "TE")])
    # The original CLI resolves tables from cwd; choose the config directory
    # for a portable desktop import and record absolute references until saved.
    for spec in data["profile_materials"].values():
        if "table_csv" in spec:
            src = Path(spec["table_csv"])
            if not src.is_absolute():
                candidate = path.parent / src
                if not candidate.is_file():
                    candidate = path.parent.parent / src
                spec["table_csv"] = str(candidate.resolve())
    data["note"] = "PH 参数截面工程；保留梯形/圆角切片。光路和 APD 配置未导入。"
    return data, path.parent


def _stack_geometry(stack):
    """Display the exact PH staircase, with the same z-up axis as process view."""
    depth = max(.5, stack.thickness_um * .4)
    solids = []
    z = 0.
    for i, layer in enumerate(stack.layers):
        for left, right, material in zip(layer.edges_um[:-1], layer.edges_um[1:], layer.materials):
            if material != stack.top:
                solids.append({"material": material, "polygon_xy_um": [[left,-.5],[right,-.5],[right,.5],[left,.5]],
                               "z_min_um": z-layer.thickness_um, "z_max_um": z, "name": f"截面层{i+1}",
                               "source_step_id": f"layer{i}", "source_step_index": i})
        z -= layer.thickness_um
    solids.append({"material": stack.bottom, "polygon_xy_um": [[0,-.5],[stack.pitch_um,-.5],[stack.pitch_um,.5],[0,.5]],
                   "z_min_um": z-depth, "z_max_um": z, "name": "半无限衬底（显示厚度）", "source_step_id": "substrate"})
    steps = [{"id": f"layer{i}", "name": f"截面层 {i+1}", "operation": "add", "enabled": True} for i in range(len(stack.layers))]
    steps.append({"id":"substrate", "name":"半无限衬底（显示厚度）", "operation":"create_substrate", "enabled":True})
    return {"schema_version":"1.0", "model_version":"ph-staircase-view-1", "representation":"parametric_solids", "units":"um",
            "period_x_um":stack.pitch_um, "invariant_y":True, "background_material":stack.top,
            "substrate_material":stack.bottom, "substrate_extend_bottom":True,
            "cell_bounds_um":[[0,stack.pitch_um],[-.5,.5],[z-depth,max(.2,stack.thickness_um*.2)]],
            "solids":solids, "provenance":{"recipe":{"steps":steps}}, "process_history":[]}


def prepare_model(project, base_dir):
    from .modeling import build_geometry, geometry_to_stack, materials_from_library
    if project.get("model_kind", "process") == "process":
        geometry = build_geometry(project["process"], base_dir=base_dir)
        stack = geometry_to_stack(geometry)
        used = {stack.top, stack.bottom} | {m for layer in stack.layers for m in layer.materials}
        library = project["materials"]
        missing = used - set(library["materials"])
        if missing:
            raise ValueError(f"缺少材料：{sorted(missing)}")
        materials = materials_from_library({**library,"materials":{name:library["materials"][name] for name in used}})
    else:
        from .cli import build_geometry as build_profile
        from .materials import from_config
        specs = copy.deepcopy(project.get("profile_materials", {}))
        for spec in specs.values():
            if "table_csv" in spec and not Path(spec["table_csv"]).is_absolute():
                spec["table_csv"] = str(Path(base_dir) / spec["table_csv"])
        materials = from_config(specs)
        stack = build_profile(project["profile"], materials)
        geometry = _stack_geometry(stack)
    used = {stack.top, stack.bottom} | {m for layer in stack.layers for m in layer.materials}
    if used - set(materials):
        raise ValueError(f"缺少材料：{sorted(used-set(materials))}")
    return geometry, stack, materials


def simulation_points(spec):
    from .solvers import Simulation
    pols = spec.get("polarizations", ["TE", "TM"])
    if not isinstance(pols, list) or not pols or len(set(pols)) != len(pols) or any(p not in ("TE", "TM") for p in pols):
        raise ValueError("偏振应为 TE、TM 或两者")
    wave = float(spec["wavelength_nm"])
    theta = float(spec.get("theta_deg", 0))
    order = spec.get("fourier_order", 31)
    if isinstance(order, bool) or not isinstance(order, int):
        raise ValueError("傅里叶阶数必须为整数")
    waves, angles = [wave], [theta]
    sweep = spec.get("sweep", {})
    if sweep.get("enabled"):
        count = sweep.get("samples", 11)
        if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= 501:
            raise ValueError("扫描点数应为 2 到 501 的整数")
        lo, hi = float(sweep["start"]), float(sweep["stop"])
        if not np.isfinite([lo,hi]).all() or lo >= hi:
            raise ValueError("扫描起点应小于终点，且均为有限数值")
        values = np.linspace(lo, hi, count).tolist()
        if sweep.get("parameter") == "wavelength_nm":
            waves = values
        elif sweep.get("parameter") == "theta_deg":
            angles = values
        elif not str(sweep.get("parameter", "")).startswith("steps."):
            raise ValueError("不支持的扫描参数")
    points = []
    geometry_values = values if sweep.get("enabled") and str(sweep.get("parameter", "")).startswith("steps.") else [None]
    for val in geometry_values:
        for lam in waves:
            for angle in angles:
                for pol in pols:
                    sim = Simulation(wavelength_um=lam/1000, theta_deg=angle, polarization=pol, fourier_order=order, na=0)
                    points.append((sim, val))
    return points


def _geometry_point(project, parameter, value):
    data = copy.deepcopy(project)
    parts = parameter.split(".")
    if len(parts) != 3 or parts[0] != "steps" or parts[2] not in {"thickness_um", "delta_x_um", "cd_bias_um", "etch_depth_um", "sidewall_thickness_um"}:
        raise ValueError("几何扫描应指定工艺步骤及厚度、偏移、CD 偏置或刻蚀深度")
    index = int(parts[1])
    if index < 0 or index >= len(data["process"]["steps"]):
        raise ValueError("扫描步骤不存在")
    step = data["process"]["steps"][index]
    step[parts[2]] = value
    if parts[2] == "thickness_um" and "z_max_um" in step:
        step["z_max_um"] = float(step.get("z_min_um", 0.)) + value
    return data


def reference_result(result, stack, reference_z_um):
    """Fix incident/reflected phases in z-up coordinates; retain exit interface.

    Incident amplitude is normalized at a fixed upper plane. Transmission is
    evaluated at its actual substrate boundary, whose absolute z is exported.
    Since the incident medium is lossless, port power fractions do not change.
    """
    top = float(stack.metadata.get("top_z_um",0.))
    distance = float(reference_z_um)-top
    if not np.isfinite(distance) or distance < -1e-12:
        raise ValueError("相位参考面必须位于所有结构顶面之上（物理 z 向上）")
    distance = max(0.,distance)
    incident_kz = result.n_top * 2*np.pi/result.wavelength_um * np.cos(np.radians(result.theta_deg))
    return replace(result, amplitude=result.amplitude*np.exp(1j*(result.kz+incident_kz)*distance),
                   transmission_amplitude=result.transmission_amplitude*np.exp(1j*incident_kz*distance),
                   reference_plane_um=float(reference_z_um))


def run_simulation(project, base_dir, output_parent, *, cancel_event=None, progress=None, converge=False):
    from .solvers import MeentSolver
    from .cli import write_result
    from .modeling import preview_geometry
    project = copy.deepcopy(project)
    points = simulation_points(project["simulation"])
    if converge and project["simulation"].get("sweep", {}).get("enabled"):
        raise ValueError("收敛检查请先关闭参数扫描")
    run = Path(output_parent) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6])
    run.mkdir(parents=True, exist_ok=False)
    digest = fingerprint(project)
    manifest = {"schema_version": "1.0", "solver":"Meent 0.13.2 + mark-rcwa", "input_sha256":digest,
                "status":"running", "point_count":len(points), "created_utc":datetime.now(timezone.utc).isoformat(),
                "phase_convention":"exp(-i omega t); physical z up, solver depth down; physical x=0 origin; incident/reflected at fixed upper z, transmitted at each recorded exit interface",
                "convergence_verified":False}
    write_json(run / "input.json", project)
    write_json(run / "manifest.json", manifest)
    records = []
    solver = MeentSolver()
    def check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("任务已取消")
    try:
        geometry, stack, materials = prepare_model(project, base_dir)
        upper_reference = project["simulation"].get("reference_z_um")
        upper_reference = float(geometry["cell_bounds_um"][2][1] if upper_reference is None else upper_reference)
        if not np.isfinite(upper_reference):
            raise ValueError("相位参考面必须为有限数值")
        manifest["geometry_to_solver"] = copy.deepcopy(stack.metadata)
        manifest["phase_reference_incident_z_um"] = upper_reference
        manifest["phase_reference_reflection_z_um"] = upper_reference
        write_json(run / "manifest.json", manifest)
        preview_geometry(geometry, run / "geometry")
        write_json(run / "geometry.json", geometry)
        for index, (sim, geometry_value) in enumerate(points):
            check_cancel()
            if geometry_value is not None:
                geometry, stack, materials = prepare_model(_geometry_point(project, project["simulation"]["sweep"]["parameter"], geometry_value), base_dir)
            if progress:
                progress({"message":f"{index+1}/{len(points)} · {sim.polarization} · {sim.wavelength_um*1000:g} nm · {sim.theta_deg:g}°", "fraction":index/len(points)})
            report = None
            if converge:
                from .validation.general_convergence import general_convergence
                maximum = project["simulation"].get("convergence_max_order",127)
                if isinstance(maximum,bool) or not isinstance(maximum,int) or not 15 <= maximum <= 255:
                    raise ValueError("收敛最大半阶数应为 15 到 255 的整数")
                orders = tuple(n for n in (7,11,15,21,31,41,55,71,95,127,155,191,223,255) if n<=maximum)
                if orders[-1]!=maximum:
                    orders = (*orders,maximum)
                report = general_convergence(stack, materials, sim, orders=orders, cancel_event=cancel_event,
                    progress=(lambda row: progress({"message":f"{sim.polarization} 收敛检查 N={row['N']}", "fraction":index/len(points)})) if progress else None)
                sim = replace(sim, fourier_order=report["final_N"])
            result = solver.solve(stack, materials, sim)
            result = reference_result(result,stack,upper_reference)
            # Stack edges use x'=x-x_min; restore the physical model x origin.
            result = result.translated(float(stack.metadata.get("x_origin_um", 0.)))
            check_cancel()
            folder = run / f"point_{index:04d}_{sim.polarization}"
            write_result(result, folder)
            exit_z = float(stack.metadata.get("bottom_z_um",stack.metadata.get("top_z_um",0.)-stack.thickness_um))
            write_json(folder/"phase_reference.json",{"incident_z_um":upper_reference,"reflection_z_um":upper_reference,
                       "transmission_z_um":exit_z,"coordinate_system":"absolute physical z, positive upward",
                       "transmission_power":"net power crossing the exit interface into the substrate"})
            if geometry_value is not None:
                write_json(folder / "geometry.json", geometry)
            if report is not None:
                write_json(folder / "convergence.json", report)
            material_snapshot = {name:{"n":materials.index(name,sim.wavelength_um).real,
                                      "k":materials.index(name,sim.wavelength_um).imag,
                                      "source":materials[name].source} for name in sorted({stack.top,stack.bottom}|{m for s in stack.layers for m in s.materials})}
            row = {"point":index, "polarization":sim.polarization, "wavelength_nm":sim.wavelength_um*1000,
                   "theta_deg":sim.theta_deg, "fourier_order":sim.fourier_order, "geometry_value_um":geometry_value,
                   "R":result.reflectance, "T":result.transmittance, "A_flux_residual":result.absorption,
                   "phase_reference_incident_z_um":upper_reference,"phase_reference_reflection_z_um":upper_reference,"phase_reference_transmission_z_um":exit_z,
                   "convergence_status":report["status"] if report else "not_checked", "orders":result.rows(),
                   "cutoff_metadata":result.cutoff_metadata,
                   "materials":material_snapshot, "folder":folder.name}
            records.append(row)
            write_json(run / "summary.json", records)
        check_cancel()
        _save_tables(run, records)
        manifest.update(status="succeeded", completed_utc=datetime.now(timezone.utc).isoformat(),
                        convergence_verified=bool(converge and all(r["convergence_status"]=="PASS" for r in records)))
        write_json(run / "manifest.json", manifest)
        return {"run_dir":str(run), "records":records, "geometry":geometry, "input_sha256":digest, "manifest":manifest}
    except Exception as exc:
        manifest.update(status="cancelled" if isinstance(exc,InterruptedError) else "failed", error=str(exc), completed_points=len(records))
        write_json(run / "manifest.json", manifest)
        raise


def _save_tables(run, records):
    from .reflection_matrix import export_reflection_excel
    export_reflection_excel(records, run / "R_matrix.xlsx")
    import csv
    with (run / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        keys = [k for k in records[0] if k not in {"orders","materials","cutoff_metadata"}]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in records)
    import h5py
    with h5py.File(run / "results.h5", "w") as handle:
        handle.attrs["schema_version"] = "mark-rcwa-results-1"
        handle.attrs["phase_convention"] = "exp(-i omega t); physical x origin; absolute z-up planes recorded per point"
        for record in records:
            group = handle.create_group(f"point_{record['point']:04d}")
            for key in ("polarization","wavelength_nm","theta_deg","fourier_order","R","T","A_flux_residual","convergence_status","phase_reference_incident_z_um","phase_reference_reflection_z_um","phase_reference_transmission_z_um"):
                group.attrs[key] = record[key]
            group.attrs["materials_json"] = json.dumps(record["materials"], ensure_ascii=False)
            group.attrs["cutoff_metadata_json"] = json.dumps(record.get("cutoff_metadata"), ensure_ascii=False, allow_nan=False)
            rows = record["orders"]
            group.create_dataset("orders", data=[r["order"] for r in rows])
            for key in ("reflected_efficiency","transmitted_efficiency","amplitude_real","amplitude_imag","transmitted_amplitude_real","transmitted_amplitude_imag"):
                if key in rows[0]:
                    group.create_dataset(key, data=[r[key] for r in rows])
    figure = scan_figure(records)
    if figure is not None:
        figure.savefig(run / "scan.png", dpi=160)


def scan_figure(records):
    """Return a publication-independent overview of a single varying input."""
    variable = next((key for key in ("wavelength_nm","theta_deg","geometry_value_um")
                     if len({r.get(key) for r in records})>1), None)
    if variable is None:
        return None
    from matplotlib.figure import Figure
    figure = Figure(figsize=(10,4),dpi=100)
    axes = figure.subplots(1,2)
    colors = {"TE":"#0f62fe","TM":"#8a3ffc"}
    for pol in sorted({r["polarization"] for r in records}):
        selected = [r for r in records if r["polarization"]==pol]
        x = [r[variable] for r in selected]
        for name,style in (("R","-"),("T","--")):
            axes[0].plot(x,[r[name] for r in selected],style+"o",markersize=3,label=f"{pol} {name}",color=colors.get(pol))
        axes[1].plot(x,[r["A_flux_residual"] for r in selected],"o-",markersize=3,label=pol,color=colors.get(pol))
    label = {"wavelength_nm":"Vacuum wavelength / nm","theta_deg":"Incidence angle / degree","geometry_value_um":"Scanned geometry value / um"}[variable]
    axes[0].set(xlabel=label,ylabel="Power / incident power",title="Reflection and transmission")
    axes[1].set(xlabel=label,ylabel="Finite-stack flux residual",title="Absorption")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=.2)
    figure.tight_layout()
    figure.set_layout_engine("constrained")
    return figure


def self_test(output):
    """Small actual solves and offline assets; safe for a clean frozen runtime."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    from .modeling import default_recipe, geometry_to_stack
    project = default_project()
    project["simulation"]["fourier_order"] = 11
    result = run_simulation(project, output, output / "runs")
    from .model import Stack
    from .solvers import MeentSolver, Simulation
    from .materials import Materials, Material
    media = Materials({"a":Material("a",1),"b":Material("b",1.5)})
    fresnel = MeentSolver().solve(Stack(.2,(),top="a",bottom="b"), media, Simulation(fourier_order=3,na=0))
    if not np.isclose(fresnel.reflectance,.04,atol=1e-12):
        raise AssertionError("Fresnel verification failed")
    portable = output / "portable" / "验证工程.mrcwa.json"
    saved = save_project(project, portable, base_dir=output)
    if fingerprint(load_project(portable)) != fingerprint(saved):
        raise AssertionError("Project roundtrip failed")
    report = {"ok":True, "status":"PASS", "fresnel_R":fresnel.reflectance, "points":len(result["records"]),
              "run_dir":result["run_dir"], "checks":["RCWA TE/TM","Fresnel interface","GDS Boolean geometry","offline 3D HTML","CSV/JSON/HDF5","portable project"]}
    write_json(output / "self_test.json", report)
    return report
