"""Public Lua API adapter for the external, checksum-pinned S4 executable.

No S4 solver implementation or GPL binary is distributed in this package.
The wrapper and generated Lua use the documented public API only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from mark_rcwa.model import positive

S4_ARCHIVE_URL = "https://web.stanford.edu/group/fan/S4/files/S4-1.1.1-win32.7z"
S4_ARCHIVE_SHA256 = "5d2e3feef1f1833021d4bcc13dbd299d6751398bd8d80d4ae58e8f2f98e7c59c"
S4_EXECUTABLE_SHA256 = "5590bad0e24a643f70f346b5cc91660299fe7f06caaa6756afe59c00c079e8c1"
S4_VERSION = "1.1.1 official Windows Lua binary"


def locate_s4(executable=None):
    """Resolve explicit path, MARK_RCWA_S4, cwd, then source checkout path."""
    explicit = executable or os.environ.get("MARK_RCWA_S4")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            raise ValueError("S4 explicit executable / MARK_RCWA_S4 must be an absolute path")
        return path.resolve()
    candidates = [Path.cwd()/"vendor"/"S4"/"S4.exe",
                  Path(__file__).resolve().parents[3]/"vendor"/"S4"/"S4.exe"]
    return next((path.resolve() for path in candidates if path.is_file()), candidates[0].resolve())


@dataclass
class S4ReferenceResult:
    orders: np.ndarray
    amplitude: np.ndarray
    electric: np.ndarray
    efficiency: np.ndarray
    transmission_efficiency: np.ndarray
    kx: np.ndarray
    kz: np.ndarray
    propagating: np.ndarray
    collected: np.ndarray
    metadata: dict

    @property
    def reflectance(self):
        return float(self.efficiency.sum())

    @property
    def transmittance(self):
        return float(self.transmission_efficiency.sum())


class S4ReferenceSolver:
    """Independent 1D reference with explicit optional passive regularization.

    Exactly lossless translated patterns are numerically unreliable in this
    legacy binary. regularization_k defaults to zero: the caller must explicitly
    request a changed internal index and report the supplied/effective values.
    Exterior half-spaces retain the specified material constants.
    """

    def __init__(self, executable=None, input_directory=None, timeout_s=120):
        self.executable = locate_s4(executable)
        if not self.executable.is_file():
            raise FileNotFoundError(f"S4 executable unavailable: {self.executable}; run scripts/install_s4.ps1")
        digest = hashlib.sha256(self.executable.read_bytes()).hexdigest()
        if digest != S4_EXECUTABLE_SHA256:
            raise ValueError(f"S4 executable SHA256 mismatch: actual {digest}, expected {S4_EXECUTABLE_SHA256}")
        self.input_directory = Path(input_directory or Path.cwd()/"reports"/"s4_inputs").resolve()
        self.timeout_s = positive(timeout_s, "S4 timeout_s")

    def solve(self, stack, materials, sim, regularization_k=0.0, reference_offset_um=0.0):
        regularization_k = positive(regularization_k, "S4 regularization_k", allow_zero=True)
        if not np.isfinite(reference_offset_um) or reference_offset_um > 0:
            raise ValueError("S4 reference_offset_um must be finite and <=0 in the incident half-space")
        nt = materials.index(stack.top, sim.wavelength_um)
        nb = materials.index(stack.bottom, sim.wavelength_um)
        if nt.imag != 0:
            raise ValueError("S4 incident half-space must be lossless")
        if sim.na > nt.real:
            raise ValueError("S4 NA cannot exceed incident-medium refractive index")
        n = sim.fourier_order
        expected_orders = np.arange(-n, n+1)
        k0 = 2*np.pi/sim.wavelength_um
        kx = nt.real*k0*np.sin(np.radians(sim.theta_deg))+expected_orders*2*np.pi/stack.pitch_um
        kz = np.sqrt((nt.real*k0)**2-kx**2+0j)
        if np.any(abs(kz) < 1e-10*k0):
            raise ValueError("S4 exact Rayleigh cutoff has singular electric modal normalization")
        physical = {name: materials.index(name, sim.wavelength_um)
                    for layer in stack.layers for name in layer.materials}
        effective = {name: index+(1j*regularization_k if index.imag == 0 else 0)
                     for name, index in physical.items()}
        lines = ["local S=S4.NewSimulation()", f"S:SetLattice({stack.pitch_um:.17g})",
                 f"S:SetNumG({2*n+1})", "S:SetVerbosity(0)",
                 "S:UseLanczosSmoothing(false)", "S:UsePolarizationDecomposition(true)"]

        def add_material(name, index):
            epsilon = index**2
            lines.append(f"S:AddMaterial('{name}',{{{epsilon.real:.17g},{epsilon.imag:.17g}}})")

        add_material("top", nt)
        add_material("bottom", nb)
        lines.append("S:AddLayer('inc',0,'top')")
        for index, layer in enumerate(stack.layers):
            lengths = {name: sum(right-left for left, right, material in
                       zip(layer.edges_um[:-1], layer.edges_um[1:], layer.materials) if material == name)
                       for name in sorted(set(layer.materials))}
            background = max(lengths, key=lengths.get)
            layer_name = f"layer{index}"
            add_material(f"background{index}", effective[background])
            lines.append(f"S:AddLayer('{layer_name}',{layer.thickness_um:.17g},'background{index}')")
            for interval, (left, right, name) in enumerate(zip(layer.edges_um[:-1], layer.edges_um[1:], layer.materials)):
                if name == background:
                    continue
                region = f"region{index}_{interval}"
                add_material(region, effective[name])
                # S4's 1D convention uses a unit-area dummy y cell.
                lines.append(f"S:SetLayerPatternRectangle('{layer_name}','{region}',"
                             f"{{{(left+right)/2:.17g},0}},0,"
                             f"{{{(right-left)/2:.17g},{.5/stack.pitch_um:.17g}}})")
        lines.append("S:AddLayer('out',0,'bottom')")
        te = sim.polarization == "TE"
        lines.extend([f"S:SetFrequency({1/sim.wavelength_um:.17g})",
            f"S:SetExcitationPlanewave({{{sim.theta_deg:.17g},0}},{{{int(te)},0}},{{{int(not te)},0}})",
            f"local a,b=S:GetAmplitudes('inc',{reference_offset_um:.17g})",
            f"local R=S:GetPowerFluxByOrder('inc',{reference_offset_um:.17g})",
            "local T=S:GetPowerFluxByOrder('out',0)", "local G=S:GetGList()",
            "for i=1,#G do",
            " print(string.format('ROW %d %d %.17g %.17g %.17g %.17g %.17g %.17g',",
            " G[i][1],G[i][2],b[i][1],b[i][2],b[i+#G][1],b[i+#G][2],-R[i][2],T[i][1]))",
            "end"])
        self.input_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.input_directory, prefix="s4_", suffix=".lua",
                                         mode="w", encoding="ascii", delete=False) as stream:
            stream.write("\n".join(lines)+"\n")
            lua_file = Path(stream.name)
        try:
            completed = subprocess.run([str(self.executable), str(lua_file)], capture_output=True,
                                       text=True, timeout=self.timeout_s, check=False)
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"S4 solve timed out; retained input {lua_file}") from exc
        if completed.returncode:
            raise ValueError(f"S4 exited {completed.returncode}; input {lua_file}; {completed.stderr.strip()} {completed.stdout.strip()}")
        try:
            rows = np.asarray([[float(value) for value in line.split()[1:]]
                               for line in completed.stdout.splitlines() if line.startswith("ROW ")])
        except ValueError as exc:
            raise ValueError(f"S4 malformed numeric output; input {lua_file}") from exc
        if rows.shape != (2*n+1, 8) or not np.isfinite(rows).all():
            raise ValueError(f"S4 unexpected/invalid mode output {rows.shape}; input {lua_file}")
        rows = rows[np.argsort(rows[:, 0])]
        if np.any(rows[:, 1] != 0) or not np.array_equal(rows[:, 0], expected_orders):
            raise ValueError(f"S4 reciprocal-order set differs from requested 1D set; input {lua_file}")
        hx = rows[:, 2]+1j*rows[:, 3]
        hy = rows[:, 4]+1j*rows[:, 5]
        ex = -hy*kz/(k0*nt.real**2)
        ey = hx*k0/kz
        ez = -kx*hy/(k0*nt.real**2)
        electric = np.column_stack([ex, ey, ez])
        amplitude = ey if te else -hy/nt.real
        flux_inc = nt.real*np.cos(np.radians(sim.theta_deg))
        efficiency = rows[:, 6]/flux_inc
        transmission = rows[:, 7]/flux_inc
        propagating = (abs(kz.imag) < 1e-10*k0) & (kz.real > 1e-10*k0)
        efficiency[~propagating] = 0
        collected = propagating & (abs(kx)/k0 <= sim.na+1e-12)
        if min(efficiency.min(), transmission.min()) < -1e-10 or efficiency.sum()+transmission.sum() > 1+1e-7:
            raise ValueError(f"S4 nonphysical passive flux; R+T={efficiency.sum()+transmission.sum():.17g}; "
                             f"legacy lossless modal-branch issue may apply; input {lua_file}; "
                             "regularization requires explicit caller input and a measured perturbation check")
        if not np.isfinite(amplitude).all() or not np.isfinite(electric).all():
            raise ValueError(f"S4 non-finite reflected electric fields; input {lua_file}")
        metadata = {
            "solver": f"S4 {S4_VERSION}", "executable": str(self.executable),
            "executable_sha256": S4_EXECUTABLE_SHA256, "archive_sha256": S4_ARCHIVE_SHA256,
            "input_lua": str(lua_file), "regularization_k": regularization_k,
            "physical_internal_indices": {name: [index.real, index.imag] for name, index in physical.items()},
            "effective_internal_indices": {name: [index.real, index.imag] for name, index in effective.items()},
            "physical_half_spaces": {"top": [nt.real, nt.imag], "bottom": [nb.real, nb.imag]},
            "reference_plane_um": float(reference_offset_um), "time_factor": "exp(-i omega t)",
            "amplitude_normalization": "incident E=1; fixed laboratory reflected s/p electric basis",
            "origin_um": 0.0, "fourier_order": n,
        }
        return S4ReferenceResult(expected_orders, amplitude, electric, efficiency,
                                 transmission, kx.astype(complex), kz, propagating,
                                 collected, metadata)
