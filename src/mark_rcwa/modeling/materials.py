"""Offline optical TXT tables; exp(-i*omega*t), n_complex=n+i*k.

Exchange tables use wavelength_nm. The RCWA registry uses wavelength_um.
"""
from __future__ import annotations

import codecs
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np

SCHEMA_VERSION = "1.0"
MODEL_VERSION = "txt-linear-rcwa-1.0"


def _decode_material_text(raw, filename):
    """Decode without replacing bytes; a BOM is authoritative when present.

    Without a BOM, compatible legacy encodings cannot always be distinguished.
    The returned name records the decoder used, not the original editor setting.
    """
    # UTF-32 LE begins with the UTF-16 LE BOM, so check UTF-32 first.
    for bom, encoding in (
        (codecs.BOM_UTF32_LE, "utf-32-le"),
        (codecs.BOM_UTF32_BE, "utf-32-be"),
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
    ):
        if raw.startswith(bom):
            try:
                text = raw[len(bom):].decode(encoding)
            except UnicodeDecodeError:
                raise ValueError(
                    f"{filename}: 文件带有 {encoding} 编码标记，但内容不完整或编码不一致；"
                    "请用文本编辑器重新另存为 UTF-8 后导入。"
                ) from None
            return text, encoding, "bom"
    if b"\x00" in raw:
        raise ValueError(
            f"{filename}: 文件含有空字节，可能是没有编码标记的 UTF-16/UTF-32 或非文本文件；"
            "请用文本编辑器另存为 UTF-8 后导入，不要只修改扩展名。"
        )
    for encoding in ("utf-8", "gbk", "gb18030"):
        try:
            return raw.decode(encoding), encoding, "compatible"
        except UnicodeDecodeError:
            continue
    raise ValueError(
        f"{filename}: 无法按 UTF-8、GBK 或 GB18030 完整读取材料文本；"
        "请用文本编辑器另存为 UTF-8 后导入。"
    )


def import_material_txt(path, *, wavelength_unit, columns=None, assume_lossless=False):
    """Read explicit column mapping; absent k stays unknown until authorized.

    columns is a zero-based mapping e.g. {'wavelength':0,'n':1,'k':2}.
    With columns=None exactly two or three columns are accepted. Identical
    duplicate wavelengths are collapsed; conflicting duplicates are rejected.
    """
    if type(assume_lossless) is not bool:
        raise ValueError("assume_lossless must be a boolean, not a string or number")
    path = Path(path).resolve()
    factors = {"nm": 1.0, "um": 1000.0, "µm": 1000.0, "m": 1e9}
    if wavelength_unit not in factors:
        raise ValueError("Explicit wavelength_unit must be nm, um or m")
    raw = path.read_bytes()
    decoded, encoding, encoding_detection = _decode_material_text(raw, path.name)
    rows, numbers, header = [], [], None
    for number, text in enumerate(decoded.splitlines(), 1):
        text = re.split(r"#|//|;", text, maxsplit=1)[0].strip()
        if not text:
            continue
        if re.search(r",\s*,|^,|,$", text):
            raise ValueError(f"{path.name}:{number}: missing comma-delimited value")
        tokens = re.split(r"[\s,]+", text)
        try:
            row = [float(token) for token in tokens]
        except ValueError:
            if not rows and header is None and any("lambda" in t.lower() or "wavelength" in t.lower() for t in tokens):
                header = text
                continue
            raise ValueError(f"{path.name}:{number}: nonnumeric/missing material value") from None
        if not np.all(np.isfinite(row)):
            raise ValueError(f"{path.name}:{number}: nonfinite material value")
        rows.append(row)
        numbers.append(number)
    if not rows:
        raise ValueError(f"{path.name}: no numeric data")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"{path.name}: inconsistent columns/missing values")
    width = widths.pop()
    if columns is None:
        if width not in (2, 3):
            raise ValueError("Default import requires lambda,n or lambda,n,k; provide explicit columns")
        columns = {"wavelength": 0, "n": 1, **({"k": 2} if width == 3 else {})}
    columns = dict(columns)
    if "wavelength" not in columns or "n" not in columns:
        raise ValueError("Column mapping must specify wavelength and n")
    if set(columns) - {"wavelength", "n", "k"}:
        raise ValueError("Unknown column mapping; allowed keys are wavelength, n, k")
    if len(set(columns.values())) != len(columns) or any(type(i) is not int or i < 0 or i >= width for i in columns.values()):
        raise ValueError("Column indices must be distinct valid zero-based integers")
    data = np.asarray(rows)
    with np.errstate(over="ignore", invalid="ignore"):
        wave = data[:, columns["wavelength"]] * factors[wavelength_unit]
    n = data[:, columns["n"]]
    k = data[:, columns["k"]] if "k" in columns else None
    if not np.isfinite(wave).all() or np.any(wave <= 0) or np.any(n <= 0) or (k is not None and np.any(k < 0)):
        raise ValueError("Require vacuum wavelength>0, n>0 and passive k>=0")
    order = np.argsort(wave, kind="stable")
    wave, n = wave[order], n[order]
    if k is not None:
        k = k[order]
    keep, duplicates = [], 0
    for i in range(len(wave)):
        if keep and wave[i] == wave[keep[-1]]:
            j = keep[-1]
            if n[i] != n[j] or (k is not None and k[i] != k[j]):
                raise ValueError(f"Conflicting duplicate wavelength {wave[i]:g} nm")
            duplicates += 1
        else:
            keep.append(i)
    if len(keep) < 2:
        raise ValueError("At least two distinct wavelength samples are required")
    return {
        "wavelength_nm": wave[keep].tolist(), "n": n[keep].tolist(),
        "k": None if k is None else k[keep].tolist(),
        "k_status": "unknown" if k is None else "provided",
        "assume_lossless": bool(assume_lossless), "source": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "valid_range_nm": [float(wave[0]), float(wave[-1])],
        "import": {"wavelength_unit": wavelength_unit, "columns": columns,
                   "encoding": encoding, "encoding_detection": encoding_detection,
                   "header": header, "duplicate_identical_rows_removed": duplicates,
                   "original_numeric_rows": rows, "original_line_numbers": numbers},
    }


def import_material_folder(folder, *, wavelength_unit, assume_lossless=False, options=None):
    """Import all TXT files; per-stem options can override mapping and units."""
    files = sorted(Path(folder).glob("*.txt"))
    if not files:
        raise ValueError("Material folder contains no .txt files")
    options = options or {}
    materials = {p.stem: import_material_txt(p, **{
                "wavelength_unit": wavelength_unit, "assume_lossless": assume_lossless,
                **options.get(p.stem, {})}) for p in files}
    return {"schema_version": SCHEMA_VERSION, "model_version": MODEL_VERSION,
            "generation_method": "local_txt_import", "software_versions": {"numpy": np.__version__},
            "input_hash": hashlib.sha256(json.dumps(materials, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "materials": materials}


def save_library(library, path):
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("Material library output must be .json; raw material TXT files are read-only inputs")
    destination = path.resolve()
    if any(spec.get("source") and Path(spec["source"]).resolve() == destination for spec in library.get("materials", {}).values()):
        raise ValueError("Refusing to overwrite an original material source")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(library, indent=2, ensure_ascii=False, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_library(path):
    library = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if library.get("schema_version") != SCHEMA_VERSION or not library.get("materials"):
        raise ValueError("Unsupported or empty MaterialLibrary")
    return library


def materials_from_library(library):
    """Validate a MarkSim-format library and preserve n+ik in RCWA materials.

    Missing k is never silently set to zero. A per-material explicit
    ``assume_lossless=True`` authorizes that approximation. Negative real
    permittivity (e.g. a metal) is supported without conductivity conversion.
    """
    from mark_rcwa.materials import Material, Materials

    if library.get('schema_version') != SCHEMA_VERSION or not library.get('materials'):
        raise ValueError('Unsupported or empty MaterialLibrary')
    result = Materials()
    for name, spec in library['materials'].items():
        if not isinstance(name, str) or not name:
            raise ValueError('Material identifiers must be nonempty strings')
        if type(spec.get('assume_lossless', False)) is not bool:
            raise ValueError(f'{name}: assume_lossless must be a boolean')
        wave = np.asarray(spec['wavelength_nm'], float)
        ns = np.asarray(spec['n'], float)
        if wave.ndim != 1 or ns.shape != wave.shape or len(wave) < 2:
            raise ValueError(f'{name}: invalid wavelength/n table shapes')
        if spec.get('k') is None:
            if spec.get('assume_lossless') is not True:
                raise ValueError(f'{name}: k unknown; explicitly choose assume_lossless before simulation')
            ks = np.zeros(wave.shape)
            approximation = '; k unknown: explicit user-selected lossless approximation'
        else:
            ks = np.asarray(spec['k'], float)
            approximation = ''
        if ks.shape != wave.shape:
            raise ValueError(f'{name}: invalid k table shape')
        source = str(spec.get('source') or 'user optical table') + approximation
        if spec.get('synthetic_demo'):
            source += '; SYNTHETIC DEMONSTRATION ONLY; not calibrated material data'
        result[name] = Material(name, wavelength_um=tuple(wave / 1000.0),
                                n_table=tuple(complex(n, k) for n, k in zip(ns, ks)),
                                source=source)
    return result


def evaluate_library(library, lambda_vacuum_nm):
    """Return interpolated n,k and complex epsilon; no extrapolation or fitting."""
    wavelength_um = float(lambda_vacuum_nm) / 1000.0
    return {name: {'n': (nk := material.index(wavelength_um)).real,
                   'k': nk.imag, 'epsilon_real': (nk * nk).real,
                   'epsilon_imag': (nk * nk).imag, 'source': material.source}
            for name, material in materials_from_library(library).items()}
