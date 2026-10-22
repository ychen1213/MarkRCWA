"""Run the editable local source with an existing Python environment."""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
# This must precede importing any project code, including an installed copy.
sys.path.insert(0, str(SOURCE))

# Only Meent is an exact requirement here. Other versions are reference values
# from pyproject.toml; compatible company environments need not match them all.
DEPENDENCIES = (
    ("meent", "meent", "0.13.2", True),
    ("numpy", "numpy", "2.2.6", False),
    ("scipy", "scipy", "1.15.3", False),
    ("matplotlib", "matplotlib", "3.10.3", False),
    ("yaml", "PyYAML", "6.0.2", False),
    ("threadpoolctl", "threadpoolctl", "3.6.0", False),
    ("psutil", "psutil", "7.0.0", False),
    ("gdstk", "gdstk", "1.0.2", False),
    ("plotly", "plotly", "7.1.0", False),
    ("h5py", "h5py", "3.16.0", False),
    ("PIL", "Pillow", None, False),
)


def check_environment(*, verbose: bool = True) -> bool:
    """Report import failures without installing packages or using the network."""
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Local source: {SOURCE}")
    errors: list[str] = []
    if not (3, 11) <= sys.version_info[:2] < (3, 13):
        errors.append("Python 3.11 or 3.12 is required by this project.")

    expected = SOURCE / "mark_rcwa" / "__init__.py"
    if not expected.is_file():
        errors.append(f"Local source is missing: {expected}")
    else:
        try:
            spec = importlib.util.find_spec("mark_rcwa")
            actual = Path(spec.origin).resolve() if spec and spec.origin else None
            if actual != expected.resolve():
                errors.append(f"mark_rcwa resolved outside local source: {actual}")
            else:
                print(f"mark_rcwa import: {actual}")
        except Exception as exc:
            errors.append(f"Cannot locate local mark_rcwa: {exc}")

    for module_name, distribution, reference, exact in DEPENDENCIES:
        try:
            module = importlib.import_module(module_name)
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                version = str(getattr(module, "__version__", "unknown"))
            wrong_version = exact and version != reference
            if wrong_version:
                errors.append(
                    f"{distribution}: found {version}; required exact version {reference}."
                )
            if verbose:
                constraint = (
                    f"required {reference}" if exact else
                    f"reference {reference}, not an exact requirement" if reference else
                    "import required"
                )
                status = "ERROR" if wrong_version else "OK"
                print(f"{status} {distribution}: {version} ({constraint})")
        except Exception as exc:
            errors.append(f"{distribution}: cannot import {module_name}: {type(exc).__name__}: {exc}")

    try:
        import tkinter

        tcl_version = tkinter.Tcl().eval("info patchlevel")
        if verbose:
            print(f"OK tkinter: Tcl {tcl_version}, Tk {tkinter.TkVersion}")
    except Exception as exc:
        errors.append(f"tkinter/Tcl: cannot initialize: {type(exc).__name__}: {exc}")

    if errors:
        print("\nEnvironment check FAILED:")
        for error in errors:
            print(f"  - {error}")
        print("Select your existing company Python environment using MARKRCWA_PYTHON")
        print("in the launcher, or run this file with that environment's python.exe.")
        print("No packages were installed or downloaded.")
        return False
    if verbose:
        print("\nEnvironment check PASSED. Local editable source will be used.")
        print("Import checks do not certify numerical compatibility of other library versions.")
    return True


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    check_only = "--check-env" in args
    if check_only and args != ["--check-env"]:
        print("Use --check-env by itself; other arguments are passed to the desktop application.")
        return 2
    if not check_environment(verbose=check_only):
        return 1
    if check_only:
        return 0
    try:
        from mark_rcwa.desktop import main as desktop_main

        result = desktop_main(args)
        return int(result) if result is not None else 0
    except Exception:
        print("\nMarkRCWA failed to start from local source:", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
