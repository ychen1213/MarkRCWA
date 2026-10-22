"""Resolve inputs against the current project; never search other projects."""
from pathlib import Path, PureWindowsPath, PurePosixPath


def resolve_input_path(value, *, base_dir=None):
    path = Path(value).expanduser()
    text = str(value)
    if (base_dir is not None and not path.is_absolute()
            and not PureWindowsPath(text).is_absolute()
            and not PurePosixPath(text).is_absolute()):
        path = Path(base_dir) / path
    return path
