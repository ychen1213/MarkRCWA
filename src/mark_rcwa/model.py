from dataclasses import dataclass, field
import numpy as np


def positive(value, name, allow_zero=False):
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return float(value)


@dataclass(frozen=True)
class Slice:
    """Piecewise constant material regions on [0, pitch], with exact edges."""
    thickness_um: float
    edges_um: tuple[float, ...]
    materials: tuple[str, ...]

    def __post_init__(self):
        positive(self.thickness_um, 'layer thickness_um')
        e = np.asarray(self.edges_um, float)
        if len(e) != len(self.materials) + 1 or len(e) < 2:
            raise ValueError('edges must have one more element than materials')
        if not np.isfinite(e).all() or e[0] != 0 or np.any(np.diff(e) <= 0):
            raise ValueError('layer edges must increase strictly from 0')
        if any(not isinstance(m, str) or not m for m in self.materials):
            raise ValueError('material identifiers must be nonempty strings')


@dataclass(frozen=True)
class Stack:
    pitch_um: float
    layers: tuple[Slice, ...]
    top: str = 'air'
    bottom: str = 'glass_demo'
    original_pitch_um: float | None = None
    metadata: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        positive(self.pitch_um, 'pitch_um')
        for layer in self.layers:
            if abs(layer.edges_um[-1] - self.pitch_um) > 1e-12 * self.pitch_um:
                raise ValueError('every layer must end at exactly the stack pitch')
        if self.original_pitch_um is not None:
            positive(self.original_pitch_um, 'original_pitch_um')
            ratio = self.pitch_um / self.original_pitch_um
            if abs(ratio - round(ratio)) > 1e-10:
                raise ValueError('supercell pitch must be an integer multiple of original pitch')

    @property
    def thickness_um(self):
        return sum(s.thickness_um for s in self.layers)
