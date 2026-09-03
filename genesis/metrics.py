"""Small, deterministic measurements used by early Genesis experiments."""

from __future__ import annotations

import numpy as np


def mass(world: np.ndarray) -> np.ndarray:
    """Total active matter per batch member."""
    return np.asarray(world).sum(axis=(-2, -1))


def occupied_fraction(world: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Fraction of cells whose state exceeds ``threshold``."""
    return (np.asarray(world) > threshold).mean(axis=(-2, -1))


def survival(world: np.ndarray, minimum_mass: float = 1.0) -> np.ndarray:
    """Whether each world retains non-trivial active matter."""
    return mass(world) >= minimum_mass


def center_of_mass(world: np.ndarray) -> np.ndarray:
    """Toroidal centre of mass as ``(y, x)`` for every batch member."""
    state = np.asarray(world)
    size_y, size_x = state.shape[-2:]
    result = []
    for axis, size in ((-2, size_y), (-1, size_x)):
        marginal = state.sum(axis=-1 if axis == -2 else -2)
        angles = np.arange(size) * (2.0 * np.pi / size)
        sin_mean = (marginal * np.sin(angles)).sum(axis=-1)
        cos_mean = (marginal * np.cos(angles)).sum(axis=-1)
        coordinate = np.mod(np.arctan2(sin_mean, cos_mean), 2.0 * np.pi)
        result.append(coordinate * size / (2.0 * np.pi))
    return np.stack(result, axis=-1)


def toroidal_displacement(start: np.ndarray, end: np.ndarray, size: int) -> np.ndarray:
    """Shortest signed displacement between toroidal coordinates."""
    return (np.asarray(end) - np.asarray(start) + size / 2) % size - size / 2
