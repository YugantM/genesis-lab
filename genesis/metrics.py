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


def aligned_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Maximum cosine similarity over every toroidal translation.

    This measures shape independently of where the organism happens to be in
    the periodic world. Inputs are single two-dimensional states.
    """
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.ndim != 2 or right_array.shape != left_array.shape:
        raise ValueError("aligned similarity requires equally shaped 2D states")
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if not denominator:
        return 0.0
    correlation = np.fft.ifft2(
        np.fft.fft2(left_array) * np.conj(np.fft.fft2(right_array))
    ).real
    return float(np.clip(correlation.max() / denominator, 0.0, 1.0))


def recovery_time(
    steps: np.ndarray | list[float],
    observed: np.ndarray | list[float],
    matched_control: np.ndarray | list[float],
    *,
    relative_tolerance: float = 0.02,
    consecutive_samples: int = 3,
    minimum_control: float = 1.0,
) -> float | None:
    """First step entering and sustaining a band around a matched control.

    A return value of ``None`` means the trajectory never remained within the
    requested relative tolerance for the required number of samples.
    """
    step_values = np.asarray(steps, dtype=float)
    values = np.asarray(observed, dtype=float)
    controls = np.asarray(matched_control, dtype=float)
    if step_values.ndim != 1 or values.shape != step_values.shape or controls.shape != step_values.shape:
        raise ValueError("steps, observed, and matched_control must be equal 1D arrays")
    if (not np.isfinite(relative_tolerance) or relative_tolerance < 0
            or not isinstance(consecutive_samples, (int, np.integer)) or consecutive_samples < 1):
        raise ValueError("tolerance must be non-negative and consecutive_samples positive")
    if not np.isfinite(step_values).all() or np.any(np.diff(step_values) <= 0):
        raise ValueError("steps must be finite and strictly increasing")
    if not np.isfinite(minimum_control) or minimum_control <= 0:
        raise ValueError("minimum_control must be finite and positive")
    scale = np.maximum(np.abs(controls), 1e-12)
    within = (
        np.isfinite(values) & np.isfinite(controls) & (controls >= minimum_control)
        & (np.abs(values - controls) <= relative_tolerance * scale)
    )
    for start in range(len(within) - consecutive_samples + 1):
        if bool(within[start : start + consecutive_samples].all()):
            return float(step_values[start])
    return None


def threshold_recovery_time(
    steps: np.ndarray | list[float],
    values: np.ndarray | list[float],
    *,
    threshold: float,
    consecutive_samples: int = 3,
) -> float | None:
    """First step at or above a threshold for consecutive samples."""
    step_values = np.asarray(steps, dtype=float)
    measured = np.asarray(values, dtype=float)
    if step_values.ndim != 1 or measured.shape != step_values.shape:
        raise ValueError("steps and values must be equal 1D arrays")
    if (not isinstance(consecutive_samples, (int, np.integer))
            or consecutive_samples < 1 or not np.isfinite(threshold)):
        raise ValueError("consecutive_samples must be positive")
    if not np.isfinite(step_values).all() or np.any(np.diff(step_values) <= 0):
        raise ValueError("steps must be finite and strictly increasing")
    passed = np.isfinite(measured) & (measured >= threshold)
    for start in range(len(passed) - consecutive_samples + 1):
        if bool(passed[start : start + consecutive_samples].all()):
            return float(step_values[start])
    return None


def functional_recovery(
    *, control_valid: bool, mass_ratio: float, motion_ratio: float,
    occupied: float, maximum_occupied: float = 0.03,
) -> bool:
    """Control-relative functional endpoint; anatomy is a separate outcome."""
    return bool(
        control_valid
        and np.isfinite([mass_ratio, motion_ratio, occupied]).all()
        and 0.75 <= mass_ratio <= 1.25
        and motion_ratio >= 0.25
        and 0 <= occupied <= maximum_occupied
    )


def functional_score(mass_ratio: float, motion_ratio: float, control_valid: bool) -> float:
    """Mass closeness × retained locomotion, without a shape reward."""
    if not control_valid or not np.isfinite([mass_ratio, motion_ratio]).all():
        return 0.0
    return float(max(0.0, min(mass_ratio, 1 / max(mass_ratio, 1e-12), 1.0))
                 * np.clip(motion_ratio, 0.0, 1.0))


def functional_recovery_time(
    steps, mass_ratios, motion_ratios, occupied, control_valid,
    *, maximum_occupied: float = 0.03, consecutive_samples: int = 3,
) -> float | None:
    """First sustained functional endpoint using completed rolling motion windows.

    Callers provide samples only once a complete post-injury motion window is
    available (50 steps in the current protocols). No anatomical gate is used.
    """
    arrays = [np.asarray(values) for values in
              (steps, mass_ratios, motion_ratios, occupied, control_valid)]
    if any(values.ndim != 1 or values.shape != arrays[0].shape for values in arrays):
        raise ValueError("functional trajectories must be equally shaped 1D arrays")
    passed = [functional_recovery(
        control_valid=bool(valid), mass_ratio=float(m), motion_ratio=float(v),
        occupied=float(area), maximum_occupied=maximum_occupied,
    ) for m, v, area, valid in zip(*arrays[1:], strict=True)]
    return threshold_recovery_time(
        arrays[0], passed, threshold=1.0, consecutive_samples=consecutive_samples,
    )
