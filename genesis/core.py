"""Batched continuous cellular automata implemented in MLX.

The update is deliberately compact and inspectable. Worlds wrap at their
boundaries, convolution is performed in Fourier space, and all stochasticity
is derived from an explicit seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .genome import Specimen


@dataclass(frozen=True)
class GenesisConfig:
    size: int = 128
    batch: int = 64
    radius: float = 13.0
    growth_center: float = 0.15
    growth_width: float = 0.025
    dt: float = 0.1
    seed: int = 1


def _kernel(config: GenesisConfig) -> mx.array:
    """Construct a normalized, smooth ring kernel at FFT origin."""
    n = config.size
    axis = np.minimum(np.arange(n), n - np.arange(n)).astype(np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    r = np.sqrt(xx * xx + yy * yy) / config.radius
    # A smooth compact ring. It is zero at the centre and outside the radius.
    k = np.zeros_like(r)
    mask = (r > 0.0) & (r < 1.0)
    k[mask] = np.exp(4.0 - 1.0 / (r[mask] * (1.0 - r[mask])))
    k /= k.sum()
    return mx.array(k)


class GenesisWorld:
    """A reproducible batch of worlds sharing one local update rule."""

    def __init__(self, config: GenesisConfig = GenesisConfig()):
        self.config = config
        rng = np.random.default_rng(config.seed)
        state = np.zeros((config.batch, config.size, config.size), np.float32)
        patch = max(8, config.size // 8)
        y0 = config.size // 2 - patch // 2
        x0 = config.size // 2 - patch // 2
        state[:, y0 : y0 + patch, x0 : x0 + patch] = rng.random(
            (config.batch, patch, patch), dtype=np.float32
        )
        self.state = mx.array(state)
        self._kernel_fft = mx.fft.fft2(_kernel(config))
        self._growth_center: float | mx.array = config.growth_center
        self._growth_width: float | mx.array = config.growth_width

    @classmethod
    def from_specimen(
        cls, specimen: Specimen, *, size: int = 128, batch: int = 1
    ) -> "GenesisWorld":
        """Create centred copies of a versioned specimen."""
        p = specimen.parameters
        config = GenesisConfig(
            size=size,
            batch=batch,
            radius=float(p["radius"]),
            growth_center=float(p["growth_center"]),
            growth_width=float(p["growth_width"]),
            dt=1.0 / float(p["time_resolution"]),
        )
        world = cls(config)
        cells = specimen.cells
        if cells.shape[0] > size or cells.shape[1] > size:
            raise ValueError("specimen is larger than the requested world")
        state = np.zeros((batch, size, size), dtype=np.float32)
        top, left = (size - cells.shape[0]) // 2, (size - cells.shape[1]) // 2
        state[:, top : top + cells.shape[0], left : left + cells.shape[1]] = cells
        world.state = mx.array(state)
        world._growth_center = config.growth_center
        world._growth_width = config.growth_width
        mx.eval(world.state)
        return world

    def set_growth_parameters(
        self, centers: np.ndarray | list[float], widths: np.ndarray | list[float]
    ) -> None:
        """Assign per-world growth parameters for batched parameter search."""
        centers_array = np.asarray(centers, dtype=np.float32)
        widths_array = np.asarray(widths, dtype=np.float32)
        expected = (self.config.batch,)
        if centers_array.shape != expected or widths_array.shape != expected:
            raise ValueError(f"growth parameter arrays must have shape {expected}")
        if np.any(widths_array <= 0):
            raise ValueError("growth widths must be positive")
        self._growth_center = mx.array(centers_array[:, None, None])
        self._growth_width = mx.array(widths_array[:, None, None])

    def step(self, count: int = 1) -> mx.array:
        """Advance every world and return the current state."""
        c = self.config
        for _ in range(count):
            neighbourhood = mx.fft.ifft2(
                mx.fft.fft2(self.state) * self._kernel_fft
            ).real
            growth = 2.0 * mx.exp(
                -((neighbourhood - self._growth_center) ** 2)
                / (2.0 * self._growth_width**2)
            ) - 1.0
            self.state = mx.clip(self.state + c.dt * growth, 0.0, 1.0)
        mx.eval(self.state)
        return self.state

    def damage(self, fraction: float = 0.25) -> None:
        """Remove a centred vertical band from every world."""
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be between zero and one")
        width = round(self.config.size * fraction)
        left = (self.config.size - width) // 2
        mask = np.ones((1, self.config.size, self.config.size), np.float32)
        mask[:, :, left : left + width] = 0.0
        self.state = self.state * mx.array(mask)
        mx.eval(self.state)

    def damage_disk(self, center: tuple[float, float], radius: float) -> None:
        """Excise a disk, using (y, x) coordinates in world cells."""
        if radius < 0:
            raise ValueError("radius must be non-negative")
        yy, xx = np.ogrid[: self.config.size, : self.config.size]
        dy = np.minimum(abs(yy - center[0]), self.config.size - abs(yy - center[0]))
        dx = np.minimum(abs(xx - center[1]), self.config.size - abs(xx - center[1]))
        mask = ((dy * dy + dx * dx) > radius * radius).astype(np.float32)
        self.state = self.state * mx.array(mask[None, :, :])
        mx.eval(self.state)

    def numpy(self) -> np.ndarray:
        return np.asarray(self.state)
