"""Two-channel Lenia with evolvable responses at inter-species boundaries.

The original coupled model applies the same pressure to every organism.  This
extension lets each channel evolve three bounded traits: pressure sensitivity,
a boundary response, and memory of recent contact. Boundary response attenuates
pressure only where another organism is locally detected or was just detected.
The response and memory have a maintenance cost, which keeps the search from
obtaining free coexistence by simply disabling competition.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .core import GenesisConfig, _kernel


@dataclass(frozen=True)
class InteractionConfig:
    size: int = 128
    dt: float = 0.1
    radius: float = 13.0
    competition: float = 1.0
    signal_scale: float = 0.12
    maintenance_cost: float = 0.035


class ResponsiveCoupledWorld:
    """A batch of paired worlds with independently evolvable interaction genes."""

    def __init__(
        self,
        states: np.ndarray,
        centers: np.ndarray,
        widths: np.ndarray,
        sensitivities: np.ndarray,
        boundary_responses: np.ndarray,
        memory_persistence: np.ndarray,
        config: InteractionConfig = InteractionConfig(),
    ) -> None:
        if states.ndim != 4 or states.shape[1] != 2:
            raise ValueError("states must have shape (batch, 2, size, size)")
        expected = states.shape[:2]
        for name, values in (
            ("centers", centers),
            ("widths", widths),
            ("sensitivities", sensitivities),
            ("boundary_responses", boundary_responses),
            ("memory_persistence", memory_persistence),
        ):
            if values.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if states.shape[-2:] != (config.size, config.size):
            raise ValueError("state size does not match configuration")
        if np.any(widths <= 0):
            raise ValueError("growth widths must be positive")
        if (
            np.any(sensitivities < 0)
            or np.any(boundary_responses < 0)
            or np.any(memory_persistence < 0)
            or np.any(memory_persistence >= 1)
        ):
            raise ValueError("interaction traits must be non-negative")
        if config.competition < 0 or config.signal_scale <= 0:
            raise ValueError("competition must be non-negative and signal scale positive")

        self.config = config
        self.state = mx.array(states.astype(np.float32, copy=False))
        self.centers = mx.array(centers.astype(np.float32))[:, :, None, None]
        self.widths = mx.array(widths.astype(np.float32))[:, :, None, None]
        self.sensitivities = mx.array(sensitivities.astype(np.float32))[:, :, None, None]
        self.boundary_responses = mx.array(
            boundary_responses.astype(np.float32)
        )[:, :, None, None]
        self.memory_persistence = mx.array(
            memory_persistence.astype(np.float32)
        )[:, :, None, None]
        self.signal_memory = mx.zeros_like(self.state)
        kernel_config = GenesisConfig(size=config.size, radius=config.radius)
        self.kernel_fft = mx.fft.fft2(_kernel(kernel_config))[None, None]
        mx.eval(
            self.state,
            self.centers,
            self.widths,
            self.sensitivities,
            self.boundary_responses,
            self.memory_persistence,
            self.signal_memory,
            self.kernel_fft,
        )

    def step(self, count: int = 1) -> mx.array:
        c = self.config
        for _ in range(count):
            neighbourhood = mx.fft.ifft2(
                mx.fft.fft2(self.state) * self.kernel_fft
            ).real
            intrinsic = 2.0 * mx.exp(
                -((neighbourhood - self.centers) ** 2) / (2.0 * self.widths**2)
            ) - 1.0
            other = mx.sum(neighbourhood, axis=1, keepdims=True) - neighbourhood
            signal = mx.clip(other / c.signal_scale, 0.0, 1.0)
            self.signal_memory = mx.maximum(
                signal, self.memory_persistence * self.signal_memory
            )
            protected_pressure = (
                c.competition
                * self.sensitivities
                * other
                * (1.0 - self.boundary_responses * self.signal_memory)
            )
            maintenance = (
                c.maintenance_cost
                * self.boundary_responses**2
                * (1.0 + 0.4 * self.memory_persistence)
                * self.state
            )
            self.state = mx.clip(
                self.state + c.dt * (intrinsic - protected_pressure - maintenance),
                0.0,
                1.0,
            )
        mx.eval(self.state)
        return self.state

    def numpy(self) -> np.ndarray:
        return np.asarray(self.state)
