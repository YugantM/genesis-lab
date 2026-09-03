"""Run a short MLX throughput and health check."""

from __future__ import annotations

import time

import mlx.core as mx

from .core import GenesisConfig, GenesisWorld
from .metrics import mass


def main() -> None:
    config = GenesisConfig()
    world = GenesisWorld(config)
    world.step(5)
    started = time.perf_counter()
    steps = 100
    world.step(steps)
    elapsed = time.perf_counter() - started
    worlds_per_second = config.batch * steps / elapsed
    print(f"Device: {mx.default_device()}")
    print(f"Worlds: {config.batch} × {config.size}²")
    print(f"Throughput: {worlds_per_second:,.0f} world-steps/s")
    print(f"Mean final mass: {mass(world.numpy()).mean():,.2f}")


if __name__ == "__main__":
    main()

