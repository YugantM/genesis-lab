"""Versioned Genesis specimen records and Lenia RLE decoding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Specimen:
    id: str
    name: str
    simulator: str
    parameters: dict
    cells: np.ndarray
    provenance: dict


def _cell_value(code: str) -> float:
    if code in (".", "b"):
        return 0.0
    if code == "o":
        return 1.0
    if len(code) == 1:
        value = ord(code) - ord("A") + 1
    else:
        value = (ord(code[0]) - ord("p")) * 24 + ord(code[1]) - ord("A") + 25
    return value / 255.0


def decode_lenia_rle(encoded: str) -> np.ndarray:
    """Decode the two-dimensional RLE used by the original Lenia catalogue."""
    rows: list[list[float]] = []
    row: list[float] = []
    count, prefix = "", ""
    for char in encoded.rstrip("!") + "$":
        if char.isdigit():
            count += char
        elif char in "pqrstuvwxy@":
            prefix = char
        elif char == "$":
            rows.append(row)
            rows.extend([] for _ in range((int(count) if count else 1) - 1))
            row, count, prefix = [], "", ""
        else:
            row.extend([_cell_value(prefix + char)] * (int(count) if count else 1))
            count, prefix = "", ""
    width = max(map(len, rows))
    return np.asarray([r + [0.0] * (width - len(r)) for r in rows], dtype=np.float32)


def load_specimen(path: str | Path) -> Specimen:
    record = json.loads(Path(path).read_text())
    if record.get("schema") != "genesis.specimen/v1":
        raise ValueError("unsupported specimen schema")
    initial = record["initial_state"]
    if initial["encoding"] != "lenia-rle-v1":
        raise ValueError("unsupported initial-state encoding")
    return Specimen(
        id=record["id"],
        name=record["name"],
        simulator=record["simulator"],
        parameters=record["parameters"],
        cells=decode_lenia_rle(initial["cells"]),
        provenance=record["provenance"],
    )

