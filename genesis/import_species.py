"""Import a pinned, compatible subset of the original Lenia catalogue."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path


SOURCE_COMMIT = "adfc542939266de7f4bb7ebb552e8499701ee107"
SOURCE_URL = (
    "https://raw.githubusercontent.com/Chakazul/Lenia/"
    f"{SOURCE_COMMIT}/Python/animals.json"
)

# Eight stable organisms spanning seven catalogue genera. They deliberately
# share the exact simulator family used by Genesis: one channel, one bump
# kernel, one Gaussian growth function, R=13, and T=10.
SPECIES_CODES = (
    "O2u",   # Orbium
    "OG2g",  # Gyrorbium
    "O4i",   # Synorbium
    "S1s",   # Scutium
    "P4cl",  # Paraptera
    "H3s",   # Helicium
    "H3cp",  # Helicium, alternate morphology
    "C0v",   # Circium
)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def compatible(record: dict) -> bool:
    params = record.get("params", {})
    return (
        params.get("R") == 13
        and params.get("T") == 10
        and params.get("b") == "1"
        and params.get("kn") == 1
        and params.get("gn") == 1
        and isinstance(params.get("m"), (int, float))
        and isinstance(params.get("s"), (int, float))
    )


def import_catalog(source: list[dict], destination: Path) -> dict:
    selected: list[dict] = []
    seen: set[str] = set()
    for record in source:
        code = record.get("code")
        if code in SPECIES_CODES and code not in seen and compatible(record):
            selected.append(record)
            seen.add(code)
    missing = set(SPECIES_CODES) - seen
    if missing:
        raise RuntimeError(f"missing compatible catalogue entries: {sorted(missing)}")
    selected.sort(key=lambda item: SPECIES_CODES.index(item["code"]))

    catalogue_dir = destination / "catalog"
    catalogue_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, source_record in enumerate(selected):
        params = source_record["params"]
        slug = slugify(source_record["name"])
        filename = f"catalog/{slug}.json"
        specimen = {
            "schema": "genesis.specimen/v1",
            "id": f"catalog-{source_record['code'].lower()}",
            "name": source_record["name"],
            "provenance": {
                "author": "Bert Wang-Chak Chan",
                "catalogue_code": source_record["code"],
                "source": SOURCE_URL,
                "source_commit": SOURCE_COMMIT,
            },
            "simulator": "lenia/classic-v1",
            "parameters": {
                "radius": params["R"],
                "time_resolution": params["T"],
                "growth_center": params["m"],
                "growth_width": params["s"],
                "kernel": "bump4",
                "kernel_peaks": [1.0],
            },
            "initial_state": {
                "encoding": "lenia-rle-v1",
                "cells": source_record["cells"],
            },
            "benchmark": {"index": index, "role": "parental-species"},
        }
        path = destination / filename
        path.write_text(json.dumps(specimen, indent=2) + "\n")
        entries.append(
            {
                "code": source_record["code"],
                "name": source_record["name"],
                "specimen_id": specimen["id"],
                "path": filename,
            }
        )

    manifest = {
        "schema": "genesis.species-manifest/v1",
        "source": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "compatibility": {
            "simulator": "lenia/classic-v1",
            "channels": 1,
            "kernels": 1,
            "kernel": "bump4",
            "growth": "gaussian",
            "radius": 13,
            "time_resolution": 10,
        },
        "entries": entries,
    }
    (destination / "species-benchmark.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="optional local animals.json")
    parser.add_argument("--destination", default="web/specimens")
    args = parser.parse_args()

    if args.source:
        payload = Path(args.source).read_bytes()
    else:
        with urllib.request.urlopen(SOURCE_URL) as response:
            payload = response.read()
    source = json.loads(payload)
    manifest = import_catalog(source, Path(args.destination))
    print(
        json.dumps(
            {
                "imported": len(manifest["entries"]),
                "source_sha256": hashlib.sha256(payload).hexdigest(),
                "source_commit": SOURCE_COMMIT,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
