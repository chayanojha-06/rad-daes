"""Inspect the configured data sources and demonstrate the synthetic generator.

This script never downloads data and never uses credentials. It reports whether any real
radiation data are present locally, and writes a small audit record.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rad_daes.data.synthetic import FAULT_MODES, REGIMES, generate_stream  # noqa: E402
from rad_daes.evaluation.runner import environment_manifest, load_configs  # noqa: E402


def _candidate_real_data_dirs(root: Path) -> list[Path]:
    return [
        root / "data",
        root / "data" / "raw",
        root / "data" / "external",
        root / "data" / "radai",
        root / "data" / "pyriid",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    root = args.root.resolve()
    data_cfg, model_cfg, exp_cfg = load_configs(root)

    stream = generate_stream(
        seed=args.seed,
        regime="stable",
        fault_condition="clean",
        duration_seconds=min(21600, int(data_cfg["sampling"]["duration_seconds"])),
    )

    real_dirs = []
    for path in _candidate_real_data_dirs(root):
        files = sorted(str(p.relative_to(root)) for p in path.glob("*") if p.is_file()) if path.exists() else []
        real_dirs.append({"path": str(path.relative_to(root)), "exists": path.exists(), "files": files})

    audit = {
        "environment": environment_manifest(),
        "platform": platform.platform(),
        "root": str(root),
        "configs": {
            "sampling": data_cfg["sampling"],
            "windows": data_cfg["windows"],
            "seeds": data_cfg["seeds"],
            "methods": exp_cfg["methods"],
        },
        "real_data_candidates": real_dirs,
        "real_data_present": any(item["files"] for item in real_dirs),
        "synthetic_generator": stream.metadata,
        "synthetic_regimes": list(REGIMES),
        "synthetic_fault_modes": list(FAULT_MODES),
        "synthetic_stream_summary": {
            "seed": int(stream.seed),
            "duration_seconds": int(stream.duration_seconds),
            "shape": list(stream.x.shape),
            "source_samples": int(stream.y.sum()),
            "source_events": int(len(np.unique(stream.event_id)) - 1),
            "fault_samples": int(np.sum(stream.fault != "none")),
            "nan_samples": int(np.sum(~np.isfinite(stream.x).all(axis=1))),
            "nominal_cps": stream.metadata["nominal_cps"],
        },
        "evidence_boundary": (
            "No real radiation data are used. The benchmark is synthetic-only. "
            "The original DAES implementation was read from an external snapshot; it is not "
            "vendored or modified here."
        ),
    }

    output = args.output or (root / "results" / "raw" / "data_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit["synthetic_stream_summary"], indent=2))
    print(f"real_data_present={audit['real_data_present']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
