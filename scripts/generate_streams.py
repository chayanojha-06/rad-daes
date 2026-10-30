"""Generate and save a small set of labelled synthetic streams for inspection.

The saved arrays are inputs for sanity checks and figures. They are not measured data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rad_daes.data.synthetic import FAULT_MODES, REGIMES, generate_stream  # noqa: E402
from rad_daes.evaluation.runner import load_configs  # noqa: E402


def _save(stream, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=stream.x,
        t_obs=stream.t_obs,
        mu_true=stream.mu_true,
        source=stream.source,
        y=stream.y,
        fault=stream.fault.astype(str),
        event_id=stream.event_id,
        drift_events=np.array(json.dumps(stream.drift_events)),
        metadata=np.array(json.dumps(stream.metadata)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    root = args.root.resolve()
    data_cfg, _, _ = load_configs(root)
    duration = int(args.duration or data_cfg["sampling"]["duration_seconds"])
    output = (args.output or (root / "results" / "raw" / "streams")).resolve()

    rows = []
    for regime in REGIMES:
        stream = generate_stream(
            seed=args.seed,
            regime=regime,
            fault_condition="clean",
            duration_seconds=duration,
        )
        path = output / f"{regime}_clean_seed{args.seed}.npz"
        _save(stream, path)
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "seed": args.seed,
                "regime": regime,
                "fault_condition": "clean",
                "duration_seconds": duration,
                "source_samples": int(stream.y.sum()),
                "source_events": int(len(np.unique(stream.event_id)) - 1),
            }
        )

    for mode in FAULT_MODES:
        stream = generate_stream(
            seed=args.seed,
            regime="stable",
            fault_condition=f"{mode}_10",
            duration_seconds=duration,
        )
        path = output / f"stable_{mode}_10_seed{args.seed}.npz"
        _save(stream, path)
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "seed": args.seed,
                "regime": "stable",
                "fault_condition": f"{mode}_10",
                "duration_seconds": duration,
                "source_samples": int(stream.y.sum()),
                "source_events": int(len(np.unique(stream.event_id)) - 1),
                "fault_samples": int(np.sum(stream.fault != "none")),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    (output / "index.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} streams to {output}")


if __name__ == "__main__":
    main()
