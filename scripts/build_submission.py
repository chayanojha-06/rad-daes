"""Assemble the submission bundle.

The script reads the frozen repository and writes ``submission/`` plus a zip.
It never modifies results, configuration, or the manuscript.  Re-running it is
safe: the generated ``submission/`` directory is rebuilt from scratch.

Usage:
    python scripts/build_submission.py
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import time
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submission"
STAMP = date.today().isoformat()

EXCLUDE_DIRS = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    "__pycache__",
    "build",
    "dist",
    "submission",
    "results_smoke",
}
EXCLUDE_SUFFIXES = {".aux", ".log", ".out", ".blg", ".bbl", ".synctex.gz", ".pyc"}
EXCLUDE_NAMES = {"cover_letter.docx"}  # copied once, at the bundle root


README = """# Rad-DAES submission bundle ({stamp})

Manuscript: *Guarded Drift-Aware Background Tracking for Synthetic
Radiation-Portal Count Streams: A Reproducible Benchmark and a Negative Result*

## Contents

| Path | What it is |
| --- | --- |
| `manuscript.pdf` | Compiled manuscript (15 pages) |
| `cover_letter.pdf` / `.docx` | Venue-neutral draft cover letter |
| `supplementary_README.md` | Supplementary-material note |
| `repository/` | Full self-contained study: source, configs, tests, docs, frozen results, tables, figures |
| `CHECKSUMS.sha256` | SHA-256 for every file in this bundle |

## Scope

- Synthetic data only. No real radiation stream was obtained; access blockers
  are documented in `repository/docs/submission_checklist.md` and in Section 6
  of the manuscript.
- No novelty claim and no field-readiness claim. The manuscript reports a
  negative result and the mechanism behind it.
- Author names, affiliations, and the target venue are placeholders. Nothing
  has been submitted or published externally.

## Verify

From this directory:

```powershell
Get-Content CHECKSUMS.sha256 | ForEach-Object {
  $hash, $path = $_ -split '  ', 2
  $actual = (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower()
  if ($actual -ne $hash) { "MISMATCH: $path" }
}
```

or, where `sha256sum` is available:

```bash
sha256sum -c CHECKSUMS.sha256
```

## Reproduce

Inside `repository/`:

```bash
python -m pip install -e .
python -m pytest -q
python scripts/run_benchmarks.py --mode full --output results/raw
python scripts/aggregate_results.py
python scripts/make_figures.py
python scripts/audit_results.py
```

The full benchmark took about 34 minutes on the development machine. The
aggregation, figure, and audit steps are fast and do not retrain anything.

## Note on frozen results

`repository/results/raw/` is the frozen test matrix (1,440 runs). It was not
regenerated or edited after the held-out evaluation. The only post-run edit was
rewriting the trace-path column of `trace_index.csv` when the raw files moved
into `results/raw/`; this is documented in
`repository/results/raw/README.md`.
"""


def keep(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDE_DIRS or part.endswith(".egg-info") for part in rel.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    if path.name in EXCLUDE_NAMES:
        return False
    return True


def copy_tree(src: Path, dst: Path) -> None:
    for item in sorted(src.rglob("*")):
        if item.is_dir() or not keep(item):
            continue
        target = dst / item.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def force_remove(path: Path) -> None:
    """Remove a tree even when OneDrive or a read-only bit blocks deletion."""
    for item in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            os.chmod(item, stat.S_IWRITE)
        except OSError:
            pass
    try:
        os.chmod(path, stat.S_IWRITE)
    except OSError:
        pass
    for attempt in range(4):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            # OneDrive can leave an empty reparse point after its children go.
            try:
                os.rmdir(path)
                return
            except OSError:
                pass
            if attempt == 3:
                raise
            time.sleep(0.5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if SUB.parent != ROOT or SUB.name != "submission":
        raise SystemExit(f"refusing to clean unexpected path: {SUB}")
    if SUB.exists():
        force_remove(SUB)
    SUB.mkdir()

    repo = SUB / "repository"
    copy_tree(ROOT, repo)

    shutil.copy2(ROOT / "paper" / "main.pdf", SUB / "manuscript.pdf")
    shutil.copy2(ROOT / "paper" / "cover_letter.pdf", SUB / "cover_letter.pdf")
    shutil.copy2(ROOT / "paper" / "cover_letter.docx", SUB / "cover_letter.docx")
    shutil.copy2(
        ROOT / "paper" / "supplementary" / "README.md",
        SUB / "supplementary_README.md",
    )

    (SUB / "README.md").write_text(README.replace("{stamp}", STAMP), encoding="utf-8")

    lines = []
    for path in sorted(SUB.rglob("*")):
        if path.is_file() and path.name != "CHECKSUMS.sha256" and path.suffix != ".zip":
            lines.append(f"{sha256(path)}  {path.relative_to(SUB).as_posix()}")
    (SUB / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    zip_path = SUB / f"Rad-DAES_submission_{STAMP}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(SUB.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(SUB).as_posix())

    n_files = len(lines)
    size_mb = zip_path.stat().st_size / (1 << 20)
    print(f"bundle: {SUB}")
    print(f"files hashed: {n_files}")
    print(f"zip: {zip_path.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
