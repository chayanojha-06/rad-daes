"""Render the cover letter as a DOCX file.

The PDF is the primary artifact; this script exists so the editable version
stays in sync with it.  It reads no results and touches no frozen artifacts.

Usage:
    python scripts/make_cover_letter_docx.py
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "cover_letter.docx"

PARAGRAPHS: list[tuple[str, str]] = [
    ("right", "Rad-DAES Project"),
    ("right", "28 October 2026"),
    ("body", "Dear Editor,"),
    (
        "body",
        "Please consider the enclosed manuscript, Guarded Drift-Aware Background "
        "Tracking for Synthetic Radiation-Portal Count Streams: A Reproducible "
        "Benchmark and a Negative Result, as a first-draft research submission.",
    ),
    (
        "body",
        "The manuscript asks whether a guarded drift-aware step-size controller "
        "improves the detection/false-alarm trade-off for radiation-portal count "
        "streams relative to static, EWMA, RLS, Adam, ungated DAES, fixed-step "
        "gated DAES, and an oracle gate. It does not. On a frozen 32-scenario, "
        "five-seed, nine-method matrix (1,440 runs), no deployable method achieves "
        "both useful event detection and an acceptable false-alarm rate. We report "
        "that outcome directly, together with the mechanism visible in the saved "
        "traces, rather than searching for a configuration that would reverse it.",
    ),
    ("body", "The submission is deliberately narrow:"),
    (
        "bullet",
        "Synthetic data only. No real radiation stream was obtained; the "
        "data-access blockers are documented in the manuscript and repository.",
    ),
    (
        "bullet",
        "No novelty claim. The candidate is presented as a hypothesis under test, "
        "and the related-work audit is described as rapid and non-systematic.",
    ),
    (
        "bullet",
        "No field-readiness claim. The result is a benchmark and an explicit "
        "evidence boundary, not an operational validation.",
    ),
    (
        "body",
        "What we believe is reusable is the protocol and the audit trail: a frozen "
        "development/test split with a held-out drift regime, a pre-specified "
        "selection rule, 44 passing tests, and a 22-check reproducibility audit "
        "that recomputes saved rows from seeds and verifies every table, figure, "
        "and macro used in the manuscript.",
    ),
    (
        "body",
        "The manuscript, supplementary note, configuration files, source, tests, "
        "result tables, figures, and raw artifacts are included in the submission "
        "bundle. The repository is self-contained; the command sequence in "
        "Section 7 regenerates every reported number.",
    ),
    (
        "body",
        "We suggest reviewers focus on three points: (i) whether the development "
        "selection rule is a fair test of the candidate, given that four of nine "
        "methods fell back; (ii) whether the gate-deadlock explanation is "
        "adequately supported by a single representative trace; and (iii) whether "
        "the negative result is scoped tightly enough to be useful.",
    ),
    (
        "body",
        "Author names, affiliations, and the target venue are placeholders in this "
        "draft. No external publication has been performed.",
    ),
    ("body", "Sincerely,"),
    ("body", "Rad-DAES Project"),
    ("body", "[Corresponding author: name, affiliation, email]"),
]


def main() -> None:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    for kind, text in PARAGRAPHS:
        if kind == "right":
            para = doc.add_paragraph(text)
            para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        elif kind == "bullet":
            doc.add_paragraph(text, style="List Bullet")
        else:
            doc.add_paragraph(text)

    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
