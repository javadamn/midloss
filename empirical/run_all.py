#!/usr/bin/env python3
"""One-command reproduction of the regional blurring manuscript.

Usage
-----
    python run_all.py --clean

The command starts from the raw files in ``data/`` and regenerates the QC
exports, nested-LOSO model predictions, exact tests, participant bootstraps,
sensitivity analyses, figures, manuscript tables, tracked and clean LaTeX manuscripts/PDFs, numeric traceability table,
and SHA-256 artifact manifest.
"""
from __future__ import annotations
import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent


def run(script: str) -> None:
    print(f"\n>>> {script}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest() -> None:
    rows = []
    root_files = [ROOT / name for name in ["README.md", "SCIENTIFIC_AUDIT.md", "COAUTHOR_COMMENT_RESPONSES.md", "CHANGELOG.md", "NEW_REFERENCES.txt", "config.json", "requirements.txt", "run_all.py"]]
    candidates = [p for p in root_files if p.exists()]
    for folder in ["data", "src", "scripts", "templates", "outputs", "manuscript", "audit"]:
        base = ROOT / folder
        if base.exists():
            candidates.extend(sorted(base.rglob("*")))
    for p in candidates:
        if p.is_file() and "rendered" not in p.parts and not p.name.endswith((".aux", ".log", ".out")) and "contact_sheet" not in p.name:
            rows.append({
                "relative_path": str(p.relative_to(ROOT)),
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            })
    pd.DataFrame(rows).to_csv(ROOT / "PACKAGE_MANIFEST.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="remove generated outputs before rerunning")
    args = ap.parse_args()
    if args.clean:
        for path in [ROOT / "outputs", ROOT / "manuscript"]:
            if path.exists():
                shutil.rmtree(path)
        (ROOT / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
        (ROOT / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
        (ROOT / "manuscript").mkdir(parents=True, exist_ok=True)

    run("01_primary_analysis.py")
    run("03_fast_decision_bootstrap.py")
    run("04_sensitivities_and_checks.py")
    run("05_make_figures.py")
    run("06_generate_manuscript.py")
    run("07_compile_manuscript.py")
    manifest()

    checks = pd.read_csv(ROOT / "outputs" / "tables" / "workflow_validation_checks.csv")
    if not checks["passed"].all():
        raise RuntimeError("At least one workflow validation check failed.")
    print(f"\nReproduction complete: {int(checks['passed'].sum())}/{len(checks)} validation checks passed.")


if __name__ == "__main__":
    main()
