"""
prepare_submission.py — Build the HackerRank Orchestrate submission archive.

Note: forces stdout to UTF-8 so coloured output renders on Windows terminals.

Run from the repo root:
    python prepare_submission.py

Creates submission.zip containing exactly the required code/ files:
    code/main.py
    code/router.py
    code/generator.py
    code/ingest.py
    code/requirements.txt
    code/README.md

Explicitly EXCLUDED (per HackerRank rules):
    • code/chroma_db/         (local vector DB — too large, not needed)
    • code/.env               (secrets — never commit)
    • code/__pycache__/       (build artefacts)
    • code/*.pyc              (compiled bytecode)
    • data/                   (corpus — already in the repo, not re-submitted)
    • support_tickets/*.csv   (graded separately as "Predictions CSV")
    • .venv / venv /          (virtualenvs)
    • *.egg-info / dist/      (packaging artefacts)
"""

import io
import os
import sys
import zipfile
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 encode errors for box chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Script lives in the repo root — resolve paths relative to it
REPO_ROOT   = Path(__file__).parent.resolve()
CODE_DIR    = REPO_ROOT / "code"
OUTPUT_ZIP  = REPO_ROOT / "submission.zip"

# The ONLY files that go into the zip (arcname = path inside the zip)
INCLUDE_FILES: list[tuple[Path, str]] = [
    (CODE_DIR / "main.py",          "code/main.py"),
    (CODE_DIR / "router.py",        "code/router.py"),
    (CODE_DIR / "generator.py",     "code/generator.py"),
    (CODE_DIR / "ingest.py",        "code/ingest.py"),
    (CODE_DIR / "requirements.txt", "code/requirements.txt"),
    (CODE_DIR / "README.md",        "code/README.md"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"


def _fmt_size(n_bytes: int) -> str:
    """Return a human-readable size string (B / KB / MB)."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1_048_576:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes / 1_048_576:.2f} MB"


def _check_missing(files: list[tuple[Path, str]]) -> list[str]:
    """Return a list of human-readable errors for any missing source files."""
    errors = []
    for src, arcname in files:
        if not src.exists():
            errors.append(f"  ✗ MISSING: {src}  (expected at {arcname})")
    return errors


def _build_zip(
    files: list[tuple[Path, str]],
    output: Path,
) -> list[tuple[str, int]]:
    """Create the zip and return a manifest of (arcname, compressed_size) pairs."""
    manifest: list[tuple[str, int]] = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for src, arcname in files:
            zf.write(src, arcname=arcname)
            info = zf.getinfo(arcname)
            manifest.append((arcname, info.compress_size))
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    SEP = "-" * 54
    print(f"\n{BOLD}{SEP}{RESET}")
    print(f"{BOLD}  HackerRank Orchestrate -- Submission Packager{RESET}")
    print(f"{BOLD}{SEP}{RESET}\n")

    # ── Pre-flight: verify all source files exist ─────────────────────────────
    errors = _check_missing(INCLUDE_FILES)
    if errors:
        print(f"{RED}[ERROR] Some source files are missing:{RESET}")
        for e in errors:
            print(e)
        print(
            f"\n  Make sure you have run the full pipeline first:\n"
            f"  1. python code/ingest.py\n"
            f"  2. python code/main.py\n"
        )
        sys.exit(1)

    # ── Warn if chroma_db is accidentally present inside code/ ────────────────
    chroma_path = CODE_DIR / "chroma_db"
    if chroma_path.exists():
        print(
            f"{YELLOW}[INFO] code/chroma_db/ exists and will be "
            f"EXCLUDED from the zip (correct behaviour).{RESET}"
        )

    # ── Warn if .env is present (should never be committed) ───────────────────
    env_path = CODE_DIR / ".env"
    if env_path.exists():
        print(
            f"{YELLOW}[INFO] code/.env exists and will be "
            f"EXCLUDED from the zip (secrets stay local).{RESET}"
        )

    print(f"\nBuilding {BOLD}submission.zip{RESET} …\n")

    # ── Build the zip ─────────────────────────────────────────────────────────
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()   # overwrite cleanly

    manifest = _build_zip(INCLUDE_FILES, OUTPUT_ZIP)

    # ── Print manifest ────────────────────────────────────────────────────────
    col_w = max(len(a) for a, _ in manifest) + 2
    print(f"  {'File':<{col_w}}  {'Compressed'}")
    print(f"  {'-' * col_w}  {'-' * 10}")
    for arcname, comp_size in manifest:
        print(f"  {GREEN}{arcname:<{col_w}}{RESET}  {_fmt_size(comp_size)}")

    # ── Final stats ───────────────────────────────────────────────────────────
    zip_bytes      = OUTPUT_ZIP.stat().st_size
    zip_kb         = zip_bytes / 1024
    n_files        = len(manifest)

    print(f"\n  {'-' * (col_w + 14)}")
    print(f"  {BOLD}Files included :{RESET} {n_files}")
    print(f"  {BOLD}Archive size   :{RESET} {CYAN}{zip_kb:.1f} KB{RESET}  ({_fmt_size(zip_bytes)})")
    print(f"  {BOLD}Saved to       :{RESET} {OUTPUT_ZIP}\n")

    # ── Exclusion summary ─────────────────────────────────────────────────────
    print(f"  {YELLOW}Excluded (per HackerRank rules):{RESET}")
    excluded = [
        "code/chroma_db/         ← local vector DB",
        "code/.env               ← secrets",
        "code/__pycache__/       ← build artefacts",
        "code/*.pyc              ← compiled bytecode",
        "data/                   ← corpus (in repo, not re-submitted)",
        "support_tickets/*.csv   ← graded as separate 'Predictions CSV'",
        ".venv / venv /          ← virtualenv",
    ]
    for item in excluded:
        print(f"    ✗  {item}")

    print(f"\n{GREEN}{BOLD}[OK] submission.zip is ready.{RESET}")
    print(
        f"\n  {BOLD}Submit at:{RESET}\n"
        f"  https://www.hackerrank.com/contests/hackerrank-orchestrate-may26"
        f"/challenges/support-agent/submission\n"
        f"\n  Upload these 3 files:\n"
        f"    1. {BOLD}submission.zip{RESET}          ← your code/ archive (just built)\n"
        f"    2. {BOLD}support_tickets/output.csv{RESET} ← your predictions\n"
        f"    3. {BOLD}~/hackerrank_orchestrate/log.txt{RESET} ← chat transcript\n"
    )
    print(f"{BOLD}{SEP}{RESET}\n")


if __name__ == "__main__":
    main()
