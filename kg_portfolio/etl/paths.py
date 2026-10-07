"""Where the inputs live.

Every path the pipeline reads is resolved here, so nothing downstream carries a
machine-specific string.

The source repositories are expected to sit side by side in one directory:

    <root>/ARC-data/
    <root>/ARC-behavioral/
    <root>/arc-error-grouping/
    <root>/cogarc-human-analysis/kg_portfolio/   <- this one
    <root>/cogarc-kg-site/

`<root>` defaults to the directory two levels above this repository, which is
already correct for that layout. Override any of it with environment variables:

    COGARC_ROOT             the directory holding the repositories
    COGARC_ARC_DATA         ARC-data checkout
    COGARC_ARC_BEHAVIORAL   ARC-behavioral checkout
    COGARC_ERROR_GROUPING   arc-error-grouping checkout
    COGARC_SITE_REPO        cogarc-kg-site checkout (export target)

Call `require(path, ...)` before reading something, so a wrong layout fails with
a message naming the variable to set rather than a bare FileNotFoundError.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# etl/paths.py -> etl -> kg_portfolio -> cogarc-human-analysis -> <root>
HERE = Path(__file__).resolve().parent
PORTFOLIO = HERE.parent
ROOT = Path(os.environ.get("COGARC_ROOT", PORTFOLIO.parent.parent))


def _repo(env: str, name: str) -> Path:
    return Path(os.environ[env]).expanduser() if env in os.environ else ROOT / name


ARC_DATA = _repo("COGARC_ARC_DATA", "ARC-data")
ARC_BEHAVIORAL = _repo("COGARC_ARC_BEHAVIORAL", "ARC-behavioral")
ERROR_GROUPING = _repo("COGARC_ERROR_GROUPING", "arc-error-grouping")
SITE = _repo("COGARC_SITE_REPO", "cogarc-kg-site")

# ── inputs ──────────────────────────────────────────────────────────────────
KEYWORD_STUDY = ARC_DATA / "MTurk-Keywords"                        # reports, submissions
KEYWORD_TESTSET = ARC_DATA / "ChatGPT-Keywords" / "keywords-testset"  # puzzles as shown
KEYWORD_PNG = ARC_DATA / "ChatGPT-Keywords" / "keywords-png"       # the rendered stimuli
ARC_TRAINING = ARC_BEHAVIORAL / "data" / "training"                # the ARC tasks
LLM_KEYWORDS = ERROR_GROUPING / "Keywords"                         # the 2024 model runs
KMEANS_LABELS = LLM_KEYWORDS / "keyword_cluster_labels_K6.csv"     # published partition

# ── outputs, all inside this repository ─────────────────────────────────────
DB = PORTFOLIO / "cogarc_kg.kuzu"
DATA = PORTFOLIO / "data"
QUERIES = PORTFOLIO / "queries"
SITE_DATA = SITE / "src" / "data"

_ENV_FOR = {
    ARC_DATA: "COGARC_ARC_DATA",
    ARC_BEHAVIORAL: "COGARC_ARC_BEHAVIORAL",
    ERROR_GROUPING: "COGARC_ERROR_GROUPING",
    SITE: "COGARC_SITE_REPO",
}


def require(*paths: Path) -> None:
    """Exit with an actionable message if an input is not where we expect."""
    missing = [p for p in paths if not Path(p).exists()]
    if not missing:
        return
    lines = ["Cannot find these inputs:"]
    for p in missing:
        p = Path(p)
        env = next((v for repo, v in _ENV_FOR.items() if repo in p.parents or repo == p),
                   None)
        lines.append(f"  {p}" + (f"\n    set {env} to the right checkout" if env else ""))
    lines.append(f"\nLooking for the repositories under: {ROOT}")
    lines.append("Set COGARC_ROOT if they live somewhere else.")
    sys.exit("\n".join(lines))
