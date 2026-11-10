"""Optional migration utility: remove obsolete, numbered filenames after an update.

Dry-run by default. Never touches application data, secrets, virtualenvs,
notebooks or user-created files. Only removes a known superseded filename
when the replacement is present in the project root.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Narrow filename patterns; these are *project naming* patterns, not dependency versions.
RENAMES = (
    ("docs", r"EVIDENCE_V\d+_LEARNING_GUIDE\.md", "docs/EVIDENCE_VALIDATION_GUIDE.md"),
    ("docs", r"LLM_V\d+_LEARNING_GUIDE\.md", "docs/LLM_GENERATION_GUIDE.md"),
    ("docs", r"PERSONALIZATION_V\d+_LEARNING_GUIDE\.md", "docs/PERSONALIZATION_GUIDE.md"),
    ("docs", r"RETRIEVAL_V\d+_LEARNING_GUIDE\.md", "docs/RETRIEVAL_LEARNING_GUIDE.md"),
    ("tests", r"test_evaluation_v\d+\.py", "tests/test_evaluation.py"),
    ("tests", r"test_evidence_v\d+\.py", "tests/test_evidence.py"),
    ("tests", r"test_llm_v\d+\.py", "tests/test_llm.py"),
    ("tests", r"test_personalization_v\d+\.py", "tests/test_personalization.py"),
    ("tests", r"test_release_v\d+\.py", "tests/test_publication.py"),
    ("data/evaluation", r"llm_qa_v\d+\.json", "data/evaluation/llm_qa.json"),
    ("data/evaluation", r"retrieval_challenge_v\d+\.json", "data/evaluation/retrieval_challenge.json"),
    ("data/evaluation", r"retrieval_eval_v\d+\.json", "data/evaluation/retrieval_regression.json"),
    ("reports", r"retrieval_v\d+_observed\.json", "reports/retrieval_observed.json"),
    (".", r"CHANGELOG\.md", "DEVELOPMENT_NOTES.md"),
)


def obsolete_files(root: Path = ROOT) -> list[Path]:
    """List only superseded files with a verified replacement in the same root."""
    obsolete: list[Path] = []
    for folder, pattern, replacement in RENAMES:
        directory = root / folder
        if not (root / replacement).is_file() or not directory.is_dir():
            continue
        matcher = re.compile(pattern, re.IGNORECASE)
        obsolete.extend(path for path in directory.iterdir()
                        if path.is_file() and matcher.fullmatch(path.name))
    return sorted(obsolete)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Delete only the listed obsolete files")
    args = parser.parse_args()
    files = obsolete_files()
    if not files:
        print("Nothing to clean. Project filenames are current.")
        return 0
    for path in files:
        print(("Removing: " if args.apply else "Would remove: ") + str(path.relative_to(ROOT)))
        if args.apply:
            path.unlink()
    if not args.apply:
        print("Dry run. Add --apply to remove these files after making a backup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
