"""Offline sanity checks before committing a public GitHub release.

Checks source packaging, basic secret patterns, and mandatory ignore rules.
No network, provider keys, Streamlit or large model dependencies required.
This is defense-in-depth; review `git diff --cached` before every push.
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md", "LICENSE", "requirements.txt", "requirements-ci.txt",
    ".env.example", ".gitignore", ".github/workflows/ci.yml",
    "app.py", "src/assessment.py", "src/rag/ask.py",
    "data/archetype_profiles.json", "data/knowledge/curated_facts.json",
    "docs/ARCHITECTURE.md", "docs/EVALUATION.md", "docs/PRIVACY.md",
)
IGNORED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints", ".idea", ".vscode",
}
TEXT_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".csv", ".ipynb"}
SECRET_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{25,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{35,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{25,}\b"),
    re.compile(r"(?im)^\s*OPENAI_API_KEY\s*=\s*['\"]?(?:sk-[^\s#'\"]{15,})"),
)


def contains_possible_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def release_text_files(root: Path) -> list[Path]:
    """List GitHub-publishable text (including notebook JSON); ignore local-only files."""
    selected = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in IGNORED_DIRS for part in relative.parts[:-1]):
            continue
        if path.name in (".env", "secrets.toml") or (
            path.name.startswith(".env.") and path.name != ".env.example"
        ):
            continue
        if relative.parts[:2] == ("data", "rag_index"):
            continue
        if relative.parts[:2] == ("reports", "local_"):
            continue
        if relative.parts[0] == "reports" and (
            path.name.startswith("local_") or "_live" in path.name
        ):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in (".gitignore", ".env.example"):
            selected.append(path)
    return selected


def check_release(root: Path = PROJECT_ROOT) -> list[str]:
    errors = []
    for filename in REQUIRED:
        if not (root / filename).is_file():
            errors.append(f"Missing required release file: {filename}")
    ignored = (root / ".gitignore")
    if ignored.exists():
        patterns = set(ignored.read_text(encoding="utf-8").splitlines())
        for pattern in (".env", ".env.*", "!.env.example", ".venv/"):
            if pattern not in patterns:
                errors.append(f"Missing safety ignore rule: {pattern}")
    for path in release_text_files(root):
        content = path.read_text(encoding="utf-8", errors="replace")
        if contains_possible_secret(content):
            errors.append(f"Potential credential in: {path.relative_to(root).as_posix()}")
    return errors


def main() -> int:
    errors = check_release()
    if errors:
        for error in errors:
            print("FAIL:", error)
        return 1
    print(f"PASS: release structure, ignore rules, and basic credential patterns; "
          f"scanned {len(release_text_files(PROJECT_ROOT))} source documents")
    print("Note: still review `git status` and `git diff --cached` before pushing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
