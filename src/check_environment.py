"""Check that the project's Python environment has its declared dependencies.

Run from project root: .venv\\Scripts\\python -m src.check_environment
This check does not download embedding models or call an LLM API.
"""
from __future__ import annotations

import importlib
import importlib.util
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

DEPENDENCIES = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "pytest": "pytest",
    "jupyterlab": "jupyterlab",
    "sentence_transformers": "sentence-transformers",
    "faiss": "faiss-cpu",
    "rapidfuzz": "rapidfuzz",
    "openai": "openai",
    "dotenv": "python-dotenv",
}


def check_environment() -> int:
    print(f"Python: {sys.version.split()[0]} | {platform.system()} {platform.machine()}")
    missing = []
    for module, distribution in DEPENDENCIES.items():
        try:
            located = importlib.util.find_spec(module) is not None
        except (ModuleNotFoundError, ValueError):
            located = False
        if not located:
            missing.append(distribution)
            print(f"MISSING  {distribution} (import: {module})")
            continue
        try:
            installed = version(distribution)
        except PackageNotFoundError:
            installed = "unknown"
        if module == "faiss":
            try:
                importlib.import_module("faiss")
            except Exception as exc:
                print(f"BROKEN   faiss-cpu ({type(exc).__name__}: {exc})")
                missing.append("faiss-cpu")
                continue
        print(f"OK       {distribution}=={installed}")
    if missing:
        print("\nFix missing or broken packages in THIS environment:")
        print(r".venv\Scripts\python -m pip install -r requirements.txt")
        return 1
    print("\nAll declared packages are present. This does not test model download or RAG answer quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(check_environment())
