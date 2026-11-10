"""Offline CI checks for publication and manual QA review utilities."""
from __future__ import annotations

import csv
import json

import pytest

from tools.check_release import PROJECT_ROOT, check_release, contains_possible_secret, release_text_files
from src.rag.faithfulness_review import export_review_csv, make_review_rows, summarize_review_csv


def test_release_scans_clean_repository():
    assert check_release(PROJECT_ROOT) == []
    names = {p.relative_to(PROJECT_ROOT).as_posix() for p in release_text_files(PROJECT_ROOT)}
    assert ".env.example" in names
    assert "src/rag/ask.py" in names
    assert "notebooks/01_data_exploration.ipynb" in names


def test_key_heuristic_does_not_mistake_placeholder_for_secret():
    assert not contains_possible_secret("OPENAI_API_KEY=your_actual_api_key")
    assert contains_possible_secret("OPENAI_API_KEY=sk-" + "a" * 35)
    assert contains_possible_secret("ghp_" + "b" * 40)


def test_export_and_summary_requires_human_grading(tmp_path):
    report = {"results": [
        {"question": "Who is Athena?", "expected_supported": True, "mode": "llm",
         "answer": "Athena is linked with wisdom [S1].", "citation_ids": ["S1"],
         "background_urls": ["https://example.org"], "supporting_passage": "Athena is linked with wisdom.",
         "review_hint": "wisdom"},
        {"question": "Zeus smartphone?", "expected_supported": False, "mode": "insufficient_evidence",
         "answer": "No evidence", "citation_ids": [], "background_urls": [], "review_hint": "abstain"},
    ]}
    rows = make_review_rows(report)
    assert rows[0]["review_label"] == "not_reviewed"
    assert rows[1]["review_label"] == "not_applicable"
    assert rows[0]["approved_passage"] == "Athena is linked with wisdom."
    path = tmp_path / "local_review.csv"
    assert export_review_csv(report, path) == 2
    result = summarize_review_csv(path)
    assert result["faithfulness_rate_on_binary_reviewed_answers"] is None
    assert result["not_reviewed_count"] == 1
    with path.open(encoding="utf-8-sig", newline="") as handle:
        saved = list(csv.DictReader(handle))
    saved[0]["review_label"] = "supported"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=saved[0].keys())
        writer.writeheader()
        writer.writerows(saved)
    result = summarize_review_csv(path)
    assert result["faithfulness_rate_on_binary_reviewed_answers"] == 1.0
    assert result["manually_reviewed_binary_count"] == 1


def test_invalid_review_labels_are_rejected(tmp_path):
    path = tmp_path / "local_invalid.csv"
    path.write_text("question,mode,review_label\nX,llm,perfect\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid review_label"):
        summarize_review_csv(path)


def test_malformed_audit_fails_closed():
    with pytest.raises(ValueError):
        make_review_rows({"results": "wrong type"})


def test_notebooks_are_scanned_for_api_keys_in_cells_and_outputs(tmp_path):
    # Jupyter notebooks are JSON text, including saved source cells and outputs.
    folder = tmp_path / "notebooks"
    folder.mkdir()
    notebook = folder / "demo.ipynb"
    payload = {
        "cells": [{"cell_type": "code", "source": ["print('hello')"],
                   "outputs": [{"output_type": "stream", "text": ["OK"]}]}],
        "nbformat": 4, "nbformat_minor": 5, "metadata": {},
    }
    notebook.write_text(json.dumps(payload), encoding="utf-8")
    assert notebook in release_text_files(tmp_path)
    assert not any("Potential credential" in item for item in check_release(tmp_path))

    # Simulate a key accidentally saved in the source cell.
    payload["cells"][0]["source"] = ["OPENAI_API_KEY=sk-" + "a" * 35]
    notebook.write_text(json.dumps(payload), encoding="utf-8")
    assert "Potential credential in: notebooks/demo.ipynb" in check_release(tmp_path)

    # A key in a cell output must also be found.
    payload["cells"][0]["source"] = ["print('safe')"]
    payload["cells"][0]["outputs"][0]["text"] = ["sk-" + "b" * 35]
    notebook.write_text(json.dumps(payload), encoding="utf-8")
    assert "Potential credential in: notebooks/demo.ipynb" in check_release(tmp_path)


def test_notebook_scanner_ignores_notebook_checkpoints(tmp_path):
    ignored = tmp_path / "notebooks" / ".ipynb_checkpoints"
    ignored.mkdir(parents=True)
    notebook = ignored / "hidden.ipynb"
    notebook.write_text('"sk-' + 'x' * 35 + '"', encoding="utf-8")
    assert notebook not in release_text_files(tmp_path)
