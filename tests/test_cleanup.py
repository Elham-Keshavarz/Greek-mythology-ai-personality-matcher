"""Offline migration checks: protect secrets and user files during layout cleanup."""
from __future__ import annotations

from tools.cleanup_legacy_names import obsolete_files


def test_cleanup_lists_only_known_superseded_paths(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs" / "EVIDENCE_V05_LEARNING_GUIDE.md").write_text("old")
    (tmp_path / "docs" / "EVIDENCE_VALIDATION_GUIDE.md").write_text("new")
    (tmp_path / "tests" / "test_llm_v06.py").write_text("old")
    (tmp_path / "tests" / "test_llm.py").write_text("new")
    (tmp_path / "tests" / "test_custom_v03.py").write_text("keep")
    (tmp_path / ".env").write_text("keep")
    assert {p.relative_to(tmp_path).as_posix() for p in obsolete_files(tmp_path)} == {
        "docs/EVIDENCE_V05_LEARNING_GUIDE.md", "tests/test_llm_v06.py"
    }


def test_cleanup_refuses_to_target_files_without_replacement(tmp_path):
    (tmp_path / "data" / "evaluation").mkdir(parents=True)
    p = tmp_path / "data" / "evaluation" / "llm_qa_v06.json"
    p.write_text("old")
    assert obsolete_files(tmp_path) == []
