"""Load small, transparently attributed, author-written mythology knowledge documents."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "data" / "archetype_profiles.json"
FACT_PATH = ROOT / "data" / "knowledge" / "curated_facts.json"


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source_url: str
    source_type: str
    source_note: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        return cls(**data)


def load_documents(profile_path: Path = PROFILE_PATH, fact_path: Path = FACT_PATH) -> list[Document]:
    """Return the 45 short deity profiles plus independently curated factual notes."""
    profiles = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    facts = json.loads(Path(fact_path).read_text(encoding="utf-8"))
    documents: list[Document] = []
    for profile in profiles:
        name = profile["name"]
        documents.append(Document(
            id="profile-" + name.lower().replace(" ", "-"),
            title=f"{name}: concise mythological profile",
            text=f"{name} is classified here as a {profile['category']} deity. {profile['description']}",
            source_url=profile["source_url"],
            source_type="project_profile",
            source_note="Project-authored overview; linked category page is background reading, not a claim-level source.",
        ))
    for fact in facts:
        documents.append(Document(
            id=fact["id"], title=fact["title"], text=fact["text"],
            source_url=fact["source_url"], source_type="curated_summary",
            source_note=fact["source_note"],
        ))
    if len({doc.id for doc in documents}) != len(documents):
        raise ValueError("Knowledge document IDs must be unique")
    if any(not d.text.strip() or not d.source_url.startswith("https://") for d in documents):
        raise ValueError("Each knowledge document requires text and an HTTPS source URL")
    return documents
