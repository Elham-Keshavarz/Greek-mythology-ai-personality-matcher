"""Shared tokenization and lightweight vocabulary used for retrieval ranking."""
from __future__ import annotations
import re

WORDS = re.compile(r"[a-z]+")
STOPWORDS = frozenset('''a an the and or of to for in on at with without by about from who what when where which why how is are was were be been being am do does did can could should would have has had this that these those there here tell me you your i my us they she he her him their its it them greek mythology myth myths god gods goddess goddesses deity deities olympian olympians known as called associated represents represent representing described describe according please more explain find often something some any all between into after before tell known use uses used using'''.split())
TOPIC_SYNONYMS = {
    "traveling": "travel", "travelling": "travel", "travels": "travel", "traveler": "travel",
    "communication": "language", "communicate": "language", "speaking": "language",
    "messenger": "messenger", "messengers": "messenger", "messages": "messenger",
    "clever": "wisdom", "wise": "wisdom", "intelligence": "wisdom",
    "relate": "relationship", "related": "relationship", "relation": "relationship", "relationships": "relationship",
    "sibling": "family", "siblings": "family", "brother": "family", "brothers": "family",
    "sister": "family", "sisters": "family", "twins": "family", "twin": "family",
    "parent": "family", "parents": "family", "father": "family", "mother": "family",
    "children": "family", "child": "family", "son": "family", "daughter": "family",
    "birth": "birth", "born": "birth", "emerges": "birth",
    "hunts": "hunt", "hunting": "hunt", "hunter": "hunt",
    "seas": "sea", "ocean": "sea", "oceans": "sea",
    "thunderstorms": "thunder", "lightning": "thunder",
    "symbols": "symbol", "emblem": "symbol", "emblems": "symbol",
    "crafts": "craft", "craftsmanship": "craft", "craftsman": "craft",
    "trading": "trade", "trades": "trade", "healer": "healing", "heal": "healing",
    "seasonal": "season", "seasons": "season", "agriculture": "grain",
    "musician": "music", "musical": "music",
    "phone": "smartphone", "phones": "smartphone", "mobile": "smartphone",
}
RELATION_TERMS = {"relationship", "family"}
RELATION_EVIDENCE = {"family", "birth", "wife", "husband", "consort", "married", "marriage", "ancestry"}


def canonical(word: str) -> str:
    word = word.lower()
    if word in TOPIC_SYNONYMS:
        return TOPIC_SYNONYMS[word]
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return TOPIC_SYNONYMS.get(word, word)


def content_tokens(text: str, names: tuple[str, ...]) -> set[str]:
    name_tokens = {n.lower() for n in names}
    return {canonical(token) for token in WORDS.findall(text.lower())
            if (len(token) >= 4 or token in {"sea", "sun", "war", "owl"}) and token not in STOPWORDS and token not in name_tokens}


def named_entities(text: str, names: tuple[str, ...]) -> set[str]:
    words = set(WORDS.findall(text.lower()))
    return {name.lower() for name in names if name.lower() in words}


