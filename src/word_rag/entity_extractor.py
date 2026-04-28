from __future__ import annotations

import re

DAX_PATTERN = re.compile(r"\bDAX-\d+\b", re.IGNORECASE)
AX_CLASS_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9_]{3,}\b")


def extract_entities(text: str) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []

    for value in DAX_PATTERN.findall(text):
        entities.append({"type": "dax_task", "value": value.upper()})

    for value in AX_CLASS_PATTERN.findall(text):
        if "_" in value or value.startswith(("WMS", "LFL", "ALK", "Invent", "Sales")):
            entities.append({"type": "ax_object", "value": value})

    unique: dict[tuple[str, str], dict[str, str]] = {}
    for entity in entities:
        key = (entity["type"], entity["value"])
        unique[key] = entity

    return list(unique.values())


def extract_dax_numbers(text: str) -> list[str]:
    return sorted({v.upper() for v in DAX_PATTERN.findall(text)})
