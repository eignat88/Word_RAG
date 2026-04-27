from __future__ import annotations

BAD_TEXTS = {"", "нет", "нет.", "-", "n/a", "none"}
NOISE_SECTIONS = {
    "история изменений документа",
    "отчеты и выходные формы",
    "ограничение доступа",
    "допущения и ограничения",
}


def normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def should_skip_chunk(text: str, section: str, min_chars: int = 100) -> bool:
    clean = normalize(text)
    sec = normalize(section)

    if clean in BAD_TEXTS:
        return True

    if len(clean) < min_chars:
        return True

    if sec in NOISE_SECTIONS and clean in BAD_TEXTS:
        return True

    return False
