"""Presentation cleanup only; course identity and deduplication belong to retrieval."""
import re


def clean_course_title(title: str) -> str:
    # Scraper UI artifacts are not part of the course title.
    title = re.sub(
        r"\b(?:warning_amber_outline|info insegnamenti(?: erogati)?|salva pdf|cambia|percorso comune)\b",
        " ", title, flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", title).strip()
