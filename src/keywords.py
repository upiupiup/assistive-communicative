"""
Loader untuk kosakata/istilah teknis per mata kuliah (dari
`data/keywords_hotwords_matkul.json`), dipakai sebagai `hotwords` di
faster-whisper (lihat asr_engine.transcribe_array).

Kenapa modul terpisah dari config.py: datanya besar (puluhan-ratusan
istilah per matkul) dan sumbernya file eksternal yang bisa kamu update
kapan aja tanpa sentuh kode Python, jadi dipisah biar jelas mana
"konfigurasi" (config.py) vs "data kosakata" (file ini + JSON-nya).
"""

import json
from functools import lru_cache

from . import config

KEYWORDS_PATH = config.BASE_DIR / "data" / "keywords_hotwords_matkul.json"
MAX_HOTWORDS_TERMS = 8


@lru_cache(maxsize=1)
def _load_all() -> dict:
    if not KEYWORDS_PATH.exists():
        return {}
    with open(KEYWORDS_PATH, encoding="utf-8") as f:
        return json.load(f)


def list_matkul() -> list[dict]:
    data = _load_all()
    return [
        {"kode": kode, "nama_mk": entry.get("nama_mk", kode), "sks": entry.get("sks")}
        for kode, entry in data.items()
    ]


def get_terms(kode_mk: str, max_terms: int = MAX_HOTWORDS_TERMS) -> list[str]:
    data = _load_all()
    entry = data.get(kode_mk)
    if not entry:
        return []
    return entry.get("keywords", [])[:max_terms]


def get_hotwords(kode_mk: str, max_terms: int = MAX_HOTWORDS_TERMS) -> str:
    terms = get_terms(kode_mk, max_terms)
    if not terms:
        return ""
    return "Istilah teknis yang mungkin disebut di sesi ini: " + ", ".join(terms) + "."
