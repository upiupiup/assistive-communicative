"""
Penyimpanan sesi kuliah — JSON per sesi (bukan database) supaya gampang
diperiksa manual pas demo ke dosen, dan gampang di-upgrade ke SQLite/Postgres
nanti kalau platform ini lanjut ke tahap production.

Struktur folder per sesi:
data/sessions/{session_id}/
    meta.json           -> mata kuliah, dosen, tanggal, dibuat_at
    transcript.txt       -> transkrip penuh (plain text, editable)
    summary.md            -> rangkuman hasil LLM (kalau sudah digenerate)
    materials/            -> file PPT/PDF yang diupload
"""

import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from . import config


def _session_dir(session_id: str) -> Path:
    return config.SESSIONS_DIR / session_id


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip()
    name = name.rstrip(". ")
    return name or "material"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def create_session(
    mata_kuliah: str, dosen: str, topik: str, tanggal: str, mk_kode: str = ""
) -> str:
    session_id = f"{tanggal}_{uuid.uuid4().hex[:8]}"
    d = _session_dir(session_id)
    (d / "materials").mkdir(parents=True, exist_ok=True)

    meta = {
        "session_id": session_id,
        "mata_kuliah": mata_kuliah,
        "mk_kode": mk_kode,
        "dosen": dosen,
        "topik": topik,
        "tanggal": tanggal,
        "dibuat_at": datetime.now().isoformat(timespec="seconds"),
    }
    (d / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (d / "transcript.txt").write_text("", encoding="utf-8")
    return session_id


def list_sessions() -> list[dict]:
    sessions = []
    for d in sorted(config.SESSIONS_DIR.iterdir(), reverse=True):
        meta_path = d / "meta.json"
        if meta_path.exists():
            sessions.append(json.loads(meta_path.read_text(encoding="utf-8")))
    return sessions


def load_meta(session_id: str) -> dict:
    return json.loads(
        (_session_dir(session_id) / "meta.json").read_text(encoding="utf-8")
    )


def get_transcript(session_id: str) -> str:
    p = _session_dir(session_id) / "transcript.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def append_transcript(session_id: str, text_chunk: str) -> None:
    if not text_chunk:
        return
    p = _session_dir(session_id) / "transcript.txt"
    existing = p.read_text(encoding="utf-8") if p.exists() else ""
    updated = (existing + " " + text_chunk).strip() if existing else text_chunk
    p.write_text(updated, encoding="utf-8")


def overwrite_transcript(session_id: str, full_text: str) -> None:
    (_session_dir(session_id) / "transcript.txt").write_text(
        full_text, encoding="utf-8"
    )


def save_material(session_id: str, uploaded_file) -> Path:
    materials_dir = _session_dir(session_id) / "materials"
    materials_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_path(materials_dir / _safe_filename(uploaded_file.name))
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def list_materials(session_id: str) -> list[Path]:
    materials_dir = _session_dir(session_id) / "materials"
    if not materials_dir.exists():
        return []
    return sorted(materials_dir.iterdir())


def save_summary(session_id: str, summary_md: str) -> None:
    (_session_dir(session_id) / "summary.md").write_text(summary_md, encoding="utf-8")


def get_summary(session_id: str) -> str:
    p = _session_dir(session_id) / "summary.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def delete_session(session_id: str) -> None:
    shutil.rmtree(_session_dir(session_id), ignore_errors=True)
