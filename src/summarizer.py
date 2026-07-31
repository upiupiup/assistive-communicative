import os

import streamlit as st

from . import config

SYSTEM_PROMPT = """Kamu adalah asisten yang membuat catatan kuliah (lecture notes) yang rapi
dan terstruktur dari transkrip perkuliahan Bahasa Indonesia, untuk mahasiswa Fakultas Ilmu
Komputer. Transkrip berasal dari sistem ASR sehingga bisa ada typo atau istilah teknis yang
salah tangkap — gunakan materi slide (kalau tersedia) untuk mengoreksi istilah tersebut dari
konteks. Jangan mengarang informasi yang tidak ada di transkrip atau materi."""

USER_PROMPT_TEMPLATE = """Buat rangkuman kuliah dalam format Markdown terstruktur dari data berikut.

## Info Sesi
- Mata Kuliah : {mata_kuliah}
- Dosen       : {dosen}
- Topik       : {topik}
- Tanggal     : {tanggal}

## Transkrip (hasil ASR real-time)
{transcript}

## Materi Pendukung (kalau ada, dari slide/PPT/PDF)
{materials_text}

---
Susun output dengan struktur berikut (pakai heading Markdown):
1. **Topik Utama** — 2-4 poin ringkas
2. **Definisi & Konsep Kunci** — istilah teknis yang dibahas beserta penjelasan singkatnya
3. **Contoh Kode / Algoritma** — kalau disebutkan di transkrip atau materi, tuliskan ulang
4. **Ringkasan Alur Pembahasan** — narasi singkat urutan materi disampaikan
5. **Catatan Tambahan** — hal yang perlu ditindaklanjuti mahasiswa (tugas, bacaan, dsb) kalau disebutkan

Tulis dengan bahasa yang jelas dan genuine, bukan bombastis. Kalau suatu bagian tidak ada
datanya di transkrip/materi, tulis singkat "Tidak disebutkan di sesi ini" — jangan mengarang.
"""


def _get_api_key() -> str | None:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    from google import genai
    from google.genai import types

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY belum diset. Tambahkan ke .streamlit/secrets.toml "
            "atau environment variable GEMINI_API_KEY. Bikin API key (pay-as-you-go) "
            "di Google AI Studio: aistudio.google.com/apikey."
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
        ),
    )
    return (response.text or "").strip()


def generate_summary(transcript: str, materials_text: str, meta: dict) -> str:
    if not transcript.strip():
        return "_Transkrip masih kosong — rekam sesi dulu sebelum generate rangkuman._"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        mata_kuliah=meta.get("mata_kuliah", "-"),
        dosen=meta.get("dosen", "-"),
        topik=meta.get("topik", "-"),
        tanggal=meta.get("tanggal", "-"),
        transcript=transcript.strip(),
        materials_text=materials_text.strip() or "(tidak ada materi diupload)",
    )

    try:
        return _call_llm(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        return f"⚠️ Gagal generate rangkuman: {e}"
