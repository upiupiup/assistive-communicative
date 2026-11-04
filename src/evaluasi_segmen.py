"""Transkripsi segmen validasi untuk pemeriksaan kesetaraan konversi (keputusan 3, Gambar 4.8).

Satu proses per model, supaya memori GPU model sebelumnya benar-benar bebas:

    python -m src.evaluasi_segmen <konfigurasi.json>

Kondisinya sama untuk semua model: float16, greedy, tanpa timestamp, satu segmen
validasi (<= 30 detik) per masukan.
- ``mode = "hf"``  : transformers, dengan fungsi yang sama dengan validasi saat
                     pelatihan (src/latih_lora.py). ``adapter`` diisi untuk adapter
                     tanpa digabung; dikosongkan untuk model dasar atau model hasil gabung.
- ``mode = "ct2"`` : faster-whisper atas model CTranslate2, dengan beam_size 1,
                     temperatur 0 saja, tanpa timestamp, tanpa VAD, tanpa riwayat teks, dan
                     batas token baru yang sama dengan jalur transformers.

Keluaran: satu baris JSONL per segmen (seg_latih_id, hipotesis).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PARAMETER_CT2 = {"language": "id", "task": "transcribe", "beam_size": 1, "temperature": 0.0,
                 "without_timestamps": True, "vad_filter": False, "condition_on_previous_text": False}


def jalankan(cfg: dict) -> None:
    keluaran = Path(cfg["keluaran"])
    keluaran.parent.mkdir(parents=True, exist_ok=True)
    seg = cfg["segmen"]
    t0 = time.perf_counter()
    if cfg["mode"] == "hf":
        from . import latih_lora as ll
        model, processor = ll.muat_model(cfg["model"], cfg["perangkat"], adapter=cfg.get("adapter"))
        fitur = [ll.fitur_audio(processor, s["path"]).astype("float16") for s in seg]
        hyp = ll.transkripsi_hf(model, processor, fitur, cfg["eval_batch"], cfg["maks_token_baru"], cfg["perangkat"])
    elif cfg["mode"] == "ct2":
        import soundfile as sf
        from faster_whisper import WhisperModel
        model = WhisperModel(cfg["model"], device=cfg["perangkat"], compute_type=cfg["compute_type"])
        hyp = []
        for s in seg:
            # dibaca dengan soundfile, sama seperti jalur transformers, supaya masukannya identik
            audio, sr = sf.read(s["path"], dtype="float32")
            assert sr == 16000, f"{s['path']}: {sr} Hz"
            out, _ = model.transcribe(audio, **PARAMETER_CT2, max_new_tokens=cfg["maks_token_baru"])
            hyp.append(" ".join(x.text.strip() for x in out).strip())
    else:
        raise ValueError(cfg["mode"])
    with open(keluaran, "w", encoding="utf-8") as fh:
        for s, h in zip(seg, hyp):
            fh.write(json.dumps({"seg_latih_id": s["seg_latih_id"], "hipotesis": h}, ensure_ascii=False) + "\n")
    print(f"[{cfg['nama']}] {len(seg)} segmen dalam {time.perf_counter() - t0:.0f} detik -> {keluaran.name}",
          flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    jalankan(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
