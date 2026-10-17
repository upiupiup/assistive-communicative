"""Worker transkripsi dan pengukuran waktu (subbab 3.6.4 dan 3.6.8 butir 3).

Dijalankan sebagai proses terpisah, satu proses per konfigurasi:

    python -m src.transkripsi <konfigurasi.json>

Proses terpisah diperlukan karena CTranslate2 tidak memakai alokator PyTorch,
sehingga memori GPU model sebelumnya baru benar-benar bebas ketika prosesnya
berakhir. Tanpa itu, varian berikutnya bisa kehabisan memori atau melambat.

Urutan kerja:
1. Muat model, waktunya dicatat terpisah (tidak masuk RTF).
2. Pemanasan atas potongan awal rekaman terpendek, hasilnya dibuang. Puncak
   memori GPU diukur hanya selama pemanasan, supaya pemantauan tidak ikut
   membebani pengukuran waktu.
3. Untuk tiap ulangan dan tiap rekaman: transkripsi atas rekaman utuh, dari
   berkas WAV sampai seluruh segmen selesai dibangkitkan. Waktu ini T_proses
   pada Persamaan 2.14, dan penyebutnya durasi rekaman utuh.
4. Keluaran mentah ulangan pertama diarsipkan. Ulangan berikutnya hanya
   dicatat sidik teksnya untuk memeriksa bahwa keluarannya identik.

Setiap pasangan (ulangan, rekaman) yang selesai langsung ditulis satu baris ke
berkas JSONL, sehingga bila proses terputus, menjalankan ulang hanya
mengerjakan yang belum ada. Pemanasan tetap dijalankan setiap kali model dimuat.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path


def vram_terpakai_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL)
        return int(out.decode(errors="replace").strip().splitlines()[0])
    except Exception:
        return -1


def transkripsi(model, audio, parameter: dict) -> tuple[list[dict], float, dict]:
    """Transkripsi satu masukan. Mengembalikan (segmen, detik_proses, info)."""
    t0 = time.perf_counter()
    segmen_gen, info = model.transcribe(audio, **parameter)
    segmen = [{
        "id": s.id, "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text,
        "temperature": s.temperature, "avg_logprob": round(s.avg_logprob, 4),
        "compression_ratio": round(s.compression_ratio, 3),
        "no_speech_prob": round(s.no_speech_prob, 4),
    } for s in segmen_gen]
    detik = time.perf_counter() - t0
    return segmen, detik, {"duration": info.duration,
                           "duration_after_vad": getattr(info, "duration_after_vad", None),
                           "language": info.language}


def sidik(segmen: list[dict]) -> str:
    return hashlib.sha1("\n".join(s["text"] for s in segmen).encode("utf-8")).hexdigest()[:16]


def _sudah_selesai(path: Path, nama: str) -> set:
    if not path.exists():
        return set()
    out = set()
    for baris in path.read_text(encoding="utf-8").splitlines():
        if baris.strip():
            d = json.loads(baris)
            if d["konfigurasi"] == nama:
                out.add((d["ulangan"], d["file_id"]))
    return out


def jalankan(cfg: dict, buat_model=None) -> None:
    nama = cfg["nama"]
    log_path = Path(cfg["log_waktu"])
    arsip = Path(cfg["arsip_mentah"])
    arsip.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    selesai = _sudah_selesai(log_path, nama)
    tugas = [(u, r) for u in range(1, cfg["ulangan"] + 1) for r in cfg["rekaman"]
             if (u, r["file_id"]) not in selesai]
    if not tugas:
        print(f"[{nama}] semua ulangan sudah ada, dilewati", flush=True)
        return
    print(f"[{nama}] {len(selesai)} sudah ada, {len(tugas)} dikerjakan", flush=True)

    if buat_model is None:
        from faster_whisper import WhisperModel

        def buat_model():
            return WhisperModel(cfg["model"], device=cfg["device"], compute_type=cfg["compute_type"])

    vram_awal = vram_terpakai_mb()
    t0 = time.perf_counter()
    model = buat_model()
    muat_s = time.perf_counter() - t0
    print(f"[{nama}] model dimuat dalam {muat_s:.1f} detik", flush=True)

    # Pemanasan dengan pemantauan memori GPU.
    puncak = [vram_awal]
    henti = threading.Event()

    def pantau():
        while not henti.is_set():
            v = vram_terpakai_mb()
            puncak[0] = max(puncak[0], v)
            time.sleep(0.2)

    th = threading.Thread(target=pantau, daemon=True)
    th.start()
    import soundfile as sf
    pem = cfg["pemanasan"]
    # WAV hasil praproses sudah 16 kHz mono PCM 16-bit, jadi cukup dibaca sebagai float32.
    audio_pem, sr = sf.read(pem["path"], dtype="float32", frames=int(pem["detik"] * 16000))
    assert sr == 16000, f"{pem['path']}: sampling rate {sr}"
    _, pem_s, _ = transkripsi(model, audio_pem, cfg["parameter"])
    henti.set()
    th.join()
    print(f"[{nama}] pemanasan {pem['detik']:.0f} detik audio selesai dalam {pem_s:.1f} detik", flush=True)

    sesi = {"muat_s": round(muat_s, 2), "pemanasan_s": round(pem_s, 2),
            "vram_awal_mb": vram_awal, "vram_puncak_mb": puncak[0]}

    for u, r in tugas:
        segmen, detik, info = transkripsi(model, r["path"], cfg["parameter"])
        fid = r["file_id"]
        if u == 1:
            (arsip / f"{fid}.json").write_text(json.dumps({
                "file_id": fid, "konfigurasi": nama, "model": cfg["model"],
                "compute_type": cfg["compute_type"], "parameter": cfg["parameter"],
                "info": info, "segmen": segmen}, ensure_ascii=False, indent=1), encoding="utf-8")
        baris = {"konfigurasi": nama, "ulangan": u, "file_id": fid,
                 "durasi_s": r["durasi"], "proses_s": round(detik, 3),
                 "rtf": round(detik / r["durasi"], 5), "n_segmen": len(segmen),
                 "sidik": sidik(segmen), "durasi_setelah_vad_s": info["duration_after_vad"],
                 "waktu": time.strftime("%Y-%m-%dT%H:%M:%S"), **sesi}
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(baris) + "\n")
        print(f"[{nama}] ulangan {u}/{cfg['ulangan']} {fid:9s} {detik:7.1f} detik  RTF {baris['rtf']:.4f}",
              flush=True)


if __name__ == "__main__":
    jalankan(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
