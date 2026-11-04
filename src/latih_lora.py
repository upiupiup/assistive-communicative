"""Worker pelatihan LoRA Tahap 3 (subbab 3.6.7, Tabel 3.6, Gambar 4.8).

Dijalankan sebagai proses terpisah, dari sel notebook atau langsung dari terminal:

    python -m src.latih_lora <konfigurasi.json>

Menjalankan ulang perintah yang sama otomatis melanjutkan dari state terakhir,
sehingga pelatihan dapat dicicil (misalnya dengan ``batas_jam``) atau dilanjutkan
setelah terputus.

Alur:
1. Muat model dasar (bobot float16 beku), pasang LoRA (float32), seperti
   src/uji_kelayakan.py. Inisialisasi LoRA memakai ``seed``.
2. Urutan segmen latih tiap epoch adalah permutasi dari ``seed + epoch``, sehingga
   posisi data cukup ditentukan oleh nomor langkah optimizer. Satu langkah optimizer
   mengakumulasi ``akumulasi`` batch; langkah terakhir tiap epoch berisi sisa segmen
   dan loss-nya dirata-rata atas jumlah segmen yang sebenarnya.
3. Laju belajar naik linear selama ``rasio_pemanasan`` x total langkah, lalu turun
   linear ke nol. Autocast float16 dengan GradScaler.
4. WER validasi (greedy, tanpa timestamp, tingkat korpus atas seluruh segmen
   validasi) dihitung di langkah 0 sebagai pembanding, tiap ``eval_tiap`` langkah,
   dan di langkah terakhir bila ``eval_langkah_akhir``. Langkah 0 tidak ikut dipilih.
   Loss validasi ikut dicatat untuk memantau konvergensi, tidak untuk memilih checkpoint.
5. Setiap evaluasi: adapter disimpan ke ``adapter_terbaik/`` bila WER-nya terendah
   sejauh ini, lalu state lengkap disimpan (adapter, optimizer, scheduler, GradScaler,
   status RNG, langkah, WER terbaik).

Berkas di ``folder`` (kecil, dibawa pulang bila dilatih di server):
- ``konfigurasi.json``      konfigurasi terkunci beserta rencana langkah
- ``log_langkah.jsonl``     satu baris per langkah optimizer: loss, laju belajar, norma gradien
- ``log_validasi.jsonl``    satu baris per evaluasi: WER, S, D, I, N, loss validasi, waktu, memori GPU
- ``validasi_segmen.jsonl`` hipotesis setiap segmen validasi pada setiap evaluasi
- ``sesi.jsonl``            satu baris per sesi: perangkat, langkah awal dan akhir, durasi
- ``adapter_terbaik/``      adapter checkpoint terbaik beserta ``terbaik.json``
State lengkap ditulis ke ``folder_state`` (besar, tidak perlu dibawa pulang).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np

__all__ = ["rencana", "urutan_epoch", "label_ids", "hitung_wer", "wer_korpus", "nilai", "muat_segmen",
           "muat_model", "fitur_audio", "transkripsi_hf", "baca_jsonl", "jalankan"]

# Kunci konfigurasi yang tidak boleh berubah setelah pelatihan dimulai (Tabel 3.6).
KUNCI_TERKUNCI = ["model_id", "lora", "laju_belajar", "epoch", "batch", "akumulasi", "rasio_pemanasan",
                  "gradient_checkpointing", "seed", "eval_tiap", "eval_langkah_akhir", "eval_batch", "maks_token_baru",
                  "adamw", "segmen_latih", "kamus", "glosarium"]
MAKS_TARGET = 448  # max_target_positions Whisper


# ---------------------------------------------------------------------------
# Rencana dan urutan data
# ---------------------------------------------------------------------------

def rencana(n_latih: int, batch: int, akumulasi: int, epoch: int, rasio_pemanasan: float) -> dict:
    """Jumlah langkah optimizer. Sisa segmen tiap epoch menjadi satu langkah tersendiri."""
    per_langkah = batch * akumulasi
    per_epoch = math.ceil(n_latih / per_langkah)
    total = per_epoch * epoch
    return {"segmen_per_langkah": per_langkah, "langkah_per_epoch": per_epoch, "total_langkah": total,
            "langkah_pemanasan": math.ceil(rasio_pemanasan * total),
            "batch_maju_mundur": math.ceil(n_latih / batch) * epoch}


def urutan_epoch(n: int, seed: int, epoch: int) -> np.ndarray:
    """Permutasi indeks segmen latih untuk satu epoch (epoch mulai dari 0)."""
    return np.random.default_rng(seed + epoch).permutation(n)


def indeks_langkah(g: int, n: int, seed: int, per_langkah: int, per_epoch: int) -> tuple[int, np.ndarray]:
    """Epoch dan indeks segmen untuk langkah optimizer ke-g (mulai dari 0)."""
    e, k = divmod(g, per_epoch)
    perm = urutan_epoch(n, seed, e)
    return e, perm[k * per_langkah:(k + 1) * per_langkah]


# ---------------------------------------------------------------------------
# Label, audio, dan WER
# ---------------------------------------------------------------------------

def label_ids(tokenizer, teks: str) -> list[int]:
    """Token label tanpa timestamp, seperti src/uji_kelayakan.py.

    ``set_prefix_tokens(language="indonesian", task="transcribe")`` harus sudah dipanggil.
    Token awal transkrip dibuang karena model menambahkannya sendiri sebagai
    decoder_start_token_id.
    """
    ids = tokenizer(teks).input_ids
    mulai = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    return ids[1:] if ids[0] == mulai else ids


def fitur_audio(processor, path) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32")
    assert sr == 16000 and x.ndim == 1, f"{path}: {sr} Hz, dimensi {x.ndim}"
    return processor.feature_extractor(x, sampling_rate=16000, return_tensors="np").input_features[0]


def hitung_wer(acuan: str, hipotesis: str) -> dict:
    """S, D, I, N satu segmen dengan jiwer atas teks yang sudah dinormalisasi."""
    import jiwer
    N = len(acuan.split())
    assert N > 0, "acuan kosong"
    if not hipotesis.strip():
        return {"N": N, "S": 0, "D": N, "I": 0}
    o = jiwer.process_words(acuan, hipotesis)
    return {"N": N, "S": o.substitutions, "D": o.deletions, "I": o.insertions}


def wer_korpus(baris: list[dict]) -> dict:
    t = {k: sum(b[k] for b in baris) for k in ("N", "S", "D", "I")}
    t["WER"] = (t["S"] + t["D"] + t["I"]) / t["N"]
    return t


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def muat_model(model_id: str, perangkat: str, lora: dict | None = None, adapter: str | None = None,
               gradient_checkpointing: bool = False, seed: int | None = None):
    """Model Whisper dengan presisi Tabel 3.6.

    lora    : pasang modul LoRA baru (pelatihan).
    adapter : muat adapter tersimpan (evaluasi tanpa digabung).
    Tanpa keduanya: model apa adanya (model dasar atau model hasil gabung).
    Di CUDA bobot dasar float16; parameter LoRA selalu float32.
    """
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dtype = torch.float16 if perangkat.startswith("cuda") else torch.float32
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if lora is not None:
        from peft import LoraConfig, get_peft_model
        if seed is not None:
            torch.manual_seed(seed)
        model = get_peft_model(model, LoraConfig(
            r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"], bias="none"))
    elif adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()
    model.to(perangkat)
    return model, processor


def transkripsi_hf(model, processor, fitur: list[np.ndarray], batch: int, maks_token_baru: int,
                   perangkat: str) -> list[str]:
    """Greedy decoding tanpa timestamp dengan transformers (dipakai validasi dan pemeriksaan setara)."""
    import torch
    cuda = perangkat.startswith("cuda")
    dtype = torch.float16 if cuda else torch.float32
    hasil = []
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
        for i in range(0, len(fitur), batch):
            x = torch.tensor(np.stack(fitur[i:i + batch]), dtype=dtype, device=perangkat)
            out = model.generate(input_features=x, language="id", task="transcribe", return_timestamps=False,
                                 do_sample=False, num_beams=1, max_new_tokens=maks_token_baru, use_cache=True)
            hasil.extend(t.strip() for t in processor.batch_decode(out, skip_special_tokens=True))
    return hasil


# ---------------------------------------------------------------------------
# Berkas
# ---------------------------------------------------------------------------

def baca_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(b) for b in path.read_text(encoding="utf-8").splitlines() if b.strip()]


def _tambah(path: Path, baris: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(baris, ensure_ascii=False) + "\n")


def _potong_log(path: Path, g: int) -> int:
    """Buang baris dengan langkah > g (dikerjakan ulang setelah dilanjutkan)."""
    lama = baca_jsonl(path)
    sisa = [b for b in lama if b["langkah"] <= g]
    if len(sisa) != len(lama):
        path.write_text("".join(json.dumps(b, ensure_ascii=False) + "\n" for b in sisa), encoding="utf-8")
    return len(lama) - len(sisa)


def _sidik_konfigurasi(cfg: dict) -> str:
    return hashlib.sha1(json.dumps({k: cfg[k] for k in KUNCI_TERKUNCI}, sort_keys=True).encode()).hexdigest()[:16]


def _simpan_atomik_folder(model, tujuan: Path, info: dict) -> None:
    tmp = tujuan.with_name(tujuan.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(str(tmp))
    (tmp / "terbaik.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    if tujuan.exists():
        shutil.rmtree(tujuan)
    os.replace(tmp, tujuan)


def _gpu() -> dict:
    import torch
    d = {"host": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
         "torch": torch.__version__, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if torch.cuda.is_available():
        bebas, total = torch.cuda.mem_get_info()
        d |= {"gpu": torch.cuda.get_device_name(0), "vram_total_mb": round(total / 2**20),
              "vram_terpakai_awal_mb": round((total - bebas) / 2**20)}
    return d


# ---------------------------------------------------------------------------
# Pelatihan
# ---------------------------------------------------------------------------

def muat_segmen(cfg: dict, root: str | Path = "."):
    """Segmen latih dan validasi beserta acuan validasi yang sudah dinormalisasi
    (kamus terkunci + glosarium mata kuliahnya, sama dengan acuan data uji)."""
    import pandas as pd
    from . import praproses_teks as pt
    root = Path(root)
    seg = pd.read_csv(root / cfg["segmen_latih"], dtype={"seg_latih_id": str}, keep_default_na=False,
                      encoding="utf-8-sig").sort_values("seg_latih_id")
    audio = root / cfg["audio_dir"]
    seg["path"] = [str(audio / s / f"{i}.wav") for s, i in zip(seg.split, seg.seg_latih_id)]
    latih = seg[seg.split == "latih"].reset_index(drop=True)
    val = seg[seg.split == "validasi"].reset_index(drop=True)
    kamus = pt.muat_kamus(root / cfg["kamus"])
    assert kamus.terkunci, "kamus normalisasi belum terkunci"
    glos = pt.muat_glosarium(root / cfg["glosarium"])
    norm = {mk: pt.Normalisasi(kamus, glos[mk]) for mk in seg.mata_kuliah.unique()}
    val["acuan_norm"] = [norm[mk].teks(t) for mk, t in zip(val.mata_kuliah, val.teks)]
    return latih, val, norm


def nilai(val, norm, hipotesis: list[str]) -> tuple[dict, list[dict]]:
    """WER korpus dan baris per segmen untuk hipotesis yang urutannya sama dengan ``val``."""
    assert len(hipotesis) == len(val)
    baris = []
    for r, h in zip(val.itertuples(), hipotesis):
        hn = norm[r.mata_kuliah].teks(h)
        baris.append({"seg_latih_id": r.seg_latih_id, "file_id": r.file_id, "hipotesis": h,
                      "hipotesis_norm": hn, **hitung_wer(r.acuan_norm, hn)})
    return wer_korpus(baris), baris


def loss_validasi(model, fitur: list[np.ndarray], label: list[list[int]], batch: int, perangkat: str) -> float:
    """Loss validasi dengan teacher forcing, dirata-rata per segmen seperti loss latih.

    Hanya untuk memantau konvergensi dan overfitting; checkpoint tetap dipilih dengan WER.
    """
    import torch
    import torch.nn.functional as F
    cuda = perangkat.startswith("cuda")
    total, n = 0.0, 0
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
        for i in range(0, len(fitur), batch):
            x = torch.tensor(np.stack(fitur[i:i + batch]), dtype=torch.float16 if cuda else torch.float32,
                             device=perangkat)
            lb = label[i:i + batch]
            y = torch.full((len(lb), max(map(len, lb))), -100, dtype=torch.long)
            for r, ids in enumerate(lb):
                y[r, :len(ids)] = torch.tensor(ids)
            y = y.to(perangkat)
            logits = model(input_features=x, labels=y).logits.float()
            ce = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100, reduction="none")
            per_segmen = ce.sum(1) / (y != -100).sum(1)
            total += float(per_segmen.sum())
            n += len(lb)
    return total / n


def evaluasi(model, processor, val, fitur, norm, cfg, perangkat, label_val=None) -> tuple[dict, list[dict]]:
    t0 = time.perf_counter()
    m, baris = nilai(val, norm, transkripsi_hf(model, processor, fitur, cfg["eval_batch"],
                                               cfg["maks_token_baru"], perangkat))
    if label_val is not None:
        m["loss_validasi"] = round(loss_validasi(model, fitur, label_val, cfg["eval_batch"], perangkat), 5)
    return {**m, "detik_eval": round(time.perf_counter() - t0, 1)}, baris


def jalankan(cfg: dict) -> str:
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import get_linear_schedule_with_warmup

    t_sesi = time.perf_counter()
    waktu_mulai = time.strftime("%Y-%m-%dT%H:%M:%S")
    folder, folder_state = Path(cfg["folder"]), Path(cfg["folder_state"])
    folder.mkdir(parents=True, exist_ok=True)
    folder_state.mkdir(parents=True, exist_ok=True)
    p_cfg, p_state = folder / "konfigurasi.json", folder_state / "state_terakhir.pt"
    p_langkah, p_val = folder / "log_langkah.jsonl", folder / "log_validasi.jsonl"
    p_valseg, p_sesi, p_terbaik = folder / "validasi_segmen.jsonl", folder / "sesi.jsonl", folder / "adapter_terbaik"
    perangkat = cfg.get("perangkat", "cuda")
    cuda = perangkat.startswith("cuda")

    latih, val, norm = muat_segmen(cfg)
    R = rencana(len(latih), cfg["batch"], cfg["akumulasi"], cfg["epoch"], cfg["rasio_pemanasan"])
    sidik = _sidik_konfigurasi(cfg)
    if p_cfg.exists():
        lama = json.loads(p_cfg.read_text(encoding="utf-8"))
        assert lama["sidik"] == sidik, (
            "konfigurasi berbeda dengan pelatihan yang sudah berjalan "
            f"({[k for k in KUNCI_TERKUNCI if lama['konfigurasi'].get(k) != cfg.get(k)]}). "
            "Tabel 3.6 tidak boleh diubah di tengah pelatihan.")
    else:
        p_cfg.write_text(json.dumps({"sidik": sidik, "konfigurasi": {k: cfg[k] for k in KUNCI_TERKUNCI},
                                     "rencana": R, "n_latih": len(latih), "n_validasi": len(val),
                                     "dibuat": waktu_mulai}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{cfg['nama']}] {len(latih)} segmen latih, {len(val)} segmen validasi; "
          f"{R['langkah_per_epoch']} langkah/epoch x {cfg['epoch']} = {R['total_langkah']} langkah optimizer, "
          f"pemanasan {R['langkah_pemanasan']}", flush=True)

    if p_state.exists():
        st = torch.load(p_state, map_location="cpu", weights_only=False)
        if st["sidik"] == sidik and st["langkah"] >= R["total_langkah"]:
            print(f"[{cfg['nama']}] pelatihan sudah selesai ({st['langkah']} langkah); "
                  f"terbaik di langkah {st['terbaik']['langkah']}", flush=True)
            return "selesai"
        del st

    model, processor = muat_model(cfg["model_id"], perangkat, lora=cfg["lora"],
                                  gradient_checkpointing=cfg["gradient_checkpointing"], seed=cfg["seed"])
    tok = processor.tokenizer
    tok.set_prefix_tokens(language="indonesian", task="transcribe")
    label = [label_ids(tok, t) for t in latih.teks]
    assert max(map(len, label)) < MAKS_TARGET, "ada label melebihi 448 token"
    fitur_val = [fitur_audio(processor, p).astype(np.float16) for p in val.path]
    label_val = [label_ids(tok, t) for t in val.teks]

    terlatih = [p for p in model.parameters() if p.requires_grad]
    A = cfg["adamw"]
    opt = torch.optim.AdamW(terlatih, lr=cfg["laju_belajar"], betas=tuple(A["betas"]), eps=A["eps"],
                            weight_decay=A["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, R["langkah_pemanasan"], R["total_langkah"])
    skala = torch.amp.GradScaler("cuda", enabled=cuda)

    g, terbaik = 0, {"WER": math.inf, "langkah": None}
    if p_state.exists():
        st = torch.load(p_state, map_location="cpu", weights_only=False)
        assert st["sidik"] == sidik, "state berasal dari konfigurasi lain"
        set_peft_model_state_dict(model, st["adapter"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        skala.load_state_dict(st["scaler"])
        torch.set_rng_state(st["rng"]["torch"])
        if cuda and st["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(st["rng"]["cuda"])
        np.random.set_state(st["rng"]["numpy"])
        random.setstate(st["rng"]["python"])
        g, terbaik = st["langkah"], st["terbaik"]
        print(f"[{cfg['nama']}] dilanjutkan dari langkah {g} (epoch {g // R['langkah_per_epoch'] + 1}); "
              f"WER terbaik {terbaik['WER']:.4f} di langkah {terbaik['langkah']}", flush=True)
    dibuang = {p.name: _potong_log(p, g) for p in (p_langkah, p_val, p_valseg)}
    if any(dibuang.values()):
        print(f"[{cfg['nama']}] baris log sesudah langkah {g} dibuang karena dikerjakan ulang: {dibuang}", flush=True)
    g_awal = g

    def simpan_state():
        tmp = p_state.with_name(p_state.name + ".tmp")
        torch.save({"sidik": sidik, "langkah": g, "epoch": g // R["langkah_per_epoch"],
                    "indeks_dalam_epoch": (g % R["langkah_per_epoch"]) * R["segmen_per_langkah"],
                    "terbaik": terbaik, "adapter": get_peft_model_state_dict(model),
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": skala.state_dict(),
                    "rng": {"torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if cuda else None,
                            "numpy": np.random.get_state(), "python": random.getstate()},
                    "waktu": time.strftime("%Y-%m-%dT%H:%M:%S")}, tmp)
        os.replace(tmp, p_state)

    def validasi(boleh_terpilih: bool):
        nonlocal terbaik
        m, seg = evaluasi(model, processor, val, fitur_val, norm, cfg, perangkat, label_val)
        jadi_terbaik = boleh_terpilih and m["WER"] < terbaik["WER"]
        if jadi_terbaik:
            terbaik = {"WER": m["WER"], "langkah": g}
            _simpan_atomik_folder(model, p_terbaik, {"langkah": g, "epoch": g / R["langkah_per_epoch"],
                                                      **m, "sidik": sidik})
        b = {"langkah": g, "epoch": round(g / R["langkah_per_epoch"], 3), **m, "boleh_terpilih": boleh_terpilih,
             "terbaik": jadi_terbaik, "waktu": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if cuda:
            b |= {"vram_puncak_dialokasikan_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                  "vram_puncak_dicadangkan_mb": round(torch.cuda.max_memory_reserved() / 2**20)}
        _tambah(p_val, b)
        for s in seg:
            _tambah(p_valseg, {"langkah": g, **s})
        print(f"[{cfg['nama']}] validasi langkah {g:5d}: WER {100 * m['WER']:.2f}%, loss {m['loss_validasi']:.4f} "
              f"({m['detik_eval']:.0f} detik){'  <- terbaik' if jadi_terbaik else ''}", flush=True)
        model.train()

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    if g == 0 and not any(b["langkah"] == 0 for b in baca_jsonl(p_val)):
        validasi(boleh_terpilih=False)
        simpan_state()

    status, t_cek = "selesai", time.perf_counter()
    batas = cfg.get("batas_jam")
    model.train()
    jendela = []
    while g < R["total_langkah"]:
        t0 = time.perf_counter()
        e, idx = indeks_langkah(g, len(latih), cfg["seed"], R["segmen_per_langkah"], R["langkah_per_epoch"])
        loss_langkah = 0.0
        for i in range(0, len(idx), cfg["batch"]):
            mb = idx[i:i + cfg["batch"]]
            x = torch.tensor(np.stack([fitur_audio(processor, latih.path[j]) for j in mb]),
                             dtype=torch.float16 if cuda else torch.float32, device=perangkat)
            panjang = max(len(label[j]) for j in mb)
            y = torch.full((len(mb), panjang), -100, dtype=torch.long)
            for r, j in enumerate(mb):
                y[r, :len(label[j])] = torch.tensor(label[j])
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                loss = model(input_features=x, labels=y.to(perangkat)).loss
            bobot = len(mb) / len(idx)
            skala.scale(loss * bobot).backward()
            loss_langkah += float(loss.detach()) * bobot
        skala.unscale_(opt)
        norma = float(torch.nn.utils.clip_grad_norm_(terlatih, math.inf))
        s0 = skala.get_scale() if cuda else 1.0
        skala.step(opt)
        skala.update()
        lr = sched.get_last_lr()[0]
        sched.step()
        opt.zero_grad(set_to_none=True)
        g += 1
        lewat = cuda and skala.get_scale() < s0
        detik = time.perf_counter() - t0
        _tambah(p_langkah, {"langkah": g, "epoch": e + 1, "n_segmen": len(idx), "loss": round(loss_langkah, 5),
                            "laju_belajar": lr, "norma_gradien": round(norma, 5) if math.isfinite(norma) else None,
                            "skala": s0, "dilewati": lewat, "detik": round(detik, 3)})
        jendela = (jendela + [(loss_langkah, detik)])[-50:]
        if g % 50 == 0:
            sisa = (R["total_langkah"] - g) * np.mean([d for _, d in jendela]) / 3600
            print(f"[{cfg['nama']}] langkah {g:5d}/{R['total_langkah']} epoch {e + 1:2d} "
                  f"loss {np.mean([l for l, _ in jendela]):.4f} lr {lr:.2e} "
                  f"{np.mean([d for _, d in jendela]):.2f} detik/langkah, sisa +-{sisa:.1f} jam", flush=True)

        akhir = g == R["total_langkah"]
        if g % cfg["eval_tiap"] == 0 or (akhir and cfg["eval_langkah_akhir"]):
            validasi(boleh_terpilih=True)
            simpan_state()
            if batas and not akhir and (time.perf_counter() - t_sesi) / 3600 >= batas:
                status = "berhenti_sementara"
                break
        elif akhir:
            simpan_state()

    sesi = {"mulai": waktu_mulai, "selesai": time.strftime("%Y-%m-%dT%H:%M:%S"), "langkah_awal": g_awal,
            "langkah_akhir": g, "status": status, "durasi_jam": round((time.perf_counter() - t_sesi) / 3600, 3),
            **_gpu()}
    if cuda:
        sesi["vram_puncak_dicadangkan_mb"] = round(torch.cuda.max_memory_reserved() / 2**20)
    _tambah(p_sesi, sesi)
    print(f"[{cfg['nama']}] {status} di langkah {g}/{R['total_langkah']}; "
          f"terbaik WER {100 * terbaik['WER']:.2f}% di langkah {terbaik['langkah']}", flush=True)
    return status


if __name__ == "__main__":
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    jalankan(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
