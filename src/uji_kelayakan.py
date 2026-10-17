"""Worker uji kelayakan pelatihan LoRA (subbab 3.6.5).

Dijalankan sebagai proses terpisah, satu proses per percobaan (varian dan
ukuran batch), supaya memori GPU percobaan sebelumnya benar-benar bebas:

    python -m src.uji_kelayakan <konfigurasi.json>

Satu percobaan:
1. Muat model dasar (bobot float16, beku) dan pasang modul LoRA sesuai
   Tabel 3.6. Parameter LoRA disimpan dalam float32, perhitungan maju memakai
   autocast float16 dengan penskala gradien.
2. Susun satu batch dari segmen latih yang labelnya paling panjang, sebab
   masukan audio Whisper selalu setara 30 detik sehingga yang membedakan
   kebutuhan memori antarsegmen hanya panjang label.
3. Jalankan beberapa langkah pemanasan, lalu beberapa langkah yang diukur.
   Batch yang sama dipakai di setiap langkah.
4. Catat puncak memori GPU, waktu per langkah, jumlah parameter terlatih, dan
   ukuran berkas adapter.

Hasil ditulis ke stdout sebagai satu baris JSON berawalan ###HASIL###.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path


def susun_batch(processor, segmen: list[dict], audio_dir: Path, ukuran: int, perangkat: str):
    import numpy as np
    import soundfile as sf
    import torch

    tok = processor.tokenizer
    tok.set_prefix_tokens(language="indonesian", task="transcribe")
    calon = []
    for s in segmen:
        ids = tok(s["teks"]).input_ids
        calon.append((len(ids), s, ids))
    calon.sort(key=lambda x: -x[0])
    pilih = calon[:ukuran]

    fitur = []
    for _, s, _ in pilih:
        x, sr = sf.read(str(audio_dir / f"{s['seg_latih_id']}.wav"), dtype="float32")
        assert sr == 16000, f"{s['seg_latih_id']}: sampling rate {sr}"
        fitur.append(processor.feature_extractor(x, sampling_rate=16000, return_tensors="np").input_features[0])
    input_features = torch.tensor(np.stack(fitur), dtype=torch.float16, device=perangkat)

    # Token awal transkrip dibuang dari label karena model menambahkannya sendiri
    # sebagai decoder_start_token_id, seperti pada pengumpul data fine-tuning Whisper.
    mulai = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    label = [ids[1:] if ids[0] == mulai else ids for _, _, ids in pilih]
    panjang = max(len(x) for x in label)
    labels = torch.full((len(label), panjang), -100, dtype=torch.long)
    for i, x in enumerate(label):
        labels[i, :len(x)] = torch.tensor(x)
    return input_features, labels.to(perangkat), [s["seg_latih_id"] for _, s, _ in pilih], panjang


def jalankan(cfg: dict) -> dict:
    hasil = {"varian": cfg["varian"], "model_id": cfg["model_id"], "batch": cfg["batch"],
             "gradient_checkpointing": cfg["gradient_checkpointing"], "status": "gagal"}
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        perangkat = "cuda"
        torch.cuda.init()
        bebas, total = torch.cuda.mem_get_info()
        hasil["vram_total_mb"] = round(total / 2**20)
        hasil["vram_terpakai_awal_mb"] = round((total - bebas) / 2**20)

        processor = WhisperProcessor.from_pretrained(cfg["model_id"])
        model = WhisperForConditionalGeneration.from_pretrained(cfg["model_id"], torch_dtype=torch.float16)
        model.config.forced_decoder_ids = None
        model.generation_config.forced_decoder_ids = None
        model.config.use_cache = False
        if cfg["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()

        L = cfg["lora"]
        model = get_peft_model(model, LoraConfig(
            r=L["r"], lora_alpha=L["alpha"], lora_dropout=L["dropout"],
            target_modules=L["target_modules"], bias="none"))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        model.to(perangkat)
        model.train()

        hasil["parameter_total"] = sum(p.numel() for p in model.parameters())
        hasil["parameter_terlatih"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        hasil["modul_lora"] = sum(1 for n, _ in model.named_modules() if n.endswith(".lora_A"))

        segmen = json.loads(Path(cfg["segmen_json"]).read_text(encoding="utf-8"))
        feats, labels, seg_ids, panjang = susun_batch(
            processor, segmen, Path(cfg["audio_dir"]), cfg["batch"], perangkat)
        hasil["panjang_label_token"] = panjang
        hasil["segmen_dipakai"] = seg_ids

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=cfg["laju_belajar"])
        skala = torch.amp.GradScaler("cuda")
        torch.cuda.reset_peak_memory_stats()

        def langkah():
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_features=feats, labels=labels).loss
            skala.scale(loss).backward()
            skala.step(opt)
            skala.update()
            opt.zero_grad(set_to_none=True)
            return float(loss)

        for _ in range(cfg["langkah_pemanasan"]):
            langkah()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(cfg["langkah_ukur"]):
            loss = langkah()
        torch.cuda.synchronize()
        hasil["detik_per_langkah"] = round((time.perf_counter() - t0) / cfg["langkah_ukur"], 3)
        hasil["loss_akhir"] = round(loss, 4)
        hasil["puncak_dialokasikan_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
        hasil["puncak_dicadangkan_mb"] = round(torch.cuda.max_memory_reserved() / 2**20)

        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)
            hasil["ukuran_adapter_mb"] = round(
                sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file()) / 2**20, 2)
        hasil["status"] = "berhasil"
    except Exception as e:  # noqa: BLE001
        nama = type(e).__name__
        hasil["status"] = "memori habis" if "OutOfMemory" in nama or "out of memory" in str(e) else "gagal"
        hasil["pesan"] = f"{nama}: {e}"[:400]
    return hasil


if __name__ == "__main__":
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    r = jalankan(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    sys.stdout.write("###HASIL###" + json.dumps(r) + "\n")
