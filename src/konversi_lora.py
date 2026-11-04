"""Penggabungan LoRA dan konversi ke CTranslate2 (subbab 3.6.8 butir c, Gambar 4.8).

Dijalankan sebagai proses terpisah supaya memori model dilepas setelah selesai:

    python -m src.konversi_lora <konfigurasi.json>

1. Muat model dasar dalam float32 di CPU. Bila ``adapter`` diisi, pasang adapter
   lalu gabungkan dengan baris kedua Persamaan 2.12, W = W0 + (alpha/r) B A.
   Penggabungan dalam float32 supaya pembulatan hanya terjadi sekali, saat model
   disimpan dalam float16.
2. Periksa penggabungan: untuk setiap modul LoRA, bobot hasil gabung dibandingkan
   dengan W0 + (alpha/r) B A yang dihitung sendiri. Selisih maksimum dicatat.
3. Simpan model hasil gabung (float16) beserta tokenizer dan preprocessor.
4. Konversi ke CTranslate2 dengan kuantisasi float16, format yang sama dengan
   model dasar Tahap 1 dan Tahap 2. ``preprocessor_config.json`` ikut disalin karena
   faster-whisper membaca jumlah mel bin (128 untuk large-v3) dari berkas itu.

Tanpa ``adapter``, langkah yang sama dijalankan atas model dasar (kontrol konversi).
Hasil ditulis ke stdout sebagai satu baris JSON berawalan ###HASIL###.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path


def _pengonversi():
    from ctranslate2.converters import TransformersConverter

    class Pengonversi(TransformersConverter):
        """ctranslate2 4.7 memanggil from_pretrained(dtype=...), sedangkan transformers 4.45
        hanya mengenal torch_dtype dan menolak argumen dtype."""

        def load_model(self, model_class, model_name_or_path, **kwargs):
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            return super().load_model(model_class, model_name_or_path, **kwargs)

    return Pengonversi


def ukuran_mb(folder) -> float:
    return round(sum(f.stat().st_size for f in Path(folder).rglob("*") if f.is_file()) / 2**20, 1)


def jalankan(cfg: dict) -> dict:
    import torch
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizerFast

    t0 = time.perf_counter()
    hf_dir, ct2_dir = Path(cfg["hf_keluaran"]), Path(cfg["ct2_keluaran"])
    hasil = {"nama": cfg["nama"], "model_id": cfg["model_id"], "adapter": cfg.get("adapter")}

    model = WhisperForConditionalGeneration.from_pretrained(cfg["model_id"], torch_dtype=torch.float32)
    if cfg.get("adapter"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg["adapter"])
        lapis = {n: m for n, m in model.named_modules() if hasattr(m, "lora_A") and "default" in m.lora_A}
        harapan = {}
        for n, m in lapis.items():
            s = m.scaling["default"]
            BA = m.lora_B["default"].weight.detach() @ m.lora_A["default"].weight.detach()
            harapan[n] = (m.base_layer.weight.detach() + s * BA).clone()
        hasil["skala_alpha_per_r"] = sorted({m.scaling["default"] for m in lapis.values()})
        model = model.merge_and_unload()
        sisa = [n for n, _ in model.named_modules() if "lora_" in n]
        assert not sisa, f"masih ada modul LoRA setelah digabung: {sisa[:3]}"
        bobot = dict(model.named_parameters())
        selisih = 0.0
        for n, w in harapan.items():
            nama = n.replace("base_model.model.", "", 1) + ".weight"
            selisih = max(selisih, float((bobot[nama].detach() - w).abs().max()))
        hasil |= {"modul_digabung": len(lapis), "selisih_gabung_maks": selisih}
        del harapan, bobot
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = True

    if hf_dir.exists():
        shutil.rmtree(hf_dir)
    model.half().save_pretrained(str(hf_dir), safe_serialization=True)
    WhisperTokenizerFast.from_pretrained(cfg["model_id"]).save_pretrained(str(hf_dir))
    WhisperFeatureExtractor.from_pretrained(cfg["model_id"]).save_pretrained(str(hf_dir))
    for f in ("tokenizer.json", "preprocessor_config.json", "generation_config.json"):
        assert (hf_dir / f).exists(), f"{f} tidak tersimpan"
    del model
    gc.collect()

    Pengonversi = _pengonversi()
    Pengonversi(str(hf_dir), copy_files=["tokenizer.json", "preprocessor_config.json"]).convert(
        str(ct2_dir), quantization=cfg.get("quantization", "float16"), force=True)
    hasil |= {"hf_mb": ukuran_mb(hf_dir), "ct2_mb": ukuran_mb(ct2_dir),
              "n_mels": json.loads((ct2_dir / "preprocessor_config.json").read_text())["feature_size"],
              "detik": round(time.perf_counter() - t0, 1)}
    return hasil


if __name__ == "__main__":
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    r = jalankan(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    print(f"[{r['nama']}] selesai dalam {r['detik']} detik", flush=True)
    sys.stdout.write("###HASIL###" + json.dumps(r) + "\n")
