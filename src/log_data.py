"""Log per data untuk setiap konfigurasi (Tahap 1 sampai Tahap 3).

Log ini hanya mencatat, tidak menganalisis. Tujuannya supaya untuk setiap
rekaman, segmen, kata, dan istilah terlihat apa yang salah dan dalam kondisi
decoding seperti apa kesalahan itu terjadi.

Berkas yang ditulis ``tulis_log`` ke ``<folder_tahap>/log/``:

- ``rekaman.csv``            satu baris per konfigurasi x rekaman: metrik, mata kuliah,
                             jalur acuan, durasi, RTF tiap ulangan.
- ``segmen_hipotesis.csv``   satu baris per segmen keluaran model: teks mentah, waktu,
                             dipakai atau dibuang, dan info decoding Whisper.
- ``kemunculan_istilah.csv`` satu baris per kemunculan istilah glosarium pada acuan:
                             benar atau salah, hipotesisnya, jenis operasinya.
- ``penyelarasan/<konfigurasi>/<ID>.tsv``
                             satu baris per operasi penyelarasan (kata): kata acuan,
                             kata hipotesis, operasi, blok, waktu, istilah, dan info
                             decoding segmen hipotesisnya.

Perhitungannya memakai fungsi yang sama dengan notebook tahap (src/metrik.py),
dan ``periksa_terhadap`` memastikan angkanya identik dengan metrik_rekaman.csv
yang sudah ada sebelum log ditulis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrik as mt

__all__ = ["log_konfigurasi", "periksa_terhadap", "tulis_log"]

INFO_DECODING = ["temperature", "avg_logprob", "compression_ratio", "no_speech_prob"]
JUMLAH = ["N", "S", "D", "I", "err_kata", "err_kar", "N_kar", "E_G", "G"]
KONTEKS = 5


def _token_per_segmen(segmen: list[dict], norm) -> tuple[list[str], list[int]]:
    """Normalisasi tiap segmen dan catat asal segmen setiap token."""
    token, asal = [], []
    for k, s in enumerate(segmen):
        t = norm(s["text"])
        token.extend(t)
        asal.extend([k] * len(t))
    return token, asal


def log_konfigurasi(konfigurasi: str, kode: str, mentah_dir: Path, acuan: dict, normalisasi: dict,
                    durasi: dict, manifest: pd.DataFrame, waktu: pd.DataFrame,
                    kanonik: dict) -> dict:
    """Bangun log satu konfigurasi atas seluruh rekaman uji.

    acuan       : {file_id: isi acuan/<ID>.json}
    normalisasi : {file_id: objek pt.Normalisasi untuk mata kuliah rekaman itu}
    durasi      : {file_id: durasi rekaman utuh dalam detik}
    manifest    : manifest_rekaman.csv, terindeks file_id
    waktu       : baris waktu_transkripsi.jsonl untuk konfigurasi ini
    kanonik     : {entri_id: bentuk_kanonik}
    """
    rek, seg_rows, ist_rows, selaras = [], [], [], {}
    for fid, a in acuan.items():
        mentah = json.loads((Path(mentah_dir) / f"{fid}.json").read_text(encoding="utf-8"))
        seg = mentah["segmen"]
        pakai, buang = mt.saring_segmen(seg, a["segmen"])
        id_pakai = {id(s) for s in pakai}
        hyp, asal = _token_per_segmen(pakai, normalisasi[fid])
        blok, ops = mt.hitung_rekaman(a, hyp, durasi[fid])

        # --- segmen hipotesis ---
        for s in seg:
            seg_rows.append({"konfigurasi": konfigurasi, "file_id": fid, "seg_id": s["id"],
                             "mulai": s["start"], "selesai": s["end"],
                             "blok": int(s["start"] // mt.PANJANG_BLOK),
                             "dipakai": id(s) in id_pakai, "teks": s["text"].strip(),
                             **{k: s.get(k) for k in INFO_DECODING}})

        # --- penyelarasan per kata ---
        ref, N = a["token"], len(a["token"])
        seg_ref = np.full(N, -1)
        for k, s in enumerate(a["segmen"]):
            seg_ref[s["i0"]:s["i1"]] = k
        istilah_tok = {}
        for x in a["istilah"]:
            for k in range(x["i0"], x["i1"]):
                istilah_tok[k] = x["entri_id"]
        G = set(a["G"])
        tepi = (N - 1) in G
        baris = []
        for o in ops:
            k = min(o.i_src, N - 1)
            sr = a["segmen"][seg_ref[k]]
            sh = pakai[asal[o.i_res]] if o.i_res is not None else None
            baris.append({
                "operasi": o.jenis,
                "i_acuan": o.i_src if o.jenis != "I" else None,
                "kata_acuan": ref[o.i_src] if o.jenis != "I" else "",
                "i_hipotesis": o.i_res,
                "kata_hipotesis": hyp[o.i_res] if o.i_res is not None else "",
                "blok": int(sr["start"] // mt.PANJANG_BLOK),
                "acuan_mulai": sr["start"], "acuan_selesai": sr["end"],
                "istilah": istilah_tok.get(o.i_src, "") if o.jenis != "I" else "",
                "masuk_E_G": o.jenis != "C" and (o.i_src in G or (o.i_src >= N and tepi)),
                "seg_hipotesis": sh["id"] if sh else None,
                "hipotesis_mulai": sh["start"] if sh else None,
                "hipotesis_selesai": sh["end"] if sh else None,
                **{f"hip_{k2}": (sh.get(k2) if sh else None) for k2 in INFO_DECODING},
            })
        selaras[fid] = pd.DataFrame(baris)

        # --- kemunculan istilah ---
        for x in a["istilah"]:
            o = mt.rentang_istilah_hipotesis(ops, x["i0"], x["i1"], N, tepi)
            salah = [y for y in o if y.jenis != "C"]
            segh = sorted({pakai[asal[y.i_res]]["id"] for y in o if y.i_res is not None})
            info = [pakai[asal[y.i_res]] for y in o if y.i_res is not None]
            sr = a["segmen"][seg_ref[x["i0"]]]
            sekitar = [y for y in ops if x["i0"] - KONTEKS <= y.i_src < x["i1"] + KONTEKS
                       and y.i_res is not None]
            ist_rows.append({
                "konfigurasi": konfigurasi, "file_id": fid, "mata_kuliah": a["mata_kuliah"],
                "blok": int(sr["start"] // mt.PANJANG_BLOK), "acuan_mulai": sr["start"],
                "i0": x["i0"], "i1": x["i1"], "entri_id": x["entri_id"],
                "bentuk_kanonik": kanonik.get(x["entri_id"], ""),
                "istilah_acuan": " ".join(ref[x["i0"]:x["i1"]]),
                "hipotesis": " ".join(hyp[y.i_res] for y in o if y.i_res is not None),
                "benar": not salah,
                "S": sum(y.jenis == "S" for y in salah), "D": sum(y.jenis == "D" for y in salah),
                "I": sum(y.jenis == "I" for y in salah),
                "seg_hipotesis": ";".join(map(str, segh)),
                **{f"hip_{k2}": (min(s[k2] for s in info) if k2 == "avg_logprob" else max(s[k2] for s in info))
                   if info else None for k2 in INFO_DECODING},
                "konteks_acuan": " ".join(ref[max(0, x["i0"] - KONTEKS):x["i1"] + KONTEKS]),
                "konteks_hipotesis": " ".join(hyp[y.i_res] for y in sekitar),
            })

        # --- rekaman ---
        tot = pd.DataFrame(blok)[JUMLAH].sum()
        m = manifest.loc[fid]
        w = waktu[waktu.file_id == fid].set_index("ulangan")
        rek.append({
            "konfigurasi": konfigurasi, "kode": kode, "file_id": fid,
            "mata_kuliah": a["mata_kuliah"], "jalur_acuan": m.get("jalur_acuan", ""),
            "penutur": m.get("penutur", ""), "durasi_s": round(durasi[fid], 3),
            **{k: int(tot[k]) for k in JUMLAH},
            "WER": tot.err_kata / tot.N, "CER": tot.err_kar / tot.N_kar,
            "TTER": tot.E_G / tot.G if tot.G else np.nan,
            "istilah_muncul": len(a["istilah"]),
            "istilah_salah": sum(1 for r in ist_rows if r["file_id"] == fid and not r["benar"]),
            "segmen": len(seg), "segmen_dibuang": len(buang),
            "durasi_setelah_vad_s": w.durasi_setelah_vad_s.iloc[0] if len(w) else np.nan,
            **{f"rtf_u{u}": w.rtf.get(u, np.nan) for u in sorted(waktu.ulangan.unique())},
            "rtf_rerata": w.rtf.mean() if len(w) else np.nan,
            "sidik_sama_antarulangan": w.sidik.nunique() == 1 if len(w) else np.nan,
        })

    return {"rekaman": pd.DataFrame(rek), "segmen": pd.DataFrame(seg_rows),
            "istilah": pd.DataFrame(ist_rows), "penyelarasan": selaras}


def periksa_terhadap(rekaman: pd.DataFrame, metrik_rekaman_csv: Path) -> int:
    """Pastikan pembilang dan penyebut log identik dengan metrik_rekaman.csv yang sudah ada."""
    lama = pd.read_csv(metrik_rekaman_csv, encoding="utf-8-sig")
    gab = rekaman.merge(lama, left_on=["konfigurasi", "file_id"], right_on=["varian", "file_id"],
                        suffixes=("", "_lama"))
    assert len(gab) == len(rekaman), "ada rekaman log yang tidak ditemukan di metrik_rekaman.csv"
    beda = [k for k in JUMLAH if not (gab[k] == gab[f"{k}_lama"]).all()]
    assert not beda, f"log tidak identik dengan {metrik_rekaman_csv.name} pada kolom {beda}"
    return len(gab)


def tulis_log(hasil: list[dict], folder_log: Path) -> dict:
    """Gabungkan log beberapa konfigurasi dan tulis ke folder_log. Mengembalikan jumlah baris."""
    folder_log = Path(folder_log)
    folder_log.mkdir(parents=True, exist_ok=True)
    jumlah = {}
    for nama, berkas in (("rekaman", "rekaman.csv"), ("segmen", "segmen_hipotesis.csv"),
                         ("istilah", "kemunculan_istilah.csv")):
        df = pd.concat([h[nama] for h in hasil], ignore_index=True)
        df.to_csv(folder_log / berkas, index=False, encoding="utf-8-sig")
        jumlah[berkas] = len(df)
    n = 0
    for h in hasil:
        konf = h["rekaman"].konfigurasi.iloc[0]
        d = folder_log / "penyelarasan" / konf
        d.mkdir(parents=True, exist_ok=True)
        for fid, df in h["penyelarasan"].items():
            df.to_csv(d / f"{fid}.tsv", sep="\t", index=False, encoding="utf-8-sig")
            n += 1
    jumlah["penyelarasan/*.tsv"] = n
    return jumlah
