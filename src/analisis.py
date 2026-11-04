"""Fungsi analisis yang dipakai bersama notebook Tahap 2, Tahap 3, dan analisis gabungan (subbab 3.4).

Isinya:
- ``gabung``: penggabungan antarblok atau antarrekaman dengan menjumlahkan pembilang
  dan penyebut lebih dahulu (subbab 2.8.4). Dipindah dari notebook 09.
- ``uji_wilcoxon``: uji peringkat bertanda Wilcoxon dua sisi dengan statistik W
  Persamaan 2.19. Dipindah dari notebook 09.
- ``deskriptif``: agregat, rerata, simpangan baku, minimum, dan maksimum atas blok.
- ``bandingkan_istilah``: status tiap kemunculan istilah terhadap baseline
  (diperbaiki, dirusak, tetap benar, tetap salah).
- ``kesalahan_istilah`` dan ``kandidat_halusinasi``: tabel untuk dilabeli manual
  (subbab 3.4 butir 5), diturunkan dari log src/log_data.py dengan aturan yang
  sama dengan sel 5.4 notebook 09.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

__all__ = ["JUMLAH", "METRIK", "metrik_dari_jumlah", "gabung", "uji_wilcoxon", "deskriptif", "bandingkan_istilah",
           "kesalahan_istilah", "kandidat_halusinasi"]

def _bool(s: pd.Series) -> pd.Series:
    """Kolom benar/salah yang terbaca sebagai bool maupun teks 'True'/'False'."""
    return s.astype(str).eq("True")


JUMLAH = ["N", "S", "D", "I", "err_kata", "err_kar", "N_kar", "E_G", "G"]
METRIK = ["WER", "CER", "TTER"]


def metrik_dari_jumlah(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung ulang WER, CER, TTER dari pembilang dan penyebut bilangan bulat.

    Dipakai sebelum membandingkan nilai yang dibaca dari CSV dengan nilai yang baru
    dihitung, supaya selisih pembulatan desimal tidak terbaca sebagai perbedaan oleh
    uji Wilcoxon. Penyebut nol menghasilkan NaN, sama seperti src/metrik.py.
    """
    df = df.copy()
    for m, a, b in (("WER", "err_kata", "N"), ("CER", "err_kar", "N_kar"), ("TTER", "E_G", "G")):
        df[m] = (df[a] / df[b].where(df[b] > 0)).astype(float)
    return df


def gabung(df: pd.DataFrame, by) -> pd.DataFrame:
    """Jumlahkan pembilang dan penyebut per kelompok, lalu hitung WER, CER, TTER."""
    g = df.groupby(by, sort=False)[JUMLAH].sum()
    g["WER"] = g.err_kata / g.N
    g["CER"] = g.err_kar / g.N_kar
    g["TTER"] = g.E_G / g.G
    return g


def uji_wilcoxon(x, y) -> dict:
    """Wilcoxon dua sisi, pasangan berselisih nol dibuang. Selisih median dalam poin persentase (x - y)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    dn = d[d != 0]
    if len(dn) == 0:
        return {"n": len(d), "n_tak_nol": 0, "W": 0.0, "p": 1.0, "selisih_median": 0.0}
    R = rankdata(np.abs(dn))
    p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue
    return {"n": len(d), "n_tak_nol": len(dn), "W": float(np.sum(np.sign(dn) * R)), "p": float(p),
            "selisih_median": float(np.median(100 * d))}


def deskriptif(blok: pd.DataFrame, by: str = "kode") -> pd.DataFrame:
    """Statistik deskriptif atas blok tiga menit, dalam persen.

    Nilai agregat dihitung dengan ``gabung``; rerata, simpangan baku, minimum, dan
    maksimum dihitung atas nilai tiap blok. Blok tanpa istilah tidak ikut TTER.
    """
    agr = gabung(blok, by)
    baris = []
    for k, b in blok.groupby(by, sort=False):
        d = {by: k}
        for m in METRIK:
            x = 100 * b[m].dropna()
            d |= {f"{m}_agregat": 100 * agr.at[k, m], f"{m}_rerata": x.mean(), f"{m}_sd": x.std(ddof=1),
                  f"{m}_min": x.min(), f"{m}_maks": x.max(), f"{m}_n_blok": len(x)}
        baris.append(d)
    return pd.DataFrame(baris).set_index(by)


def bandingkan_istilah(dasar: pd.DataFrame, perlakuan: pd.DataFrame) -> pd.DataFrame:
    """Pasangkan kemunculan istilah (log kemunculan_istilah.csv) baseline dan perlakuan.

    Kunci pasangan adalah (file_id, i0), yaitu posisi istilah pada acuan, sehingga
    kedua konfigurasi dinilai pada kemunculan yang sama.
    """
    kunci = ["file_id", "i0"]
    tetap = ["mata_kuliah", "blok", "acuan_mulai", "entri_id", "bentuk_kanonik", "istilah_acuan"]
    a = dasar.set_index(kunci).sort_index()
    b = perlakuan.set_index(kunci).sort_index()
    assert a.index.equals(b.index), "kemunculan istilah baseline dan perlakuan tidak sama"
    out = a[tetap].copy()
    out["hipotesis_dasar"] = a.hipotesis
    out["hipotesis_perlakuan"] = b.hipotesis
    out["benar_dasar"] = _bool(a.benar)
    out["benar_perlakuan"] = _bool(b.benar)
    out["status"] = np.select(
        [out.benar_dasar & out.benar_perlakuan, ~out.benar_dasar & out.benar_perlakuan,
         out.benar_dasar & ~out.benar_perlakuan],
        ["tetap benar", "diperbaiki", "dirusak"], "tetap salah")
    return out.reset_index()


def kesalahan_istilah(kemunculan: pd.DataFrame) -> pd.DataFrame:
    """Kemunculan istilah yang salah, siap dilabeli (format sel 5.4 notebook 09).

    Baris yang seluruh operasinya penghapusan langsung diisi 'penghapusan'.
    """
    k = kemunculan[~_bool(kemunculan.benar)].copy()
    k["golongan"] = np.where((k.S.astype(int) == 0) & (k.I.astype(int) == 0), "penghapusan", "")
    k["catatan"] = ""
    kolom = ["konfigurasi", "file_id", "mata_kuliah", "blok", "i0", "entri_id", "bentuk_kanonik",
             "istilah_acuan", "hipotesis", "S", "D", "I", "konteks_acuan", "konteks_hipotesis",
             "golongan", "catatan"]
    return k[kolom].reset_index(drop=True)


def kandidat_halusinasi(folder_log: Path, konfigurasi: str, min_sisip: int = 5,
                        konteks: int = 5) -> pd.DataFrame:
    """Penyisipan beruntun >= min_sisip kata dan segmen yang dibuang karena di luar acuan."""
    folder_log = Path(folder_log)
    baris = []
    for f in sorted((folder_log / "penyelarasan" / konfigurasi).glob("*.tsv")):
        # keep_default_na=False supaya kata seperti "na" atau "null" tidak terbaca sebagai kosong
        p = pd.read_csv(f, sep="\t", encoding="utf-8-sig", keep_default_na=False)
        p["i_acuan"] = pd.to_numeric(p.i_acuan, errors="coerce")
        ref = p.loc[p.operasi != "I"].sort_values("i_acuan")
        ref_tok = ref.kata_acuan.astype(str).tolist()
        N = len(ref_tok)
        ops = p.operasi.tolist() + ["C"]
        run = []
        for i, o in enumerate(ops):
            if o == "I":
                run.append(i)
                continue
            if len(run) >= min_sisip:
                # posisi acuan sesudah penyisipan; sesudah kata terakhir dipetakan ke N - 1
                k = int(p.i_acuan.iloc[i]) if i < len(p) else N - 1
                baris.append({"konfigurasi": konfigurasi, "file_id": f.stem, "sumber": "penyisipan beruntun",
                              "blok": int(p.blok.iloc[run[0]]), "n_kata": len(run),
                              "teks": " ".join(p.kata_hipotesis.iloc[run].astype(str)),
                              "konteks_acuan": " ".join(ref_tok[max(0, k - konteks):k + konteks]),
                              "halusinasi": "", "catatan": ""})
            run = []
    seg = pd.read_csv(folder_log / "segmen_hipotesis.csv", encoding="utf-8-sig", keep_default_na=False)
    seg = seg[(seg.konfigurasi == konfigurasi) & ~_bool(seg.dipakai)]
    for s in seg.itertuples():
        baris.append({"konfigurasi": konfigurasi, "file_id": s.file_id, "sumber": "segmen di luar acuan",
                      "blok": int(s.blok), "n_kata": len(str(s.teks).split()), "teks": str(s.teks).strip(),
                      "konteks_acuan": f"{float(s.mulai):.2f}-{float(s.selesai):.2f} detik", "halusinasi": "", "catatan": ""})
    return pd.DataFrame(baris, columns=["konfigurasi", "file_id", "sumber", "blok", "n_kata", "teks",
                                        "konteks_acuan", "halusinasi", "catatan"])
