"""Perhitungan WER, CER, dan TTER (subbab 2.8 dan 3.6.8).

Dipakai notebook Tahap 1 sampai Tahap 3 supaya ketiga tahap dihitung dengan
prosedur yang identik.

Alur untuk satu rekaman dan satu konfigurasi:

1. ``saring_segmen``: segmen hipotesis yang tidak beririsan waktu dengan satu
   pun segmen acuan dibuang, sebab bagian rekaman itu tidak punya acuan.
2. ``normalisasi_hipotesis``: tiap segmen hipotesis dinormalisasi tersendiri
   dengan fungsi yang sama dengan acuan (src/praproses_teks.py).
3. ``selaraskan``: satu penyelarasan tingkat kata per rekaman (jarak
   Levenshtein dengan penelusuran balik). Hasilnya himpunan operasi sunting E
   pada Persamaan 2.8, berupa (jenis, i_src, i_res).
4. ``hitung_rekaman``: operasi dibagi ke blok tiga menit menurut stempel
   waktu segmen acuan, lalu WER dan TTER dihitung dari operasi itu dan CER
   dihitung dengan jiwer pada teks tiap blok.

Ketentuan posisi operasi:
- Substitusi, kecocokan, dan penghapusan memakai indeks kata acuan yang
  bersangkutan.
- Penyisipan diberi i_src = indeks kata acuan sesudahnya. Penyisipan sesudah
  kata acuan terakhir diberi i_src = N (panjang acuan). Aturan tepi Ugan, et
  al. (2025): operasi dengan i_src >= N ikut masuk E_G bila kata acuan
  terakhir termasuk G.
- Penelusuran balik mendahulukan langkah diagonal (cocok/substitusi), lalu
  penghapusan, lalu penyisipan. Jumlah operasinya selalu sama dengan jarak
  minimum, sehingga WER-nya sama dengan jiwer (diperiksa ``periksa_jiwer``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Operasi", "saring_segmen", "normalisasi_hipotesis", "selaraskan",
    "hitung_rekaman", "periksa_jiwer", "rentang_istilah_hipotesis",
    "PANJANG_BLOK",
]

PANJANG_BLOK = 180.0  # detik, subbab 3.4 butir 1


@dataclass(frozen=True)
class Operasi:
    jenis: str          # "C" cocok, "S" substitusi, "D" penghapusan, "I" penyisipan
    i_src: int          # indeks kata acuan (penyisipan: indeks kata acuan sesudahnya)
    i_res: int | None   # indeks kata hipotesis (None untuk penghapusan)


# ---------------------------------------------------------------------------
# Hipotesis
# ---------------------------------------------------------------------------

def saring_segmen(seg_hipotesis: list[dict], seg_acuan: list[dict]) -> tuple[list[dict], list[dict]]:
    """Pisahkan segmen hipotesis yang beririsan waktu dengan acuan dari yang tidak.

    Irisan dihitung terbuka: segmen yang hanya bersentuhan di satu titik
    dianggap tidak beririsan. Mengembalikan (dipakai, dibuang).
    """
    acuan = sorted((float(s["start"]), float(s["end"])) for s in seg_acuan)
    awal = np.array([a for a, _ in acuan])
    akhir_maks = np.maximum.accumulate(np.array([b for _, b in acuan]))
    dipakai, dibuang = [], []
    for s in seg_hipotesis:
        st, en = float(s["start"]), float(s["end"])
        # segmen acuan dengan start < en; cukup periksa akhir terjauh di antaranya
        k = int(np.searchsorted(awal, en, side="left"))
        beririsan = k > 0 and akhir_maks[k - 1] > st
        (dipakai if beririsan else dibuang).append(s)
    return dipakai, dibuang


def normalisasi_hipotesis(seg_hipotesis: list[dict], norm) -> list[str]:
    """Normalisasi tiap segmen tersendiri lalu sambung, sama seperti acuan."""
    token: list[str] = []
    for s in seg_hipotesis:
        token.extend(norm(s["text"]))
    return token


# ---------------------------------------------------------------------------
# Penyelarasan
# ---------------------------------------------------------------------------

def _matriks_jarak(r: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Matriks jarak Levenshtein (n+1) x (m+1), dihitung per baris dengan numpy.

    Untuk baris i, nilai tanpa penyisipan A[j] = min(substitusi/cocok, hapus)
    dihitung sekaligus, lalu penyisipan berantai diselesaikan dengan
    D[i, j] = min_k<=j (A[k] + j - k) = j + cummin(A[k] - k).
    """
    n, m = len(r), len(h)
    D = np.empty((n + 1, m + 1), dtype=np.int32)
    j = np.arange(m + 1, dtype=np.int32)
    D[0] = j
    for i in range(1, n + 1):
        prev = D[i - 1]
        A = np.empty(m + 1, dtype=np.int32)
        A[0] = i
        if m:
            A[1:] = np.minimum(prev[:-1] + (h != r[i - 1]), prev[1:] + 1)
        D[i] = j + np.minimum.accumulate(A - j)
    return D


def selaraskan(acuan: list[str], hipotesis: list[str]) -> list[Operasi]:
    """Selaraskan dua deret token dan kembalikan operasi sunting berurutan."""
    kosakata: dict[str, int] = {}
    r = np.array([kosakata.setdefault(t, len(kosakata)) for t in acuan], dtype=np.int64)
    h = np.array([kosakata.setdefault(t, len(kosakata)) for t in hipotesis], dtype=np.int64)
    D = _matriks_jarak(r, h)
    ops: list[Operasi] = []
    i, j = len(r), len(h)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            beda = int(r[i - 1] != h[j - 1])
            if D[i, j] == D[i - 1, j - 1] + beda:
                ops.append(Operasi("S" if beda else "C", i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(Operasi("D", i - 1, None))
            i -= 1
            continue
        ops.append(Operasi("I", i, j - 1))
        j -= 1
    ops.reverse()
    assert sum(o.jenis != "C" for o in ops) == D[-1, -1]
    return ops


def periksa_jiwer(acuan: list[str], hipotesis: list[str], ops: list[Operasi]) -> dict:
    """Bandingkan jumlah operasi hasil penyelarasan sendiri dengan jiwer."""
    import jiwer
    o = jiwer.process_words(" ".join(acuan), " ".join(hipotesis))
    sendiri = {j: sum(x.jenis == j for x in ops) for j in "SDI"}
    wer_sendiri = (sendiri["S"] + sendiri["D"] + sendiri["I"]) / len(acuan)
    return {
        "wer_sendiri": wer_sendiri, "wer_jiwer": o.wer,
        "sama": math.isclose(wer_sendiri, o.wer, abs_tol=1e-12),
        "S": sendiri["S"], "D": sendiri["D"], "I": sendiri["I"],
        "S_jiwer": o.substitutions, "D_jiwer": o.deletions, "I_jiwer": o.insertions,
    }


# ---------------------------------------------------------------------------
# Metrik per blok dan per rekaman
# ---------------------------------------------------------------------------

def _blok_token(acuan: dict, panjang_blok: float) -> np.ndarray:
    """Indeks blok tiap token acuan, menurut waktu mulai segmen acuannya."""
    blok = np.zeros(len(acuan["token"]), dtype=np.int64)
    for s in acuan["segmen"]:
        blok[s["i0"]:s["i1"]] = int(float(s["start"]) // panjang_blok)
    return blok


def hitung_rekaman(acuan: dict, hipotesis: list[str], durasi: float,
                   panjang_blok: float = PANJANG_BLOK) -> tuple[list[dict], list[Operasi]]:
    """Hitung WER, CER, TTER per blok tiga menit untuk satu rekaman.

    `acuan` adalah isi acuan/<ID>.json (token, segmen, G). Mengembalikan
    (baris per blok, operasi). Tiap baris memuat pembilang dan penyebut agar
    penggabungan antarblok dan antarrekaman dilakukan dengan menjumlahkan
    pembilang dan penyebut lebih dahulu (subbab 2.8.4).
    """
    import jiwer

    ref = acuan["token"]
    N = len(ref)
    G = set(acuan["G"])
    ops = selaraskan(ref, hipotesis)
    blok_tok = _blok_token(acuan, panjang_blok)
    n_blok = max(int(math.ceil(durasi / panjang_blok)), int(blok_tok.max()) + 1 if N else 1)

    def blok_op(o: Operasi) -> int:
        return int(blok_tok[min(o.i_src, N - 1)])

    tepi = (N - 1) in G  # aturan tepi penyisipan Ugan, et al. (2025)

    baris = []
    for b in range(n_blok):
        idx = np.flatnonzero(blok_tok == b)
        ops_b = [o for o in ops if blok_op(o) == b]
        n_ref = len(idx)
        S = sum(o.jenis == "S" for o in ops_b)
        Dl = sum(o.jenis == "D" for o in ops_b)
        I = sum(o.jenis == "I" for o in ops_b)
        G_b = [k for k in idx if k in G]
        E_G = sum(1 for o in ops_b if o.jenis != "C" and (o.i_src in G or (o.i_src >= N and tepi)))
        teks_ref = " ".join(ref[k] for k in idx)
        teks_hyp = " ".join(hipotesis[o.i_res] for o in ops_b if o.i_res is not None)
        if n_ref:
            c = jiwer.process_characters(teks_ref, teks_hyp)
            c_err, c_n = c.substitutions + c.deletions + c.insertions, c.hits + c.substitutions + c.deletions
        else:
            c_err, c_n = len(teks_hyp), 0
        baris.append({
            "blok": b, "mulai": b * panjang_blok, "selesai": min((b + 1) * panjang_blok, durasi),
            "N": n_ref, "S": S, "D": Dl, "I": I,
            "err_kata": S + Dl + I, "err_kar": c_err, "N_kar": c_n,
            "E_G": E_G, "G": len(G_b),
            "WER": (S + Dl + I) / n_ref if n_ref else np.nan,
            "CER": c_err / c_n if c_n else np.nan,
            "TTER": E_G / len(G_b) if G_b else np.nan,
        })
    return baris, ops


def rentang_istilah_hipotesis(ops: list[Operasi], i0: int, i1: int, N: int, tepi: bool) -> list[Operasi]:
    """Operasi yang jatuh pada rentang istilah [i0, i1), termasuk penyisipan di dalamnya.

    Penyisipan dengan i_src = i0 (tepat sebelum istilah) ikut, sama seperti
    ketentuan E_G. Penyisipan sesudah kata acuan terakhir ikut bila i1 == N
    dan aturan tepi berlaku.
    """
    out = []
    for o in ops:
        if i0 <= o.i_src < i1 or (o.i_src >= N and tepi and i1 == N):
            out.append(o)
    return out
