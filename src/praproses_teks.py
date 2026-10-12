"""Praproses teks dan penyiapan segmen latih.

Modul ini dipakai bersama oleh notebook 08 (transkrip acuan) dan notebook
Tahap 1 sampai Tahap 3 (transkrip hipotesis), supaya normalisasi yang
dikenakan pada acuan dan hipotesis benar-benar identik (subbab 3.6.3).

Urutan normalisasi untuk satu teks:

1. Pra-BasicTextNormalizer
   - NFKC.
   - "%" sesudah angka diganti " persen", agar "10%" dan "sepuluh persen"
     berakhir sama.
   - Titik pemisah ribuan pada angka ("2.000", "1.000.000") dihapus, agar
     "2.000" dan "dua ribu" berakhir sama.
2. BasicTextNormalizer bawaan Whisper (pustaka transformers): huruf kecil,
   teks di dalam kurung dibuang, seluruh tanda baca dan simbol termasuk tanda
   hubung diganti spasi (Tabel 3.4).
3. Aturan berbasis pola, berlaku untuk seluruh token termasuk istilah
   glosarium:
   - kata bilangan menjadi angka ("seribu sembilan ratus empat puluh"
     menjadi "1940"; "2 ribu" menjadi "2000");
   - "titik" dan "koma" di antara dua angka dibuang, sebab pada keluaran
     model keduanya tertulis sebagai tanda baca yang sudah dibuang langkah 2
     ("802 titik 11" dan "802.11" sama-sama menjadi "802 11");
   - huruf tunggal berurutan dirapatkan ("u m l" menjadi "uml", "i o" dari
     "I/O" menjadi "io").
4. Pencocokan kamus tertutup, hanya untuk token di luar istilah glosarium:
   - jeda terisi dibuang;
   - varian ejaan bentuk lisan diseragamkan;
   - singkatan tulis dipanjangkan.

Bentuk permukaan glosarium dinormalisasi dengan fungsi yang sama, sehingga
indeks posisinya sejalan dengan transkrip.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from transformers.models.whisper.english_normalizer import BasicTextNormalizer  # noqa: E402

__all__ = [
    "Kamus", "Normalisasi", "muat_kamus", "simpan_kamus",
    "muat_glosarium", "tandai_glosarium", "gabung_segmen",
    "kata_bilangan_ke_angka", "himpunan_g", "potong_audio", "KAMUS_DRAF",
]

_BASIC = BasicTextNormalizer()


# ---------------------------------------------------------------------------
# Kamus tertutup
# ---------------------------------------------------------------------------

KAMUS_DRAF = {
    "versi": 1,
    "status": "draf",
    "tanggal_kunci": None,
    "disusun_dari": "transkrip acuan himpunan latih dan validasi, ditambah contoh "
                    "pada subbab 3.6.3",
    "catatan": "Ubah status menjadi 'terkunci' dan isi tanggal_kunci setelah ditinjau. "
               "Sesudah terkunci, isi kamus tidak boleh diubah selama pengukuran.",
    # Kelompok 1: penyeragaman varian ejaan bentuk lisan.
    # Target 'enggak' mengikuti lema KBBI.
    "ejaan_lisan": {
        "gak": "enggak",
        "ga": "enggak",
        "engga": "enggak",
        "ngga": "enggak",
        "nggak": "enggak",
    },
    # Kelompok 2: pemanjangan singkatan tulis.
    "singkatan": {
        "yg": "yang",
        "dgn": "dengan",
        "tdk": "tidak",
        "utk": "untuk",
        "dll": "dan lain lain",
        "dsb": "dan sebagainya",
        "dst": "dan seterusnya",
        "tsb": "tersebut",
    },
    # Kelompok 3: jeda terisi, dibuang.
    "jeda_terisi": ["e", "ee", "eee", "eeee", "eem", "em", "emm", "hm", "hmm", "mm"],
}


@dataclass(frozen=True)
class Kamus:
    ejaan_lisan: dict
    singkatan: dict
    jeda_terisi: frozenset
    status: str = "draf"
    tanggal_kunci: str | None = None

    @classmethod
    def dari_dict(cls, d: dict) -> "Kamus":
        def _cek(peta, nama):
            for k, v in peta.items():
                if k != _BASIC(k).strip() or " " in k:
                    raise ValueError(f"kunci {nama} '{k}' harus satu token hasil normalisasi dasar")
        _cek(d["ejaan_lisan"], "ejaan_lisan")
        _cek(d["singkatan"], "singkatan")
        return cls(
            ejaan_lisan=dict(d["ejaan_lisan"]),
            singkatan=dict(d["singkatan"]),
            jeda_terisi=frozenset(d["jeda_terisi"]),
            status=d.get("status", "draf"),
            tanggal_kunci=d.get("tanggal_kunci"),
        )

    @property
    def terkunci(self) -> bool:
        return self.status == "terkunci" and bool(self.tanggal_kunci)


def muat_kamus(path: str | Path) -> Kamus:
    with open(path, encoding="utf-8") as fh:
        return Kamus.dari_dict(json.load(fh))


def simpan_kamus(d: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Bilangan
# ---------------------------------------------------------------------------

_SATUAN = {"nol": 0, "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5,
           "enam": 6, "tujuh": 7, "delapan": 8, "sembilan": 9}
_SE = {"sepuluh": 10, "sebelas": 11, "seratus": 100, "seribu": 1000, "sejuta": 10**6}
_BESAR = {"ribu": 10**3, "juta": 10**6, "miliar": 10**9, "triliun": 10**12}
_KATA_BILANGAN = set(_SATUAN) | set(_SE) | set(_BESAR) | {"belas", "puluh", "ratus"}


def kata_bilangan_ke_angka(tokens: list[str]) -> list[str]:
    """Ubah deret kata bilangan menjadi angka.

    Bagian di bawah seribu diisi berurutan dari ratusan, puluhan, lalu satuan,
    dan satu bilangan berakhir ketika urutan itu tidak dapat dilanjutkan.
    Karena itu "dua puluh lima" menjadi "25", sedangkan "satu dua tiga" menjadi
    "1 2 3" dan "delapan nol dua" menjadi "8 0 2". Angka yang langsung diikuti
    ribu/juta/miliar ikut dikalikan ("2 ribu" menjadi "2000"). Kata pengali
    tanpa satuan di depannya ("beberapa ratus") dibiarkan sebagai kata.
    """
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        j, nilai = _baca_bilangan(tokens, i)
        if j == i:
            out.append(tokens[i])
            i += 1
        else:
            out.append(str(nilai))
            i = j
    return out


def _baca_bilangan(tokens: list[str], i: int) -> tuple[int, int]:
    """Baca satu bilangan mulai dari indeks i. Kembalikan (indeks_sesudahnya, nilai)."""
    n = len(tokens)
    besar, kecil = 0, 0
    posisi = 3          # 3: ratusan masih boleh, 2: puluhan, 1: satuan, 0: penuh
    pengali_akhir = None
    j = i
    # Angka yang diikuti ribu/juta/miliar, misalnya "2 ribu".
    if tokens[i].isdigit():
        if i + 1 < n and tokens[i + 1] in _BESAR and 0 < int(tokens[i]) < 1000:
            kecil, posisi, j = int(tokens[i]), 0, i + 1
        else:
            return i, 0
    while j < n:
        w = tokens[j]
        berikut = tokens[j + 1] if j + 1 < n else None
        if w in _SATUAN:
            u = _SATUAN[w]
            if berikut == "ratus":
                if posisi < 3 or u == 0:
                    break
                kecil += u * 100; posisi = 2; j += 2
            elif berikut == "puluh":
                if posisi < 2 or u == 0:
                    break
                kecil += u * 10; posisi = 1; j += 2
            elif berikut == "belas":
                if posisi < 2 or u == 0:
                    break
                kecil += 10 + u; posisi = 0; j += 2
            elif u == 0:
                if j != i:
                    break
                return j + 1, 0
            else:
                if posisi < 1:
                    break
                kecil += u; posisi = 0; j += 1
        elif w == "seratus":
            if posisi < 3:
                break
            kecil += 100; posisi = 2; j += 1
        elif w in ("sepuluh", "sebelas"):
            if posisi < 2:
                break
            kecil += _SE[w]; posisi = 0; j += 1
        elif w in ("seribu", "sejuta"):
            v = _SE[w]
            if kecil or (pengali_akhir is not None and pengali_akhir <= v):
                break
            besar += v; pengali_akhir = v; posisi = 3; j += 1
        elif w in _BESAR:
            v = _BESAR[w]
            if not kecil or (pengali_akhir is not None and pengali_akhir <= v):
                break
            besar += kecil * v; kecil = 0; pengali_akhir = v; posisi = 3; j += 1
        else:
            break
    if j == i:
        return i, 0
    return j, besar + kecil


def _buang_titik_koma_antarangka(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for k, t in enumerate(tokens):
        if (t in ("titik", "koma") and out and out[-1].isdigit()
                and k + 1 < len(tokens) and tokens[k + 1].isdigit()):
            continue
        out.append(t)
    return out


_HURUF = re.compile(r"^[a-z]$")


def _rapatkan_huruf(tokens: list[str]) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for t in tokens + [None]:
        if t is not None and _HURUF.match(t):
            buf.append(t)
            continue
        if len(buf) >= 2:
            out.append("".join(buf))
        else:
            out.extend(buf)
        buf = []
        if t is not None:
            out.append(t)
    return out


_PERSEN = re.compile(r"(?<=\d)\s*%")
_RIBUAN = re.compile(r"(?<![\d.,])\d{1,3}(?:\.\d{3})+(?![\d.,]*\d)")


def _pra(teks: str) -> str:
    t = unicodedata.normalize("NFKC", str(teks))
    t = _PERSEN.sub(" persen", t)
    t = _RIBUAN.sub(lambda m: m.group(0).replace(".", ""), t)
    return t


def normalisasi_pola(teks: str) -> list[str]:
    """Langkah 1 sampai 3: BasicTextNormalizer dan aturan berbasis pola."""
    tokens = _BASIC(_pra(teks)).split()
    tokens = kata_bilangan_ke_angka(tokens)
    tokens = _buang_titik_koma_antarangka(tokens)
    tokens = _rapatkan_huruf(tokens)
    return tokens


# ---------------------------------------------------------------------------
# Glosarium
# ---------------------------------------------------------------------------

@dataclass
class BentukGlosarium:
    token: tuple
    entri_id: str
    bentuk_permukaan: str


@dataclass
class GlosariumMK:
    mata_kuliah: str
    bentuk: dict = field(default_factory=dict)   # token tuple -> BentukGlosarium
    panjang_maks: int = 0

    def tambah(self, b: BentukGlosarium) -> None:
        lama = self.bentuk.get(b.token)
        if lama is not None and lama.entri_id != b.entri_id:
            raise ValueError(f"{self.mata_kuliah}: bentuk '{' '.join(b.token)}' milik dua entri "
                             f"({lama.entri_id}, {b.entri_id}) setelah normalisasi")
        self.bentuk.setdefault(b.token, b)
        self.panjang_maks = max(self.panjang_maks, len(b.token))


def muat_glosarium(path: str | Path) -> dict[str, GlosariumMK]:
    """Baca glosarium_semua.csv dan normalisasi tiap bentuk permukaan."""
    import csv
    hasil: dict[str, GlosariumMK] = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            mk = r["mata_kuliah"]
            tok = tuple(normalisasi_pola(r["bentuk_permukaan"]))
            if not tok:
                continue
            g = hasil.setdefault(mk, GlosariumMK(mk))
            g.tambah(BentukGlosarium(tok, r["entri_id"], r["bentuk_permukaan"]))
    return hasil


def tandai_glosarium(tokens: list[str], glosarium: GlosariumMK | None) -> list[tuple[int, int, str]]:
    """Cari kemunculan istilah, istilah terpanjang didahulukan.

    Mengembalikan daftar (i_awal, i_akhir_eksklusif, entri_id). Himpunan G pada
    Persamaan 2.7 adalah gabungan seluruh rentang ini.
    """
    if glosarium is None or not glosarium.bentuk:
        return []
    out, i, n = [], 0, len(tokens)
    while i < n:
        for L in range(min(glosarium.panjang_maks, n - i), 0, -1):
            b = glosarium.bentuk.get(tuple(tokens[i:i + L]))
            if b is not None:
                out.append((i, i + L, b.entri_id))
                i += L
                break
        else:
            i += 1
    return out


def himpunan_g(rentang: Iterable[tuple[int, int, str]]) -> list[int]:
    return sorted({k for a, b, _ in rentang for k in range(a, b)})


# ---------------------------------------------------------------------------
# Normalisasi lengkap
# ---------------------------------------------------------------------------

class Normalisasi:
    """Fungsi normalisasi untuk satu mata kuliah.

    Pakai objek yang sama untuk acuan dan hipotesis:

        norm = Normalisasi(kamus, glosarium["AOK"])
        token = norm(teks)
    """

    def __init__(self, kamus: Kamus, glosarium: GlosariumMK | None = None):
        self.kamus = kamus
        self.glosarium = glosarium

    def __call__(self, teks: str) -> list[str]:
        tokens = normalisasi_pola(teks)
        lindung = [False] * len(tokens)
        for a, b, _ in tandai_glosarium(tokens, self.glosarium):
            for k in range(a, b):
                lindung[k] = True
        out: list[str] = []
        for t, dilindungi in zip(tokens, lindung):
            if dilindungi:
                out.append(t)
            elif t in self.kamus.jeda_terisi:
                continue
            elif t in self.kamus.ejaan_lisan:
                out.append(self.kamus.ejaan_lisan[t])
            elif t in self.kamus.singkatan:
                out.extend(self.kamus.singkatan[t].split())
            else:
                out.append(t)
        return out

    def teks(self, teks: str) -> str:
        return " ".join(self(teks))


# ---------------------------------------------------------------------------
# Segmen latih (subbab 3.6.2)
# ---------------------------------------------------------------------------

def gabung_segmen(segmen, maks_durasi: float = 30.0, maks_celah: float = 1.0) -> list[dict]:
    """Gabungkan segmen acuan berurutan dari satu rekaman sampai mendekati 30 detik.

    `segmen` adalah iterable dict/baris dengan kunci seg_id, start, end, teks_acuan,
    sudah urut menurut waktu dan berasal dari satu rekaman. Penggabungan berhenti
    bila rentang gabungan akan melebihi `maks_durasi`, atau bila celah ke segmen
    berikutnya melebihi `maks_celah` detik, sebab celah panjang dapat memuat
    ujaran yang tidak bertranskrip.
    """
    hasil, cur = [], None
    for s in segmen:
        st, en = float(s["start"]), float(s["end"])
        teks = str(s["teks_acuan"]).strip()
        if cur is not None and en - cur["start"] <= maks_durasi and st - cur["end"] <= maks_celah:
            cur["end"] = en
            cur["seg_id_akhir"] = s["seg_id"]
            cur["n_segmen_asal"] += 1
            cur["teks"].append(teks)
        else:
            if cur is not None:
                hasil.append(cur)
            cur = {"start": st, "end": en, "seg_id_awal": s["seg_id"], "seg_id_akhir": s["seg_id"],
                   "n_segmen_asal": 1, "teks": [teks]}
    if cur is not None:
        hasil.append(cur)
    for h in hasil:
        h["teks"] = " ".join(t for t in h["teks"] if t)
        h["dur"] = h["end"] - h["start"]
    return hasil


def potong_audio(sumber: str | Path, start: float, end: float, tujuan: str | Path) -> float:
    """Potong rentang [start, end] detik dari WAV sumber ke WAV tujuan.

    Format keluaran sama dengan sumber (16 kHz, mono, PCM 16-bit pada data ini).
    Mengembalikan durasi hasil dalam detik.
    """
    import soundfile as sf
    info = sf.info(str(sumber))
    a = max(0, int(round(start * info.samplerate)))
    b = min(info.frames, int(round(end * info.samplerate)))
    data, sr = sf.read(str(sumber), start=a, stop=b, dtype="int16", always_2d=False)
    Path(tujuan).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(tujuan), data, sr, subtype="PCM_16")
    return (b - a) / sr
