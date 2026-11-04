"""Ringkasan dan kurva pelatihan LoRA dari log src/latih_lora.py.

Dipakai notebook 11, 11a (pemantauan dan uji coba), dan 11b.

- ``muat``          : konfigurasi, log langkah, log validasi, dan sesi satu folder pelatihan.
- ``per_epoch``     : loss latih rerata tiap epoch beserta validasi terakhir di dalam epoch itu.
- ``ringkas_konvergensi`` : perubahan loss dan WER di akhir pelatihan, untuk membaca konvergensi.
- ``gambar_langkah``: loss per langkah optimizer dan WER validasi per evaluasi.
- ``gambar_epoch``  : loss latih dan loss validasi per epoch, serta WER validasi per epoch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .latih_lora import baca_jsonl

__all__ = ["muat", "per_epoch", "ringkas_konvergensi", "gambar_langkah", "gambar_epoch"]

BIRU, ABU, ABU_TUA, HITAM, JINGGA = "#2a78d6", "#c3c2b7", "#52514e", "#0b0b0b", "#d9822b"


def muat(folder) -> dict:
    folder = Path(folder)
    konf = json.loads((folder / "konfigurasi.json").read_text(encoding="utf-8"))
    langkah = pd.DataFrame(baca_jsonl(folder / "log_langkah.jsonl"))
    val = pd.DataFrame(baca_jsonl(folder / "log_validasi.jsonl"))
    per = konf["rencana"]["langkah_per_epoch"]
    if len(val):
        val["epoch"] = val.langkah / per
        if "loss_validasi" not in val:
            val["loss_validasi"] = np.nan
    return {"konfigurasi": konf, "langkah": langkah, "validasi": val,
            "sesi": pd.DataFrame(baca_jsonl(folder / "sesi.jsonl")), "langkah_per_epoch": per}


def per_epoch(d: dict) -> pd.DataFrame:
    """Satu baris per epoch yang sudah selesai atau sedang berjalan.

    loss_latih adalah rerata loss langkah optimizer dibobot jumlah segmen, sehingga
    sama dengan rerata loss per segmen dalam epoch itu. Validasi tidak jatuh tepat di
    akhir epoch (tiap 200 langkah, sedangkan satu epoch 251 langkah), jadi kolom
    validasi berasal dari evaluasi terakhir yang langkahnya tidak melewati akhir epoch.
    """
    L, V, per = d["langkah"], d["validasi"], d["langkah_per_epoch"]
    if not len(L):
        return pd.DataFrame()
    g = L.groupby("epoch")
    e = pd.DataFrame({"langkah_akhir": g.langkah.max(), "n_langkah": g.size(),
                      "loss_latih": g.apply(lambda x: np.average(x.loss, weights=x.n_segmen), include_groups=False),
                      "loss_latih_sd": g.loss.std(ddof=1)})
    e["lengkap"] = e.n_langkah == per
    e["perubahan_loss_%"] = 100 * e.loss_latih.pct_change()
    if len(V):
        for k, r in e.iterrows():
            v = V[V.langkah <= k * per]
            if len(v):
                v = v.iloc[-1]
                e.loc[k, ["langkah_validasi", "WER_validasi", "loss_validasi"]] = [v.langkah, 100 * v.WER,
                                                                                  v.loss_validasi]
    return e


def ringkas_konvergensi(d: dict, jendela: int = 5) -> pd.Series:
    """Perubahan di akhir pelatihan. Deskriptif, bukan uji statistik."""
    e = per_epoch(d)
    e = e[e.lengkap]
    V = d["validasi"]
    kandidat = V[V.boleh_terpilih] if len(V) else V
    out = {"epoch_lengkap": len(e)}
    if len(e) > jendela:
        a, b = e.loss_latih.iloc[-jendela - 1], e.loss_latih.iloc[-1]
        out[f"perubahan loss latih {jendela} epoch terakhir (%)"] = round(100 * (b - a) / a, 2)
        out["perubahan loss latih per epoch, rerata (%)"] = round(e["perubahan_loss_%"].iloc[-jendela:].mean(), 2)
    if len(kandidat):
        t = kandidat.loc[kandidat.WER.idxmin()]
        akhir = kandidat.tail(jendela)
        out |= {"WER validasi terbaik (%)": round(100 * t.WER, 2), "epoch terbaik": round(t.epoch, 2),
                f"WER {jendela} validasi terakhir (%)": f"{100 * akhir.WER.min():.2f}-{100 * akhir.WER.max():.2f}"}
        lv = kandidat.loss_validasi.dropna()
        if len(lv):
            i = lv.idxmin()
            out |= {"loss validasi terendah": round(lv.min(), 4), "epoch loss validasi terendah": round(V.epoch[i], 2),
                    "loss validasi terakhir": round(lv.iloc[-1], 4)}
    return pd.Series(out)


def _rapikan(ax):
    for a in ax:
        a.grid(alpha=0.3, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)


def _panel_wer(a, V, x: str, label_x: str):
    w0 = 100 * V.loc[V.langkah == 0, "WER"].iloc[0] if (V.langkah == 0).any() else None
    a.plot(V[x], 100 * V.WER, color=BIRU, lw=2, marker="o", ms=4)
    if w0 is not None:
        a.axhline(w0, color=ABU_TUA, lw=1, ls="--")
        a.annotate(f"langkah 0: {w0:.2f}%", (0, w0), textcoords="offset points", xytext=(4, 4), color=ABU_TUA)
    k = V[V.boleh_terpilih]
    if len(k):
        t = k.loc[k.WER.idxmin()]
        a.plot(t[x], 100 * t.WER, "o", ms=9, mfc="none", mec=HITAM)
        a.annotate(f"terbaik: {100 * t.WER:.2f}% (langkah {int(t.langkah)})", (t[x], 100 * t.WER),
                   textcoords="offset points", xytext=(4, -14))
    a.set(title="WER validasi (greedy, 149 segmen)", xlabel=label_x, ylabel="WER (%)")


def gambar_langkah(d: dict, path_png=None):
    import matplotlib.pyplot as plt
    L, V = d["langkah"], d["validasi"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    if len(L):
        ax[0].plot(L.langkah, L.loss, color=ABU, lw=0.6, label="per langkah")
        ax[0].plot(L.langkah, L.loss.rolling(50, min_periods=1).mean(), color=BIRU, lw=2, label="rerata 50 langkah")
        ax[0].legend(frameon=False)
    ax[0].set(title="Loss latih", xlabel="langkah optimizer", ylabel="loss")
    if len(V):
        _panel_wer(ax[1], V, "langkah", "langkah optimizer")
    _rapikan(ax)
    fig.tight_layout()
    if path_png:
        fig.savefig(path_png, dpi=150)
    return fig


def gambar_epoch(d: dict, path_png=None):
    """Loss latih rerata per epoch dan loss validasi pada satu sumbu (keduanya loss per segmen),
    serta WER validasi terhadap epoch. Titik validasi diletakkan pada posisi epoch pecahannya,
    sedangkan rerata loss latih satu epoch diletakkan di tengah epoch itu."""
    import matplotlib.pyplot as plt
    e, V = per_epoch(d), d["validasi"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    if len(e):
        # rerata satu epoch digambar di tengah epoch itu, supaya sejajar dengan titik validasi
        ax[0].plot(e.index - 0.5, e.loss_latih, color=BIRU, lw=2, marker="o", ms=4,
                   label="loss latih (rerata epoch, di tengah epoch)")
    if len(V) and V.loss_validasi.notna().any():
        ax[0].plot(V.epoch, V.loss_validasi, color=JINGGA, lw=2, marker="s", ms=4, label="loss validasi")
    ax[0].set(title="Loss per epoch", xlabel="epoch", ylabel="loss per segmen")
    ax[0].legend(frameon=False)
    if len(V):
        _panel_wer(ax[1], V, "epoch", "epoch")
    _rapikan(ax)
    fig.tight_layout()
    if path_png:
        fig.savefig(path_png, dpi=150)
    return fig
