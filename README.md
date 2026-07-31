# Assistive Lecture Captioning — Prototype End-to-End

Prototipe pipeline penuh sesuai arahan dosen: rekam suara dosen real-time → caption
langsung tampil & tersimpan → tambahkan materi (PPT/PDF) → generate rangkuman kuliah
rapi pakai LLM. Semua dibungkus jadi satu web app Streamlit.

## Alur pemakaian

1. **Sidebar** — buat sesi baru (mata kuliah, dosen, topik, tanggal) atau pilih sesi yang sudah ada.
2. **Tab 🎙️ Live Caption** — klik START, izinkan akses mic browser. Audio ditranskrip
   per potongan ~4 detik dan caption tampil real-time (font besar, kontras tinggi,
   sesuai KPI aksesibilitas di draft riset kamu). Transkrip otomatis tersimpan ke disk.
3. **Tab 📎 Materi** — upload slide (`.pptx`) atau catatan (`.pdf`) punya sesi itu.
   Teksnya diekstrak otomatis dan dipakai sebagai konteks tambahan buat rangkuman.
4. **Tab 📝 Transkrip & Rangkuman** — transkrip bisa diedit manual (koreksi typo ASR),
   lalu klik **Generate Rangkuman** buat dapet catatan kuliah terstruktur dari LLM,
   bisa didownload sebagai `.md`.

## Kenapa didesain begini (mapping ke riset kamu)

| Komponen | Pilihan | Alasan |
|---|---|---|
| Model ASR | faster-whisper **medium** | Hasil `02-baseline-presentation-kaggle.ipynb`: WER & TTER paling rendah dibanding large-v3-turbo, Wav2Vec2 XLSR, Qwen2-Audio pada domain kuliah kamu. |
| Preprocessing | denoise non-stationary + normalize amplitude | Diporting langsung dari `03_preprocessing_exploration.ipynb` (`full_preprocess`). Bisa dimatikan lewat `config.PREPROCESS_ENABLED` kalau kesimpulan notebook kamu beda atau RTF real-time kelewat tinggi. |
| Streaming | chunk-based, sliding window ~4 detik | Sesuai draft riset (chunk-based inference, sliding window 3-5 detik). "Overlap" konteks diimplementasikan di level **teks** (`initial_prompt` ke Whisper dari potongan transkrip sebelumnya), bukan overlap audio mentah — biar gak perlu logic dedup kata yang rawan duplikat di titik sambungan chunk. |
| Capture audio browser | `streamlit-webrtc` | Standar de-facto buat real-time mic capture di Streamlit, aktif dikembangkan, gak perlu bikin backend FastAPI/WebSocket terpisah seperti di draft riset — semuanya jalan dalam satu proses Streamlit. |
| LLM rangkuman | Gemini **2.5 Flash**, pay-as-you-go | $0.30/1M token input, $2.50/1M token output (Jun 2026) — rangkuman 1 sesi kuliah biasanya cuma beberapa ribu token, jauh di bawah $1. Gampang diganti provider lain — cukup edit `src/summarizer.py`. |
| Storage | JSON per sesi | Prototipe demo, gampang diperiksa manual pas presentasi ke dosen. Upgrade natural ke SQLite/Postgres kalau lanjut ke tahap production. |

## Setup

### 1. Dependency sistem
`pydub` butuh **ffmpeg** terinstall di sistem:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg
# Mac
brew install ffmpeg
```

### 2. Dependency Python
```bash
pip install -r requirements.txt
```
> Model `faster-whisper medium` (~1.5GB) otomatis terdownload dari Hugging Face saat
> pertama kali `load_whisper_model()` dipanggil. Pastikan koneksi internet aktif di
> run pertama; setelah itu tersimpan di cache lokal.

### 3. API key Gemini (pay-as-you-go)
API key Gemini dibuat lewat **Google AI Studio** (bukan "model hub" — itu istilah lain,
biasanya merujuk ke Hugging Face Hub tempat kamu download model open-source seperti
Whisper/Wav2Vec2. Gemini itu model tertutup punya Google, aksesnya lewat API key aja):

1. Buka [aistudio.google.com/apikey](https://aistudio.google.com/apikey), login pakai akun Google.
2. Klik **Create API key**, pilih/buat sebuah Google Cloud project.
3. Supaya bisa pakai model berbayar (bukan cuma kuota gratis harian yang kecil),
   aktifkan **billing** di project itu lewat [Google Cloud Console](https://console.cloud.google.com/billing) —
   isi kartu, top up sesuai kebutuhan (mis. $1). Setelah billing aktif, API key otomatis
   kepake tarif pay-as-you-go Gemini 2.5 Flash.
4. Copy API key-nya, lalu:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml, isi GEMINI_API_KEY
```
Atau pakai environment variable `GEMINI_API_KEY` kalau lebih suka.

> Estimasi biaya: 1 sesi kuliah ~1-2 jam biasanya transkrip + materi cuma sekitar
> 5.000-15.000 token, jadi 1x generate rangkuman jauh di bawah $0.01. Uang $1 kamu
> cukup buat ratusan kali generate.

### 4. Jalankan
```bash
streamlit run app.py
```

## Struktur folder

```
app.py                      # UI Streamlit (sidebar sesi + 3 tab)
src/
  config.py                 # semua konstanta & pilihan model/preprocessing di satu tempat
  audio_preprocessing.py    # denoise & normalize (dari 03_preprocessing_exploration.ipynb)
  asr_engine.py              # load model + StreamingTranscriber (sliding window chunking)
  keywords.py                 # loader kosakata teknis per matkul -> hotwords faster-whisper
  data/keywords_hotwords_matkul.json  # kosakata per matkul (kode MK -> daftar istilah)
  realtime.py                # konversi av.AudioFrame (webrtc) -> numpy buat Whisper
  session_store.py            # simpan/baca sesi (JSON per sesi)
  materials.py                # ekstrak teks dari PPTX/PDF
  summarizer.py                # prompt building + panggil Groq buat rangkuman
data/sessions/{session_id}/    # transcript.txt, summary.md, materials/, meta.json — dibuat otomatis
```

## Vocabulary boosting (hotwords) per mata kuliah

Sesi baru sekarang bisa dipilih dari daftar matkul di `src/data/keywords_hotwords_matkul.json`
(bukan cuma ketik manual). Kalau dipilih dari daftar, `mk_kode` disimpan di `meta.json`
sesi itu, dan saat tab Live Caption dibuka, `src/keywords.get_hotwords(mk_kode)` narik
daftar istilah teknis matkul itu dan dipasang sebagai `hotwords` faster-whisper —
selain `initial_prompt` (context overlap antar-chunk) yang sudah ada sebelumnya.
Status hotwords aktif (jumlah istilah + kode matkul) tampil di caption bawah judul
tab Live Caption, biar kelihatan pas demo apakah hotwords beneran kepasang.

Kalau matkul belum ada di daftar JSON, sesi tetap bisa dibuat lewat input manual —
cuma jalan tanpa hotwords (perilaku sama seperti sebelumnya).

**Catatan token budget**: faster-whisper motong gabungan `hotwords` + `initial_prompt`
ke sekitar setengah `max_length` model (~223 token buat Whisper standar). Beberapa
matkul (mis. Arsitektur dan Organisasi Komputer) punya 60+ istilah di JSON-nya —
`keywords.MAX_HOTWORDS_TERMS` (default 40) motong dari depan list biar predictable,
bukan kepotong diam-diam oleh faster-whisper. Naikin/turunin nilainya di
`src/keywords.py` kalau context overlap antar-chunk kelihatan mulai kepotong.

## Keterbatasan yang perlu kamu tahu (buat demo/laporan ke dosen)

- **UI freeze saat merekam**: pola `streamlit-webrtc` yang dipakai di sini
  (loop `while webrtc_ctx.state.playing`) memblokir eksekusi script Streamlit
  selama recording jalan — jadi jangan pindah tab dulu sampai klik STOP. ini
  batasan arsitektural Streamlit + WebRTC, bukan bug; kalau mau UX yang lebih
  mulus, tahap berikutnya bisa pisah ke backend FastAPI + WebSocket kayak draft
  riset original.
- **STUN vs TURN**: konfigurasi WebRTC di sini cuma pakai STUN server publik
  (Google). Ini cukup buat testing lokal atau jaringan kampus biasa. Kalau nanti
  di-deploy ke cloud (mis. Streamlit Community Cloud) dan mic gak connect, kamu
  butuh TURN server (opsi gratis: Open Relay Project; opsi stabil: Twilio).
- **RTF real-time**: beam_size & preprocessing di-set buat prioritas latensi
  (`WHISPER_BEAM_SIZE_STREAMING = 2`, denoising non-stationary aktif). Kalau di
  hardware kamu ternyata RTF > 1 (transkripsi lebih lambat dari audio real-time),
  matikan `PREPROCESS_ENABLED` dulu atau turunin `CHUNK_SECONDS` — nilai real-time
  factor chunk terakhir ditampilkan langsung di tab Live Caption buat kamu pantau.
- **Belum ada speaker diarization / evaluasi WER otomatis** di jalur real-time
  ini — sesuai scope draft riset kamu, itu emang out of scope untuk versi ini.

## Yang bisa langsung kamu utak-atik

Semua keputusan model/preprocessing/LLM ada di satu file: `src/config.py`. Ganti
`WHISPER_MODEL_SIZE`, `PREPROCESS_ENABLED`, `VAD_PARAMS`, `CHUNK_SECONDS`, atau
`GROQ_MODEL` di situ — gak perlu sentuh file lain.
