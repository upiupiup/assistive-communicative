"""
Konfigurasi terpusat.

Nilai default di sini diturunkan langsung dari hasil eksplorasi kamu:
- Model: 02-baseline-presentation-kaggle.ipynb -> whisper medium paling stabil
  (WER & TTER terendah dibanding large-v3-turbo, Wav2Vec2 XLSR, Qwen2-Audio).
- Preprocessing & VAD: 03_preprocessing_exploration.ipynb -> full_preprocess
  (non-stationary denoise + amplitude normalize) + vad_default.
  Kalau kesimpulan notebook kamu beda (best combo lain), tinggal ubah
  PREPROCESS_ENABLED / VAD_PARAMS di bawah, gak perlu ubah kode lain.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 4
MIN_FLUSH_SECONDS = 0.6

WHISPER_MODEL_SIZE = "medium"
WHISPER_DEVICE = "auto"
WHISPER_COMPUTE_TYPE = "auto"
WHISPER_LANGUAGE = "id"
WHISPER_BEAM_SIZE_STREAMING = 2

VAD_PARAMS = dict(min_silence_duration_ms=300)
NO_SPEECH_THRESHOLD = 0.45
COMPRESSION_RATIO_THRESHOLD = 2.0

PREPROCESS_ENABLED = True
DENOISE_PROP_DECREASE = 0.75
TARGET_DB = -20.0

TECH_TERMS = [
    "algoritma",
    "fungsi",
    "variabel",
    "array",
    "rekursi",
    "iterasi",
    "stack",
    "queue",
    "tree",
    "graph",
    "node",
    "edge",
    "branch",
    "class",
    "abstract",
    "constructor",
    "method",
    "attribute",
    "interface",
    "inheritance",
    "override",
    "polymorphism",
    "encapsulation",
    "instansiasi",
    "object",
    "diagram",
    "backtracking",
    "coloring",
    "pewarnaan",
    "simpul",
    "depth first search",
    "dfs",
    "eliminasi",
    "preprocessing",
    "classification",
    "segmentasi",
    "ekstraksi",
    "dataset",
    "convolutional",
    "cnn",
    "feature extraction",
    "pipeline",
    "implementasi",
    "sistem",
    "parameter",
]


GEMINI_MODEL = "gemini-2.5-flash"
CAPTION_FONT_SIZE_PX = 40
CAPTION_BG_COLOR = "#0B0F14"
CAPTION_TEXT_COLOR = "#FFFFFF"
