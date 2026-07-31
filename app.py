"""
Assistive Lecture Captioning — prototype end-to-end pipeline.

Alur: rekam suara dosen real-time -> caption langsung tampil & tersimpan ->
tambahkan materi (PPT/PDF) -> generate rangkuman kuliah rapi pakai LLM.

Jalankan:
    streamlit run app.py
"""

import queue
from datetime import date

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src import asr_engine, config, keywords, materials, realtime, session_store, summarizer

st.set_page_config(
    page_title="Assistive Lecture Captioning", page_icon="🎓", layout="wide"
)

CAPTION_CSS = f"""
<style>
.live-caption {{
    background-color: {config.CAPTION_BG_COLOR};
    color: {config.CAPTION_TEXT_COLOR};
    font-size: {config.CAPTION_FONT_SIZE_PX}px;
    line-height: 1.45;
    padding: 24px 28px;
    border-radius: 12px;
    min-height: 160px;
    max-height: 480px;
    overflow-y: auto;
    font-weight: 500;
}}
</style>
"""
st.markdown(CAPTION_CSS, unsafe_allow_html=True)


def _trim_tail(text: str, max_chars: int = 700) -> str:
    """Ambil `max_chars` karakter terakhir tanpa motong di tengah kata.

    Dipakai cuma buat tampilan live caption pas LAGI merekam (biar caption
    box gak usah discroll pas kuliah masih jalan — selalu nunjukin bagian
    paling baru). Transkrip lengkapnya tetep aman tersimpan utuh di disk;
    ini cuma soal apa yang ditampilkan di caption box saat itu.
    """
    if len(text) <= max_chars:
        return text
    trimmed = text[-max_chars:]
    if " " in trimmed:
        trimmed = trimmed.split(" ", 1)[1]
    return trimmed.strip()

# ── Sidebar: manajemen sesi ────────────────────────────────────────────
st.sidebar.title("🎓 Sesi Kuliah")

sessions = session_store.list_sessions()

with st.sidebar.expander("+ Buat sesi baru", expanded=not sessions):
    with st.form("new_session_form", clear_on_submit=True):
        matkul_list = keywords.list_matkul()
        no_pilihan = "(belum pilih dari daftar)"
        pilihan_labels = [no_pilihan] + [
            f"{m['nama_mk']} ({m['kode']})" for m in matkul_list
        ]
        pilihan = st.selectbox(
            "Mata Kuliah — pilih dari daftar biar hotwords ASR auto-keisi",
            pilihan_labels,
        )
        mata_kuliah_manual = st.text_input(
            "Atau ketik manual (kalau matkul belum ada di daftar / mau ganti nama)"
        )
        dosen = st.text_input("Dosen")
        topik = st.text_input("Topik Pertemuan")
        tanggal = st.date_input("Tanggal", value=date.today())
        submitted = st.form_submit_button("Buat Sesi")
        if submitted:
            if pilihan != no_pilihan:
                dipilih = matkul_list[pilihan_labels.index(pilihan) - 1]
                mk_kode = dipilih["kode"]
                mata_kuliah = mata_kuliah_manual.strip() or dipilih["nama_mk"]
            else:
                mk_kode = ""
                mata_kuliah = mata_kuliah_manual.strip()

            if not mata_kuliah:
                st.warning(
                    "Nama mata kuliah wajib diisi (pilih dari daftar atau ketik manual)."
                )
            else:
                new_id = session_store.create_session(
                    mata_kuliah, dosen.strip(), topik.strip(), str(tanggal), mk_kode=mk_kode
                )
                st.session_state["current_session_id"] = new_id
                st.rerun()

if sessions:
    options = {
        f"{m['tanggal']} · {m['mata_kuliah']} · {m['topik'] or '-'}": m["session_id"]
        for m in sessions
    }
    labels = list(options.keys())
    current_id = st.session_state.get("current_session_id")
    current_label = next(
        (lbl for lbl, sid in options.items() if sid == current_id), labels[0]
    )
    chosen_label = st.sidebar.selectbox(
        "Sesi aktif", labels, index=labels.index(current_label)
    )
    st.session_state["current_session_id"] = options[chosen_label]

    if st.sidebar.button("🗑️ Hapus sesi ini", use_container_width=True):
        session_store.delete_session(st.session_state["current_session_id"])
        st.session_state["current_session_id"] = None
        st.rerun()
else:
    st.sidebar.info("Belum ada sesi. Buat sesi baru dulu di atas.")

with st.sidebar.expander("⚙️ Setup LLM rangkuman"):
    st.caption(
        "Rangkuman pakai Groq (free tier). Daftar API key gratis di "
        "[console.groq.com](https://console.groq.com), lalu isi `GROQ_API_KEY` di "
        "`.streamlit/secrets.toml` atau environment variable."
    )

session_id = st.session_state.get("current_session_id")
if not session_id:
    st.title("🎓 Assistive Lecture Captioning")
    st.info("👈 Buat atau pilih sesi kuliah di sidebar untuk mulai.")
    st.stop()

meta = session_store.load_meta(session_id)

# ── Header ──────────────────────────────────────────────────────────
st.title(f"🎓 {meta['mata_kuliah']}")
st.caption(
    f"{meta['topik'] or 'Tanpa topik'} · {meta['dosen'] or '-'} · {meta['tanggal']}"
)

tab_live, tab_materi, tab_transkrip = st.tabs(
    ["🎙️ Live Caption", "📎 Materi", "📝 Transkrip & Rangkuman"]
)

# ── Tab 1: Live Caption ────────────────────────────────────────────
with tab_live:
    model = asr_engine.load_whisper_model()

    if st.session_state.get("transcriber_session") != session_id:
        hotwords_str = keywords.get_hotwords(meta.get("mk_kode", ""))
        st.session_state["transcriber"] = asr_engine.StreamingTranscriber(
            model=model, hotwords=hotwords_str
        )
        st.session_state["transcriber_session"] = session_id

    active_terms = keywords.get_terms(meta.get("mk_kode", ""))
    hotwords_status = (
        f"hotwords: {len(active_terms)} istilah ({meta.get('mk_kode')})"
        if active_terms
        else "hotwords: tidak ada (matkul ini belum ada di daftar keyword)"
    )
    st.caption(
        f"Model: faster-whisper {config.WHISPER_MODEL_SIZE} · "
        f"chunk {config.CHUNK_SECONDS}s · preprocessing "
        f"{'ON' if config.PREPROCESS_ENABLED else 'OFF'} · {hotwords_status}"
    )

    reset_confirm_key = f"confirm_reset_{session_id}"
    if st.session_state.get(reset_confirm_key):
        st.warning("Yakin mau hapus semua transkrip sesi ini? Gak bisa dibatalin.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Ya, hapus", key=f"reset-yes-{session_id}", use_container_width=True):
                session_store.overwrite_transcript(session_id, "")
                transcriber_to_clear = st.session_state.get("transcriber")
                if transcriber_to_clear is not None:
                    transcriber_to_clear.reset()
                st.session_state[reset_confirm_key] = False
                st.rerun()
        with c2:
            if st.button("Batal", key=f"reset-no-{session_id}", use_container_width=True):
                st.session_state[reset_confirm_key] = False
                st.rerun()
    else:
        if st.button("🔄 Mulai dari awal (hapus transkrip sesi ini)", key=f"reset-{session_id}"):
            st.session_state[reset_confirm_key] = True
            st.rerun()

    webrtc_ctx = webrtc_streamer(
        key=f"live-caption-{session_id}",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    status_placeholder = st.empty()
    caption_placeholder = st.empty()
    rtf_placeholder = st.empty()

    saved_transcript = session_store.get_transcript(session_id)

    if webrtc_ctx.state.playing:
        status_placeholder.success(
            "🔴 Merekam & mentranskrip — jangan pindah tab dulu sampai berhenti."
        )
        transcriber: asr_engine.StreamingTranscriber = st.session_state["transcriber"]

        while webrtc_ctx.state.playing:
            if webrtc_ctx.audio_receiver is None:
                break
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                continue

            samples = realtime.frames_to_float32(audio_frames)
            if len(samples) == 0:
                continue

            new_text = transcriber.push_samples(samples)
            if new_text:
                session_store.append_transcript(session_id, new_text)

            live_full = (
                saved_transcript + " " + transcriber.get_full_transcript()
            ).strip()
            # Live: tampilin tail-nya aja (gak usah scroll pas lagi jalan, biar
            # yang paling baru selalu kelihatan). Transkrip penuhnya tetep utuh
            # tersimpan di disk & bisa direview lengkap begitu direkam dihentikan.
            caption_placeholder.markdown(
                f'<div class="live-caption">{_trim_tail(live_full) or "…mendengarkan…"}</div>',
                unsafe_allow_html=True,
            )
            rtf_placeholder.caption(
                f"Real-time factor chunk terakhir: {transcriber.last_rtf:.2f}x"
            )
    else:
        transcriber = st.session_state.get("transcriber")
        if transcriber is not None:
            tail = transcriber.flush()
            if tail:
                session_store.append_transcript(session_id, tail)
                transcriber.reset()

        status_placeholder.caption("⏸️ Klik START untuk mulai merekam sesi kuliah ini.")
        full = session_store.get_transcript(session_id)
        # Udah berhenti -> gak ada alasan buat motong. Tampilin transkrip
        # lengkap, box-nya scrollable (lihat CAPTION_CSS) biar bisa direview.
        caption_placeholder.markdown(
            f'<div class="live-caption">{full or "Belum ada transkrip."}</div>',
            unsafe_allow_html=True,
        )

# ── Tab 2: Materi ───────────────────────────────────────────────────
with tab_materi:
    st.subheader("Materi Pendukung Sesi")
    st.caption(
        "Upload slide (PPTX) atau catatan (PDF) — teksnya dipakai sebagai konteks tambahan buat rangkuman."
    )

    uploaded_files = st.file_uploader(
        "Upload materi",
        type=["pptx", "pdf"],
        accept_multiple_files=True,
        key=f"uploader-{session_id}",
    )
    if uploaded_files:
        processed_key = f"processed-material-uploads-{session_id}"
        processed_uploads = st.session_state.setdefault(processed_key, set())
        new_files = []

        for f in uploaded_files:
            upload_id = (f.name, f.size, f.type)
            if upload_id not in processed_uploads:
                session_store.save_material(session_id, f)
                processed_uploads.add(upload_id)
                new_files.append(f)

        if new_files:
            st.session_state[processed_key] = processed_uploads
            st.success(f"{len(new_files)} file tersimpan.")
            st.rerun()

    existing_materials = session_store.list_materials(session_id)
    if existing_materials:
        st.write(f"**{len(existing_materials)} materi tersimpan:**")
        for p in existing_materials:
            with st.expander(p.name):
                extracted = materials.extract_text(p)
                if extracted:
                    preview = extracted[:2000] + (
                        "..." if len(extracted) > 2000 else ""
                    )
                    st.text(preview)
                else:
                    st.caption("(tidak ada teks yang bisa diekstrak dari file ini)")
    else:
        st.caption("Belum ada materi diupload untuk sesi ini.")

# ── Tab 3: Transkrip & Rangkuman ────────────────────────────────────
with tab_transkrip:
    st.subheader("Transkrip Sesi")
    current_transcript = session_store.get_transcript(session_id)
    edited_transcript = st.text_area(
        "Transkrip (bisa diedit manual buat koreksi typo hasil ASR)",
        value=current_transcript,
        height=260,
        key=f"transcript-edit-{session_id}",
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Simpan perubahan transkrip", use_container_width=True):
            session_store.overwrite_transcript(session_id, edited_transcript)
            st.success("Transkrip tersimpan.")
    with col2:
        st.download_button(
            "⬇️ Download transkrip (.txt)",
            data=current_transcript,
            file_name=f"transkrip_{session_id}.txt",
            use_container_width=True,
        )

    st.divider()
    st.subheader("Rangkuman Otomatis (LLM)")

    if st.button("✨ Generate Rangkuman", type="primary"):
        materials_text = materials.extract_all(session_store.list_materials(session_id))
        with st.spinner("Menghasilkan rangkuman..."):
            summary_md = summarizer.generate_summary(
                session_store.get_transcript(session_id), materials_text, meta
            )
        session_store.save_summary(session_id, summary_md)
        st.rerun()

    existing_summary = session_store.get_summary(session_id)
    if existing_summary:
        st.markdown(existing_summary)
        st.download_button(
            "⬇️ Download rangkuman (.md)",
            data=existing_summary,
            file_name=f"rangkuman_{session_id}.md",
        )
    else:
        st.caption("Belum ada rangkuman. Isi transkrip dulu, lalu klik tombol di atas.")
