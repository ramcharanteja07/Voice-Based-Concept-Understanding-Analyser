import streamlit as st

from audio_utils import save_temp_audio, load_audio, normalize_audio, plot_waveform, compute_pause_ratio, compute_rms_energy
from speech_to_text import transcribe_with_confidence
from semantic_eval import compute_similarity
from scoring_engine import compute_concept_score, compute_filler_counts, compute_filler_ratio, compute_fluency_score, compute_audio_confidence_score, compute_final_score, classify_understanding
from report_generator import build_report

REFERENCE_TEXT = "Machine Learning is a subset of artificial intelligence that allows systems to learn patterns from data and improve performance without being explicitly programmed."
KEYWORDS = ["machine learning", "artificial intelligence", "data", "patterns", "learn"]
FILLER_WORDS = ["um", "uh", "like", "you know", "so", "actually"]

if "run_analysis" not in st.session_state:
    st.session_state["run_analysis"] = False

st.set_page_config(page_title="Concept Analyser", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg-0: #0a0e14;
    --bg-1: #0d1219;
    --card: #11161f;
    --border: #1e2732;
    --border-soft: #171e28;
    --text-hi: #e9eef3;
    --text-lo: #7c8896;
    --signal: #5eead4;
    --signal-dim: rgba(94, 234, 212, 0.12);
}

html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace; }

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(94,234,212,0.05), transparent 40%),
        radial-gradient(circle at 85% 100%, rgba(240,168,104,0.04), transparent 40%),
        linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
}

#MainMenu, header, footer { visibility: hidden; }

.hero {
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid var(--border-soft);
    margin-bottom: 2rem;
}
.hero-eyebrow {
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    color: var(--signal);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    color: var(--text-hi);
    letter-spacing: -0.01em;
    margin: 0;
    line-height: 1.15;
}
.hero-title span { color: var(--signal); }
.hero-sub {
    color: var(--text-lo);
    font-size: 0.95rem;
    margin-top: 0.75rem;
    max-width: 640px;
}

.waveform { display: flex; align-items: flex-end; gap: 3px; height: 34px; margin-top: 1.3rem; }
.waveform span {
    display: block;
    width: 3px;
    background: var(--signal);
    border-radius: 2px;
    opacity: 0.85;
    animation: pulse 1.6s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scaleY(0.35); opacity: 0.45; }
    50% { transform: scaleY(1); opacity: 0.9; }
}

.st-key-upload_panel, .st-key-reference_panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.6rem 1.7rem 1.8rem 1.7rem;
}

.panel-label {
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-lo);
    margin-bottom: 0.25rem;
}
.panel-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1.2rem;
    color: var(--text-hi);
    margin-bottom: 1rem;
}

.stFileUploader > div > div {
    background: var(--bg-1) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 8px !important;
}
.stFileUploader label { color: var(--text-lo) !important; }
.stFileUploader small { color: var(--text-lo) !important; }

.concept-block {
    background: var(--bg-1);
    border-left: 3px solid var(--signal);
    border-radius: 6px;
    padding: 1rem 1.15rem;
    color: var(--text-hi);
    font-size: 0.92rem;
    line-height: 1.6;
}

.status-bar {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    background: var(--signal-dim);
    border: 1px solid rgba(94,234,212,0.28);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    color: var(--text-hi);
    font-size: 0.88rem;
    margin-top: 2rem;
}
.status-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--signal);
    box-shadow: 0 0 10px var(--signal);
    flex-shrink: 0;
}
.status-bar b { color: var(--signal); }

[data-testid="stCaptionContainer"] { color: var(--text-lo) !important; }
</style>
""", unsafe_allow_html=True)

bar_heights = [14, 22, 30, 18, 34, 12, 26, 20, 32, 16, 24, 10, 28, 18, 22, 14, 30, 20, 12, 26, 34, 16, 22, 10, 28, 18]
bars_html = "".join(f'<span style="height:{h}px; animation-delay:{i*0.06}s;"></span>' for i, h in enumerate(bar_heights))

st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">SIGNAL · SPEECH · UNDERSTANDING</div>
    <div class="hero-title">Voice-Based <span>Concept</span> Understanding Analyser</div>
    <div class="hero-sub">
        Upload a spoken explanation and the reference concept is compared against it —
        automated evaluation of conceptual understanding from raw audio.
    </div>
    <div class="waveform">{bars_html}</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    with st.container(key="upload_panel"):
        st.markdown('<div class="panel-label">Input · 01</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📤 Upload Student Audio</div>', unsafe_allow_html=True)
        audio_file = st.file_uploader("Upload audio file", type=['wav', 'mp3'], label_visibility="collapsed")
        st.caption("Supports WAV, MP3 · Max size 200MB")
        if audio_file is not None:
            if audio_file.type not in ["audio/wav", "audio/mpeg"]:
                st.error("❌ Unsupported file format. Please upload WAV or MP3.")
                st.stop()
                st.audio(audio_file, format="audio/wav")
        if st.button("Run Analysis"):
            st.session_state["run_analysis"] = True

with col2:
    with st.container(key="reference_panel"):
        st.markdown('<div class="panel-label">Reference · 02</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🧠 Concept Reference</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="concept-block">Machine Learning is a subset of artificial intelligence '
            'that allows systems to learn patterns from data and improve performance without '
            'being explicitly programmed.</div>',
            unsafe_allow_html=True
        )

if audio_file is None:

    st.markdown("""
    <div class="status-bar">
        <div class="status-dot"></div>
        <div><b>Ready ·</b> upload an audio file to begin the analysis.</div>
    </div>
    """, unsafe_allow_html=True)

elif not st.session_state["run_analysis"]:
    st.markdown("""
    <div class="status-bar">
        <div class="status-dot"></div>
        <div><b>Loaded ·</b> click 'Run Analysis' to process the audio.</div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="status-bar">
        <div class="status-dot"></div>
        <div><b>Processing ·</b> analyzing speech, extracting features...</div>
    </div>
    """, unsafe_allow_html=True)


if audio_file is not None and st.session_state["run_analysis"]:

    st.subheader("Transcription")

    audio_bytes = audio_file.read()
    audio_path = save_temp_audio(audio_bytes)

    y, sr = load_audio(audio_path)
    y = normalize_audio(y)
    fig = plot_waveform(y, sr)
    st.pyplot(fig)

    try:
        text, transcription_confidence = transcribe_with_confidence(audio_path)
    except Exception as e:
        st.error("❌ Error during transcription. Try another audio file.")
        st.stop()

    concept_score = compute_concept_score(text, KEYWORDS)
    filler_count = compute_filler_counts(text, FILLER_WORDS)
    filler_ratio = compute_filler_ratio(text, filler_count)
    fluency_score = compute_fluency_score(filler_ratio)
    similarity = compute_similarity(REFERENCE_TEXT, text)
    pause_ratio = compute_pause_ratio(y)
    rms_energy = compute_rms_energy(y)
    audio_confidence_score = compute_audio_confidence_score(rms_energy, pause_ratio)
    final_score = compute_final_score(similarity, fluency_score, audio_confidence_score)
    understanding = classify_understanding(final_score)

    st.session_state["transcript"] = text
    st.session_state["filler_count"] = filler_count
    st.session_state["fluency_score"] = fluency_score
    st.session_state["concept_score"] = concept_score

    st.markdown("### 📝 Transcribed Explanation")

    st.markdown(f"""
    <div class="concept-block">
    {text}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("## 📊 Analysis Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🎧 Audio Confidence Score", f"{audio_confidence_score} / 10")

    with col2:
        st.metric("🗣️ Fluency Score", f"{fluency_score} / 10")

    with col3:
        st.metric("🔁 Filler Words", sum(filler_count.values()))

    st.markdown("### 🔍 Filler Word Breakdown")
    st.write(filler_count)

    if st.button("Download Report"):
        waveform_path = "waveform.png"
        plot_waveform(y, sr, save_path=waveform_path)

        metrics = {
            "Semantic Similarity": similarity,
            "Filler Word Ratio": filler_ratio,
            "Pause Ratio": pause_ratio,
            "RMS Energy": rms_energy,
            "Audio Confidence Score": audio_confidence_score,
            "Final Score": f"{final_score}/100",
            "Understanding Level": understanding,
        }

        report_path = build_report(REFERENCE_TEXT, text, waveform_path, metrics)

        with open(report_path, "rb") as f:
            st.download_button("📄 Download PDF", f, file_name="report.pdf")