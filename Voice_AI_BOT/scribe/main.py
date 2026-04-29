"""
AI Medical Scribe v3 — EHR-Ready
=================================
Tab 1: Upload audio → Transcribe → Clinical Notes
Tab 2: Live AI Doctor consultation → Clinical Notes
Tab 3: Consultation History
"""
import os, sys, glob, time, json, re, asyncio, tempfile
import streamlit as st
from pathlib import Path
from datetime import datetime

# Load .env locally (optional — Streamlit Cloud uses st.secrets)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # dotenv not needed on Streamlit Cloud

from clinical_engine import (generate_clinical_output, save_consultation,
    load_consultation_history, severity_badge, triage_badge,
    check_guardrails, GUARDRAIL_DISCLAIMER)

SCRIBE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIBE_DIR.parent

def _cfg(key, default=""):
    """Get config from env vars or Streamlit secrets."""
    val = os.getenv(key, "")
    if not val:
        try: val = st.secrets.get(key, default)
        except: val = default
    return val or default

GROQ_API_KEY = _cfg("GROQ_API_KEY")
GROQ_MODEL = _cfg("GROQ_MODEL", "llama-3.3-70b-versatile")
WHISPER_MODEL_SIZE = _cfg("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = _cfg("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = _cfg("WHISPER_COMPUTE_TYPE", "int8")
# ── Language Configuration ──
LANG_CONFIG = {
    "English": {
        "code": "en",
        "tts_voice": os.getenv("EDGE_TTS_ENGLISH_VOICE", "en-US-JennyNeural"),
        "flag": "🇺🇸",
        "doctor_lang_instruction": "Respond ONLY in English.",
    },
    "Telugu (తెలుగు)": {
        "code": "te",
        "tts_voice": os.getenv("EDGE_TTS_TELUGU_VOICE", "te-IN-ShrutiNeural"),
        "flag": "🇮🇳",
        "doctor_lang_instruction": "Respond ONLY in Telugu (తెలుగు). Use Telugu script. Patient may speak in Telugu or English — always reply in Telugu.",
    },
    "Hindi (हिन्दी)": {
        "code": "hi",
        "tts_voice": "hi-IN-SwaraNeural",
        "flag": "🇮🇳",
        "doctor_lang_instruction": "Respond ONLY in Hindi (हिन्दी). Use Devanagari script. Patient may speak in Hindi or English — always reply in Hindi.",
    },
}

st.set_page_config(page_title="AI Medical Scribe", page_icon="🩺", layout="wide", initial_sidebar_state="collapsed")

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.stApp { font-family:'Inter',sans-serif; background:#ffffff!important; color:#1e293b!important; }
.stApp>header { background:#ffffff!important; }
[data-testid="stSidebar"] { background:#f8fafc!important; }
.stMarkdown,.stMarkdown p,.stMarkdown li,.stMarkdown span,[data-testid="stText"],h1,h2,h3,h4,h5,h6 { color:#1e293b!important; }
[data-testid="stStatusWidget"] { background:#f1f5f9!important; }
hr { border-color:#e2e8f0!important; }
.scribe-header { background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#0f766e 100%); padding:2rem 2.5rem; border-radius:16px; margin-bottom:2rem; position:relative; overflow:hidden; }
.scribe-header::before { content:''; position:absolute; top:-50%; right:-20%; width:400px; height:400px; background:radial-gradient(circle,rgba(14,165,150,0.15) 0%,transparent 70%); border-radius:50%; }
.soap-card { background:#ffffff; border-radius:12px; padding:1.5rem; margin-bottom:1rem; border-left:5px solid; box-shadow:0 1px 3px rgba(0,0,0,0.08),0 4px 12px rgba(0,0,0,0.04); transition:transform 0.2s,box-shadow 0.2s; }
.soap-card:hover { transform:translateY(-2px); box-shadow:0 4px 12px rgba(0,0,0,0.12); }
.soap-s { border-left-color:#3b82f6; } .soap-o { border-left-color:#10b981; }
.soap-a { border-left-color:#f59e0b; } .soap-p { border-left-color:#8b5cf6; }
.soap-card h3 { font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:0.75rem; }
.soap-s h3{color:#3b82f6;} .soap-o h3{color:#10b981;} .soap-a h3{color:#f59e0b;} .soap-p h3{color:#8b5cf6;}
.soap-card .content { font-size:0.95rem; line-height:1.7; color:#334155; }
.badge { display:inline-block; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-doctor { background:#dbeafe; color:#1d4ed8; } .badge-patient { background:#d1fae5; color:#065f46; }
.transcript-line { padding:0.6rem 1rem; margin-bottom:0.4rem; border-radius:8px; font-size:0.9rem; line-height:1.6; color:#1e293b; }
.transcript-doctor { background:#eff6ff; border-left:3px solid #3b82f6; }
.transcript-patient { background:#ecfdf5; border-left:3px solid #10b981; }
.metrics-row { display:flex; gap:1rem; margin:1rem 0; }
.metric-card { flex:1; background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:1rem 1.25rem; text-align:center; }
.metric-card .value { font-size:1.5rem; font-weight:700; color:#0f172a; }
.metric-card .label { font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; margin-top:0.25rem; }
.chat-doctor { background:#eff6ff; border-left:4px solid #3b82f6; padding:1rem 1.25rem; border-radius:0 12px 12px 0; margin:0.5rem 0; }
.chat-patient { background:#ecfdf5; border-left:4px solid #10b981; padding:1rem 1.25rem; border-radius:0 12px 12px 0; margin:0.5rem 0; }
.chat-label { font-size:0.75rem; font-weight:600; margin-bottom:0.3rem; }
.chat-text { font-size:0.95rem; line-height:1.6; color:#1e293b; }
.stDownloadButton>button { background:linear-gradient(135deg,#0f766e,#1e3a5f)!important; color:#fff!important; border:none!important; border-radius:10px!important; font-weight:600!important; }
.stDownloadButton>button:hover { opacity:0.9!important; }
#MainMenu{visibility:hidden;} footer{visibility:hidden;} header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    from faster_whisper import WhisperModel
    return WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

def get_selected_lang():
    """Get the currently selected language config."""
    lang_name = st.session_state.get("selected_language", "English")
    return LANG_CONFIG.get(lang_name, LANG_CONFIG["English"])

def transcribe_audio_bytes(audio_bytes, lang_code=None):
    """Transcribe audio bytes using faster-whisper."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=str(SCRIBE_DIR))
    tmp.write(audio_bytes)
    tmp.close()
    model = load_whisper_model()
    lc = lang_code or get_selected_lang()["code"]
    segments, info = model.transcribe(tmp.name, beam_size=5, language=lc, vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments)
    os.unlink(tmp.name)
    return text

def transcribe_audio_file(audio_path):
    """Transcribe audio file with speaker diarization."""
    model = load_whisper_model()
    lc = get_selected_lang()["code"]
    segs, info = model.transcribe(audio_path, beam_size=5, language=lc, vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=600, speech_pad_ms=200))
    all_segs = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segs]
    labeled = []
    speaker = "Doctor"
    for i, seg in enumerate(all_segs):
        if i > 0 and seg["start"] - all_segs[i-1]["end"] >= 1.5:
            speaker = "Patient" if speaker == "Doctor" else "Doctor"
        labeled.append({**seg, "speaker": speaker})
    full_text = "\n".join(f"{s['speaker']}: {s['text']}" for s in labeled)
    return {"segments": labeled, "full_text": full_text, "duration": info.duration, "language": info.language}

def render_clinical_output(data):
    """Render full clinical output with severity, SOAP, and structured data."""
    # ── Disclaimer ──
    st.markdown(f'<div style="background:#fef3c7;border:1px solid #f59e0b;border-radius:10px;padding:0.75rem 1rem;margin-bottom:1rem;font-size:0.85rem;color:#92400e;">{GUARDRAIL_DISCLAIMER}</div>', unsafe_allow_html=True)

    # ── Severity + Triage Row ──
    sev = data.get("severity", "unknown")
    tri = data.get("triage", "unknown")
    cc = data.get("chief_complaint", "N/A")
    st.markdown(f"""<div class="metrics-row">
        <div class="metric-card"><div class="value">{severity_badge(sev)}</div><div class="label">Severity</div></div>
        <div class="metric-card"><div class="value">{triage_badge(tri)}</div><div class="label">Triage</div></div>
        <div class="metric-card"><div class="value" style="font-size:1rem;">{cc}</div><div class="label">Chief Complaint</div></div>
    </div>""", unsafe_allow_html=True)

    # ── Dual View Tabs ──
    soap_tab, struct_tab = st.tabs(["📋 SOAP Notes", "🔬 Structured Data (EHR)"])

    with soap_tab:
        sections = [("Subjective","subjective","soap-s","💬"), ("Objective","objective","soap-o","🔬"),
                    ("Assessment","assessment","soap-a","🧠"), ("Plan","plan","soap-p","📋")]
        for title, key, css, icon in sections:
            content = data.get(key, "N/A")
            if not isinstance(content, str): content = str(content)
            conf = data.get("confidence", {}).get(key, "")
            conf_badge = f' <span style="background:#e0f2fe;color:#0369a1;padding:0.1rem 0.5rem;border-radius:10px;font-size:0.7rem;">🎯 {conf}</span>' if conf else ""
            content_html = content.replace("\n", "<br>")
            st.markdown(f'<div class="soap-card {css}"><h3>{icon} {title}{conf_badge}</h3><div class="content">{content_html}</div></div>', unsafe_allow_html=True)

    with struct_tab:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🩺 Symptoms Present**")
            for s in data.get("symptoms_present", []):
                st.markdown(f"- ✅ {s}")
            st.markdown("**❌ Symptoms Denied**")
            for s in data.get("symptoms_denied", []):
                st.markdown(f"- ❌ {s}")
            st.markdown("**💊 Medications Taken**")
            for m in data.get("medications_taken", []):
                st.markdown(f"- 💊 {m}")
        with col2:
            st.markdown("**🔍 Differential Diagnosis**")
            for d in data.get("differential_diagnosis", []):
                st.markdown(f"- 🔹 {d}")
            st.markdown("**🚩 Red Flags Screened**")
            for r in data.get("red_flags_screened", []):
                st.markdown(f"- 🚩 {r}")
            fu = data.get("follow_up", "Not specified")
            st.markdown(f"**📅 Follow-up:** {fu}")
        st.divider()
        st.markdown("**📦 Raw EHR-Ready JSON**")
        st.json(data)



def tts_speak(text, voice=None):
    """Convert text to speech using Edge TTS, return audio file path."""
    import edge_tts
    if voice is None:
        voice = get_selected_lang()["tts_voice"]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=str(SCRIBE_DIR))
    tmp.close()
    async def _gen():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(tmp.name)
    asyncio.run(_gen())
    return tmp.name

def render_header():
    st.markdown("""
    <div class="scribe-header">
        <div style="color:#ffffff;font-size:2.2rem;font-weight:700;margin:0 0 0.5rem 0;letter-spacing:-0.5px;text-shadow:0 2px 4px rgba(0,0,0,0.3);">🩺 AI Medical Scribe</div>
        <div style="color:rgba(255,255,255,0.9);font-size:1.05rem;margin:0;font-weight:300;">Automated SOAP notes from doctor-patient conversations</div>
    </div>""", unsafe_allow_html=True)


def fmt_time(sec): m,s=divmod(int(sec),60); return f"{m:02d}:{s:02d}"


# ══════════════════════════════════════════════════════════════
# AI DOCTOR - SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

def get_doctor_prompt():
    """Generate doctor system prompt in the selected language."""
    lang = get_selected_lang()
    lang_instruction = lang["doctor_lang_instruction"]
    return f"""You are Dr. AI, a professional, empathetic physician conducting a medical consultation.

LANGUAGE: {lang_instruction}

RULES:
1. Start with a warm greeting and ask the chief complaint.
2. Ask only 1-2 focused follow-up questions per turn. Do NOT overwhelm the patient.
3. Follow proper medical history taking:
   - History of Present Illness (onset, duration, severity, location)
   - Associated symptoms
   - Red flag screening (breathing difficulty, chest pain, etc.)
   - Past medical history, medications, allergies
4. After gathering enough info (usually 4-6 exchanges), provide:
   - Summary of findings
   - Likely diagnosis
   - Basic treatment advice
   - When to seek urgent care
5. Keep responses concise (2-3 sentences max).
6. Be warm, professional, and reassuring.
7. When the consultation is naturally complete, end your response with [CONSULTATION_COMPLETE]"""

# ══════════════════════════════════════════════════════════════
# TAB 1: FILE UPLOAD MODE
# ══════════════════════════════════════════════════════════════

def tab_file_upload():
    audio_files = sorted(glob.glob(str(SCRIBE_DIR / "*.wav")) + glob.glob(str(SCRIBE_DIR / "*.mp3")))
    if not audio_files:
        st.warning("⚠️ No audio files in `scribe/` folder.")
        return
    selected = audio_files[0] if len(audio_files) == 1 else st.selectbox("Choose recording", audio_files, format_func=lambda x: Path(x).name)
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown(f"### 🎙️ Recording: `{Path(selected).name}`")
        st.audio(selected)
    with col2:
        sz = os.path.getsize(selected)/(1024*1024)
        st.markdown(f'<div class="metric-card" style="margin-top:2rem;"><div class="value">{sz:.1f} MB</div><div class="label">File Size</div></div>', unsafe_allow_html=True)
    st.divider()
    if st.button("🚀 Generate SOAP Notes", type="primary", use_container_width=True, key="btn_file"):
        with st.status("🔄 Processing...", expanded=True) as status:
            st.write("🎧 Transcribing audio...")
            t0 = time.time()
            result = transcribe_audio_file(selected)
            t_stt = time.time()-t0
            st.write(f"✅ Transcribed {len(result['segments'])} segments in {t_stt:.1f}s")
            st.write("🧠 Generating clinical notes...")
            t1 = time.time()
            clinical = generate_clinical_output(result["full_text"])
            t_llm = time.time()-t1
            st.write(f"✅ Clinical notes in {t_llm:.1f}s")
            status.update(label="✅ Complete!", state="complete", expanded=False)
        st.session_state["file_result"] = result
        st.session_state["file_clinical"] = clinical
        st.session_state["file_timings"] = {"stt": t_stt, "llm": t_llm}

    if "file_clinical" in st.session_state:
        result = st.session_state["file_result"]
        clinical = st.session_state["file_clinical"]
        t = st.session_state["file_timings"]
        st.markdown(f"""<div class="metrics-row">
            <div class="metric-card"><div class="value">{fmt_time(result['duration'])}</div><div class="label">Duration</div></div>
            <div class="metric-card"><div class="value">{len(result['segments'])}</div><div class="label">Segments</div></div>
            <div class="metric-card"><div class="value">{t['stt']:.1f}s</div><div class="label">STT Time</div></div>
            <div class="metric-card"><div class="value">{t['llm']:.1f}s</div><div class="label">LLM Time</div></div>
            </div>""", unsafe_allow_html=True)
        st.markdown("### 📝 Transcript")
        with st.container(height=250):
            for seg in result["segments"]:
                sp = seg["speaker"]
                cls = "transcript-doctor" if sp=="Doctor" else "transcript-patient"
                bcls = "badge-doctor" if sp=="Doctor" else "badge-patient"
                ico = "👨‍⚕️" if sp=="Doctor" else "🧑"
                st.markdown(f'<div class="transcript-line {cls}"><span class="badge {bcls}">{ico} {sp}</span><div style="margin-top:0.4rem;">{seg["text"]}</div></div>', unsafe_allow_html=True)
        st.divider()
        render_clinical_output(clinical)
        st.divider()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("📥 Download Full Clinical JSON", json.dumps(clinical, indent=2, ensure_ascii=False), f"clinical_{ts}.json", "application/json", use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 2: LIVE CONSULTATION MODE
# ══════════════════════════════════════════════════════════════

def get_doctor_response(chat_history):
    """Get AI Doctor response from Groq."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    messages = [{"role": "system", "content": get_doctor_prompt()}] + chat_history
    resp = client.chat.completions.create(model=GROQ_MODEL, messages=messages, temperature=0.3, max_tokens=500)
    return resp.choices[0].message.content.strip()

def render_chat_history(messages):
    """Display chat messages in styled bubbles."""
    for msg in messages:
        if msg["role"] == "assistant":
            text = msg["content"].replace("[CONSULTATION_COMPLETE]", "").strip()
            st.markdown(f'<div class="chat-doctor"><div class="chat-label" style="color:#3b82f6;">👨‍⚕️ Dr. AI</div><div class="chat-text">{text}</div></div>', unsafe_allow_html=True)
        elif msg["role"] == "user":
            st.markdown(f'<div class="chat-patient"><div class="chat-label" style="color:#059669;">🧑 Patient (You)</div><div class="chat-text">{msg["content"]}</div></div>', unsafe_allow_html=True)

def tab_live_consultation():
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "consultation_done" not in st.session_state:
        st.session_state.consultation_done = False
    if "consultation_started" not in st.session_state:
        st.session_state.consultation_started = False

    # ── Language Selector ──
    lang_options = list(LANG_CONFIG.keys())
    selected_lang = st.selectbox(
        "🌐 Select Language",
        lang_options,
        index=0,
        key="selected_language",
        disabled=st.session_state.consultation_started,
    )
    lang = LANG_CONFIG[selected_lang]
    st.caption(f"{lang['flag']} Doctor will speak in **{selected_lang}** | TTS: `{lang['tts_voice']}` | STT: `{lang['code']}`")

    # ── Start / Reset buttons ──
    col_a, col_b = st.columns([1,1])
    with col_a:
        if not st.session_state.consultation_started:
            if st.button("🟢 Start Consultation", type="primary", use_container_width=True, key="start_btn"):
                st.session_state.chat_history = []
                st.session_state.consultation_done = False
                st.session_state.consultation_started = True
                # Doctor greeting
                greeting = get_doctor_response([])
                st.session_state.chat_history.append({"role": "assistant", "content": greeting})
                # Generate TTS for greeting
                audio_path = tts_speak(greeting.replace("[CONSULTATION_COMPLETE]", ""))
                st.session_state.last_doctor_audio = audio_path
                st.rerun()
    with col_b:
        if st.session_state.consultation_started:
            if st.button("🔴 End & Generate SOAP", use_container_width=True, key="end_btn"):
                st.session_state.consultation_done = True
                st.session_state.consultation_started = False
                st.rerun()

    if not st.session_state.consultation_started and not st.session_state.consultation_done:
        st.info("👆 Click **Start Consultation** to begin talking with the AI Doctor. Speak your symptoms and the doctor will ask follow-up questions.")
        # Reset button if there's old data
        if st.session_state.chat_history:
            if st.button("🔄 New Consultation", key="reset_btn"):
                st.session_state.chat_history = []
                st.session_state.consultation_done = False
                st.rerun()
        return

    # ── Active Consultation ──
    if st.session_state.consultation_started and not st.session_state.consultation_done:
        # Display chat history
        st.markdown("### 💬 Consultation")
        with st.container(height=400):
            render_chat_history(st.session_state.chat_history)

        # Play last doctor audio
        if "last_doctor_audio" in st.session_state and st.session_state.last_doctor_audio:
            if os.path.exists(st.session_state.last_doctor_audio):
                st.audio(st.session_state.last_doctor_audio, format="audio/mp3", autoplay=True)

        # Check if consultation auto-completed
        if st.session_state.chat_history and "[CONSULTATION_COMPLETE]" in st.session_state.chat_history[-1].get("content", ""):
            st.success("✅ The doctor has completed the consultation. Click **End & Generate SOAP** to get your notes.")

        st.markdown("---")
        st.markdown("**🎤 Your turn — speak or type your response:**")

        # Two input methods: mic or text
        input_col1, input_col2 = st.columns([1, 1])

        with input_col1:
            audio_input = st.audio_input("🎤 Record your response", key=f"mic_{len(st.session_state.chat_history)}")
            if audio_input:
                with st.spinner("🔄 Transcribing..."):
                    patient_text = transcribe_audio_bytes(audio_input.read())
                if patient_text.strip():
                    st.session_state.chat_history.append({"role": "user", "content": patient_text.strip()})
                    with st.spinner("👨‍⚕️ Doctor is thinking..."):
                        doc_response = get_doctor_response(st.session_state.chat_history)
                    st.session_state.chat_history.append({"role": "assistant", "content": doc_response})
                    with st.spinner("🔊 Doctor is speaking..."):
                        audio_path = tts_speak(doc_response.replace("[CONSULTATION_COMPLETE]", ""))
                        st.session_state.last_doctor_audio = audio_path
                    st.rerun()

        with input_col2:
            text_input = st.text_input("⌨️ Or type here", key=f"text_{len(st.session_state.chat_history)}", placeholder="Type your symptoms...")
            if text_input:
                st.session_state.chat_history.append({"role": "user", "content": text_input})
                with st.spinner("👨‍⚕️ Doctor is thinking..."):
                    doc_response = get_doctor_response(st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "assistant", "content": doc_response})
                with st.spinner("🔊 Doctor is speaking..."):
                    audio_path = tts_speak(doc_response.replace("[CONSULTATION_COMPLETE]", ""))
                    st.session_state.last_doctor_audio = audio_path
                st.rerun()

    # ── Consultation Done → SOAP Notes ──
    if st.session_state.consultation_done and st.session_state.chat_history:
        st.markdown("### 💬 Consultation Transcript")
        with st.container(height=300):
            render_chat_history(st.session_state.chat_history)

        st.divider()

        # Build transcript text
        transcript = "\n".join(
            f"{'Doctor' if m['role']=='assistant' else 'Patient'}: {m['content'].replace('[CONSULTATION_COMPLETE]','').strip()}"
            for m in st.session_state.chat_history if m["role"] in ("assistant", "user")
        )

        if "live_clinical" not in st.session_state:
            with st.status("🧠 Generating clinical notes...", expanded=True) as status:
                clinical = generate_clinical_output(transcript)
                # Add patient info if available
                pi = st.session_state.get("patient_info", {})
                if pi:
                    clinical["patient_info"] = pi
                st.session_state.live_clinical = clinical
                st.session_state.live_transcript = transcript
                # Save to history
                save_consultation(pi, clinical, transcript, st.session_state.chat_history)
                status.update(label="✅ Clinical notes ready!", state="complete", expanded=False)
            st.rerun()
        else:
            clinical = st.session_state.live_clinical
            transcript = st.session_state.live_transcript

            render_clinical_output(clinical)

            st.divider()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button("📥 Download Full Clinical JSON", json.dumps(clinical, indent=2, ensure_ascii=False),
                f"clinical_{ts}.json", "application/json", use_container_width=True, key="dl_live_json")

            regen_col, new_col = st.columns(2)
            with regen_col:
                if st.button("🔄 Regenerate Notes", use_container_width=True, key="regen_soap"):
                    st.session_state.pop("live_clinical", None)
                    st.rerun()
            with new_col:
                if st.button("🆕 New Consultation", type="primary", use_container_width=True, key="new_consult"):
                    for k in ["chat_history", "consultation_done", "consultation_started", "live_clinical", "live_transcript", "last_doctor_audio", "patient_info"]:
                        st.session_state.pop(k, None)
                    st.rerun()

# ══════════════════════════════════════════════════════════════
# TAB 3: CONSULTATION HISTORY
# ══════════════════════════════════════════════════════════════

def tab_history():
    records = load_consultation_history()
    if not records:
        st.info("📭 No consultations saved yet. Complete a consultation to see it here.")
        return
    st.markdown(f"### 📚 {len(records)} Saved Consultations")
    for i, rec in enumerate(records):
        ts = rec.get("timestamp", "Unknown")
        pt = rec.get("patient", {})
        cd = rec.get("clinical_data", {})
        name = pt.get("name", "Unknown")
        sev = cd.get("severity", "unknown")
        cc = cd.get("chief_complaint", "N/A")
        with st.expander(f"🕐 {ts} — {name} — {cc} {severity_badge(sev)}", expanded=False):
            if pt:
                st.markdown(f"**Patient:** {name} | Age: {pt.get('age','?')} | Gender: {pt.get('gender','?')}")
            render_clinical_output(cd)
            st.download_button("📥 Download JSON", json.dumps(rec, indent=2, ensure_ascii=False),
                f"history_{i}.json", "application/json", key=f"hist_dl_{i}")

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    render_header()
    tab1, tab2, tab3 = st.tabs(["📂 Upload Audio", "🎤 Live AI Doctor", "📚 History"])
    with tab1:
        tab_file_upload()
    with tab2:
        # Patient Info (before consultation starts)
        if not st.session_state.get("consultation_started") and not st.session_state.get("consultation_done"):
            with st.expander("👤 Patient Information (Optional)", expanded=False):
                c1, c2, c3 = st.columns(3)
                with c1:
                    p_name = st.text_input("Name", key="p_name", placeholder="Patient name")
                with c2:
                    p_age = st.number_input("Age", min_value=0, max_value=150, value=0, key="p_age")
                with c3:
                    p_gender = st.selectbox("Gender", ["", "Male", "Female", "Other"], key="p_gender")
                if p_name or p_age or p_gender:
                    st.session_state["patient_info"] = {"name": p_name, "age": p_age if p_age > 0 else None, "gender": p_gender or None}
        tab_live_consultation()
    with tab3:
        tab_history()

if __name__ == "__main__":
    main()
