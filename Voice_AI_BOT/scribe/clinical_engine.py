"""Clinical engine — structured output, severity, guardrails."""
import json, re, os
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
HISTORY_DIR = Path(__file__).resolve().parent / "history"
HISTORY_DIR.mkdir(exist_ok=True)

CLINICAL_PROMPT = """You are an expert medical scribe AI. Extract ALL clinical data from this doctor-patient transcript into a structured format.

RULES:
- Extract ONLY what is explicitly stated in the transcript
- DO NOT hallucinate or invent any information
- For each field, rate confidence: "high", "medium", or "low"
- If not mentioned, use null

Return ONLY this exact JSON structure:
{{
  "patient_info": {{"age": null, "gender": null}},
  "chief_complaint": "",
  "history_of_present_illness": {{
    "onset": "", "duration": "", "severity": "", "character": "", "location": ""
  }},
  "symptoms_present": ["symptom1", "symptom2"],
  "symptoms_denied": ["symptom1"],
  "vitals": {{"temperature": null, "bp": null, "pulse": null, "spo2": null}},
  "medications_taken": ["med1"],
  "past_medical_history": "",
  "allergies": "",
  "subjective": "• bullet point summary of patient complaints",
  "objective": "• clinical observations or 'Telemedicine consultation — no physical exam'",
  "assessment": "• diagnosis with clinical reasoning",
  "plan": "• treatment steps",
  "differential_diagnosis": ["diagnosis1", "diagnosis2"],
  "red_flags_screened": ["symptom checked - negative"],
  "confidence": {{"subjective": "high", "objective": "medium", "assessment": "high", "plan": "high"}},
  "severity": "mild|moderate|severe|emergency",
  "triage": "self-care|outpatient|urgent|emergency",
  "follow_up": ""
}}

TRANSCRIPT:
{transcript}"""


def generate_clinical_output(transcript):
    """Generate structured clinical output from transcript."""
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise medical scribe. Output ONLY valid JSON. Never hallucinate. Extract only from transcript."},
            {"role": "user", "content": CLINICAL_PROMPT.format(transcript=transcript)}
        ],
        temperature=0.1, max_tokens=3000,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content.strip()
    match = re.search(r'\{[\s\S]*\}', raw)
    if match: raw = match.group(0)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"error": raw[:500], "subjective": raw, "objective": "Parse error", "assessment": "Parse error", "plan": "Parse error",
                "severity": "unknown", "triage": "unknown", "confidence": {}, "symptoms_present": [], "symptoms_denied": [],
                "differential_diagnosis": [], "medications_taken": [], "red_flags_screened": []}
    # Ensure all string fields are actually strings
    for k in ["subjective", "objective", "assessment", "plan", "chief_complaint", "follow_up", "severity", "triage"]:
        v = data.get(k)
        if isinstance(v, dict): data[k] = "\n".join(f"• {dk}: {dv}" for dk, dv in v.items())
        elif isinstance(v, list): data[k] = "\n".join(f"• {i}" for i in v)
        elif v is None: data[k] = "Not mentioned"
        elif not isinstance(v, str): data[k] = str(v)
    for k in ["symptoms_present", "symptoms_denied", "differential_diagnosis", "medications_taken", "red_flags_screened"]:
        if not isinstance(data.get(k), list): data[k] = []
    if "confidence" not in data or not isinstance(data["confidence"], dict): data["confidence"] = {}
    return data


GUARDRAIL_DISCLAIMER = "⚠️ AI-Generated Clinical Notes — Must be reviewed and validated by a licensed physician before use in patient care."

DANGEROUS_PATTERNS = [
    r"(?i)you definitely have",
    r"(?i)i am 100% sure",
    r"(?i)you don't need to see a doctor",
    r"(?i)stop taking your medication",
    r"(?i)this is certainly",
]

def check_guardrails(text):
    """Check for dangerous advice patterns. Returns list of warnings."""
    warnings = []
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, text):
            warnings.append(f"⚠️ Flagged pattern: {pattern}")
    return warnings


def save_consultation(patient_info, clinical_data, transcript, chat_history):
    """Save consultation to history."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {
        "timestamp": datetime.now().isoformat(),
        "patient": patient_info,
        "clinical_data": clinical_data,
        "transcript": transcript,
        "chat_history": [{"role": m["role"], "content": m["content"]} for m in chat_history],
    }
    fpath = HISTORY_DIR / f"consultation_{ts}.json"
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return str(fpath)


def load_consultation_history():
    """Load all saved consultations."""
    files = sorted(HISTORY_DIR.glob("consultation_*.json"), reverse=True)
    records = []
    for f in files[:20]:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                records.append(json.load(fh))
        except: pass
    return records


def severity_badge(severity):
    """Return HTML badge for severity."""
    badges = {
        "mild": ("🟢 Mild", "#dcfce7", "#166534"),
        "moderate": ("⚠️ Moderate", "#fef9c3", "#854d0e"),
        "severe": ("🔴 Severe", "#fee2e2", "#991b1b"),
        "emergency": ("🚨 EMERGENCY", "#fee2e2", "#7f1d1d"),
    }
    label, bg, color = badges.get(str(severity).lower(), ("❓ Unknown", "#f1f5f9", "#475569"))
    return f'<span style="background:{bg};color:{color};padding:0.3rem 0.8rem;border-radius:20px;font-weight:700;font-size:0.85rem;">{label}</span>'


def triage_badge(triage):
    badges = {
        "self-care": ("🏠 Self-Care", "#dcfce7", "#166534"),
        "outpatient": ("🏥 Outpatient", "#dbeafe", "#1e40af"),
        "urgent": ("⚡ Urgent Care", "#fef9c3", "#854d0e"),
        "emergency": ("🚑 Emergency", "#fee2e2", "#991b1b"),
    }
    label, bg, color = badges.get(str(triage).lower(), ("❓ Unknown", "#f1f5f9", "#475569"))
    return f'<span style="background:{bg};color:{color};padding:0.3rem 0.8rem;border-radius:20px;font-weight:700;font-size:0.85rem;">{label}</span>'
