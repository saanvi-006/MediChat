import streamlit as st
import requests

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediChat – Health Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
if "page"         not in st.session_state: st.session_state.page         = "landing"
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "language"     not in st.session_state: st.session_state.language     = "en"
if "show_map"     not in st.session_state: st.session_state.show_map     = False

# ══════════════════════════════════════════════════════════════════════════════
# TRANSLATIONS  (EN / HI / BN)
# ══════════════════════════════════════════════════════════════════════════════
T = {
    "en": {
        "badge":          "Health Assistant",
        "subtitle":       "Tell me what you're feeling — I'll help you understand your symptoms with clear, structured guidance.",
        "feat1_title":    "Symptom Understanding",
        "feat1_desc":     "Interprets your symptoms with context and clarity.",
        "feat2_title":    "Conversational",
        "feat2_desc":     "Natural dialogue — like speaking with a health advisor.",
        "feat3_title":    "Clear Insights",
        "feat3_desc":     "Category, severity and confidence shown when relevant.",
        "disclaimer":     "MediChat provides general health guidance only. It is not a substitute for professional medical diagnosis or treatment. Always consult a licensed healthcare provider.",
        "start_btn":      "Start Chat →",
        "welcome_h":      "Tell me how you're feeling.",
        "welcome_p":      "Describe your symptoms and I'll do my best to help.",
        "placeholder":    "Tell me what you're feeling…",
        "send_btn":       "Send →",
        "online":         "Ready to help",
        "not_doctor":     "Not a medical diagnosis",
        "analyzing":      "Analyzing…",
        "find_doctors":   "🗺️ Find nearby doctors",
        "map_title":      "Nearby Hospitals & Clinics",
        "map_note":       "Showing approximate results based on your region.",
        "err_empty":      "Please type a message first.",
        "err_conn":       "⚠️ Could not reach the MediChat server. Is the backend running on port 8000?",
        "err_timeout":    "⏱️ The request timed out. Please try again.",
        "err_generic":    "⚠️ An unexpected error occurred. Please try again.",
        "back":           "← Back",
        "category":       "Category",
        "severity":       "Severity",
        "confidence":     "Confidence",
    },
    "hi": {
        "badge":          "स्वास्थ्य सहायक",
        "subtitle":       "बताइए आप कैसा महसूस कर रहे हैं — मैं आपके लक्षणों को समझने में मदद करूँगा।",
        "feat1_title":    "लक्षण विश्लेषण",
        "feat1_desc":     "आपके लक्षणों को संदर्भ के साथ समझता है।",
        "feat2_title":    "संवादात्मक",
        "feat2_desc":     "स्वास्थ्य सलाहकार की तरह स्वाभाविक बातचीत।",
        "feat3_title":    "स्पष्ट जानकारी",
        "feat3_desc":     "श्रेणी, गंभीरता और विश्वास स्तर जब प्रासंगिक हो।",
        "disclaimer":     "MediChat केवल सामान्य स्वास्थ्य मार्गदर्शन प्रदान करता है। यह चिकित्सा निदान का विकल्प नहीं है।",
        "start_btn":      "चैट शुरू करें →",
        "welcome_h":      "आप कैसा महसूस कर रहे हैं?",
        "welcome_p":      "अपने लक्षण बताएं और मैं मदद करने की कोशिश करूँगा।",
        "placeholder":    "आप कैसा महसूस कर रहे हैं…",
        "send_btn":       "भेजें →",
        "online":         "मदद के लिए तैयार",
        "not_doctor":     "चिकित्सा निदान नहीं",
        "analyzing":      "विश्लेषण हो रहा है…",
        "find_doctors":   "🗺️ नज़दीकी डॉक्टर खोजें",
        "map_title":      "नज़दीकी अस्पताल और क्लीनिक",
        "map_note":       "आपके क्षेत्र के अनुमानित परिणाम दिखाए जा रहे हैं।",
        "err_empty":      "कृपया पहले कोई संदेश लिखें।",
        "err_conn":       "⚠️ सर्वर से कनेक्ट नहीं हो पाया।",
        "err_timeout":    "⏱️ अनुरोध समय सीमा पार हो गई। कृपया पुनः प्रयास करें।",
        "err_generic":    "⚠️ एक अप्रत्याशित त्रुटि हुई।",
        "back":           "← वापस",
        "category":       "श्रेणी",
        "severity":       "गंभीरता",
        "confidence":     "विश्वास",
    },
    "bn": {
        "badge":          "স্বাস্থ্য সহকারী",
        "subtitle":       "আপনি কেমন অনুভব করছেন বলুন — আমি আপনার লক্ষণগুলি বুঝতে সাহায্য করব।",
        "feat1_title":    "লক্ষণ বিশ্লেষণ",
        "feat1_desc":     "প্রেক্ষাপট সহ আপনার লক্ষণ বোঝে।",
        "feat2_title":    "কথোপকথনমূলক",
        "feat2_desc":     "স্বাস্থ্য পরামর্শদাতার মতো স্বাভাবিক কথোপকথন।",
        "feat3_title":    "স্পষ্ট তথ্য",
        "feat3_desc":     "প্রাসঙ্গিক হলে বিভাগ, তীব্রতা এবং আস্থা দেখানো হয়।",
        "disclaimer":     "MediChat শুধুমাত্র সাধারণ স্বাস্থ্য নির্দেশিকা প্রদান করে। এটি চিকিৎসা নির্ণয়ের বিকল্প নয়।",
        "start_btn":      "চ্যাট শুরু করুন →",
        "welcome_h":      "আপনি কেমন অনুভব করছেন?",
        "welcome_p":      "আপনার লক্ষণ বর্ণনা করুন এবং আমি সাহায্য করার চেষ্টা করব।",
        "placeholder":    "আপনি কেমন অনুভব করছেন…",
        "send_btn":       "পাঠান →",
        "online":         "সাহায্য করতে প্রস্তুত",
        "not_doctor":     "চিকিৎসা নির্ণয় নয়",
        "analyzing":      "বিশ্লেষণ করা হচ্ছে…",
        "find_doctors":   "🗺️ কাছের ডাক্তার খুঁজুন",
        "map_title":      "কাছের হাসপাতাল ও ক্লিনিক",
        "map_note":       "আপনার অঞ্চলের আনুমানিক ফলাফল দেখানো হচ্ছে।",
        "err_empty":      "অনুগ্রহ করে আগে একটি বার্তা লিখুন।",
        "err_conn":       "⚠️ সার্ভারের সাথে সংযোগ করা যায়নি।",
        "err_timeout":    "⏱️ অনুরোধের সময় শেষ হয়ে গেছে।",
        "err_generic":    "⚠️ একটি অপ্রত্যাশিত ত্রুটি ঘটেছে।",
        "back":           "← ফিরুন",
        "category":       "বিভাগ",
        "severity":       "তীব্রতা",
        "confidence":     "আস্থা",
    },
}

def tr(key: str) -> str:
    lang = st.session_state.get("language", "en")
    return T.get(lang, T["en"]).get(key, T["en"].get(key, key))


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

    /* ── Fonts ── */
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ══════════════════════════════════════════
       PREMIUM BACKGROUND  (Fix 3 + optional A)
       Layered gradient mesh with slow animation.
    ══════════════════════════════════════════ */
    .stApp {
        background:
            radial-gradient(ellipse 80% 60% at 10% 10%,  rgba(186,224,255,0.35) 0%, transparent 60%),
            radial-gradient(ellipse 70% 50% at 90% 90%,  rgba(200,230,255,0.28) 0%, transparent 55%),
            radial-gradient(ellipse 60% 40% at 80% 15%,  rgba(220,240,255,0.18) 0%, transparent 50%),
            linear-gradient(160deg, #EEF6FF 0%, #F7FBFF 50%, #EDF4FD 100%);
        background-attachment: fixed;
        animation: bgShift 18s ease-in-out infinite alternate;
    }

    @keyframes bgShift {
        0%   { background-position: 0% 0%,   100% 100%, 80% 20%, center; }
        100% { background-position: 5% 5%,   95% 95%,  75% 25%, center; }
    }

    /* ── Remove ALL white container backgrounds (Fix 2) ── */
    .stApp > div,
    .stAppViewContainer,
    section[data-testid="stAppViewContainer"],
    div[data-testid="stAppViewContainer"],
    .main,
    div[data-testid="stVerticalBlock"],
    div[data-testid="column"],
    div[data-testid="stHorizontalBlock"] {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* ── Block container ── */
    .block-container {
        background: transparent !important;
        padding-top: 0.5rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 760px;
    }

    /* ── Strip Streamlit form chrome globally ── */
    div[data-testid="stForm"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }

    /* ══════════════════════════════════════════
       LANGUAGE SELECTOR
    ══════════════════════════════════════════ */
    .lang-bar {
        display: flex;
        justify-content: flex-end;
        gap: 6px;
        margin-bottom: 0.5rem;
    }

    .lang-btn {
        background: rgba(255,255,255,0.65);
        border: 1px solid rgba(21,101,192,0.18);
        border-radius: 8px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #455A64;
        cursor: pointer;
        transition: all 0.2s;
        backdrop-filter: blur(6px);
    }
    .lang-btn:hover  { background: rgba(21,101,192,0.08); color: #1565C0; }
    .lang-btn.active { background: #1565C0; color: #fff; border-color: #1565C0; }

    /* ══════════════════════════════════════════
       LANDING PAGE
    ══════════════════════════════════════════ */
    .landing-wrapper {
        min-height: 78vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 2rem 1rem;
    }

    .landing-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(232,245,233,0.85);
        color: #2E7D32;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        padding: 5px 14px;
        border-radius: 999px;
        margin-bottom: 1.5rem;
        border: 1px solid #C8E6C9;
        backdrop-filter: blur(4px);
    }

    .landing-title {
        font-family: 'DM Serif Display', serif;
        font-size: clamp(2.6rem, 6vw, 4rem);
        color: #0D1B2A;
        line-height: 1.12;
        margin-bottom: 1rem;
        letter-spacing: -0.5px;
    }
    .landing-title span { color: #1565C0; }

    .landing-subtitle {
        font-size: clamp(0.98rem, 2.4vw, 1.15rem);
        color: #546E7A;
        font-weight: 400;
        max-width: 500px;
        line-height: 1.75;
        margin-bottom: 2.5rem;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
        gap: 1rem;
        max-width: 620px;
        width: 100%;
        margin-bottom: 1.5rem;
    }

    .feature-card {
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(21,101,192,0.1);
        border-radius: 16px;
        padding: 1.15rem 1rem;
        box-shadow: 0 2px 14px rgba(21,101,192,0.07);
        text-align: left;
        backdrop-filter: blur(8px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .feature-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(21,101,192,0.11); }
    .feature-card .icon { font-size: 1.5rem; margin-bottom: 0.5rem; }
    .feature-card h4 { font-size: 0.88rem; font-weight: 600; color: #0D1B2A; margin: 0 0 4px; }
    .feature-card p  { font-size: 0.79rem; color: #78909C; margin: 0; line-height: 1.5; }

    .disclaimer-box {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        background: rgba(255,248,225,0.80);
        border: 1px solid #FFE082;
        border-radius: 12px;
        padding: 0.6rem 0.9rem;
        max-width: 500px;
        width: 100%;
        text-align: left;
        backdrop-filter: blur(4px);
    }
    .disclaimer-box p { font-size: 0.8rem; color: #795548; margin: 0; line-height: 1.6; }

    /* ══════════════════════════════════════════
       CHAT HEADER  (Fix 1: removed "AI ·")
    ══════════════════════════════════════════ */
    .chat-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.8rem 1.3rem;
        background: rgba(255,255,255,0.80);
        border: 1px solid rgba(21,101,192,0.12);
        border-radius: 18px;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 16px rgba(21,101,192,0.07);
        backdrop-filter: blur(10px);
    }
    .chat-header-left { display: flex; align-items: center; gap: 10px; }
    .chat-header-avatar {
        width: 38px; height: 38px;
        background: linear-gradient(135deg, #1565C0, #42A5F5);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem; flex-shrink: 0;
    }
    .chat-header-info h3 { font-size: 0.94rem; font-weight: 600; color: #0D1B2A; margin: 0; }
    .chat-header-info p  { font-size: 0.72rem; color: #66BB6A; margin: 0; font-weight: 500; }

    /* "Not a medical diagnosis" tag — subtle, no "AI" word (Fix 1) */
    .header-tag {
        font-size: 0.7rem;
        color: #90A4AE;
        background: rgba(236,239,241,0.7);
        padding: 3px 10px;
        border-radius: 999px;
        border: 1px solid #E0E6EC;
        white-space: nowrap;
    }

    .online-dot {
        width: 7px; height: 7px;
        background: #66BB6A; border-radius: 50%;
        display: inline-block; margin-right: 5px;
        animation: pulse 2.4s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }

    /* ══════════════════════════════════════════
       CHAT BUBBLES  (Fix 5: depth + spacing)
    ══════════════════════════════════════════ */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 0.9rem;
        padding: 0.4rem 0 0.6rem;
    }

    #chat-bottom { height: 1px; }

    .msg-user { display: flex; justify-content: flex-end; animation: fadeUp 0.22s ease both; }

    .bubble-user {
        background: linear-gradient(135deg, #1565C0, #1E88E5);
        color: #fff;
        padding: 0.72rem 1.1rem;
        border-radius: 18px 18px 4px 18px;
        max-width: min(74%, 460px);
        font-size: 0.91rem;
        line-height: 1.65;
        box-shadow: 0 4px 18px rgba(21,101,192,0.28);
        word-break: break-word;
        overflow-wrap: anywhere;
    }

    .msg-bot { display: flex; align-items: flex-start; gap: 9px; animation: fadeUp 0.22s ease both; }

    .bot-avatar {
        width: 30px; height: 30px;
        background: linear-gradient(135deg, #1565C0, #42A5F5);
        border-radius: 9px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.85rem; flex-shrink: 0; margin-top: 3px;
    }

    .bubble-bot {
        background: rgba(255,255,255,0.88);
        color: #263238;
        padding: 0.72rem 1.1rem;
        border-radius: 18px 18px 18px 4px;
        max-width: min(74%, 460px);
        font-size: 0.91rem;
        line-height: 1.65;
        border: 1px solid rgba(21,101,192,0.10);
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        word-break: break-word;
        overflow-wrap: anywhere;
        backdrop-filter: blur(6px);
    }

    @keyframes fadeUp {
        from { opacity:0; transform:translateY(8px); }
        to   { opacity:1; transform:translateY(0); }
    }

    /* ── Meta card ── */
    .meta-card {
        background: rgba(248,251,255,0.88);
        border: 1px solid rgba(21,101,192,0.12);
        border-radius: 12px;
        padding: 0.7rem 1rem;
        margin-top: 0.3rem;
        margin-left: 39px;
        max-width: min(74%, 460px);
        display: flex;
        gap: 1.3rem;
        flex-wrap: wrap;
        animation: fadeUp 0.28s ease 0.08s both;
        backdrop-filter: blur(6px);
    }
    .meta-item { display: flex; flex-direction: column; gap: 2px; }
    .meta-label { font-size: 0.67rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #90A4AE; }
    .meta-value { font-size: 0.8rem; font-weight: 600; color: #37474F; }
    .sev-low    { color: #2E7D32; }
    .sev-medium { color: #E65100; }
    .sev-high   { color: #B71C1C; }
    .conf-bar-bg   { width: 76px; height: 5px; background: #E3EAF2; border-radius: 99px; overflow: hidden; margin-top: 4px; }
    .conf-bar-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg,#42A5F5,#1565C0); }

    /* ── Map card ── */
    .map-card {
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(21,101,192,0.13);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
        margin-left: 39px;
        max-width: min(74%, 460px);
        backdrop-filter: blur(8px);
        box-shadow: 0 2px 12px rgba(21,101,192,0.06);
        animation: fadeUp 0.3s ease both;
    }
    .map-card h4 { font-size: 0.85rem; font-weight: 600; color: #0D1B2A; margin: 0 0 4px; }
    .map-card p  { font-size: 0.75rem; color: #78909C; margin: 0 0 0.7rem; }

    /* ══════════════════════════════════════════
       WELCOME EMPTY STATE  (Enhancement B)
    ══════════════════════════════════════════ */
    .welcome-msg {
        text-align: center;
        padding: 3.5rem 1.5rem 1rem;
        animation: fadeUp 0.4s ease both;
    }
    .welcome-msg .big-icon { font-size: 2.6rem; margin-bottom: 1rem; }
    .welcome-msg h3 {
        font-family: 'DM Serif Display', serif;
        font-size: 1.45rem;
        color: #37474F;
        margin-bottom: 0.5rem;
        font-weight: 400;
    }
    .welcome-msg p { font-size: 0.87rem; color: #90A4AE; max-width: 300px; margin: 0 auto; line-height: 1.65; }

    /* ══════════════════════════════════════════
       STICKY INPUT BAR  (Fix 4: polished)
    ══════════════════════════════════════════ */
    .sticky-input-bar {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: min(760px, 100vw);
        background: rgba(240,247,255,0.92);
        border-top: 1px solid rgba(21,101,192,0.10);
        border-radius: 20px 20px 0 0;       /* rounded top corners */
        padding: 0.8rem 1.3rem 1.1rem;
        z-index: 9999;
        box-shadow: 0 -6px 28px rgba(21,101,192,0.09);
        backdrop-filter: blur(16px);
    }

    /* Strip inner form chrome */
    .sticky-input-bar div[data-testid="stForm"],
    .sticky-input-bar div[data-testid="stForm"] > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }

    /* Streamlit button overrides */
    div[data-testid="stButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {
        border-radius: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }

    /* ══════════════════════════════════════════
       MOBILE
    ══════════════════════════════════════════ */
    @media (max-width: 600px) {
        .block-container { padding-left: 0.7rem !important; padding-right: 0.7rem !important; }
        .bubble-user, .bubble-bot { max-width: 87% !important; font-size: 0.88rem; }
        .meta-card, .map-card { max-width: 87% !important; }
        .chat-header { padding: 0.6rem 0.9rem; border-radius: 14px; }
        .sticky-input-bar { padding: 0.6rem 0.75rem 0.85rem; border-radius: 16px 16px 0 0; }
        .landing-title { letter-spacing: 0; }
    }
    </style>
    """, unsafe_allow_html=True)


# ── Auto-scroll ──────────────────────────────────────────────────────────────────
def inject_autoscroll():
    st.markdown("""
    <script>
    (function(){
        function s(){ var e=document.getElementById('chat-bottom'); if(e) e.scrollIntoView({behavior:'smooth',block:'end'}); }
        s(); setTimeout(s, 220);
    })();
    </script>
    """, unsafe_allow_html=True)


# ── Language selector widget ─────────────────────────────────────────────────────
def render_lang_selector():
    """Three pill buttons; clicking sets session_state.language and reruns."""
    lang = st.session_state.language
    langs = [("en", "EN"), ("hi", "हिन्दी"), ("bn", "বাংলা")]

    cols = st.columns([6, 2, 2, 2])
    for col, (code, label) in zip(cols[1:], langs):
        with col:
            btn_type = "primary" if lang == code else "secondary"
            if st.button(label, key=f"lang_{code}", type=btn_type, use_container_width=True):
                st.session_state.language = code
                st.rerun()


# ── API call ─────────────────────────────────────────────────────────────────────
def call_api(user_message: str, history: list[dict], language: str = "en") -> dict | None:
    formatted = [
        f"{'User' if m['role'] == 'user' else 'Bot'}: {m['content']}"
        for m in history
    ]
    try:
        resp = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"message": user_message, "history": formatted, "language": language},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError: return {"error": "connection"}
    except requests.exceptions.Timeout:         return {"error": "timeout"}
    except Exception:                            return {"error": "unknown"}


# ── Helpers ──────────────────────────────────────────────────────────────────────
def severity_class(sev: str) -> str:
    s = (sev or "").lower()
    if s in ("high", "severe", "critical"):  return "sev-high"
    if s in ("medium", "moderate"):           return "sev-medium"
    return "sev-low"

def should_show_meta(idx: int, history: list[dict], meta: dict | None) -> bool:
    if not meta or meta.get("confidence", 0) < 60: return False
    last_bot = max((i for i, m in enumerate(history) if m["role"] == "bot"), default=-1)
    return idx == last_bot

def is_serious_severity(sev: str) -> bool:
    return (sev or "").lower() in ("moderate", "medium", "high", "severe", "critical")


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_landing():
    render_lang_selector()

    st.markdown(f"""
    <div class="landing-wrapper">
        <div class="landing-badge">🩺 {tr('badge')}</div>
        <h1 class="landing-title">
        <span style="color:#42A5F5;">Medi</span><span style="color:#1565C0;">Chat</span>
        </h1>
        <p class="landing-subtitle">{tr('subtitle')}</p>
        <div class="feature-grid">
            <div class="feature-card">
                <div class="icon">🧠</div>
                <h4>{tr('feat1_title')}</h4>
                <p>{tr('feat1_desc')}</p>
            </div>
            <div class="feature-card">
                <div class="icon">💬</div>
                <h4>{tr('feat2_title')}</h4>
                <p>{tr('feat2_desc')}</p>
            </div>
            <div class="feature-card">
                <div class="icon">📊</div>
                <h4>{tr('feat3_title')}</h4>
                <p>{tr('feat3_desc')}</p>
            </div>
        </div>
        <div class="disclaimer-box">
            <span>⚠️</span>
            <p>{tr('disclaimer')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        if st.button(tr("start_btn"), type="primary", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# CHAT PAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_chat():
    # ── Top bar: back button + language selector ───────────────────────────────
    col_back, col_l1, col_l2, col_l3 = st.columns([3, 1, 1, 1])
    with col_back:
        if st.button(tr("back")):
            st.session_state.page = "landing"
            st.session_state.chat_history = []
            st.session_state.show_map = False
            st.rerun()
    for col, (code, label) in zip([col_l1, col_l2, col_l3],
                                   [("en","EN"),("hi","हिन्दी"),("bn","বাংলা")]):
        with col:
            btn_type = "primary" if st.session_state.language == code else "secondary"
            if st.button(label, key=f"clang_{code}", type=btn_type, use_container_width=True):
                st.session_state.language = code
                st.rerun()

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="chat-header">
        <div class="chat-header-left">
            <div class="chat-header-avatar">🩺</div>
            <div class="chat-header-info">
                <h3>MediChat</h3>
                <p><span class="online-dot"></span>{tr('online')}</p>
            </div>
        </div>
        <span class="header-tag">{tr('not_doctor')}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Conversation ────────────────────────────────────────────────────────────
    history = st.session_state.chat_history

    if not history:
        st.markdown(f"""
        <div class="welcome-msg">
            <div class="big-icon">🩺</div>
            <h3>{tr('welcome_h')}</h3>
            <p>{tr('welcome_p')}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        html = '<div class="chat-container">'
        last_meta = None  # track meta of last bot message for map button

        for i, msg in enumerate(history):
            content = msg["content"]
            meta    = msg.get("meta")

            if msg["role"] == "user":
                html += f'<div class="msg-user"><div class="bubble-user">{content}</div></div>'
            else:
                html += f"""
                <div class="msg-bot">
                    <div class="bot-avatar">🩺</div>
                    <div class="bubble-bot">{content}</div>
                </div>"""

                if should_show_meta(i, history, meta):
                    last_meta = meta
                    cat  = meta.get("category", "—")
                    sev  = meta.get("severity", "—")
                    conf = meta.get("confidence", 0)
                    sc   = severity_class(sev)
                    bw   = min(int(conf), 100)
                    se   = {"low":"🟢","medium":"🟡","moderate":"🟡",
                            "high":"🔴","severe":"🔴","critical":"🔴"
                           }.get((sev or "").lower(), "⚪")
                    html += f"""
                    <div class="meta-card">
                        <div class="meta-item">
                            <span class="meta-label">{tr('category')}</span>
                            <span class="meta-value">💊 {cat}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">{tr('severity')}</span>
                            <span class="meta-value {sc}">{se} {sev.capitalize() if sev else '—'}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">{tr('confidence')}</span>
                            <span class="meta-value">{conf}%</span>
                            <div class="conf-bar-bg">
                                <div class="conf-bar-fill" style="width:{bw}%"></div>
                            </div>
                        </div>
                    </div>"""

        html += '<div id="chat-bottom"></div></div>'
        st.markdown(html, unsafe_allow_html=True)
        inject_autoscroll()

        # ── Map feature (Fix: New Feature 2) ───────────────────────────────────
        # Show "Find nearby doctors" button when last response has moderate+ severity
        if last_meta and is_serious_severity(last_meta.get("severity", "")):
            st.markdown("<div style='margin-left:39px;margin-top:0.4rem'>", unsafe_allow_html=True)
            if st.button(tr("find_doctors"), key="map_btn"):
                st.session_state.show_map = not st.session_state.show_map
            st.markdown("</div>", unsafe_allow_html=True)

            if st.session_state.show_map:
                st.markdown(f"""
                <div class="map-card">
                    <h4>🏥 {tr('map_title')}</h4>
                    <p>{tr('map_note')}</p>
                </div>
                """, unsafe_allow_html=True)

                # Embed an OpenStreetMap iframe centered on India (no API key needed).
                # The user can drag/zoom to their actual location.
                st.markdown("""
                <div style="margin-left:39px; margin-top:0.5rem; border-radius:14px; overflow:hidden;
                            box-shadow:0 3px 16px rgba(21,101,192,0.12); max-width:460px;">
                    <iframe
                        width="100%" height="300" style="border:0; display:block;"
                        loading="lazy" allowfullscreen
                        src="https://www.openstreetmap.org/export/embed.html?bbox=68%2C6%2C97%2C37&layer=mapnik&marker=20.5937%2C78.9629"
                    ></iframe>
                </div>
                <div style="margin-left:39px; margin-top:0.4rem;">
                    <a href="https://www.openstreetmap.org/search?query=hospital+near+me"
                       target="_blank"
                       style="font-size:0.78rem; color:#1565C0; text-decoration:none;">
                        🔍 Search hospitals near me on OpenStreetMap →
                    </a>
                </div>
                """, unsafe_allow_html=True)

    # ── Sticky input bar ────────────────────────────────────────────────────────
    st.markdown('<div class="sticky-input-bar">', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        col_in, col_btn = st.columns([6, 1])
        with col_in:
            user_input = st.text_input(
                "msg",
                placeholder=tr("placeholder"),
                label_visibility="collapsed",
            )
        with col_btn:
            send_clicked = st.form_submit_button(tr("send_btn"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Handle send ─────────────────────────────────────────────────────────────
    if send_clicked:
        text = (user_input or "").strip()
        if not text:
            st.toast(tr("err_empty"), icon="⚠️")
            return

        st.session_state.chat_history.append({"role": "user", "content": text})
        st.session_state.show_map = False  # reset map on new message

        with st.spinner(tr("analyzing")):
            result = call_api(text, st.session_state.chat_history[:-1],
                              language=st.session_state.language)

        if not result or "error" in result:
            err   = (result or {}).get("error", "unknown")
            reply = {"connection": tr("err_conn"), "timeout": tr("err_timeout")}.get(
                err, tr("err_generic"))
            st.session_state.chat_history.append({"role": "bot", "content": reply})
        else:
            reply = result.get("reply", "I'm sorry, I couldn't generate a response.")
            meta  = {
                "category":   result.get("category", ""),
                "severity":   result.get("severity", ""),
                "confidence": result.get("confidence", 0),
            }
            st.session_state.chat_history.append(
                {"role": "bot", "content": reply, "meta": meta}
            )

        st.rerun()


# ── Router ───────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    if st.session_state.page == "landing":
        render_landing()
    else:
        render_chat()


if __name__ == "__main__":
    main()
