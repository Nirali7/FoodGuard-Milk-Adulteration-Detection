from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "assets"
MODEL_PATH = BASE_DIR / "milk_adulteration_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

st.set_page_config(
    page_title="FoodGuard | Food Adulteration Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Model setup — the supplied model supports milk only and uses 18 lab features.
# ----------------------------------------------------------------------------
FEATURE_NAMES = [
    "Milk_Type", "Fat_percent", "SNF_percent", "Protein_percent",
    "Lactose_percent", "Moisture_percent", "Ash_percent",
    "Total_Solids_percent", "pH", "Specific_Gravity",
    "Acidity_Titratable_pct", "Viscosity_cP", "Freezing_Point_C",
    "Conductivity_mS_cm", "Urea_mg_per_100mL", "Melamine_ppm",
    "MBRT_hours", "Peroxidase_Activity",
]

MILK_MAPPING = {
    "Double Toned": 0,
    "Full Cream": 1,
    "Skimmed": 2,
    "Standardised": 3,
    "Toned": 4,
}

DEFAULTS = {
    "milk_type": "Full Cream", "fat": 4.0, "snf": 8.5,
    "protein": 3.2, "lactose": 4.8, "moisture": 87.0,
    "ash": 0.7, "total_solids": 13.0, "ph": 6.7,
    "gravity": 1.030, "acidity": 0.15, "viscosity": 2.0,
    "freezing": -0.54, "conductivity": 4.5, "urea": 20.0,
    "melamine": 0.0, "mbrt": 6.0, "peroxidase": 1,
}

@st.cache_resource(show_spinner=False)
def load_model_files(model_path: str, scaler_path: str):
    model_file, scaler_file = Path(model_path), Path(scaler_path)
    if not model_file.exists() or not scaler_file.exists():
        return None, None, "Model files are missing. Add milk_adulteration_model.pkl and scaler.pkl beside app.py."
    try:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        if getattr(model, "n_features_in_", len(FEATURE_NAMES)) != len(FEATURE_NAMES):
            return None, None, "The saved model does not match the 18 required milk measurements."
        if getattr(scaler, "n_features_in_", len(FEATURE_NAMES)) != len(FEATURE_NAMES):
            return None, None, "The saved scaler does not match the 18 required milk measurements."
        return model, scaler, None
    except Exception as exc:
        return None, None, f"The model could not be loaded: {exc}"

model, scaler, model_error = load_model_files(str(MODEL_PATH), str(SCALER_PATH))

# Use a queued page change so Home buttons can navigate to the correct section.
if "requested_page" in st.session_state:
    st.session_state["sidebar_navigation"] = st.session_state.pop("requested_page")
PAGES = ["Home", "Check Milk", "Milk Safety Guide"]
legacy_pages = {
    "🏠  Home": "Home",
    "🥛  Check Milk": "Check Milk",
    "📖  Milk Safety Guide": "Milk Safety Guide",
    "🧪  Check a Sample": "Check Milk",
    "📚  Food Safety Guide": "Milk Safety Guide",
}
if st.session_state.get("sidebar_navigation") in legacy_pages:
    st.session_state["sidebar_navigation"] = legacy_pages[st.session_state["sidebar_navigation"]]

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def go_to(page: str, category: str | None = None) -> None:
    st.session_state["requested_page"] = page
    if category:
        st.session_state["pending_guide_category"] = category


def make_input_frame(values: dict) -> pd.DataFrame:
    row = {
        "Milk_Type": MILK_MAPPING[values["milk_type"]],
        "Fat_percent": values["fat"],
        "SNF_percent": values["snf"],
        "Protein_percent": values["protein"],
        "Lactose_percent": values["lactose"],
        "Moisture_percent": values["moisture"],
        "Ash_percent": values["ash"],
        "Total_Solids_percent": values["total_solids"],
        "pH": values["ph"],
        "Specific_Gravity": values["gravity"],
        "Acidity_Titratable_pct": values["acidity"],
        "Viscosity_cP": values["viscosity"],
        "Freezing_Point_C": values["freezing"],
        "Conductivity_mS_cm": values["conductivity"],
        "Urea_mg_per_100mL": values["urea"],
        "Melamine_ppm": values["melamine"],
        "MBRT_hours": values["mbrt"],
        "Peroxidase_Activity": values["peroxidase"],
    }
    return pd.DataFrame([row], columns=FEATURE_NAMES)


def reset_sample() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.pop("last_prediction", None)



# ----------------------------------------------------------------------------
# Visual identity — food-label / editorial look.
# Palette: aubergine ink, parchment paper, tomato-coral action, saffron highlight.
# ----------------------------------------------------------------------------
def html(markup: str) -> None:
    """Render HTML safely through Markdown (flattens indentation so it is never treated as code)."""
    flat = "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,800;9..144,900&family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@500&display=swap');
:root{
  --ink:#2A1E2C; --ink-soft:#5B4A5C; --paper:#EFE5CE; --paper-2:#E5D8BB; --paper-3:#F7F0DF;
  --aubergine:#2E2131; --aubergine-2:#413047; --tomato:#CF432D; --tomato-d:#A8321F;
  --saffron:#EBAA33; --sage:#6F7F55; --rule:#B9A785;
  --display:'Fraunces',Georgia,serif; --body:'DM Sans',system-ui,sans-serif; --mono:'DM Mono',ui-monospace,monospace;
}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"]{ background:var(--paper) !important; color:var(--ink) !important; font-family:var(--body); }
[data-testid="stHeader"]{ background:transparent !important; }
#MainMenu, footer, [data-testid="stDecoration"]{ display:none !important; }
.block-container{ max-width:1180px; padding:1.4rem 2.2rem 3rem !important; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li{ font-family:var(--body); }
.stApp h1,.stApp h2,.stApp h3{ font-family:var(--display); color:var(--ink); }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"]{ background:var(--aubergine) !important; border-right:0 !important; min-width:280px !important; }
[data-testid="stSidebar"] > div:first-child{ padding-top:1.4rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]{ color:#F3E8D2; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"]{ display:none; }
.fg-brand{ display:flex; gap:.8rem; align-items:center; padding:.2rem .3rem 1.3rem; border-bottom:1px dashed #6a5570; margin-bottom:1.2rem; }
.fg-mark{ width:46px; height:46px; background:var(--tomato); color:#FFF3DE; display:flex; align-items:center; justify-content:center;
  font:900 1.25rem var(--display); letter-spacing:-1px; border-radius:2px; transform:rotate(-4deg); box-shadow:3px 3px 0 var(--saffron); flex-shrink:0; }
.fg-name{ font:900 1.45rem/1 var(--display); color:#FFF3DE !important; letter-spacing:-.5px; }
.fg-name i{ color:var(--saffron); font-style:normal; }
.fg-sub{ font:500 .66rem var(--mono); letter-spacing:1.3px; text-transform:uppercase; color:#BBA6BE !important; margin-top:.35rem; }
.fg-navlabel{ font:500 .66rem var(--mono); letter-spacing:1.6px; text-transform:uppercase; color:#BBA6BE !important; margin:0 0 .5rem .3rem; }
[data-testid="stSidebar"] [role="radiogroup"]{ gap:.15rem; counter-reset:nav; }
[data-testid="stSidebar"] [role="radiogroup"] label{ counter-increment:nav; position:relative; width:100%; padding:.8rem .9rem .8rem 3rem; margin:0;
  border-left:4px solid transparent; border-radius:0 6px 6px 0; cursor:pointer; transition:background .15s ease, padding-left .15s ease, border-color .15s ease; }
[data-testid="stSidebar"] [role="radiogroup"] label > div > div:not([data-testid="stMarkdownContainer"]), [data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child{ display:none !important; }
[data-testid="stSidebar"] [role="radiogroup"] label::before{ content:"0" counter(nav); position:absolute; left:.9rem; top:50%; transform:translateY(-50%);
  font:500 .78rem var(--mono); color:var(--saffron); }
[data-testid="stSidebar"] [role="radiogroup"] label p{ font:700 1.02rem var(--display) !important; letter-spacing:.1px; color:#F3E8D2 !important; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover{ background:var(--aubergine-2); padding-left:3.3rem; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked), [data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"]{ background:var(--paper); border-left-color:var(--tomato); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{ color:var(--ink) !important; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked)::before{ color:var(--tomato); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:focus-visible){ outline:2px solid var(--saffron); outline-offset:2px; }
.fg-scope{ margin-top:1.8rem; border:1px dashed #6a5570; padding:.9rem 1rem; font-size:.8rem; line-height:1.55; color:#E3D3E3 !important; }
.fg-scope b{ display:block; font:500 .66rem var(--mono); letter-spacing:1.5px; text-transform:uppercase; color:var(--saffron) !important; margin-bottom:.35rem; }
.fg-scope s{ color:#9d88a2 !important; }

/* ---------- Typography blocks ---------- */
.fg-strip{ display:flex; justify-content:space-between; align-items:center; border-top:3px solid var(--ink); border-bottom:1px solid var(--ink);
  padding:.38rem 0; margin-bottom:1.6rem; font:500 .7rem var(--mono); letter-spacing:1.4px; text-transform:uppercase; color:var(--ink); }
.fg-strip b{ color:var(--tomato); font-weight:500; }
.fg-kicker{ font:500 .72rem var(--mono); letter-spacing:1.6px; text-transform:uppercase; color:var(--tomato-d); margin-bottom:.7rem; }
.fg-h1{ font:900 clamp(2.3rem,4.4vw,3.9rem)/1.0 var(--display); letter-spacing:-1.8px; color:var(--ink); margin:0 0 1.1rem; }
.fg-h1 em{ font-style:italic; font-weight:600; color:var(--tomato); }
.fg-h2{ font:800 clamp(1.7rem,2.6vw,2.3rem)/1.05 var(--display); letter-spacing:-1px; color:var(--ink); margin:.1rem 0 .6rem; }
.fg-lede{ font-size:1.06rem; line-height:1.65; color:var(--ink-soft); max-width:34rem; margin-bottom:1.3rem; }
.fg-small{ font-size:.85rem; line-height:1.6; color:var(--ink-soft); }
.fg-mono{ font:500 .7rem var(--mono); letter-spacing:1.3px; text-transform:uppercase; color:var(--ink-soft); }

/* ---------- Ingredient-label panel ---------- */
.fg-label{ background:var(--paper-3); border:3px solid var(--ink); padding:1rem 1.15rem 1.1rem; box-shadow:7px 7px 0 var(--saffron); color:var(--ink); }
.fg-label h3{ font:900 1.9rem/1 var(--display); letter-spacing:-.8px; margin:0; padding-bottom:.45rem; border-bottom:8px solid var(--ink); }
.fg-label .row{ display:flex; justify-content:space-between; gap:1rem; padding:.5rem 0; border-bottom:1px solid var(--ink); font-size:.92rem; }
.fg-label .row.thick{ border-bottom:4px solid var(--ink); }
.fg-label .row span:first-child{ font-weight:700; }
.fg-label .row span:last-child{ text-align:right; color:var(--ink-soft); }
.fg-label .yes{ color:var(--sage) !important; font-weight:700; }
.fg-label .no{ color:#8a7a8c !important; text-decoration:line-through; text-decoration-thickness:1px; }
.fg-label .foot{ font-size:.72rem; line-height:1.5; color:var(--ink-soft); padding-top:.55rem; }

/* ---------- Contrasting band ---------- */
.fg-band{ background:var(--aubergine); color:#F3E8D2; padding:2rem 2.2rem 2.1rem; margin:2.2rem 0 0; }
.fg-band .fg-kicker{ color:var(--saffron); }
.fg-band .fg-h2{ color:#FFF3DE; max-width:40rem; }
.fg-band .cols{ display:grid; grid-template-columns:repeat(3,1fr); margin-top:1.4rem; border-top:1px solid #6a5570; }
.fg-band .col{ padding:1.1rem 1.3rem 0 0; }
.fg-band .col + .col{ padding-left:1.3rem; border-left:1px solid #6a5570; }
.fg-band .col b{ display:block; font:700 1.15rem var(--display); color:var(--saffron); margin-bottom:.35rem; }
.fg-band .col p{ font-size:.9rem; line-height:1.6; color:#E3D3E3; margin:0; }
@media(max-width:820px){ .fg-band .cols{ grid-template-columns:1fr; } .fg-band .col + .col{ border-left:0; padding-left:0; border-top:1px solid #6a5570; margin-top:1rem; } .block-container{ padding:1rem 1rem 2rem !important; } }

/* ---------- Steps ---------- */
.fg-steps{ display:flex; gap:0; margin:0 0 1.4rem; border:2px solid var(--ink); }
.fg-steps div{ flex:1; padding:.65rem .9rem; font-size:.86rem; font-weight:600; border-right:2px solid var(--ink); background:var(--paper-3); }
.fg-steps div:last-child{ border-right:0; }
.fg-steps div.on{ background:var(--ink); color:#FFF3DE; }
.fg-steps small{ display:block; font:500 .66rem var(--mono); letter-spacing:1.3px; color:var(--tomato-d); }
.fg-steps div.on small{ color:var(--saffron); }

.fg-note{ background:#F4DEA7; border-left:6px solid var(--saffron); padding:.8rem 1rem; font-size:.88rem; line-height:1.6; color:#432F10; margin:.4rem 0 1rem; }
.fg-warn{ background:#F6D9CE; border-left:6px solid var(--tomato); padding:.7rem 1rem; font-size:.88rem; line-height:1.55; color:#5A1D12; margin:.4rem 0; }

/* ---------- Result ---------- */
.fg-result{ border:3px solid var(--ink); padding:1.3rem 1.5rem 1.4rem; margin:.4rem 0 1rem; box-shadow:7px 7px 0 var(--ink); }
.fg-result.ok{ background:#E2E6C6; } .fg-result.flag{ background:#F6CFC1; }
.fg-result .tag{ display:inline-block; font:500 .68rem var(--mono); letter-spacing:1.5px; text-transform:uppercase; padding:.25rem .55rem; background:var(--ink); color:#FFF3DE; margin-bottom:.8rem; }
.fg-result h3{ font:900 clamp(1.7rem,3vw,2.5rem)/1.05 var(--display); letter-spacing:-1px; margin:0 0 .6rem; color:var(--ink); }
.fg-result p{ font-size:1rem; line-height:1.65; color:#3A2B3C; margin:0 0 .3rem; }

/* ---------- Streamlit widgets ---------- */
[data-testid="stWidgetLabel"] p, .stApp label p{ font-weight:600 !important; color:var(--ink) !important; font-size:.9rem; }
[data-testid="stNumberInput"] input, [data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-testid="stSelectbox"] [data-baseweb="select"] > div{ background:var(--paper-3) !important; color:var(--ink) !important; border-color:var(--ink) !important; border-radius:2px !important; font-family:var(--mono); }
[data-testid="stNumberInput"] > div > div{ border-radius:2px !important; border-color:var(--ink) !important; background:var(--paper-3) !important; }
[data-testid="stNumberInput"] button{ background:var(--paper-2) !important; color:var(--ink) !important; border-color:var(--ink) !important; }
[data-baseweb="select"] *{ color:var(--ink) !important; }
[data-baseweb="popover"] *{ color:var(--ink) !important; }
[data-baseweb="popover"] ul{ background:var(--paper-3) !important; }
[data-testid="stForm"]{ background:transparent; border:2px solid var(--ink); border-radius:0; padding:1.1rem 1.3rem 1.3rem; }
[data-testid="stTabs"] [data-baseweb="tab-list"]{ gap:0; border-bottom:2px solid var(--ink); }
[data-testid="stTabs"] button[role="tab"]{ background:transparent !important; border-radius:0; padding:.7rem 1.1rem; font-weight:700; color:var(--ink-soft) !important; }
[data-testid="stTabs"] button[role="tab"] p{ font:700 .95rem var(--display) !important; color:inherit !important; }
[data-testid="stTabs"] button[aria-selected="true"]{ background:var(--ink) !important; color:#FFF3DE !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{ background:var(--tomato) !important; height:4px !important; }
[data-testid="stTabs"] [data-baseweb="tab-border"]{ display:none; }

.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button{
  background:var(--tomato) !important; color:#FFF6E6 !important; border:2px solid var(--ink) !important; border-radius:0 !important;
  font:700 1rem var(--body) !important; letter-spacing:.2px; min-height:3rem; padding:.55rem 1.4rem;
  box-shadow:4px 4px 0 var(--ink); transition:transform .12s ease, box-shadow .12s ease, background .12s ease; }
.stButton > button p, .stFormSubmitButton > button p, .stDownloadButton > button p{ color:#FFF6E6 !important; font-weight:700; }
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover{ background:var(--tomato-d) !important; transform:translate(-2px,-2px); box-shadow:6px 6px 0 var(--ink); }
.stButton > button:active, .stFormSubmitButton > button:active{ transform:translate(3px,3px); box-shadow:1px 1px 0 var(--ink); }
.stButton > button:focus-visible, .stFormSubmitButton > button:focus-visible{ outline:3px solid var(--saffron) !important; outline-offset:3px; }
.fg-ghost .stButton > button, .st-key-reset_values button, .st-key-home_guide button, .st-key-guide_open_checker button{ background:var(--paper-3) !important; color:var(--ink) !important; box-shadow:3px 3px 0 var(--ink); min-height:2.5rem; }
.fg-ghost .stButton > button p, .st-key-reset_values button p, .st-key-home_guide button p, .st-key-guide_open_checker button p{ color:var(--ink) !important; }
.fg-ghost .stButton > button:hover, .st-key-reset_values button:hover, .st-key-home_guide button:hover, .st-key-guide_open_checker button:hover{ background:var(--saffron) !important; }
[data-testid="stForm"] .stFormSubmitButton > button{ width:100%; font-size:1.1rem; min-height:3.4rem; }

[data-testid="stExpander"]{ border:2px solid var(--ink) !important; border-radius:0 !important; background:var(--paper-3); margin-bottom:.6rem; }
[data-testid="stExpander"] summary{ padding:.75rem 1rem; background:var(--paper-3) !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] div, [data-testid="stSelectbox"] [data-baseweb="select"] > div{ background:var(--paper-3) !important; background-color:var(--paper-3) !important; }
[data-testid="stExpander"] summary:hover{ background:var(--saffron); }
[data-testid="stExpander"] summary p{ font:700 1.02rem var(--display) !important; color:var(--ink) !important; }
[data-testid="stExpander"] summary svg{ color:var(--ink) !important; }
[data-testid="stExpander"] details > div{ color:var(--ink); }
table.fg-table{ width:100%; border-collapse:collapse; font-size:.9rem; background:var(--paper-3); border:2px solid var(--ink); }
table.fg-table th{ text-align:left; font:500 .68rem var(--mono); letter-spacing:1.3px; text-transform:uppercase; padding:.55rem .8rem; background:var(--ink); color:#FFF3DE; }
table.fg-table td{ padding:.5rem .8rem; border-bottom:1px solid var(--rule); color:var(--ink); }
table.fg-table td:last-child{ font-family:var(--mono); text-align:right; }
.st-key-cta_strip{ background:var(--saffron) !important; border:3px solid var(--ink) !important; border-radius:0 !important; padding:.6rem 1rem; margin-top:2rem; }
.fg-g{ display:grid; grid-template-columns:3rem 1fr; gap:.2rem .6rem; padding:.55rem 0; border-bottom:1px solid var(--rule); font-size:.95rem; line-height:1.55; }
.fg-g:last-child{ border-bottom:0; }
.fg-g .n{ font:500 .8rem var(--mono); color:var(--tomato-d); padding-top:.15rem; }
[data-testid="stMarkdownContainer"] a{ color:var(--tomato-d); }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Sidebar navigation (persistent)
# ----------------------------------------------------------------------------
with st.sidebar:
    html("""
    <div class="fg-brand"><div class="fg-mark">FG</div>
    <div><div class="fg-name">Food<i>Guard</i></div><div class="fg-sub">Food adulteration detection</div></div></div>
    <div class="fg-navlabel">Contents</div>
    """)
    page = st.radio("Navigation", PAGES, key="sidebar_navigation", label_visibility="collapsed")


def masthead(number: str, title: str) -> None:
    html(f'<div class="fg-strip"><span>FoodGuard <b>/</b> Food adulteration detection</span><span>Nº {number} — {title}</span></div>')


# ----------------------------------------------------------------------------
# Home
# ----------------------------------------------------------------------------
if page == "Home":
    masthead("01", "Home")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        html("""
        <div class="fg-kicker">Know what is in your food</div>
        <div class="fg-h1">The label says one thing. <em>The lab</em> may say another.</div>
        <div class="fg-lede">Food adulteration — added water, undeclared substances, cheaper ingredients passed off as the real thing — is rarely visible to the eye.
        FoodGuard turns laboratory measurements into a clear first screening result you can understand and act on.</div>
        """)
        if st.button("Check Milk  →", key="hero_check"):
            go_to("Check Milk")
            st.rerun()
        html('<div class="fg-small" style="margin-top:.9rem;">Milk screening is available now. More foods are not supported yet.</div>')
    with right:
        html("""
        <div class="fg-label"><h3>Sample check</h3>
        <div class="row thick"><span>Serving size</span><span>One milk sample</span></div>
        <div class="row"><span>Input</span><span>18 lab measurements</span></div>
        <div class="row"><span>Output</span><span>Pure / Potentially adulterated</span></div>
        <div class="row"><span>Milk</span><span class="yes">Available</span></div>
        <div class="foot">* A screening result is not a laboratory certificate. Confirm important concerns with a validated food test.</div></div>
        """)

    html("""
    <div class="fg-band"><div class="fg-kicker">Why quality checks matter</div>
    <div class="fg-h2">You cannot judge adulteration by colour, smell or taste alone.</div>
    <div class="cols">
    <div class="col"><b>Added water</b><p>Dilution changes a product's composition. Measurements can show what the eye cannot.</p></div>
    <div class="col"><b>Undeclared substances</b><p>Substances that are not on the label cannot be reliably identified by appearance.</p></div>
    <div class="col"><b>Handling &amp; storage</b><p>Poor handling or storage can affect quality and safety even when nothing was added.</p></div>
    </div></div>
    """)

    with st.container(border=True, key="cta_strip"):
        c1, c2 = st.columns([2.2, 1], gap="large", vertical_alignment="center")
        with c1:
            html("""
            <div class="fg-mono" style="color:#2A1E2C;">Milk Safety Guide</div>
            <div class="fg-h2" style="font-size:1.55rem;margin:.2rem 0 .3rem;">Labels, storage, and when to ask a lab.</div>
            <div class="fg-small" style="color:#2A1E2C;">A short consumer guide, plus plain-language notes on the 18 measurements FoodGuard uses.</div>
            """)
        with c2:
            st.markdown('<div class="fg-ghost">', unsafe_allow_html=True)
            if st.button("Read the guide  →", key="home_guide"):
                go_to("Milk Safety Guide")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Check Milk
# ----------------------------------------------------------------------------
elif page == "Check Milk":
    masthead("02", "Check Milk")
    html("""
    <div class="fg-kicker">Milk screening</div>
    <div class="fg-h1" style="font-size:clamp(2rem,3.4vw,3rem);">Check a milk sample.</div>
    <div class="fg-lede" style="margin-bottom:1rem;">Enter the 18 measurements from a laboratory report. The saved FoodGuard model returns a screening result for that one sample. Photographs cannot be used.</div>
    """)
    has_result = "last_prediction" in st.session_state
    html(f"""
    <div class="fg-steps">
    <div class="{'' if has_result else 'on'}"><small>STEP 01</small>Enter measurements</div>
    <div><small>STEP 02</small>Analyse the sample</div>
    <div class="{'on' if has_result else ''}"><small>STEP 03</small>Read the result</div></div>
    """)
    if model_error:
        st.error(model_error)
    html('<div class="fg-note"><b>Before you begin:</b> use actual measured values from a laboratory report and check the units. The pre-filled values are only an example for trying the form — they are not a real test.</div>')

    st.markdown('<div class="fg-ghost">', unsafe_allow_html=True)
    ra, rb = st.columns([1, 3])
    with ra:
        if st.button("↺  Reset example values", key="reset_values"):
            reset_sample()
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    with st.form("milk_screening_form", clear_on_submit=False):
        tab1, tab2, tab3 = st.tabs(["1 · Composition", "2 · Physical properties", "3 · Lab tests"])
        with tab1:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                milk_type = st.selectbox("Milk type", list(MILK_MAPPING.keys()), key="milk_type")
                fat = st.number_input("Fat (%)", min_value=0.0, max_value=100.0, step=0.1, key="fat")
                snf = st.number_input("SNF (%)", min_value=0.0, max_value=100.0, step=0.1, key="snf", help="Solids-not-fat: the milk solids other than fat.")
                protein = st.number_input("Protein (%)", min_value=0.0, max_value=100.0, step=0.1, key="protein")
            with c2:
                lactose = st.number_input("Lactose (%)", min_value=0.0, max_value=100.0, step=0.1, key="lactose")
                moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, step=0.1, key="moisture")
                ash = st.number_input("Ash (%)", min_value=0.0, max_value=100.0, step=0.1, key="ash", help="Mineral residue left after burning the sample.")
                total_solids = st.number_input("Total solids (%)", min_value=0.0, max_value=100.0, step=0.1, key="total_solids")
        with tab2:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1, key="ph")
                gravity = st.number_input("Specific gravity", min_value=0.0, max_value=5.0, step=0.001, format="%.3f", key="gravity")
                acidity = st.number_input("Titratable acidity (%)", min_value=0.0, max_value=100.0, step=0.01, key="acidity")
            with c2:
                viscosity = st.number_input("Viscosity (cP)", min_value=0.0, max_value=100000.0, step=0.1, key="viscosity")
                freezing = st.number_input("Freezing point (°C)", min_value=-100.0, max_value=100.0, step=0.01, key="freezing")
                conductivity = st.number_input("Conductivity (mS/cm)", min_value=0.0, max_value=100000.0, step=0.1, key="conductivity")
        with tab3:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                urea = st.number_input("Urea (mg/100 mL)", min_value=0.0, max_value=100000.0, step=0.5, key="urea")
                melamine = st.number_input("Melamine (ppm)", min_value=0.0, max_value=100000.0, step=0.1, key="melamine")
            with c2:
                mbrt = st.number_input("MBRT (hours)", min_value=0.0, max_value=1000.0, step=0.5, key="mbrt", help="Methylene blue reduction time.")
                peroxidase = st.selectbox("Peroxidase activity", [0, 1], format_func=lambda val: "Absent (0)" if val == 0 else "Present (1)", key="peroxidase")
            html('<div class="fg-small" style="margin-top:.6rem;">Check the units on your report carefully. Do not guess values.</div>')
        st.markdown('<div class="fg-wide">', unsafe_allow_html=True)
        submitted = st.form_submit_button("Analyse milk sample  →")
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        values = {
            "milk_type": milk_type, "fat": fat, "snf": snf, "protein": protein,
            "lactose": lactose, "moisture": moisture, "ash": ash,
            "total_solids": total_solids, "ph": ph, "gravity": gravity,
            "acidity": acidity, "viscosity": viscosity, "freezing": freezing,
            "conductivity": conductivity, "urea": urea, "melamine": melamine,
            "mbrt": mbrt, "peroxidase": peroxidase,
        }
        numeric = [v for k, v in values.items() if k not in ("milk_type", "peroxidase")]
        if model is None or scaler is None:
            st.error("The checker is unavailable until the model and scaler files are placed beside app.py.")
        elif all(v == 0 for v in numeric):
            st.error("Every measurement is zero. Please enter the values from your laboratory report.")
        else:
            # No formula-based consistency warnings; use entered measurements for prediction.
            st.session_state.pop("validation_hints", None)
            try:
                with st.spinner("Analysing sample…"):
                    input_frame = make_input_frame(values)
                    scaled = scaler.transform(input_frame)
                    pred = int(model.predict(scaled)[0])
                st.session_state["last_prediction"] = {
                    "prediction": pred,
                    "input_frame": input_frame.copy(),
                    "input_values": values.copy(),
                }
                st.rerun()
            except Exception as exc:
                st.error(f"The sample could not be analysed: {exc}")

    result = st.session_state.get("last_prediction")
    if result:
        html('<div class="fg-kicker" style="margin-top:2rem;">Result</div>')
        flagged = result["prediction"] != 0
        if flagged:
            html(f"""
            <div class="fg-result flag"><span class="tag">Screening result</span>
            <h3>Potential adulteration flagged.</h3>
            <p>The model classified this sample as <b>Potentially adulterated</b>. This is a screening result, not proof.</p>
            </div>
            """)
        else:
            html(f"""
            <div class="fg-result ok"><span class="tag">Screening result</span>
            <h3>No adulteration flagged.</h3>
            <p>The model classified this sample as <b>Pure</b>. That means the model found no pattern it associates with adulteration — it does not certify the milk as safe.</p>
            </div>
            """)

        with st.expander("What does this result mean?", expanded=True):
            if flagged:
                st.markdown("The saved classifier found that these measurements resemble samples it learned to label as adulterated. Typing errors or wrong units can also produce a flag, so use the values and units from the laboratory report.")
            else:
                st.markdown("The saved classifier found that these measurements resemble samples it learned to label as pure. A *pure* label can still miss problems the model was not built to detect, such as contamination or spoilage.")
        with st.expander("Suggested next steps"):
            if flagged:
                st.markdown("- Re-check each value and unit against the laboratory report.\n- Keep the original packaging and note the batch and date.\n- Avoid relying on this result alone; arrange a validated laboratory test for the product.\n- Read the Milk Safety Guide for precautions.")
            else:
                st.markdown("- Keep milk stored as the label directs.\n- If you still have a concern about the product, arrange a validated laboratory test.\n- Read the Milk Safety Guide for what the measurements mean.")

        values = result["input_values"]
        rows = [("Milk type", values["milk_type"]), ("Fat (%)", values["fat"]), ("Protein (%)", values["protein"]),
                ("SNF (%)", values["snf"]), ("pH", values["ph"]), ("Specific gravity", values["gravity"]),
                ("Urea (mg/100 mL)", values["urea"]), ("Melamine (ppm)", values["melamine"])]
        body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
        with st.expander("Sample summary"):
            html(f'<table class="fg-table"><tr><th>Measurement</th><th style="text-align:right">Entered</th></tr>{body}</table>')
        export = result["input_frame"].copy()
        export["Predicted_Class"] = "Pure" if result["prediction"] == 0 else "Potentially adulterated"
        st.download_button("Download result summary (CSV)", data=export.to_csv(index=False).encode("utf-8"),
                           file_name="foodguard_milk_screening.csv", mime="text/csv")
        html('<div class="fg-note" style="margin-top:1rem;"><b>Caution:</b> this is an initial model-based screening result, not a certificate of purity and not laboratory confirmation. For important concerns, use an appropriate validated laboratory test.</div>')

# ----------------------------------------------------------------------------
# Milk Safety Guide
# ----------------------------------------------------------------------------
elif page == "Milk Safety Guide":
    masthead("03", "Milk Safety Guide")
    gl, gr = st.columns([1.2, 0.8], gap="large", vertical_alignment="bottom")
    with gl:
        html("""
        <div class="fg-kicker">Consumer guide</div>
        <div class="fg-h1" style="font-size:clamp(2rem,3.4vw,3rem);">Know what to check, and when to ask a lab.</div>
        <div class="fg-lede">Practical information about milk quality, possible adulteration concerns, and responsible next steps.</div>
        """)
    with gr:
        st.markdown('<div class="fg-ghost">', unsafe_allow_html=True)
        if st.button("Open the milk checker  →", key="guide_open_checker"):
            go_to("Check Milk")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        html('<div style="height:1.2rem"></div>')

    with st.expander("Concerns worth knowing about", expanded=True):
        html("""
        <div class="fg-g"><span class="n">01</span><span><b>Added water</b> can change milk composition.</span></div>
        <div class="fg-g"><span class="n">02</span><span><b>Undeclared substances</b> may be present and cannot be reliably identified by appearance alone.</span></div>
        <div class="fg-g"><span class="n">03</span><span><b>Poor handling or storage</b> can affect milk quality and safety.</span></div>
        """)
    with st.expander("Practical precautions", expanded=True):
        html("""
        <div class="fg-g"><span class="n">01</span><span>Check the seal, label, date, and storage instructions before use.</span></div>
        <div class="fg-g"><span class="n">02</span><span>Buy from a reliable supplier and keep milk refrigerated as directed.</span></div>
        <div class="fg-g"><span class="n">03</span><span>Do not rely on home tricks, colour, or smell alone to confirm purity.</span></div>
        <div class="fg-g"><span class="n">04</span><span>For a serious concern, keep the packaging and arrange a suitable laboratory test.</span></div>
        """)
    with st.expander("What the 18 measurements are"):
        html("""
        <div class="fg-g"><span class="n">A</span><span><b>Composition:</b> milk type, fat, SNF (solids-not-fat), protein, lactose, moisture, ash and total solids.</span></div>
        <div class="fg-g"><span class="n">B</span><span><b>Physical properties:</b> pH, specific gravity, titratable acidity, viscosity, freezing point and conductivity.</span></div>
        <div class="fg-g"><span class="n">C</span><span><b>Lab tests:</b> urea, melamine, MBRT (methylene blue reduction time) and peroxidase activity.</span></div>
        <div class="fg-g"><span class="n"> </span><span class="fg-small">Values come from laboratory testing. They cannot be read from a photograph or judged by eye.</span></div>
        """)
    with st.expander("How to read a FoodGuard result"):
        html("""
        <div class="fg-g"><span class="n">01</span><span>The model returns one of two classes: <b>Pure</b> or <b>Potentially adulterated</b>.</span></div>
        <div class="fg-g"><span class="n">02</span><span>It is a screening signal. It does not certify milk as safe or pure, and it does not replace a validated food-testing method.</span></div>
        <div class="fg-g"><span class="n">03</span><span>Mistyped values or wrong units can change the result, so re-check inputs before acting on a flag.</span></div>
        """)

    html('<div class="fg-note" style="margin-top:1rem;">FoodGuard provides an initial screening result from laboratory measurements. It cannot certify milk as safe or pure, and a result should not replace a validated food-testing method.</div>')