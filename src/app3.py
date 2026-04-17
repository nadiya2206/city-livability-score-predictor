import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from simulation import run_simulation
from feature_importance import show_feature_importance

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="CityScore India",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# =========================
# Custom CSS — Dark Indigo + Saffron theme
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,400&display=swap');

:root {
    --bg:      #0b0d1a;
    --surface: #111320;
    --card:    #181b2e;
    --card2:   #1e2238;
    --border:  #272b46;
    --accent:  #ff6b1a;
    --accent2: #ffb347;
    --teal:    #00d4c8;
    --text:    #e8eaf6;
    --muted:   #6b7194;
    --success: #3ecf8e;
    --warn:    #f9a825;
    --danger:  #e05a5a;
    --radius:  12px;
}

/* ── Force dark background everywhere ── */
html, body { background: var(--bg) !important; }
.stApp, .stApp > div, .main, .main > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="block-container"],
.block-container,
section[data-testid="stMain"] > div,
div[data-testid="stVerticalBlock"] {
    background: var(--bg) !important;
    background-color: var(--bg) !important;
}

/* ── Typography ── */
*, *::before, *::after {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text);
    box-sizing: border-box;
}
h1, h2, h3, h4, h5 { font-family: 'Syne', sans-serif !important; }

h1 {
    font-weight: 800 !important;
    font-size: 1.9rem !important;
    line-height: 1.2 !important;
    background: linear-gradient(100deg, var(--accent) 0%, var(--accent2) 50%, var(--teal) 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: .1rem !important;
}
h2 { font-size: 1.3rem !important; font-weight: 700 !important; color: var(--text) !important; }
h3 { font-size: 1.05rem !important; font-weight: 700 !important; color: var(--text) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] section {
    background: var(--surface) !important;
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Top toolbar / header ── */
[data-testid="stHeader"],
header[data-testid="stHeader"] {
    background: var(--bg) !important;
    border-bottom: 1px solid var(--border) !important;
}

/* ── Hide only the decoration text strip ── */
[data-testid="stDecoration"] { display: none !important; }
#MainMenu { visibility: hidden !important; }
footer    { display: none !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.2rem !important;
    transition: transform .18s, box-shadow .18s;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 24px rgba(255,107,26,.15) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: .72rem !important;
    text-transform: uppercase;
    letter-spacing: .07em;
}
[data-testid="stMetricValue"] {
    color: var(--accent2) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.65rem !important;
    font-weight: 700 !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] label { color: var(--muted) !important; font-size:.85rem !important; }
[data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
[data-testid="stSelectbox"] svg { fill: var(--accent) !important; }

/* ── Selectbox dropdown popup ── */
[data-baseweb="popover"],
[data-baseweb="popover"] *,
[data-baseweb="menu"],
[data-baseweb="menu"] *,
ul[data-baseweb="menu"],
ul[data-baseweb="menu"] li,
div[data-baseweb="popover"] > div,
div[role="listbox"],
div[role="listbox"] * {
    background-color: #1e2238 !important;
    color: #e8eaf6 !important;
}
ul[data-baseweb="menu"] li:hover,
div[role="option"]:hover,
[data-baseweb="menu"] [aria-selected="true"] {
    background-color: #2a2e4a !important;
    color: #ffb347 !important;
}
/* selected item highlight */
[data-baseweb="menu"] li[aria-selected="true"] {
    background-color: rgba(255,107,26,.18) !important;
    color: #ffb347 !important;
}

/* ── Slider ── */
[data-testid="stSlider"] label { color: var(--muted) !important; font-size:.82rem !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #0b0d1a !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    transition: opacity .18s, transform .18s;
}
.stButton > button:hover { opacity: .85; transform: translateY(-2px); }

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stDataFrame"] * { color: var(--text) !important; }

/* ── Alert boxes ── */
[data-testid="stAlert"],
div[class*="stSuccess"], div[class*="stInfo"], div[class*="stWarning"], div[class*="stError"] {
    border-radius: 10px !important;
}
div[class*="stSuccess"] { background: rgba(62,207,142,.12) !important; border-left: 3px solid var(--success) !important; }
div[class*="stInfo"]    { background: rgba(0,212,200,.10)  !important; border-left: 3px solid var(--teal)    !important; }
div[class*="stWarning"] { background: rgba(249,168,37,.12) !important; border-left: 3px solid var(--warn)    !important; }

/* ── Radio ── */
[data-testid="stRadio"] > div { gap: 4px !important; }
[data-testid="stRadio"] label {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: .5rem 1rem !important;
    cursor: pointer;
    transition: border-color .15s, background .15s;
    width: 100%;
}
[data-testid="stRadio"] label:hover { border-color: var(--accent); background: var(--card2); }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; opacity: 1 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Score badge ── */
.score-badge {
    display: inline-block;
    padding: .5rem 1.6rem;
    border-radius: 50px;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    letter-spacing: -1px;
}
.score-high   { background: linear-gradient(135deg,#3ecf8e,#00d4c8); color:#0b0d1a; }
.score-medium { background: linear-gradient(135deg,#f9a825,#ffb347); color:#0b0d1a; }
.score-low    { background: linear-gradient(135deg,#e05a5a,#ff6b1a); color:#fff; }

/* ── Progress bar ── */
.prog-wrap { margin: .4rem 0 1rem; }
.prog-label { display:flex; justify-content:space-between; font-size:.78rem; color:var(--muted); margin-bottom:.28rem; }
.prog-bg { background: var(--border); border-radius: 99px; height: 7px; overflow: hidden; }
.prog-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,var(--accent),var(--accent2)); }

/* ── Section header pill ── */
.section-pill {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    background: rgba(255,107,26,.1);
    border: 1px solid rgba(255,107,26,.25);
    border-radius: 99px;
    padding: .3rem .9rem;
    font-family: 'Syne', sans-serif;
    font-size: .8rem;
    font-weight: 700;
    color: var(--accent2);
    text-transform: uppercase;
    letter-spacing: .07em;
    margin-bottom: .7rem;
}

/* ── Info card ── */
.info-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.3rem;
}
.info-card-accent { border-left: 3px solid var(--accent); }

/* ── Streamlit pyplot background ── */
[data-testid="stImage"] img { border-radius: var(--radius); }
</style>
""", unsafe_allow_html=True)

# =========================
# Load data & model
# =========================
df = pd.read_csv("../data/made3.csv").dropna()
model = joblib.load("../models/livability_model_v3.pkl")
feature_cols = joblib.load("../models/feature_columns_v3.pkl")

drop_cols = ["IPC_Crime_Cases", "MSME_Registered_Units", "Population_Density_per_sqkm"]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

positive_cols = [
    'Literacy_Rate_%','Internet_Penetration_%','Electricity_Access_%',
    'Hospitals_per_100k','Beds_per_1000','Toilet_Access_%',
    'Road_Density_km_per_1000sqkm','Higher_Education_Institutions'
]
negative_cols = [
    'Unemployment_Rate_%','Average_AQI','Crime_Rate_per_100k',
    'Student_Teacher_Ratio','PMAY_Average_Housing_Cost_Lakhs','Population_Growth_%'
]

scaler = MinMaxScaler()
df_pos = pd.DataFrame(scaler.fit_transform(df[positive_cols]), columns=positive_cols)
df_neg = pd.DataFrame(scaler.fit_transform(df[negative_cols]), columns=negative_cols)
df_neg = 1 - df_neg
df["Livability_Index"] = (df_pos.mean(axis=1) * 0.6 + df_neg.mean(axis=1) * 0.4) * 100

X_all = df[feature_cols]
df["Predicted_Livability"] = model.predict(X_all)

def classify(score):
    if score < 45:   return "Low"
    elif score < 55: return "Medium"
    else:            return "High"

def classify_color(score):
    if score < 45:   return "#e05a5a"
    elif score < 55: return "#f9a825"
    else:            return "#4caf97"

def classify_css(score):
    if score < 45:   return "score-low"
    elif score < 55: return "score-medium"
    else:            return "score-high"

df["Class"] = df["Predicted_Livability"].apply(classify)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.2rem;border-bottom:1px solid #272b46;margin-bottom:1.2rem'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                    background:linear-gradient(90deg,#ff6b1a,#ffb347);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;letter-spacing:-.3px'>
            🏙️ CityScore India
        </div>
        <div style='font-size:.7rem;color:#6b7194;margin-top:.2rem;
                    letter-spacing:.1em;text-transform:uppercase'>
            ML Livability Predictor
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_items = [
        ("🏠 Main Dashboard",     "🏠", "Main Dashboard"),
        ("🏆 City Rankings",      "🏆", "City Rankings"),
        ("📊 Feature Importance", "📊", "Feature Importance"),
        ("🎛️ Policy Simulation",  "🎛️", "Policy Simulation"),
    ]

    menu = st.radio(
        "Navigate",
        ["🏠 Main Dashboard", "🏆 City Rankings", "📊 Feature Importance", "🎛️ Policy Simulation"],
        label_visibility="collapsed"
    )

    # Style active vs inactive nav items
    st.markdown(f"""
    <style>
    /* Hide default radio buttons */
    [data-testid="stRadio"] > div {{
        gap: 0 !important;
    }}
    [data-testid="stRadio"] [data-baseweb="radio"] {{
        display: none !important;
    }}
    [data-testid="stRadio"] label {{
        background: transparent !important;
        border: none !important;
        border-radius: 10px !important;
        padding: .62rem 1rem .62rem 1.1rem !important;
        margin: 2px 0 !important;
        cursor: pointer;
        font-family: 'Syne', sans-serif !important;
        font-size: .88rem !important;
        font-weight: 600 !important;
        color: #6b7194 !important;
        transition: all .15s ease !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        letter-spacing: .01em;
        border-left: 3px solid transparent !important;
    }}
    [data-testid="stRadio"] label:hover {{
        background: rgba(255,107,26,.08) !important;
        color: #e8eaf6 !important;
        border-left: 3px solid rgba(255,107,26,.4) !important;
    }}
    /* Active item — the checked label */
    [data-testid="stRadio"] label[data-checked="true"],
    [data-testid="stRadio"] label:has(input:checked) {{
        background: linear-gradient(90deg, rgba(255,107,26,.18), rgba(255,179,71,.06)) !important;
        color: #ffb347 !important;
        border-left: 3px solid #ff6b1a !important;
    }}
    /* Nav section label */
    [data-testid="stRadio"] > label {{
        display: none !important;
    }}
    </style>

    <div style='font-size:.68rem;color:#6b7194;text-transform:uppercase;
                letter-spacing:.1em;font-weight:700;padding:.2rem .2rem .5rem;
                margin-bottom:.2rem'>
        Navigation
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:1.4rem;padding:1rem;background:#181b2e;
                border:1px solid #272b46;border-radius:12px'>
        <div style='font-family:Syne,sans-serif;font-size:.68rem;font-weight:700;
                    color:#ffb347;text-transform:uppercase;letter-spacing:.1em;
                    margin-bottom:.8rem'>⚙️ Model Info</div>
        <div style='font-size:.75rem;color:#6b7194;line-height:2'>
            <span style='color:#e8eaf6;font-weight:600'>Algorithm</span><br/>
            <span style='color:#ffb347'>Random Forest</span><br/>
            <span style='color:#e8eaf6;font-weight:600'>Features</span><br/>
            <span style='color:#ffb347'>15 indicators</span><br/>
            <span style='color:#e8eaf6;font-weight:600'>Training period</span><br/>
            <span style='color:#ffb347'>2015 – 2024</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)
    with mc1: st.metric("R²", "0.71")
    with mc2: st.metric("MAE", "3.19")

# =========================
# HEADER
# =========================
hcol1, hcol2 = st.columns([5, 1])
with hcol1:
    st.markdown("""
    <div style='padding:.2rem 0 .4rem'>
        <h1>🏙️ Indian City Livability Predictor</h1>
        <p style='color:#6b7194;font-size:.88rem;margin:.1rem 0 0'>
            Machine-learning powered insights into urban quality of life across Indian cities
        </p>
    </div>
    """, unsafe_allow_html=True)
with hcol2:
    st.markdown("""
    <div style='text-align:right;padding-top:.6rem'>
        <span style='background:rgba(62,207,142,.12);border:1px solid rgba(62,207,142,.3);
                     border-radius:99px;padding:.25rem .75rem;font-size:.72rem;
                     font-family:Syne,sans-serif;font-weight:700;color:#3ecf8e;
                     text-transform:uppercase;letter-spacing:.06em'>R² 0.71</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin:.4rem 0 1rem'/>", unsafe_allow_html=True)

# =========================
# CITY SELECTOR + KPI ROW
# =========================
sel_col, _, __ = st.columns([2, 2, 2])
with sel_col:
    city = st.selectbox("🔍 Select a City", sorted(df["City"].unique()),
                        help="Choose any Indian city to analyse",
                        label_visibility="collapsed")

city_data    = df[df["City"] == city].sort_values("Year")
latest       = city_data.iloc[-1]
latest_score = model.predict(pd.DataFrame([latest])[feature_cols])[0]
label        = classify(latest_score)
label_color  = classify_color(latest_score)
badge_css    = classify_css(latest_score)

# Styled city pill
st.markdown(f"""
<div style='display:flex;align-items:center;gap:.6rem;margin:.2rem 0 .9rem'>
    <span style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#e8eaf6'>
        {city}
    </span>
    <span class='score-badge {badge_css}' style='font-size:.85rem;padding:.15rem .75rem'>
        {latest_score:.1f}
    </span>
    <span style='font-size:.78rem;color:{label_color};font-weight:600'>{label} Livability</span>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("Livability Score",    f"{latest_score:.1f}")
with k2: st.metric("Literacy Rate",       f"{latest['Literacy_Rate_%']:.1f}%")
with k3: st.metric("Avg AQI",             f"{latest['Average_AQI']:.0f}")
with k4: st.metric("Crime Rate /100k",    f"{latest['Crime_Rate_per_100k']:.1f}")
with k5: st.metric("Internet Access",     f"{latest['Internet_Penetration_%']:.1f}%")

st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#   PAGE ROUTING
# ─────────────────────────────────────────────

# ── MAIN DASHBOARD ──────────────────────────
if menu == "🏠 Main Dashboard":

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("### 📈 Livability Trend Over Time")

        fig, ax = plt.subplots(figsize=(8, 3.8))
        fig.patch.set_facecolor('#1c1f33')
        ax.set_facecolor('#1c1f33')

        years  = city_data["Year"].values
        values = city_data["Livability_Index"].values

        ax.fill_between(years, values, alpha=.18, color='#ff6b1a')
        ax.plot(years, values, color='#ff6b1a', linewidth=2.5, marker='o',
                markersize=7, markerfacecolor='#ffb347', markeredgecolor='#ff6b1a')

        ax.set_xlabel("Year", color='#7b80a8', fontsize=10)
        ax.set_ylabel("Livability Index", color='#7b80a8', fontsize=10)
        ax.tick_params(colors='#7b80a8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2e4a')
        ax.grid(axis='y', color='#2a2e4a', linestyle='--', linewidth=.6)
        ax.set_title(f"{city} — Historical Livability", color='#e8eaf6',
                     fontsize=13, fontweight='bold', pad=12)
        plt.tight_layout()
        st.pyplot(fig)

    with right:
        st.markdown("### 🟢 Current Classification")

        st.markdown(f"""
        <div style='text-align:center;padding:1.4rem 0;'>
            <div style='font-size:.8rem;color:#7b80a8;text-transform:uppercase;
                        letter-spacing:.1em;margin-bottom:.6rem'>2024 Predicted Score</div>
            <span class='score-badge {badge_css}'>{latest_score:.1f}</span>
            <div style='margin-top:.9rem;font-family:Syne,sans-serif;font-size:1.1rem;
                        font-weight:700;color:{label_color}'>{label} Livability</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:.5rem'>", unsafe_allow_html=True)
        for feat, val, maxv, positive in [
            ("Literacy Rate",       latest['Literacy_Rate_%'],          100, True),
            ("Internet Access",     latest['Internet_Penetration_%'],   100, True),
            ("Electricity Access",  latest['Electricity_Access_%'],     100, True),
            ("Toilet Access",       latest['Toilet_Access_%'],          100, True),
        ]:
            pct = (val / maxv) * 100
            st.markdown(f"""
            <div class='prog-wrap'>
                <div class='prog-label'><span>{feat}</span><span>{val:.1f}%</span></div>
                <div class='prog-bg'><div class='prog-fill' style='width:{pct:.0f}%'></div></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### 🔮 Future Livability Forecast (2025–2030)")

    future_years = [2025, 2026, 2027, 2028, 2029, 2030]
    future_rows  = []
    for yr in future_years:
        row = latest.copy(); row["Year"] = yr
        future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    future_df["Predicted_Livability"] = model.predict(future_df[feature_cols])
    future_df["Class"] = future_df["Predicted_Livability"].apply(classify)

    # Forecast chart
    fig2, ax2 = plt.subplots(figsize=(9, 3.2))
    fig2.patch.set_facecolor('#1c1f33')
    ax2.set_facecolor('#1c1f33')

    fy = future_df["Year"].values
    fv = future_df["Predicted_Livability"].values

    ax2.fill_between(fy, fv, alpha=.15, color='#00d4c8')
    ax2.plot(fy, fv, color='#00d4c8', linewidth=2.5, marker='D',
             markersize=7, markerfacecolor='#00d4c8')

    for x, y in zip(fy, fv):
        ax2.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', color='#e8eaf6', fontsize=8.5)

    ax2.set_xlabel("Year", color='#7b80a8', fontsize=10)
    ax2.set_ylabel("Predicted Score", color='#7b80a8', fontsize=10)
    ax2.tick_params(colors='#7b80a8')
    for sp in ax2.spines.values(): sp.set_edgecolor('#2a2e4a')
    ax2.grid(axis='y', color='#2a2e4a', linestyle='--', linewidth=.6)
    ax2.set_title(f"{city} — Forecast", color='#e8eaf6', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    st.pyplot(fig2)

    # Forecast table with styled output
    disp = future_df[["Year", "Predicted_Livability", "Class"]].copy()
    disp.columns = ["Year", "Predicted Score", "Category"]
    disp["Predicted Score"] = disp["Predicted Score"].round(2)
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ── RANKING PAGE ─────────────────────────────
elif menu == "🏆 City Rankings":

    st.markdown("### 🏆 City Livability Rankings")

    ranking = (
        df.groupby("City")["Predicted_Livability"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    ranking.columns = ["City", "Avg Predicted Score"]
    ranking["Rank"]  = range(1, len(ranking) + 1)
    ranking["Category"] = ranking["Avg Predicted Score"].apply(classify)
    ranking["Avg Predicted Score"] = ranking["Avg Predicted Score"].round(2)
    ranking = ranking[["Rank", "City", "Avg Predicted Score", "Category"]]

    top3 = ranking.head(3)
    medals = ["🥇", "🥈", "🥉"]
    medal_cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        with medal_cols[i]:
            badge = classify_css(row["Avg Predicted Score"])
            st.markdown(f"""
            <div style='background:#1c1f33;border:1px solid #2a2e4a;border-radius:14px;
                        padding:1.3rem;text-align:center;'>
                <div style='font-size:2rem'>{medals[i]}</div>
                <div style='font-family:Syne,sans-serif;font-weight:700;font-size:1.05rem;
                            margin:.4rem 0;color:#e8eaf6'>{row["City"]}</div>
                <span class='score-badge {badge}' style='font-size:1.4rem;padding:.35rem 1.1rem'>
                    {row["Avg Predicted Score"]}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Bar chart
    top_n = 20
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fig3.patch.set_facecolor('#1c1f33')
    ax3.set_facecolor('#1c1f33')

    top_cities = ranking.head(top_n)
    colors = [classify_color(s) for s in top_cities["Avg Predicted Score"]]

    bars = ax3.barh(top_cities["City"][::-1], top_cities["Avg Predicted Score"][::-1],
                    color=colors[::-1], height=.65, edgecolor='none')

    for bar, val in zip(bars, top_cities["Avg Predicted Score"][::-1]):
        ax3.text(bar.get_width() + .3, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}", va='center', ha='left', color='#e8eaf6', fontsize=8.5)

    ax3.set_xlabel("Avg Predicted Livability Score", color='#7b80a8', fontsize=10)
    ax3.tick_params(colors='#7b80a8')
    for sp in ax3.spines.values(): sp.set_edgecolor('#2a2e4a')
    ax3.grid(axis='x', color='#2a2e4a', linestyle='--', linewidth=.6)
    ax3.set_title(f"Top {top_n} Cities by Livability", color='#e8eaf6',
                  fontsize=13, fontweight='bold', pad=12)

    legend_patches = [
        mpatches.Patch(color='#4caf97', label='High (≥55)'),
        mpatches.Patch(color='#f9a825', label='Medium (45–55)'),
        mpatches.Patch(color='#e05a5a', label='Low (<45)'),
    ]
    ax3.legend(handles=legend_patches, loc='lower right',
               facecolor='#1c1f33', edgecolor='#2a2e4a', labelcolor='#e8eaf6', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.dataframe(ranking, use_container_width=True, hide_index=True)

# ── FEATURE IMPORTANCE ───────────────────────
elif menu == "📊 Feature Importance":
    show_feature_importance(model, feature_cols)

# ── POLICY SIMULATION ────────────────────────
elif menu == "🎛️ Policy Simulation":
    latest_row = city_data.iloc[-1]
    run_simulation(latest_row, model, feature_cols, city)