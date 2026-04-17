import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def run_simulation(latest, model, feature_cols, city_name):

    st.markdown("### 🎛️ Policy Simulation — What-If Analysis")
    st.markdown(
        "<p style='color:#7b80a8;font-size:.92rem;margin-top:-.4rem'>"
        "Adjust city indicators below and instantly see how livability changes.</p>",
        unsafe_allow_html=True
    )

    sim_data = latest.copy()

    # ── Current baseline ──────────────────────────────────────
    baseline_score = model.predict(
        pd.DataFrame([latest])[feature_cols]
    )[0]

    # ── Sliders in two columns ─────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-weight:700;"
            "color:#ffb347;font-size:.85rem;text-transform:uppercase;"
            "letter-spacing:.07em;margin-bottom:.8rem'>✅ Positive Indicators</div>",
            unsafe_allow_html=True
        )

        sim_data["Literacy_Rate_%"] = st.slider(
            "📚 Literacy Rate (%)",
            50.0, 100.0, float(latest["Literacy_Rate_%"]), step=0.5,
            help="Higher literacy drives better employment and civic participation"
        )

        sim_data["Internet_Penetration_%"] = st.slider(
            "🌐 Internet Penetration (%)",
            10.0, 100.0, float(latest["Internet_Penetration_%"]), step=0.5,
            help="Digital access correlates strongly with economic opportunity"
        )

        sim_data["Electricity_Access_%"] = st.slider(
            "⚡ Electricity Access (%)",
            50.0, 100.0, float(latest["Electricity_Access_%"]), step=0.5
        )

        sim_data["Toilet_Access_%"] = st.slider(
            "🚿 Toilet Access (%)",
            20.0, 100.0, float(latest["Toilet_Access_%"]), step=0.5
        )

        sim_data["Hospitals_per_100k"] = st.slider(
            "🏥 Hospitals per 100k",
            1.0, 30.0, float(latest["Hospitals_per_100k"]), step=0.1
        )

    with col_b:
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-weight:700;"
            "color:#e05a5a;font-size:.85rem;text-transform:uppercase;"
            "letter-spacing:.07em;margin-bottom:.8rem'>⚠️ Negative Indicators (Lower = Better)</div>",
            unsafe_allow_html=True
        )

        sim_data["Average_AQI"] = st.slider(
            "💨 Average AQI",
            20.0, 400.0, float(latest["Average_AQI"]), step=1.0,
            help="Lower AQI = cleaner air = better health outcomes"
        )

        sim_data["Crime_Rate_per_100k"] = st.slider(
            "🔒 Crime Rate per 100k",
            0.0, 1000.0, float(latest["Crime_Rate_per_100k"]), step=1.0
        )

        sim_data["Unemployment_Rate_%"] = st.slider(
            "📉 Unemployment Rate (%)",
            0.0, 30.0, float(latest["Unemployment_Rate_%"]), step=0.1
        )

        sim_data["PMAY_Average_Housing_Cost_Lakhs"] = st.slider(
            "🏠 Avg Housing Cost (Lakhs)",
            5.0, 150.0, float(latest["PMAY_Average_Housing_Cost_Lakhs"]), step=0.5
        )

        sim_data["Population_Growth_%"] = st.slider(
            "👥 Population Growth (%)",
            0.0, 5.0, float(latest["Population_Growth_%"]), step=0.05
        )

    # ── Predict ───────────────────────────────────────────────
    sim_X     = pd.DataFrame([sim_data])[feature_cols]
    sim_score = model.predict(sim_X)[0]
    delta     = sim_score - baseline_score

    # ── Result panel ──────────────────────────────────────────
    st.markdown("<hr/>", unsafe_allow_html=True)

    def get_badge(score):
        if score < 45:   return "score-low",    "#e05a5a", "Low"
        elif score < 55: return "score-medium",  "#f9a825", "Medium"
        else:            return "score-high",    "#4caf97",  "High"

    base_css, base_color, base_label = get_badge(baseline_score)
    sim_css,  sim_color,  sim_label  = get_badge(sim_score)

    delta_sign  = "+" if delta >= 0 else ""
    delta_color = "#4caf97" if delta >= 0 else "#e05a5a"
    delta_arrow = "▲" if delta >= 0 else "▼"

    res1, res2, res3 = st.columns(3, gap="medium")

    with res1:
        st.markdown(f"""
        <div style='background:#1c1f33;border:1px solid #2a2e4a;border-radius:14px;
                    padding:1.4rem;text-align:center;'>
            <div style='font-size:.75rem;color:#7b80a8;text-transform:uppercase;
                        letter-spacing:.08em;margin-bottom:.7rem'>Baseline Score</div>
            <span class='score-badge {base_css}' style='font-size:1.8rem;padding:.4rem 1.2rem'>
                {baseline_score:.1f}</span>
            <div style='margin-top:.6rem;color:{base_color};font-family:Syne,sans-serif;
                        font-weight:700;font-size:.95rem'>{base_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with res2:
        st.markdown(f"""
        <div style='background:#1c1f33;border:1px solid #2a2e4a;border-radius:14px;
                    padding:1.4rem;text-align:center;'>
            <div style='font-size:.75rem;color:#7b80a8;text-transform:uppercase;
                        letter-spacing:.08em;margin-bottom:.7rem'>Simulated Score</div>
            <span class='score-badge {sim_css}' style='font-size:1.8rem;padding:.4rem 1.2rem'>
                {sim_score:.1f}</span>
            <div style='margin-top:.6rem;color:{sim_color};font-family:Syne,sans-serif;
                        font-weight:700;font-size:.95rem'>{sim_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with res3:
        st.markdown(f"""
        <div style='background:#1c1f33;border:1px solid #2a2e4a;border-radius:14px;
                    padding:1.4rem;text-align:center;'>
            <div style='font-size:.75rem;color:#7b80a8;text-transform:uppercase;
                        letter-spacing:.08em;margin-bottom:.7rem'>Change</div>
            <div style='font-family:Syne,sans-serif;font-weight:800;font-size:2rem;
                        color:{delta_color}'>{delta_arrow} {delta_sign}{delta:.2f}</div>
            <div style='margin-top:.6rem;color:#7b80a8;font-size:.82rem'>points vs baseline</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Comparison bar chart ───────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 2.4))
    fig.patch.set_facecolor('#1c1f33')
    ax.set_facecolor('#1c1f33')

    bar_vals  = [baseline_score, sim_score]
    bar_lbls  = ["Baseline", "Simulated"]
    bar_colors= [base_color, sim_color]

    bars = ax.barh(bar_lbls, bar_vals, color=bar_colors, height=.45, edgecolor='none')
    for bar, val in zip(bars, bar_vals):
        ax.text(bar.get_width() + .5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va='center', ha='left', color='#e8eaf6', fontsize=11,
                fontweight='bold')

    ax.set_xlim(0, 100)
    ax.set_xlabel("Livability Score", color='#7b80a8', fontsize=10)
    ax.tick_params(colors='#7b80a8')
    for sp in ax.spines.values(): sp.set_edgecolor('#2a2e4a')
    ax.grid(axis='x', color='#2a2e4a', linestyle='--', linewidth=.6)
    plt.tight_layout()
    st.pyplot(fig)

    # ── Contextual message ────────────────────────────────────
    if abs(delta) < 0.5:
        st.info(f"ℹ️ Minimal change detected for **{city_name}**. Try adjusting more indicators to see a significant impact.")
    elif delta > 0:
        st.success(f"✅ Great! These policy changes would improve **{city_name}**'s livability score by **{delta:.2f} points**.")
    else:
        st.warning(f"⚠️ These changes would reduce **{city_name}**'s livability score by **{abs(delta):.2f} points**.")

    return sim_score