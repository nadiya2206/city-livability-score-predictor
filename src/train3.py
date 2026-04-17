import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def show_feature_importance(model, feature_cols):

    st.markdown("### 📊 Feature Importance Analysis")
    st.markdown(
        "<p style='color:#7b80a8;font-size:.92rem;margin-top:-.4rem'>"
        "Relative influence of each indicator on the predicted livability score.</p>",
        unsafe_allow_html=True
    )

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    })

    importance_df = importance_df[importance_df["Feature"] != "Is_Recent"]
    importance_df = importance_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    importance_df["Rank"] = range(1, len(importance_df) + 1)

    # ── Summary KPIs ────────────────────────────────────────────
    top_feature  = importance_df.iloc[0]["Feature"].replace("_", " ").replace("%", "").strip()
    top_val      = importance_df.iloc[0]["Importance"]
    top5_share   = importance_df.head(5)["Importance"].sum() * 100

    k1, k2, k3 = st.columns(3, gap="medium")
    with k1:
        st.metric("Most Influential Feature", top_feature.title())
    with k2:
        st.metric("Top Feature Score", f"{top_val:.4f}")
    with k3:
        st.metric("Top 5 Features Explain", f"{top5_share:.1f}%")

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Horizontal bar chart ──────────────────────────────────
    n = len(importance_df)
    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.42)))
    fig.patch.set_facecolor('#1c1f33')
    ax.set_facecolor('#1c1f33')

    # Gradient from accent → teal based on rank
    cmap   = plt.cm.get_cmap('YlOrRd')
    norm_v = importance_df["Importance"].values / importance_df["Importance"].max()
    colors = [cmap(0.35 + 0.55 * v) for v in norm_v[::-1]]

    feats  = importance_df["Feature"].str.replace("_", " ").values
    vals   = importance_df["Importance"].values

    bars = ax.barh(feats[::-1], vals[::-1], color=colors, height=0.62, edgecolor='none')

    for bar, val in zip(bars, vals[::-1]):
        ax.text(
            bar.get_width() + importance_df["Importance"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va='center', ha='left', color='#e8eaf6', fontsize=8.5
        )

    ax.set_xlabel("Importance Score", color='#7b80a8', fontsize=10)
    ax.tick_params(colors='#7b80a8', labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor('#2a2e4a')
    ax.grid(axis='x', color='#2a2e4a', linestyle='--', linewidth=0.6)
    ax.set_title("Feature Importance — Random Forest Model",
                 color='#e8eaf6', fontsize=13, fontweight='bold', pad=14)

    plt.tight_layout()
    st.pyplot(fig)

    # ── Pie / donut chart for top 6 ───────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    col_pie, col_tbl = st.columns([2, 3], gap="large")

    with col_pie:
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-weight:700;"
            "font-size:.85rem;color:#ffb347;text-transform:uppercase;"
            "letter-spacing:.07em;margin-bottom:.6rem'>Top 6 Share</div>",
            unsafe_allow_html=True
        )
        top6    = importance_df.head(6)
        others  = importance_df.iloc[6:]["Importance"].sum()
        pie_df  = pd.concat([
            top6,
            pd.DataFrame([{"Feature": "Others", "Importance": others, "Rank": 7}])
        ], ignore_index=True)

        pie_colors = ['#ff6b1a','#ffb347','#00d4c8','#4caf97','#7b80a8','#e05a5a','#2a2e4a']
        fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
        fig2.patch.set_facecolor('#1c1f33')
        ax2.set_facecolor('#1c1f33')

        wedges, texts, autotexts = ax2.pie(
            pie_df["Importance"],
            labels=None,
            autopct='%1.1f%%',
            pctdistance=0.78,
            colors=pie_colors,
            startangle=140,
            wedgeprops=dict(width=0.55, edgecolor='#1c1f33', linewidth=2)
        )
        for at in autotexts:
            at.set_color('#e8eaf6'); at.set_fontsize(8)

        ax2.legend(
            wedges,
            [f.replace("_", " ") for f in pie_df["Feature"]],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            facecolor='#1c1f33',
            edgecolor='#2a2e4a',
            labelcolor='#e8eaf6',
            fontsize=7.5
        )
        plt.tight_layout()
        st.pyplot(fig2)

    with col_tbl:
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-weight:700;"
            "font-size:.85rem;color:#ffb347;text-transform:uppercase;"
            "letter-spacing:.07em;margin-bottom:.6rem'>Full Feature Table</div>",
            unsafe_allow_html=True
        )
        disp = importance_df[["Rank", "Feature", "Importance"]].copy()
        disp["Feature"]    = disp["Feature"].str.replace("_", " ")
        disp["Importance"] = disp["Importance"].round(5)
        st.dataframe(disp, use_container_width=True, hide_index=True)