import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show_feature_importance(model, feature_cols):
    st.subheader("📊 Feature Importance Analysis")

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    })

    # ❌ Remove Is_Recent if present
    importance_df = importance_df[importance_df["Feature"] != "Is_Recent"]

    # Sort after removing
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance for Livability Prediction")

    st.pyplot(fig)

    # Show table
    st.dataframe(importance_df)
