# streamlit_app.py
# Rail Fleet Explorer — built for "LEM Data.xlsx"

import os
import io
from typing import Optional, Iterable, List, Any

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Rail Fleet Explorer", layout="wide")

# -----------------------------
# Utils
# -----------------------------
def decade_midpoint(bin_label: str) -> Optional[float]:
    """Convert a 'YYYY-YYYY' decade bin to its midpoint year."""
    if not isinstance(bin_label, str):
        return None
    parts = bin_label.split("-")
    if len(parts) != 2:
        return None
    try:
        start = int(parts[0])
        end = int(parts[1])
        return (start + end) / 2
    except Exception:
        return None

def safe_sorted_unique(values: Iterable[Any]) -> List[str]:
    """Return unique values sorted lexicographically as strings."""
    if values is None:
        return []
    ser = pd.Series(values).dropna()
    return sorted(ser.astype(str).unique().tolist(), key=lambda v: v.lower())

@st.cache_data(show_spinner=False)
def load_data(fallback_path="LEM Data.xlsx"):
    if os.path.exists(fallback_path):
        df = pd.read_excel(fallback_path, sheet_name=0)
    else:
        st.error("File **LEM Data.xlsx** not found. Place it alongside this script.")
        st.stop()

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Normalize expected columns
    rename_map = {
        "Reporting year": "Reporting year",
        "Appedix number": "Appendix number",
        "Type of locomotive": "Type of locomotive",
        "US EPA Tier Level": "US EPA Tier Level",
        "Year of Manufacture": "Year of Manufacture",
        "Engine": "Engine",
        "hp": "hp",
        "OEM": "OEM",
        "Model": "Model",
        "Class 1": "Class 1",
        "Regional": "Regional",
        "Shortlines": "Shortlines",
        "Intercity Rail": "Intercity Rail",
        "Commuter": "Commuter",
        "Tourist & Excursion": "Tourist & Excursion",
    }
    df = df.rename(columns=rename_map)

    # Coerce numeric columns where appropriate
    for col in ["hp","Reporting year","Class 1","Regional","Shortlines","Intercity Rail","Commuter","Tourist & Excursion"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Year midpoint + age (static reference year 2025)
    if "Year of Manufacture" in df.columns:
        df["Manufacture Mid Year"] = df["Year of Manufacture"].apply(decade_midpoint)
        rep_year = 2025
        df["Est. Age (yrs)"] = rep_year - df["Manufacture Mid Year"]

        # Remove negatives (future mid-years / bad parses) and future mid-years
        df.loc[df["Est. Age (yrs)"] < 0, "Est. Age (yrs)"] = np.nan
        df.loc[df["Manufacture Mid Year"] > rep_year, "Manufacture Mid Year"] = np.nan
    else:
        df["Manufacture Mid Year"] = np.nan
        df["Est. Age (yrs)"] = np.nan

    # Melt operator categories to long form
    operator_cols = [c for c in ["Class 1","Regional","Shortlines","Intercity Rail","Commuter","Tourist & Excursion"] if c in df.columns]
    if operator_cols:
        melted = df.melt(
            id_vars=[c for c in df.columns if c not in operator_cols],
            value_vars=operator_cols,
            var_name="Operator Type",
            value_name="Count",
        )
        melted = melted[pd.to_numeric(melted["Count"], errors="coerce").fillna(0) > 0]
    else:
        melted = df.copy()
        melted["Operator Type"] = "Unknown"
        melted["Count"] = 1.0

    return df, melted

def apply_filters(dd: pd.DataFrame) -> pd.DataFrame:
    out = dd.copy()
    if "OEM" in out.columns and sel_oem:
        out = out[out["OEM"].astype(str).isin(sel_oem)]
    if "Model" in out.columns and sel_model:
        out = out[out["Model"].astype(str).isin(sel_model)]
    if "US EPA Tier Level" in out.columns and sel_tier:
        out = out[out["US EPA Tier Level"].astype(str).isin(sel_tier)]
    if "Operator Type" in out.columns and sel_ops:
        out = out[out["Operator Type"].astype(str).isin(sel_ops)]
    if year_range and "Manufacture Mid Year" in out.columns:
        out = out[out["Manufacture Mid Year"].between(year_range[0], year_range[1], inclusive="both")]
    if age_range and "Est. Age (yrs)" in out.columns:
        out = out[out["Est. Age (yrs)"].between(age_range[0], age_range[1], inclusive="both")]
    if hp_range and "hp" in out.columns:
        out = out[out["hp"].between(hp_range[0], hp_range[1], inclusive="both")]
    return out

def _weighted_mean(series, weights):
    s = pd.to_numeric(series, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0).clip(lower=0)
    mask = (~s.isna()) & (w > 0)
    if mask.any():
        return float(np.average(s[mask], weights=w[mask]))
    return np.nan

# -----------------------------
# UI — Title & Sidebar
# -----------------------------
st.title("🚆 Rail Fleet Explorer")
st.caption("Interactive explorer for locomotive data (US EPA Tier, horsepower, age, operator classes). Age referenced to year **2025**.")

df_raw, df = load_data()

with st.sidebar.expander("🔎 Filters", expanded=True):
    oems  = safe_sorted_unique(df["OEM"]) if "OEM" in df.columns else []
    models = safe_sorted_unique(df["Model"]) if "Model" in df.columns else []
    tiers = safe_sorted_unique(df["US EPA Tier Level"]) if "US EPA Tier Level" in df.columns else []
    ops   = safe_sorted_unique(df["Operator Type"]) if "Operator Type" in df.columns else []

    sel_oem = st.multiselect("OEM", oems, default=[])
    sel_model = st.multiselect("Model", models, default=[])
    sel_tier = st.multiselect("US EPA Tier Level", tiers, default=[])
    sel_ops = st.multiselect("Operator Type", ops, default=ops if ops else [])

    # Manufacture Mid Year range
    if "Manufacture Mid Year" in df.columns and df["Manufacture Mid Year"].notna().any():
        y_min = int(np.nanmin(df["Manufacture Mid Year"]))
        y_max = int(np.nanmax(df["Manufacture Mid Year"]))
        year_range = st.slider("Manufacture Mid Year range", min_value=y_min, max_value=y_max, value=(y_min, y_max))
    else:
        year_range = None

    # Estimated Age (yrs) range — min always 0
    if "Est. Age (yrs)" in df.columns and df["Est. Age (yrs)"].notna().any():
        a_max = int(np.ceil(np.nanmax(df["Est. Age (yrs)"])))
        if a_max == 0:
            a_max = 1
        age_range = st.slider("Estimated Age (yrs) range", min_value=0, max_value=a_max, value=(0, a_max))
    else:
        age_range = None

    # Horsepower range
    if "hp" in df.columns and df["hp"].notna().any():
        hp_min = int(np.nanmin(df["hp"]))
        hp_max = int(np.nanmax(df["hp"]))
        if hp_min == hp_max:
            hp_max += 1
        hp_range = st.slider("Horsepower (hp) range", min_value=hp_min, max_value=hp_max, value=(hp_min, hp_max))
    else:
        hp_range = None

# Sidebar footer note
st.sidebar.markdown(
    "<hr style='margin-top:8px; margin-bottom:8px; opacity:0.3;'>"
    "<div style='font-size:12px; opacity:0.7;'>Last updated by Victor Pham, August 2025</div>",
    unsafe_allow_html=True
)

df_f = apply_filters(df)

# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
total_units = int(df_f["Count"].sum()) if "Count" in df_f.columns else len(df_f)
avg_age = _weighted_mean(df_f.get("Est. Age (yrs)"), df_f.get("Count")) if len(df_f) else np.nan
avg_hp  = _weighted_mean(df_f.get("hp"), df_f.get("Count")) if len(df_f) else np.nan
unique_models = df_f["Model"].astype(str).nunique() if "Model" in df_f.columns else np.nan

col1.metric("Total Units", f"{total_units:,}")
col2.metric("Avg. Estimated Age (yrs)", f"{avg_age:,.1f}" if not np.isnan(avg_age) else "—")
col3.metric("Avg. Horsepower", f"{avg_hp:,.0f}" if not np.isnan(avg_hp) else "—")
col4.metric("Unique Models", f"{int(unique_models):,}" if not np.isnan(unique_models) else "—")

st.divider()

# -----------------------------
# Charts & Tables
# -----------------------------
tabs = st.tabs(["Tier Distribution", "Age Distribution", "Horsepower by Operator", "OEM / Model Breakdown", "Raw Data"])

with tabs[0]:
    st.subheader("US EPA Tier Distribution")
    if "US EPA Tier Level" in df_f.columns and "Count" in df_f.columns:
        temp = df_f.copy()
        temp["US EPA Tier Level"] = temp["US EPA Tier Level"].astype(str)
        tier_df = (
            temp.groupby("US EPA Tier Level", dropna=False)["Count"]
                .sum().reset_index().rename(columns={"Count": "Units"})
                .sort_values("Units", ascending=False)
        )
        if not tier_df.empty:
            chart = alt.Chart(tier_df).mark_bar().encode(
                x=alt.X("Units:Q", title="Units"),
                y=alt.Y("US EPA Tier Level:N", sort='-x', title="Tier"),
                tooltip=["US EPA Tier Level:N", "Units:Q"],
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No data after filters.")

with tabs[1]:
    st.subheader("Estimated Age Distribution")
    if "Est. Age (yrs)" in df_f.columns and "Count" in df_f.columns:
        ages = df_f.dropna(subset=["Est. Age (yrs)"]).copy()
        ages["Count"] = pd.to_numeric(ages["Count"], errors="coerce").fillna(0).astype(int).clip(lower=1)
        ages_rep = ages.loc[ages.index.repeat(ages["Count"])]
        if not ages_rep.empty:
            hist = alt.Chart(ages_rep).transform_bin(
                "age_bin", field="Est. Age (yrs)", bin=alt.Bin(maxbins=25)
            ).mark_bar().encode(
                x=alt.X("age_bin:Q", title="Estimated Age (yrs)", scale=alt.Scale(domainMin=0)),
                y=alt.Y("count()", title="Units"),
                tooltip=[alt.Tooltip("count()", title="Units")],
            )
            st.altair_chart(hist, use_container_width=True)
        else:
            st.info("No data after filters.")

with tabs[2]:
    st.subheader("Average Horsepower by Operator Type")
    if all(c in df_f.columns for c in ["hp", "Operator Type", "Count"]):
        tmp = df_f.copy()
        tmp["hp"] = pd.to_numeric(tmp["hp"], errors="coerce")
        tmp["w"] = pd.to_numeric(tmp["Count"], errors="coerce").fillna(0).clip(lower=0)
        hp_by_op = (
            tmp.groupby("Operator Type", dropna=False)
               .apply(lambda g: _weighted_mean(g["hp"], g["w"]))
               .reset_index(name="Avg hp")
               .sort_values("Avg hp", ascending=False, na_position="last")
        )
        if hp_by_op["Avg hp"].notna().any():
            chart = alt.Chart(hp_by_op.dropna()).mark_bar().encode(
                x=alt.X("Avg hp:Q", title="Average hp"),
                y=alt.Y("Operator Type:N", sort='-x'),
                tooltip=["Operator Type:N", alt.Tooltip("Avg hp:Q", format=".0f")],
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No horsepower data after filters.")

with tabs[3]:
    st.subheader("OEM / Model Breakdown")
    if "OEM" in df_f.columns and "Model" in df_f.columns and "Count" in df_f.columns:
        pivot = df_f.pivot_table(index="OEM", columns="Model", values="Count", aggfunc="sum", fill_value=0).astype(int)
        st.dataframe(pivot, use_container_width=True)
        st.caption("Values represent unit counts by OEM × Model after filters.")
    else:
        st.info("OEM or Model columns not found.")

with tabs[4]:
    st.subheader("Raw Data (after filters)")
    show_cols = [c for c in [
        "Reporting year","Type of locomotive","OEM","Model","US EPA Tier Level","Engine","hp",
        "Year of Manufacture","Manufacture Mid Year","Est. Age (yrs)","Operator Type","Count"
    ] if c in df_f.columns]
    st.dataframe(
        (df_f[show_cols] if show_cols else df_f).reset_index(drop=True),
        use_container_width=True, height=500
    )
    buf = io.BytesIO()
    df_f.to_excel(buf, index=False)
    st.download_button("Download filtered data (Excel)", data=buf.getvalue(),
                       file_name="rail_filtered.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.divider()
st.caption("Use the sidebar to filter by OEM, model, tier, operator type, manufacture-year range, estimated age (yrs), and horsepower. Age is computed as 2025 − Manufacture Mid Year.")
