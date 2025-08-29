# streamlit_app.py
# Canadian Rail Fleet Explorer — built for "LEM Data.xlsx"

import os
import io
from typing import Optional, Iterable, List, Any
import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Canadian Rail Fleet Explorer", layout="wide")

# -----------------------------
# Constants
# -----------------------------
REF_YEAR = 2025
TIER_ORDER = ["No Tier", "Tier 0", "Tier 0+", "Tier 1", "Tier 1+", "Tier 2", "Tier 2+", "Tier 3", "Tier 4"]
HP_BANDS = ["Low (<2000)", "Medium (2000–3000)", "High (>3000)"]
ALL_OPERATORS = ["Class 1", "Commuter", "Intercity Rail", "Regional", "Shortlines", "Tourist & Excursion"]

# -----------------------------
# Utils
# -----------------------------
_DASH_PATTERN = re.compile(r"[–—−]+")
_YEAR_PATTERN = re.compile(r"(\d{4})")

def parse_mid_year(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, (int, np.integer)):
        return float(val)
    s = str(val).strip()
    if not s:
        return np.nan
    s = _DASH_PATTERN.sub("-", s)
    years = [int(m.group(1)) for m in _YEAR_PATTERN.finditer(s)]
    if len(years) >= 2:
        y1, y2 = years[0], years[1]
        return (y1 + y2) / 2
    if len(years) == 1:
        return float(years[0])
    m = re.search(r"(\d{4})\s*-\s*(\d{2})", s)
    if m:
        y1 = int(m.group(1)); yy = int(m.group(2))
        y2 = (y1 // 100) * 100 + yy
        return (y1 + y2) / 2
    return np.nan

def safe_sorted_unique(values: Iterable[Any]) -> List[str]:
    if values is None:
        return []
    ser = pd.Series(values).dropna()
    return sorted(ser.astype(str).unique().tolist(), key=lambda v: v.lower())

def _weighted_mean(series, weights):
    s = pd.to_numeric(series, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0).clip(lower=0)
    mask = (~s.isna()) & (w > 0)
    if mask.any():
        return float(np.average(s[mask], weights=w[mask]))
    return np.nan

def _pie_chart(df, cat_col, val_col, donut=False, tier_order=None):
    if df.empty:
        return None
    d = df.copy()
    total = d[val_col].sum()
    d["Pct"] = d[val_col] / total if total else 0
    base = alt.Chart(d)
    arc = (
        base.mark_arc(innerRadius=60 if donut else 0, outerRadius=120)
        .encode(
            theta=alt.Theta(f"{val_col}:Q", stack=True),
            color=alt.Color(
                f"{cat_col}:N",
                legend=alt.Legend(title=cat_col),
                scale=alt.Scale(domain=tier_order) if tier_order else alt.Undefined,
            ),
            tooltip=[
                alt.Tooltip(f"{cat_col}:N", title=cat_col),
                alt.Tooltip(f"{val_col}:Q", title="Units", format=",.0f"),
                alt.Tooltip("Pct:Q", title="Share", format=".1%"),
            ],
            order=alt.Order(f"{cat_col}:N", sort="ascending"),
        )
    )
    labels = (
        base.mark_text(radius=135, radiusOffset=10, fontSize=12)
        .encode(theta=alt.Theta(f"{val_col}:Q", stack=True), text=alt.Text("Pct:Q", format=".1%"), color=alt.value("black"))
    )
    return arc + labels

def normalize_tier(x):
    if pd.isna(x):
        return x
    s = str(x).strip()
    s = s.replace("Plus", "+")
    s = re.sub(r"\s*\+\s*", "+", s)
    s = re.sub(r"(?i)^no\s*tier$", "No Tier", s)
    s = re.sub(r"(?i)^tier\s*([0-4])\+$", r"Tier \1+", s)
    s = re.sub(r"(?i)^tier\s*([0-4])\b$", r"Tier \1", s)
    return s

def hp_to_band(v: Any) -> Optional[str]:
    v = pd.to_numeric(v, errors="coerce")
    if pd.isna(v):
        return None
    if v < 2000:
        return HP_BANDS[0]
    if 2000 <= v <= 3000:
        return HP_BANDS[1]
    return HP_BANDS[2]

@st.cache_data(show_spinner=False)
def load_data(fallback_path="LEM Data.xlsx"):
    if os.path.exists(fallback_path):
        df = pd.read_excel(fallback_path, sheet_name=0)
    else:
        st.error("File **LEM Data.xlsx** not found. Place it alongside this script.")
        st.stop()

    df.columns = [str(c).strip() for c in df.columns]

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

    for col in ["hp","Reporting year","Class 1","Regional","Shortlines","Intercity Rail","Commuter","Tourist & Excursion"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Reporting year" in df.columns:
        df["Reporting year"] = pd.to_numeric(df["Reporting year"], errors="coerce").astype("Int64")

    if "US EPA Tier Level" in df.columns:
        df["US EPA Tier Level"] = df["US EPA Tier Level"].apply(normalize_tier)

    if "Year of Manufacture" in df.columns:
        df["Manufacture Mid Year"] = df["Year of Manufacture"].apply(parse_mid_year)
        df.loc[df["Manufacture Mid Year"] > REF_YEAR, "Manufacture Mid Year"] = np.nan
        df["Est. Age (yrs)"] = REF_YEAR - df["Manufacture Mid Year"]
        df.loc[df["Est. Age (yrs)"] < 0, "Est. Age (yrs)"] = np.nan
    else:
        df["Manufacture Mid Year"] = np.nan
        df["Est. Age (yrs)"] = np.nan

    op_cols = [c for c in ["Class 1","Regional","Shortlines","Intercity Rail","Commuter","Tourist & Excursion"] if c in df.columns]
    if op_cols:
        melted = df.melt(
            id_vars=[c for c in df.columns if c not in op_cols],
            value_vars=op_cols,
            var_name="Operator Type",
            value_name="Count",
        )
        melted["Count"] = pd.to_numeric(melted["Count"], errors="coerce").fillna(0)
        melted = melted[melted["Count"] > 0]
    else:
        melted = df.copy()
        melted["Operator Type"] = "Unknown"
        melted["Count"] = 1.0

    return df, melted

def apply_filters(dd: pd.DataFrame, *, include_missing_midyear: bool,
                  year_range, age_range, hp_range, include_missing_hp,
                  sel_oem, sel_model, sel_tier, sel_ops, sel_type) -> pd.DataFrame:
    out = dd.copy()
    if "OEM" in out.columns and sel_oem:
        out = out[out["OEM"].astype(str).isin(sel_oem)]
    if "Model" in out.columns and sel_model:
        out = out[out["Model"].astype(str).isin(sel_model)]
    if "US EPA Tier Level" in out.columns and sel_tier:
        out = out[out["US EPA Tier Level"].astype(str).isin(sel_tier)]
    if "Operator Type" in out.columns and sel_ops:
        out = out[out["Operator Type"].astype(str).isin(sel_ops)]
    if "Type of locomotive" in out.columns and sel_type:
        out = out[out["Type of locomotive"].astype(str).isin(sel_type)]
    if year_range and "Manufacture Mid Year" in out.columns:
        mm = out["Manufacture Mid Year"]
        mask = mm.between(year_range[0], year_range[1], inclusive="both")
        if include_missing_midyear:
            mask = mask | mm.isna()
        out = out[mask]
    if age_range and "Est. Age (yrs)" in out.columns:
        out = out[out["Est. Age (yrs)"].between(age_range[0], age_range[1], inclusive="both")]
    if hp_range and "hp" in out.columns:
        hp_series = pd.to_numeric(out["hp"], errors="coerce")
        mask = hp_series.between(hp_range[0], hp_range[1], inclusive="both")
        if include_missing_hp:
            mask = mask | hp_series.isna()
        out = out[mask]
    return out

def build_yoy(df_long: pd.DataFrame):
    if df_long.empty or "Reporting year" not in df_long.columns:
        return None, None, None

    d = df_long.copy()
    d["Year"] = d["Reporting year"].astype("Int64")

    totals = d.groupby("Year", dropna=True)["Count"].sum().reset_index(name="Units").dropna().sort_values("Year")
    totals["YoY Δ"] = totals["Units"].diff().fillna(0).astype(int)
    totals["YoY %"] = (totals["Units"].pct_change() * 100).fillna(0)

    tiers = d.copy()
    tiers["Tier"] = tiers["US EPA Tier Level"].astype(str)
    tiers = tiers.groupby(["Year","Tier"], dropna=False)["Count"].sum().reset_index()
    tiers["Tier"] = pd.Categorical(tiers["Tier"], categories=TIER_ORDER, ordered=True)

    ops = d.groupby(["Year","Operator Type"], dropna=False)["Count"].sum().reset_index()

    return totals, tiers, ops

def yoy_charts(totals, tiers, ops, yoy_metric: str):
    totals = totals[(totals["Year"].notna()) & (totals["Year"] >= 2010)]
    tiers = tiers[(tiers["Year"].notna()) & (tiers["Year"] >= 2010)]
    ops = ops[(ops["Year"].notna()) & (ops["Year"] >= 2010)]

    if yoy_metric == "Units":
        yfield = "Units"; title = "Total Units"; fmt = ",.0f"
    elif yoy_metric == "YoY change":
        yfield = "YoY Δ"; title = "Year-over-Year Change (Units)"; fmt = ",.0f"
    else:
        yfield = "YoY %"; title = "Year-over-Year Change (%)"; fmt = ",.1f"

    plot = totals.copy()
    line = (
        alt.Chart(plot)
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:O", title="Reporting year", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{yfield}:Q", title=title, axis=alt.Axis(format=fmt)),
            tooltip=[alt.Tooltip("Year:O"), alt.Tooltip(f"{yfield}:Q", format=fmt)]
        )
        .properties(width=500)
    )
    st.altair_chart(line, use_container_width=True)

    st.subheader("Tiers")
    present_tiers = [t for t in TIER_ORDER if t in tiers["Tier"].astype(str).unique().tolist()]
    sel_tiers = st.multiselect("Show tiers", present_tiers, default=present_tiers, key="yoy_tier_include")
    tiers_plot = tiers[tiers["Tier"].astype(str).isin(sel_tiers)]

    tier_view = st.radio("Display", ["Stacked area", "Line", "Stacked bars"], horizontal=True, key="tier_yoy_view")
    if tier_view == "Stacked area":
        chart = (
            alt.Chart(tiers_plot)
            .mark_area(opacity=0.85)
            .encode(
                x=alt.X("Year:O", title="Reporting year", axis=alt.Axis(labelAngle=0, labels=True)),
                y=alt.Y("sum(Count):Q", title="Units", axis=alt.Axis(format=",.0f")),
                color=alt.Color("Tier:N", scale=alt.Scale(domain=TIER_ORDER), legend=alt.Legend(title="Tier")),
                tooltip=[alt.Tooltip("Year:O"), alt.Tooltip("Tier:N"), alt.Tooltip("sum(Count):Q", title="Units", format=",.0f")]
            )
        )
    elif tier_view == "Line":
        chart = (
            alt.Chart(tiers_plot)
            .mark_line(point=True)
            .encode(
                x=alt.X("Year:O", title="Reporting year", axis=alt.Axis(labelAngle=0, labels=True)),
                y=alt.Y("Count:Q", title="Units", axis=alt.Axis(format=",.0f")),
                color=alt.Color("Tier:N", scale=alt.Scale(domain=TIER_ORDER), legend=alt.Legend(title="Tier")),
                tooltip=[alt.Tooltip("Year:O"), alt.Tooltip("Tier:N"), alt.Tooltip("Count:Q", title="Units", format=",.0f")]
            )
        )
    else:
        chart = (
            alt.Chart(tiers_plot)
            .mark_bar()
            .encode(
                x=alt.X("Year:O", title="Reporting year", axis=alt.Axis(labelAngle=0, labels=True)),
                y=alt.Y("sum(Count):Q", title="Units", axis=alt.Axis(format=",.0f")),
                color=alt.Color("Tier:N", scale=alt.Scale(domain=TIER_ORDER), legend=alt.Legend(title="Tier")),
                tooltip=[alt.Tooltip("Year:O"), alt.Tooltip("Tier:N"), alt.Tooltip("sum(Count):Q", title="Units", format=",.0f")]
            )
        )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Operator Types")
    present_ops = safe_sorted_unique(ops["Operator Type"])
    sel_ops2 = st.multiselect("Show operator types", present_ops, default=present_ops, key="yoy_ops_include")
    ops_plot = ops[ops["Operator Type"].astype(str).isin(sel_ops2)]

    ops_view = st.radio("Display", ["Stacked bars", "Line", "Stacked area"], horizontal=True, key="ops_yoy_view")
    if ops_view == "Stacked bars":
        ops_chart = (
            alt.Chart(ops_plot)
            .mark_bar()
            .encode(
                x=alt.X("Year:O", title="Reporting year", axis=alt.Axis(labelAngle=0, labels=True)),
                y=alt.Y("sum(Count):Q", title="Units", axis=alt.Axis(format=",.0f")),
                color=alt.Color("Operator Type:N", legend=alt.Legend(title="Operator")),
                tooltip=[alt.Tooltip("Year:O"), alt.Tooltip("Operator Type:N"), alt.Tooltip("sum(Count):Q", title="Units", format=",.0f")]
            )
        )
    elif ops_view == "Line":
        ops_chart = (
            alt.Chart(ops_plot)
            .mark_line(point=True)
            .encode(
                x=alt.X("Year:O", title="Reporting year", axis=alt.Axis(labelAngle=0, labels=True)),
                y=alt.Y("Count:Q", title="Units", axis=alt.Axis(format=",.0f")),
                color=alt.Color("Operator Type:N", legend=alt.Legend(title="Operator"))
            )
        )
    else:
        ops_chart = (
            alt.Chart(ops_plot)
            .mark_area(opacity=0.85)
            .encode(
                x=alt.X("Year:O", title="Reporting year", axis=alt.Axis(labelAngle=0, labels=True)),
                y=alt.Y("sum(Count):Q", title="Units", axis=alt.Axis(format=",.0f")),
                color=alt.Color("Operator Type:N", legend=alt.Legend(title="Operator"))
            )
        )
    st.altair_chart(ops_chart, use_container_width=True)

    with st.expander("YoY table"):
        t = plot.copy()
        t["Units"] = t["Units"].astype(int)
        t["YoY Δ"] = t["YoY Δ"].astype(int)
        t["YoY %"] = t["YoY %"].round(1)
        st.dataframe(t.set_index("Year"), use_container_width=True)

def calc_cagr(totals: pd.DataFrame) -> Optional[float]:
    if totals is None or totals.empty:
        return None
    t = totals[(totals["Year"].notna())].sort_values("Year")
    t = t[t["Year"] >= 2010]
    if t.empty:
        return None
    first, last = t.iloc[0], t.iloc[-1]
    years = int(last["Year"] - first["Year"])
    if years <= 0 or first["Units"] <= 0:
        return None
    return (float(last["Units"]) / float(first["Units"])) ** (1 / years) - 1.0

# -----------------------------
# UI — Title
# -----------------------------
st.title("🚆 Canadian Rail Fleet Explorer")
st.caption(f"Age references {REF_YEAR}. Manufacture-year slider capped at 2020.")

# -----------------------------
# Sidebar — Navigation + controls
# -----------------------------
with st.sidebar:
    st.markdown("### Navigation")
    yoy_mode = st.toggle("Annual Trends View", value=False, key="yoy_mode_sidebar")
    if yoy_mode:
        yoy_metric = st.selectbox("YoY series", ["Units", "YoY change", "YoY %"], index=0, key="yoy_metric_sidebar")
    else:
        yoy_metric = "Units"

# Load data
df_raw, df_all = load_data()

# -----------------------------
# Sidebar — Year selection and filters
# -----------------------------
with st.sidebar:
    years_all = sorted(pd.Series(df_all["Reporting year"]).dropna().astype(int).unique().tolist()) if "Reporting year" in df_all.columns else []

    if not yoy_mode:
        if years_all:
            sel_year = st.selectbox("Reporting year", years_all, index=0, format_func=lambda y: f"{y}")
            compare_mode = st.checkbox("Compare with another year", value=False)
            if compare_mode:
                years_second = [y for y in years_all if y != sel_year]
                sel_year2 = st.selectbox("Compare to", years_second, index=0, format_func=lambda y: f"{y}")
                chosen_years = [sel_year, sel_year2]
            else:
                sel_year2 = None
                chosen_years = [sel_year]
        else:
            sel_year = None
            sel_year2 = None
            compare_mode = False
            chosen_years = []
    else:
        compare_mode = False
        chosen_years = years_all

with st.sidebar.expander("🔎 Filters", expanded=True):
    base_for_filters = df_all if yoy_mode else df_all[df_all["Reporting year"].isin(chosen_years)] if chosen_years else df_all

    pairs = base_for_filters[["OEM", "Model"]].dropna().astype(str)
    all_oems = safe_sorted_unique(pairs["OEM"])
    all_models = safe_sorted_unique(pairs["Model"])

    current_models = st.session_state.get("sel_model", [])
    current_oems = st.session_state.get("sel_oem", [])

    avail_oems = safe_sorted_unique(pairs[pairs["Model"].isin(current_models)]["OEM"]) if current_models else all_oems
    if "sel_oem" in st.session_state:
        st.session_state.sel_oem = [o for o in st.session_state.sel_oem if o in avail_oems]
    sel_oem = st.multiselect("OEM", options=avail_oems, key="sel_oem")

    avail_models = safe_sorted_unique(pairs[pairs["OEM"].isin(sel_oem)]["Model"]) if sel_oem else all_models
    if "sel_model" in st.session_state:
        st.session_state.sel_model = [m for m in st.session_state.sel_model if m in avail_models]
    sel_model = st.multiselect("Model", options=avail_models, key="sel_model")

    tiers = safe_sorted_unique(base_for_filters["US EPA Tier Level"]) if "US EPA Tier Level" in base_for_filters.columns else []
    sel_tier = st.multiselect("US EPA Tier Level", tiers, default=[])

    # New: Type of locomotive filter
    types = safe_sorted_unique(base_for_filters["Type of locomotive"]) if "Type of locomotive" in base_for_filters.columns else []
    sel_type = st.multiselect("Type of locomotive", options=types, default=types)

    # Operators
    avail_ops = safe_sorted_unique(base_for_filters["Operator Type"]) if "Operator Type" in base_for_filters.columns else ALL_OPERATORS
    default_ops = avail_ops if avail_ops else ALL_OPERATORS
    sel_ops = st.multiselect("Operator Type", options=default_ops, default=default_ops)

    # Manufacture Mid Year range — hard cap max at 2020
    if "Manufacture Mid Year" in base_for_filters.columns:
        if base_for_filters["Manufacture Mid Year"].notna().any():
            y_min = int(np.floor(np.nanmin(base_for_filters["Manufacture Mid Year"])))
        else:
            y_min = 1900
        y_max = 2020
        if y_min > y_max:
            y_min = y_max
        year_range = st.slider(
            "Manufacture Mid Year range (midpoint)",
            min_value=y_min,
            max_value=y_max,
            value=(y_min, y_max)
        )
    else:
        year_range = None
    include_missing_midyear = st.checkbox("Include rows with unparsed manufacture year", value=True)

    # Estimated Age range
    if "Est. Age (yrs)" in base_for_filters.columns and base_for_filters["Est. Age (yrs)"].notna().any():
        a_max = int(np.ceil(np.nanmax(base_for_filters["Est. Age (yrs)"])))
        if a_max == 0:
            a_max = 1
        age_range = st.slider("Estimated Age (yrs) range", min_value=0, max_value=a_max, value=(0, a_max))
    else:
        age_range = None

    # Horsepower range
    if "hp" in base_for_filters.columns and base_for_filters["hp"].notna().any():
        hp_min = int(np.nanmin(base_for_filters["hp"]))
        hp_max = int(np.nanmax(base_for_filters["hp"]))
        if hp_min == hp_max:
            hp_max += 1
        hp_range = st.slider("Horsepower (hp) range", min_value=hp_min, max_value=hp_max, value=(hp_min, hp_max))
    else:
        hp_range = None
    include_missing_hp = st.checkbox("Include rows with missing horsepower", value=True)

st.sidebar.markdown(
    "<hr style='margin-top:8px; margin-bottom:8px; opacity:0.3;'>"
    "<div style='font-size:12px; opacity:0.7;'>Last updated by Victor Pham, August 2025</div>",
    unsafe_allow_html=True
)

# -----------------------------
# Data after filters
# -----------------------------
if yoy_mode:
    df_filtered_all_years = apply_filters(
        df_all,
        include_missing_midyear=include_missing_midyear,
        year_range=year_range,
        age_range=age_range,
        hp_range=hp_range,
        include_missing_hp=include_missing_hp,
        sel_oem=sel_oem, sel_model=sel_model, sel_tier=sel_tier, sel_ops=sel_ops, sel_type=sel_type
    )
else:
    df_scoped = df_all[df_all["Reporting year"].isin(chosen_years)] if chosen_years else df_all
    df_filtered_all_years = apply_filters(
        df_scoped,
        include_missing_midyear=include_missing_midyear,
        year_range=year_range,
        age_range=age_range,
        hp_range=hp_range,
        include_missing_hp=include_missing_hp,
        sel_oem=sel_oem, sel_model=sel_model, sel_tier=sel_tier, sel_ops=sel_ops, sel_type=sel_type
    )

# -----------------------------
# Views
# -----------------------------
if yoy_mode:
    st.success("YoY view on. Using all reporting years. Charts start at 2010.")
    totals, tiers_df, ops_df = build_yoy(df_filtered_all_years)
    if totals is None:
        st.info("No data for YoY view after filters.")
    else:
        cagr = calc_cagr(totals)
        latest = totals[totals["Year"] >= 2010].iloc[-1] if not totals[totals["Year"] >= 2010].empty else totals.iloc[-1]
        c1, c2 = st.columns(2)
        c1.metric("Total fleet (latest)", f"{int(latest['Units']):,}")
        c2.metric("CAGR", f"{cagr*100:,.1f}%" if cagr is not None else "—")
        st.subheader("Year-over-Year")
        yoy_charts(totals, tiers_df, ops_df, yoy_metric)

    with st.expander("Raw data (after filters)"):
        show_cols = [c for c in [
            "Reporting year","Type of locomotive","OEM","Model","US EPA Tier Level","Engine","hp",
            "Year of Manufacture","Manufacture Mid Year","Est. Age (yrs)","Operator Type","Count"
        ] if c in df_filtered_all_years.columns]
        st.dataframe((df_filtered_all_years[show_cols] if show_cols else df_filtered_all_years).reset_index(drop=True),
                     use_container_width=True, height=420)
        buf = io.BytesIO()
        df_filtered_all_years.to_excel(buf, index=False)
        st.download_button("Download filtered data (Excel)", data=buf.getvalue(),
                           file_name="rail_filtered_yoy.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    df_f = df_filtered_all_years

    col1, col2, col3, col4 = st.columns(4)
    total_units = int(df_f["Count"].sum()) if "Count" in df_f.columns else len(df_f)
    avg_age = _weighted_mean(df_f.get("Est. Age (yrs)"), df_f.get("Count")) if len(df_f) else np.nan
    avg_hp  = _weighted_mean(df_f.get("hp"), df_f.get("Count")) if len(df_f) else np.nan
    unique_models = df_f["Model"].astype(str).nunique() if "Model" in df_f.columns else np.nan
    col2.metric("Avg. Estimated Age (yrs)", f"{avg_age:,.1f}" if not np.isnan(avg_age) else "—")
    col3.metric("Avg. Horsepower", f"{avg_hp:,.0f}" if not np.isnan(avg_hp) else "—")
    col4.metric("Unique Models", f"{int(unique_models):,}" if not np.isnan(unique_models) else "—")

    if compare_mode and "Reporting year" in df_f.columns:
        per_year = (
            df_f.assign(Year=df_f["Reporting year"].astype(int))
                .groupby("Year")
                .agg(
                    Units=("Count","sum"),
                    Avg_Age=("Est. Age (yrs)", lambda s: _weighted_mean(s, df_f.loc[s.index,"Count"])),
                    Avg_hp=("hp", lambda s: _weighted_mean(s, df_f.loc[s.index,"Count"])),
                    Unique_Models=("Model", "nunique")
                )
        )
        per_year["Units"] = per_year["Units"].astype(int)
        st.dataframe(per_year, use_container_width=True)

    st.divider()

    tabs = st.tabs(["Tier Distribution", "Age Distribution", "Horsepower by Operator", "OEM / Model Breakdown", "Raw Data"])

    with tabs[0]:
        st.subheader("US EPA Tier Distribution")
        chart_choice = st.radio("Chart type", ["Bar", "Pie", "Donut"], horizontal=True, key="tier_chart_choice")
        if all(c in df_f.columns for c in ["US EPA Tier Level","Count","Reporting year"]):
            temp = df_f.copy()
            temp["Year"] = temp["Reporting year"].astype("Int64")
            temp["US EPA Tier Level"] = temp["US EPA Tier Level"].astype(str).apply(normalize_tier)
            tier_df = (
                temp.groupby(["Year","US EPA Tier Level"], dropna=False)["Count"]
                    .sum()
                    .reset_index()
                    .rename(columns={"US EPA Tier Level": "Tier", "Count": "Units"})
            )
            tier_df["Tier"] = pd.Categorical(tier_df["Tier"].astype(str), categories=TIER_ORDER, ordered=True)
            present = [t for t in TIER_ORDER if t in tier_df["Tier"].astype(str).unique().tolist()]
            selected_tiers = st.multiselect("Show tiers", present, default=present, key="tier_include")
            tier_df = tier_df[tier_df["Tier"].astype(str).isin(selected_tiers)]
            if not tier_df.empty:
                if chart_choice == "Bar":
                    base = alt.Chart(tier_df).mark_bar().encode(
                        x=alt.X("Units:Q", title="Units", axis=alt.Axis(format=",.0f")),
                        y=alt.Y("Tier:N", title="Tier", scale=alt.Scale(domain=TIER_ORDER)),
                        tooltip=["Year:N","Tier:N", alt.Tooltip("Units:Q", format=",.0f")],
                        color=alt.Color("Year:N", legend=alt.Legend(title="Year")) if compare_mode else alt.value("#4c78a8"),
                    )
                    chart = base if not compare_mode else base.facet(row=alt.Row("Year:N", header=alt.Header(title="")), spacing=12).resolve_scale(x="independent")
                else:
                    pie = _pie_chart(tier_df, "Tier", "Units", donut=(chart_choice == "Donut"), tier_order=TIER_ORDER)
                    chart = pie if not compare_mode else pie.facet(row=alt.Row("Year:N", header=alt.Header(title="")), spacing=12).resolve_scale(x="independent")
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Select at least one tier.")
        else:
            st.info("Tier or Year columns not found.")

    with tabs[1]:
        st.subheader("Estimated Age Distribution")
        chart_choice_age = st.radio("Chart type", ["Histogram", "Density"], horizontal=True, key="age_chart_choice")
        if all(c in df_f.columns for c in ["Est. Age (yrs)","Count","Reporting year"]):
            ages = df_f.dropna(subset=["Est. Age (yrs)"]).copy()
            ages["Year"] = ages["Reporting year"].astype("Int64")
            ages["Count"] = pd.to_numeric(ages["Count"], errors="coerce").fillna(0).astype(int).clip(lower=1)
            ages_rep = ages.loc[ages.index.repeat(ages["Count"])]
            if not ages_rep.empty:
                if chart_choice_age == "Histogram":
                    base = (
                        alt.Chart(ages_rep)
                        .transform_bin("age_bin", field="Est. Age (yrs)", bin=alt.Bin(maxbins=25))
                        .mark_bar()
                        .encode(
                            x=alt.X("age_bin:Q", title="Estimated Age (yrs)", axis=alt.Axis(labels=True, labelAngle=0), scale=alt.Scale(domainMin=0)),
                            y=alt.Y("count()", title="Units", axis=alt.Axis(format=",.0f")),
                            tooltip=[alt.Tooltip("count()", title="Units", format=",.0f")],
                            color=alt.Color("Year:N", legend=alt.Legend(title="Year")) if compare_mode else alt.value("#4c78a8"),
                        )
                    )
                    chart = base if not compare_mode else base.facet(row=alt.Row("Year:N", header=alt.Header(title="")), spacing=12).resolve_scale(x="independent")
                else:
                    density_kwargs = {"groupby": ["Year"]} if compare_mode else {}
                    base = (
                        alt.Chart(ages_rep.rename(columns={"Est. Age (yrs)":"Age"}))
                        .transform_density("Age", as_=["Age", "density"], **density_kwargs)
                        .mark_area(opacity=0.6)
                        .encode(
                            x=alt.X("Age:Q", title="Estimated Age (yrs)", axis=alt.Axis(labels=True, labelAngle=0)),
                            y=alt.Y("density:Q", title="Density"),
                            color=alt.Color("Year:N", legend=alt.Legend(title="Year")) if compare_mode else alt.value("#4c78a8"),
                        )
                    )
                    chart = base if not compare_mode else base.facet(row=alt.Row("Year:N", header=alt.Header(title="")), spacing=12).resolve_scale(x="independent")
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No data after filters.")
        else:
            st.info("Required columns not found.")

    with tabs[2]:
        st.subheader("Horsepower by Operator Type")
        chart_choice_hp = st.radio("Chart type", ["Triple bars", "Box"], horizontal=True, index=0, key="hp_chart_choice")
        if all(c in df_f.columns for c in ["hp","Operator Type","Count","Reporting year"]):
            tmp = df_f.copy()
            tmp["hp"] = pd.to_numeric(tmp["hp"], errors="coerce")
            tmp["w"] = pd.to_numeric(tmp["Count"], errors="coerce").fillna(0).clip(lower=0)
            tmp["Year"] = tmp["Reporting year"].astype("Int64")
            tmp["HP Band"] = tmp["hp"].apply(hp_to_band)
            tmp = tmp.dropna(subset=["HP Band"])

            if chart_choice_hp == "Triple bars":
                banded = (
                    tmp.groupby(["Year","Operator Type","HP Band"], dropna=False)["w"]
                       .sum()
                       .reset_index()
                       .rename(columns={"w":"Units"})
                )
                banded["Operator Type"] = pd.Categorical(banded["Operator Type"], categories=ALL_OPERATORS, ordered=True)
                banded["HP Band"] = pd.Categorical(banded["HP Band"], categories=HP_BANDS, ordered=True)

                idx = pd.MultiIndex.from_product(
                    [banded["Year"].unique(), ALL_OPERATORS, HP_BANDS],
                    names=["Year","Operator Type","HP Band"]
                )
                banded = banded.set_index(["Year","Operator Type","HP Band"]).reindex(idx, fill_value=0).reset_index()

                base = (
                    alt.Chart(banded)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "Operator Type:N",
                            title="Operator",
                            axis=alt.Axis(labelAngle=0, labels=True),
                            sort=ALL_OPERATORS,
                            scale=alt.Scale(domain=ALL_OPERATORS, padding=0.2),
                        ),
                        y=alt.Y("Units:Q", title="Units", axis=alt.Axis(format=",.0f")),
                        color=alt.Color("HP Band:N", legend=alt.Legend(title="Horsepower band"),
                                        scale=alt.Scale(domain=HP_BANDS)),
                        xOffset=alt.XOffset("HP Band:N"),
                        tooltip=[
                            alt.Tooltip("Year:N"),
                            alt.Tooltip("Operator Type:N", title="Operator"),
                            alt.Tooltip("HP Band:N", title="Band"),
                            alt.Tooltip("Units:Q", format=",.0f")
                        ],
                    )
                ).properties(width=alt.Step(70))

                chart = base if not compare_mode else base.facet(row=alt.Row("Year:N", header=alt.Header(title="Reporting year")), spacing=16).resolve_scale(x="independent")
                st.altair_chart(chart, use_container_width=True)
            else:
                rep = tmp.dropna(subset=["hp"]).copy()
                rep["w_int"] = rep["w"].astype(int).clip(lower=1)
                rep = rep.loc[rep.index.repeat(rep["w_int"])]
                if rep.empty:
                    st.info("No horsepower data after filters.")
                else:
                    base = alt.Chart(rep).mark_boxplot().encode(
                        x=alt.X("hp:Q", title="hp", axis=alt.Axis(labels=True, labelAngle=0)),
                        y=alt.Y("Operator Type:N", title="Operator", sort=ALL_OPERATORS),
                        color=alt.Color("Year:N", legend=alt.Legend(title="Year")) if compare_mode else alt.value("#4c78a8"),
                    ).properties(width=alt.Step(70))
                    chart = base if not compare_mode else base.facet(row=alt.Row("Year:N", header=alt.Header(title="Reporting year")), spacing=16).resolve_scale(x="independent")
                    st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Required columns not found.")

    with tabs[3]:
        st.subheader("OEM / Model Breakdown")
        if all(c in df_f.columns for c in ["OEM","Model","Count","Reporting year"]):
            if compare_mode:
                c1, c2 = st.columns(2)
                y1, y2 = chosen_years
                p1 = df_f[df_f["Reporting year"] == y1].pivot_table(index="OEM", columns="Model", values="Count", aggfunc="sum", fill_value=0).astype(int)
                p2 = df_f[df_f["Reporting year"] == y2].pivot_table(index="OEM", columns="Model", values="Count", aggfunc="sum", fill_value=0).astype(int)
                with c1:
                    st.caption(f"Pivot — {int(y1)}")
                    st.dataframe(p1, use_container_width=True)
                with c2:
                    st.caption(f"Pivot — {int(y2)}")
                    st.dataframe(p2, use_container_width=True)
            else:
                pivot = df_f.pivot_table(index="OEM", columns="Model", values="Count", aggfunc="sum", fill_value=0).astype(int)
                st.dataframe(pivot, use_container_width=True)
            st.caption("Units after filters.")

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
st.caption("YoY charts start at 2010. Horsepower triple bars use Low <2000, Medium 2000–3000, High >3000. Faceted charts show x-axis labels on every row. Added Type of locomotive filter in sidebar.")
