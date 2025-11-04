# -*- coding: utf-8 -*-
"""
Streamlit App: Planet’s Fever Chart — Climate Data Storytelling (NASA GISTEMP)
Refactored from a Jupyter Notebook.
"""

# === 1. IMPORTS ===
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from prophet import Prophet
import os, re, pycountry
from matplotlib.colors import TwoSlopeNorm
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from datetime import datetime

# --- ADD THIS BLOCK TO FIX FILE PATHS ---
import os 
# Get the absolute path to the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# --- END OF BLOCK ---


# === 2. PAGE CONFIGURATION ===
st.set_page_config(
    page_title="Planet's Fever Chart",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Planet’s Fever Chart — Climate Data Storytelling")
st.markdown("""
**Goal:** Turn climate data into a compelling story using visuals + a mini interactive dashboard.
This app is a Streamlit conversion of a data storytelling notebook.
""")

# === 3. DATA LOADING FUNCTIONS (CACHED) ===

# Function to load and process GISS data (from Cell 2, 6, 9)
@st.cache_data
def load_giss_data(file_path):
    try:
        df = pd.read_csv(file_path, skiprows=1)
    except FileNotFoundError:
        st.error(f"Data file not found: {file_path}. Make sure it's in the same folder as the app.")
        return None, None
        
    df = df.rename(columns=lambda c: c.strip())
    df_full = df.copy() # Keep full data for seasonal
    
    # Process for simple anomaly (J-D)
    df_simple_builder = df_full.copy()
    df_simple_builder["Year"] = pd.to_numeric(df_simple_builder["Year"], errors='coerce')
    df_simple_builder["J-D"] = pd.to_numeric(df_simple_builder["J-D"], errors='coerce')
    df_simple_builder = df_simple_builder.dropna(subset=["Year", "J-D"])
    df_simple_builder["Year"] = df_simple_builder["Year"].astype(int)
    df_simple_builder = df_simple_builder[df_simple_builder["Year"] >= 1880]
    df_simple = df_simple_builder.rename(columns={"J-D": "Anomaly"})
    df_simple = df_simple[["Year", "Anomaly"]].reset_index(drop=True)

    # Add rolling means (from Cell 3)
    df_simple['Anomaly_5yr'] = df_simple['Anomaly'].rolling(window=5, center=True, min_periods=1).mean()
    df_simple['Anomaly_10yr'] = df_simple['Anomaly'].rolling(window=10, center=True, min_periods=1).mean()

    # --- START FIX ---
    # Process full data for seasonal (Cell 6), months (Cell 8), and J-D (Cell 11)
    
    # Define all columns that need to be numeric
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    seasonal = ["DJF", "MAM", "JJA", "SON"]
    annual = ["J-D", "D-N"] # J-D is used by Prophet in Tab 6
    
    cols_to_clean = months + seasonal + annual
    
    for c in cols_to_clean:
        if c in df_full.columns: # Check if column exists
            df_full[c] = pd.to_numeric(df_full[c], errors="coerce")
            
    # Also clean Year column
    df_full["Year"] = pd.to_numeric(df_full["Year"], errors="coerce")
    df_full = df_full.dropna(subset=["Year"]) # Drop rows where year is invalid
    df_full["Year"] = df_full["Year"].astype(int)
    # --- END FIX ---

    return df_simple, df_full

# Function to load CO2 data (from Cell 12)
@st.cache_data
def load_co2_data(file_path):
    try:
        df = pd.read_csv(file_path, comment='#')
        return df
    except FileNotFoundError:
        st.error(f"Data file not found: {file_path}. Make sure it's in the same folder as the app.")
        return None

# Function to load Prophet forecast (from Cell 11)
@st.cache_data
def get_prophet_forecast(df_prophet):
    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_mode='additive',
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.68
    )
    model.fit(df_prophet)
    future_years = pd.date_range(
        start=df_prophet['ds'].min(),
        end=pd.Timestamp(year=2100, month=7, day=1),
        freq='YS'
    )
    future = pd.DataFrame({'ds': future_years})
    forecast = model.predict(future)
    forecast['year'] = forecast['ds'].dt.year
    return forecast

# Functions for Country-Level Data (from Cell 10)
def normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("°", "")
    s = re.sub(r"[(){}\[\],:;'/\\\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "_")
    return s

def to_iso3(name):
    aliases = {
        "United States": "USA", "Russia": "RUS", "Iran": "IRN", "Syria": "SYR",
        "Vietnam": "VNM", "Bolivia": "BOL", "Venezuela": "VEN", "Congo": "COG",
        "Democratic Republic of the Congo": "COD", "Laos": "LAO", "Tanzania": "TZA",
        "Ivory Coast": "CIV", "Côte d’Ivoire": "CIV", "Cote d'Ivoire": "CIV",
        "Cape Verde": "CPV", "Czech Republic": "CZE", "Slovakia": "SVK",
        "North Macedonia": "MKD", "South Korea": "KOR", "North Korea": "PRK",
        "Myanmar": "MMR", "Eswatini": "SWZ", "Bahamas": "BHS", "Gambia": "GMB",
    }
    if name in aliases:
        return aliases[name]
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

@st.cache_data
def load_country_data(data_dir):
    try:
        files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)
                        if f.lower().endswith(".csv")])
    except FileNotFoundError:
        st.error(f"Country data folder not found: {data_dir}. Make sure it's a subfolder.")
        return None

    if not files:
        st.warning(f"No CSV files found in folder: {data_dir}")
        return None
        
    CANON_MAP = {
        "year": "year", "yr": "year", "month": "month", "mo": "month",
        "year_decimal": "year_decimal", "decimal_year": "year_decimal",
        "temperature_c": "temperature_c", "temperature": "temperature_c",
        "temp_c": "temperature_c", "temp": "temperature_c",
        "uncertainty_c": "uncertainty_c", "uncertainty": "uncertainty_c",
        "stderr_c": "uncertainty_c",
    }
    REQUIRED = {"year", "month", "temperature_c"}

    def load_one(path):
        try:
            country = os.path.basename(path).replace(".csv","").replace("-"," ").strip()
            df = pd.read_csv(path, comment="#")
        except Exception:
            return None # Skip files that can't be read
            
        norm_cols = [normalize_name(c) for c in df.columns]
        df = df.rename(columns=dict(zip(df.columns, norm_cols)))
        canon_cols = {c: CANON_MAP.get(c, c) for c in df.columns}
        df = df.rename(columns=canon_cols)

        if not REQUIRED.issubset(set(df.columns)):
            return None
        
        keep = [c for c in ["year","month","temperature_c"] if c in df.columns]
        df = df[keep].copy()
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
        df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
        df = df.dropna(subset=["year","month","temperature_c"]).copy()
        df = df[(df["year"] >= 1850) & (df["year"] <= 2100)]
        if df.empty: return None
        df["Country"] = country
        return df

    records = [d for p in files if (d := load_one(p)) is not None]
    if not records:
        st.error("No usable country data files were loaded.")
        return None

    df_monthly = pd.concat(records, ignore_index=True)
    df_annual = (df_monthly
                 .groupby(["Country","year"], as_index=False)["temperature_c"]
                 .mean()
                 .rename(columns={"year":"Year","temperature_c":"AnnualTemp_C"}))
    
    baseline = (df_annual[(df_annual["Year"]>=1951) & (df_annual["Year"]<=1980)]
                .groupby("Country", as_index=False)["AnnualTemp_C"].mean()
                .rename(columns={"AnnualTemp_C":"Baseline_1951_1980_C"}))
    df_annual = df_annual.merge(baseline, on="Country", how="left")

    cov = (df_annual[(df_annual["Year"]>=1951) & (df_annual["Year"]<=1980)]
           .groupby("Country")["Year"].nunique())
    eligible = set(cov[cov>=20].index)
    df_annual = df_annual[df_annual["Country"].isin(eligible)].copy()
    
    # Check if baseline column exists before subtracting
    if "Baseline_1951_1980_C" in df_annual.columns:
        df_annual["Anomaly_C"] = df_annual["AnnualTemp_C"] - df_annual["Baseline_1951_1980_C"]
    else:
        st.warning("Baseline column not found for country data. Anomalies will not be calculated.")
        df_annual["Anomaly_C"] = np.nan

    df_annual["ISO3"] = df_annual["Country"].apply(to_iso3)
    df_annual = df_annual.dropna(subset=["ISO3"]).copy()
    df_annual = df_annual[(df_annual["Year"] >= 1880) & (df_annual["Year"] <= 2020)].copy()
    return df_annual

# === 4. LOAD ALL DATA ON STARTUP ===

# Build the full, absolute path to the files
path_glb = os.path.join(BASE_DIR, "GLB.Ts+dSST.csv")
path_nh = os.path.join(BASE_DIR, "NH.Ts+dSST.csv")
path_sh = os.path.join(BASE_DIR, "SH.Ts+dSST.csv")
path_co2 = os.path.join(BASE_DIR, "co2_annmean_mlo.csv")

# Pass these new, full paths to the functions
df, df_full = load_giss_data(path_glb)
df_nh, _ = load_giss_data(path_nh)
df_sh, _ = load_giss_data(path_sh)
df_co2 = load_co2_data(path_co2)

# Stop if main data failed to load
if df is None:
    st.error("Failed to load main data file (GLB.Ts+dSST.csv). App cannot continue.")
    st.stop()

# === 5. SIDEBAR: DATA STORY ===
st.sidebar.title("✍️ Your Data Story")
st.sidebar.markdown("""
### Setup
Is the planet truly running a fever? We will explore NASA's global temperature data since 1880 to see what the 'fever chart' of our world looks like.

### Conflict
The data clearly shows a sharp acceleration in warming, especially after 1980. We are now consistently breaching +1.0°C and rapidly approaching the critical +1.5°C threshold set by the Paris Agreement.

### Falling Action
This isn't just a global number; it impacts regions, seasons, and countries differently. The Northern Hemisphere is warming faster than the Southern. This differential warming fuels extreme weather, impacting everything from food security to sea levels.

### Resolution
The forecast shows we are on track to cross the +1.5°C limit within the next decade. This data underscores the extreme urgency for global mitigation efforts and localized adaptation strategies.
""")

# === 6. APP LAYOUT WITH TABS ===
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Fever Charts", 
    "🎛️ Interactive Dashboard", 
    "☀️ Seasonal & Regional", 
    "🍩 Monthly Radials", 
    "🗺️ Country-Level Maps", 
    "🔮 Forecast & CO₂"
])


# --- TAB 1: STATIC FEVER CHARTS (Cells 3, 4, 5) ---
with tab1:
    st.header("The Planet's Fever Chart")
    
    # Cell 3: Quick EDA
    last_year = int(df['Year'].max())
    last_val = float(df.loc[df['Year']==last_year, 'Anomaly'].iloc[0])
    st.metric(f"Latest Anomaly (~{last_year})", f"{last_val:.2f} °C", "relative to 1951–1980 baseline")
    
    # Cell 5: Core 'Fever Chart'
    st.subheader("Global Temperature Anomaly (°C) — 1880-Present")
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=df['Year'], y=df['Anomaly'],
        mode='lines', line=dict(color='firebrick', width=2),
        name='Annual anomaly'
    ))
    fig5.add_trace(go.Scatter(
        x=df['Year'], y=df['Anomaly_10yr'],
        mode='lines', line=dict(color='black', width=3),
        name='10-year smooth'
    ))
    fig5.add_hline(y=0, line_color='gray', line_dash='dash')
    fig5.add_hline(y=1.5, line_color='blue', line_dash='dot', annotation_text="+1.5°C target", annotation_position="top left")
    fig5.update_layout(
        yaxis_title="Temp. Anomaly (°C vs 1951–1980)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig5, use_container_width=True)

    # Cell 4: Absolute temperature
    st.subheader("Observed Global Temperature (Approximate)")
    BASELINE_C = 14.0
    if "Anomaly" in df.columns:
        df["Absolute_Temp"] = BASELINE_C + df["Anomaly"]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df["Year"], y=df["Absolute_Temp"], mode="lines",
                                 name="Global mean temperature (approx)"))
        fig4.add_hline(y=BASELINE_C, line_dash="dash", annotation_text="1951–1980 mean ≈ 14.0°C")
        fig4.update_layout(yaxis_title="Temperature (°C)", template="plotly_white")
        st.plotly_chart(fig4, use_container_width=True)


# --- TAB 2: INTERACTIVE DASHBOARD (Cell 7) ---
with tab2:
    st.header("Interactive Fever Chart Explorer")
    
    years = sorted(df['Year'].unique().tolist())
    y_min, y_max = int(years[0]), int(years[-1])

    # 1. Create Streamlit Widgets
    col1, col2 = st.columns(2)
    with col1:
        start_year, end_year = st.slider(
            "Select Year Range",
            min_value=y_min,
            max_value=y_max,
            value=(y_min, y_max)
        )
    with col2:
        smooth_k = st.selectbox(
            "Select Smoothing",
            options=[("None", 0), ("5-year", 5), ("10-year", 10)],
            format_func=lambda x: x[0],
            index=2
        )[1] # Get the number (0, 5, or 10)

        thresh = st.selectbox(
            "Select Threshold",
            options=[("None", None), ("+1.5°C", 1.5), ("+2.0°C", 2.0)],
            format_func=lambda x: x[0],
            index=1
        )[1] # Get the value (None, 1.5, or 2.0)

    # 2. Filter data based on widget values
    d = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].copy()

    if d.empty:
        st.warning(f"No data between {start_year} and {end_year}.")
    else:
        if smooth_k and smooth_k > 1:
            d['Smoothed'] = d['Anomaly'].rolling(window=int(smooth_k), center=True, min_periods=1).mean()
        else:
            d['Smoothed'] = np.nan

        # 3. Create Plotly Figure
        fig_interactive = go.Figure()
        fig_interactive.add_trace(go.Scatter(x=d['Year'], y=d['Anomaly'], mode='lines',
                                     name='Annual', line=dict(color="firebrick", width=2)))

        if d['Smoothed'].notna().any():
            fig_interactive.add_trace(go.Scatter(x=d['Year'], y=d['Smoothed'], mode='lines',
                                         name=f'{smooth_k}-year smooth', line=dict(color="black", width=3)))

        fig_interactive.add_hline(y=0, line_dash='dash')
        if thresh is not None:
            fig_interactive.add_hline(y=float(thresh), line_dash='dot', annotation_text=f"+{thresh}°C")

        # 4. Display chart
        fig_interactive.update_layout(
            title="Interactive Fever Chart",
            yaxis_title="Temperature Anomaly (°C)",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_interactive, use_container_width=True)


# --- TAB 3: SEASONAL & REGIONAL (Cells 6, 9) ---
with tab3:
    st.header("Seasonal and Regional Anomalies")

    # Cell 6: Summer & Winter anomaly plots
    if df_full is not None:
        st.subheader("Global Seasonal Anomalies")
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=df_full["Year"], y=df_full["DJF"], mode="lines", name="Winter (DJF)"))
        fig6.add_trace(go.Scatter(x=df_full["Year"], y=df_full["MAM"], mode="lines", name="Spring (MAM)"))
        fig6.add_trace(go.Scatter(x=df_full["Year"], y=df_full["JJA"], mode="lines", name="Summer (JJA)"))
        fig6.add_trace(go.Scatter(x=df_full["Year"], y=df_full["SON"], mode="lines", name="Autumn (SON)"))
        fig6.add_hline(y=0, line_dash="dash")
        fig6.update_layout(yaxis_title="Temperature Anomaly (°C)", template="plotly_white")
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.warning("Could not load full GISS dataset for seasonal plots.")

    # Cell 9: Regional anomalies
    if df_nh is not None and df_sh is not None:
        st.subheader("Regional Anomalies (Annual, J–D)")
        fig9 = go.Figure()
        fig9.add_trace(go.Scatter(x=df["Year"], y=df["Anomaly"], mode="lines", name="Global"))
        fig9.add_trace(go.Scatter(x=df_nh["Year"], y=df_nh["Anomaly"], mode="lines", name="Northern Hemisphere"))
        fig9.add_trace(go.Scatter(x=df_sh["Year"], y=df_sh["Anomaly"], mode="lines", name="Southern Hemisphere"))
        fig9.add_hline(y=0, line_dash="dash")
        fig9.update_layout(yaxis_title="Anomaly (°C)", template="plotly_white")
        st.plotly_chart(fig9, use_container_width=True)
    else:
        st.warning("Could not load NH/SH data. Please add `NH.Ts+dSST.csv` and `SH.Ts+dSST.csv`.")


# --- TAB 4: MONTHLY RADIALS (Cell 8) ---
with tab4:
    st.header("Monthly Global Anomaly 'Donuts' (1951–2025)")
    st.info("Generating high-resolution radial plot... this may take a moment.")

    # Data prep from Cell 8
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_order = {m:i for i,m in enumerate(months, start=1)}
    
    if df_full is not None:
        # Because 'Anomaly' (from months) is now numeric, .dropna() will work correctly
        df_m = (df_full.melt(id_vars="Year", value_vars=months,
                        var_name="Month", value_name="Anomaly")
                  .dropna(subset=["Anomaly"]) 
                  .assign(MonthNum=lambda d: d["Month"].map(month_order))
                  .sort_values(["Year","MonthNum"])
                  .reset_index(drop=True))

        target_start = 1951
        target_years = list(range(target_start, target_start + 75))
        available_years = set(df_m["Year"].unique())
        
        # Filter df_m once to avoid large intermediate data
        df_m_filtered = df_m[df_m["Year"].isin(target_years)]
        
        if not df_m_filtered.empty:
            # THIS LINE IS NOW FIXED because "Anomaly" is numeric
            max_abs = float(np.nanmax(np.abs(df_m_filtered["Anomaly"].values)))
            vlim = round(max_abs + 0.1, 1)
        else:
            vlim = 1.0 # Default if no data
            
        vmin, vmax = -vlim, vlim
        cmap = colormaps.get_cmap("RdBu_r")
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

        ncols, nrows = 15, 5
        col_w, row_h = 1.15, 1.15
        fig_w = ncols * col_w + 0.8
        fig_h = nrows * row_h + 0.8

        fig8 = plt.figure(figsize=(fig_w, fig_h), dpi=150) # Lower DPI for web
        gs = gridspec.GridSpec(nrows, ncols + 1, width_ratios=[1]*ncols + [0.06],
                               wspace=0.06, hspace=0.10)

        theta_centers = np.linspace(0, 2*np.pi, 12, endpoint=False) - np.pi/2
        width = 2*np.pi / 12
        inner_radius, base_r = 0.33, 0.33
        bar_h  = 1.0 - inner_radius

        for idx, yr in enumerate(target_years):
            r, c = divmod(idx, ncols)
            ax = fig8.add_subplot(gs[r, c], projection="polar")
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location("N")

            vals = {m: np.nan for m in range(1,13)}
            if yr in available_years:
                d = df_m_filtered[df_m_filtered["Year"] == yr][["MonthNum","Anomaly"]]
                vals.update({int(m): float(v) for m, v in zip(d["MonthNum"], d["Anomaly"])})

            for k, ang in enumerate(theta_centers, start=1):
                val = vals[k]
                if np.isnan(val):
                    ax.bar(ang, bar_h, width=width*0.98, bottom=base_r,
                           edgecolor="#DDDDDD", linewidth=0.5, facecolor="none")
                else:
                    ax.bar(ang, bar_h, width=width*0.98, bottom=base_r,
                           color=cmap(norm(val)), edgecolor="white", linewidth=0.5)

            ax.text(0, base_r-0.06, f"{yr}", ha="center", va="center", fontsize=9)
            ax.set_yticklabels([]); ax.set_xticklabels([])
            ax.set_yticks([]);      ax.set_xticks([])
            ax.set_rlim(0, 1.0)

        sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cax = fig8.add_subplot(gs[:, -1])
        cbar = fig8.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label("Monthly anomaly (°C; vs 1951–1980)", rotation=90)
        fig8.suptitle("Monthly global mean temperature anomalies 1951–2025 (vs 1951–1980)", y=0.995, fontsize=18)
        fig8.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)
        
        # Display Matplotlib plot in Streamlit
        st.pyplot(fig8)
    else:
        st.warning("Could not load full GISS dataset to generate radial plots.")


# --- TAB 5: COUNTRY-LEVEL MAPS (Cell 10) ---
with tab5:
    st.header("Country-Level Animated Maps (1880–2020)")
    st.markdown("Data processed from Berkeley Earth. Animation may take a moment to load.")
    
    # --- HERE IS THE FIX for paths ---
    # Build the path to the data directory
    path_country_dir = os.path.join(BASE_DIR, "temperature-data")
    # Pass that path to the function
    df_annual_countries = load_country_data(path_country_dir)
    # --- END OF FIX ---
    
    # --- THIS IS THE RESTORED CODE that fixes the IndentationError ---
    if df_annual_countries is not None:
        st.subheader("Country-level Temperature Anomalies")
        fig_anom = px.choropleth(
            df_annual_countries,
            locations="ISO3",
            color="Anomaly_C",
            hover_name="Country",
            animation_frame="Year",
            color_continuous_scale="RdBu_r",
            range_color=[-3, 3],
            title="Country-level Temperature Anomalies (vs 1951–1980, 1880–2020)",
            labels={"Anomaly_C": "Anomaly (°C)"},
        )
        fig_anom.update_geos(
            projection_type="natural earth",
            showcoastlines=True, showcountries=True, showland=True, landcolor="lightgray"
        )
        st.plotly_chart(fig_anom, use_container_width=True)

        st.subheader("Country-level Absolute Annual Temperature")
        abs_min, abs_max = float(df_annual_countries["AnnualTemp_C"].min()), float(df_annual_countries["AnnualTemp_C"].max())
        fig_abs = px.choropleth(
            df_annual_countries,
            locations="ISO3",
            color="AnnualTemp_C",
            hover_name="Country",
            animation_frame="Year",
            color_continuous_scale="Turbo",
            range_color=[abs_min, abs_max],
            title="Country-level Absolute Annual Temperature (°C, 1880–2020)",
            labels={"AnnualTemp_C": "Annual Temp (°C)"},
        )
        fig_abs.update_geos(
            projection_type="natural earth",
            showcoastlines=True, showcountries=True, showland=True, landcolor="lightgray"
        )
        st.plotly_chart(fig_abs, use_container_width=True)
    else:
        st.error("Could not load country-level data. Please check the `temperature-data` folder.")


# --- TAB 6: FORECAST & CO₂ (Cells 11, 12) ---
with tab6:
    st.header("Forecast to 2100 & CO₂ Correlation")
    
    # Cell 11: Prophet Forecast
    st.subheader("Global Temperature Anomaly Forecast to 2100 (Prophet Model)")
    
    if df_full is not None:
        
        # --- START FIX ---
        # Data is pre-cleaned in load_giss_data, so this is much simpler
        prophet_data = []
        valid_prophet_data = df_full[pd.notna(df_full['Year']) & pd.notna(df_full['J-D'])].copy()
        
        for _, row in valid_prophet_data.iterrows():
            prophet_data.append({
                'ds': pd.Timestamp(year=int(row['Year']), month=7, day=1),
                'y': float(row['J-D'])
            })
        # --- END FIX ---

        df_prophet = pd.DataFrame(prophet_data)
        
        if not df_prophet.empty:
            forecast = get_prophet_forecast(df_prophet)

            # Find when 1.5C is exceeded
            exceed_1_5 = forecast[(forecast['yhat'] >= 1.5) & (forecast['year'] > df_prophet['ds'].dt.year.max())]
            year_exceed_1_5 = exceed_1_5.iloc[0]['year'] if not exceed_1_5.empty else None

            # Create the plot
            fig11, ax = plt.subplots(figsize=(15, 8))
            historical_years = df_prophet['ds'].dt.year
            ax.scatter(historical_years, df_prophet['y'], color='#3b82f6', s=20, alpha=0.7, label='Observed Data', zorder=4)
            
            forecast_years = forecast['ds'].dt.year
            ax.plot(forecast_years, forecast['yhat'], '-', color='#f97316', linewidth=2.5, label='Prophet Forecast', zorder=3)
            
            future_mask = forecast['ds'] >= df_prophet['ds'].max()
            ax.plot(forecast_years[future_mask], forecast['yhat'][future_mask], '-', color='#dc2626', linewidth=3, alpha=0.8, zorder=3)
            ax.fill_between(forecast_years[future_mask], forecast['yhat_lower'][future_mask], forecast['yhat_upper'][future_mask],
                            color='#fed7aa', alpha=0.5, label='±1 Std Dev (68% CI)', zorder=1)
            
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(y=1.5, color='red', linestyle='--', linewidth=2.5, alpha=0.8, label='1.5°C Paris Target')

            if year_exceed_1_5:
                ax.axvline(x=year_exceed_1_5, color='red', linestyle=':', linewidth=2, alpha=0.6, zorder=2)
                ax.annotate(f'1.5°C Exceeded\nin {year_exceed_1_5}',
                           xy=(year_exceed_1_5, 1.5), xytext=(year_exceed_1_5 + 8, 1.2),
                           fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', color='red', lw=2))

            ax.set_xlabel('Year', fontsize=13)
            ax.set_ylabel('Temperature Anomaly (°C)', fontsize=13)
            ax.set_title('Global Temperature Anomaly Forecast to 2100', fontsize=15)
            ax.legend(loc='upper left', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(1880, 2100)
            ax.set_ylim(-0.8, 3.0)
            
            st.pyplot(fig11)

            # Print forecast stats
            st.subheader("Forecast Summary")
            last_observed = df_prophet.iloc[-1]['y']
            last_year = df_prophet.iloc[-1]['ds'].year
            forecast_2050 = forecast[forecast['year'] == 2050]['yhat'].values[0]
            forecast_2100 = forecast[forecast['year'] == 2100]['yhat'].values[0]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Last Observed", f"{last_observed:.2f}°C", f"in {last_year}")
            col2.metric("Forecast 2050", f"{forecast_2050:.2f}°C")
            col3.metric("Forecast 2100", f"{forecast_2100:.2f}°C")
            if year_exceed_1_5:
                col4.metric("1.5°C Exceeded", f"{year_exceed_1_5}", "Warning", delta_color="inverse")
            
            st.markdown("Key forecast years:")
            key_years = [2025, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
            forecast_table = forecast[forecast['year'].isin(key_years)][['year', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecast_table.columns = ['Year', 'Forecast', 'Lower 68%', 'Upper 68%']
            st.dataframe(forecast_table.set_index('Year').style.format("{:.2f}"))
        else:
            st.warning("No data available to create Prophet forecast.")

    # Cell 12: CO2 emission rate
    st.subheader("Atmospheric CO₂ Concentrations (Mauna Loa)")
    if df_co2 is not None:
        fig12, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_co2['year'], df_co2['mean'], linewidth=2, color='#d62728', marker='o', markersize=3)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('CO₂ Concentration (ppm)', fontsize=12)
        ax.set_title('Atmospheric CO₂ Concentrations (1959-2024)', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        st.pyplot(fig12)

        # Stats
        start_co2 = df_co2['mean'].iloc[0]
        end_co2 = df_co2['mean'].iloc[-1]
        increase = end_co2 - start_co2
        percent_increase = (increase / start_co2) * 100
        
        col1, col2 = st.columns(2)
        col1.metric(f"Starting CO₂ ({df_co2['year'].iloc[0]})", f"{start_co2:.2f} ppm")
        col2.metric(f"Ending CO₂ ({df_co2['year'].iloc[-1]})", f"{end_co2:.2f} ppm", f"{increase:.2f} ppm ({percent_increase:.1f}%) increase")
    else:
        st.warning("Could not load CO2 data. Please add `co2_annmean_mlo.csv`.")
