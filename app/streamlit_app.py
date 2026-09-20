# ========================================
#  Karachi AQI Predictor 
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import shutil
from hopsworks import login
from dotenv import load_dotenv
from datetime import datetime, timedelta

# ---------------------- ENV ----------------------
load_dotenv()
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# Local fallback paths
FALLBACK_CSV_PATH = "data/processed/aq_weather_clean.csv"
CACHED_MODELS_DIR = "cached_models"  # models get copied here after every successful download

st.set_page_config(page_title="Pearls AQI Predictor", layout="wide", page_icon="🏙️")
st.sidebar.title("🏙️ Pearls AQI Predictor")
st.sidebar.caption("Air Quality Intelligence · Karachi")
page = st.sidebar.radio("Navigation", ["Forecast Dashboard", "EDA Dashboard"])

# ---------------------- AQI color & hazard ----------------------
def aqi_color_hazard(aqi):
    if aqi < 50:
        return "#5B9C6E", "Good"
    elif aqi < 100:
        return "#C9A227", "Moderate"
    elif aqi < 150:
        return "#D97F3D", "Unhealthy for Sensitive Groups"
    elif aqi < 200:
        return "#C1503E", "Unhealthy"
    elif aqi < 300:
        return "#7A4C8C", "Very Unhealthy"
    else:
        return "#5C2A34", "Hazardous"


# ---------------------- Custom styling ----------------------
def _darken(hex_color, factor=0.75):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}
.stApp {
    background: linear-gradient(180deg, #F7F9FB 0%, #EBEFF3 100%);
}
section[data-testid="stSidebar"] {
    background-color: #121826;
}
section[data-testid="stSidebar"] * {
    color: #D3D9E0 !important;
}
.aqi-scale-strip {
    height: 6px;
    width: 100%;
    border-radius: 3px;
    margin: 4px 0 20px 0;
    background: linear-gradient(90deg, #5B9C6E 0%, #C9A227 20%, #D97F3D 40%, #C1503E 60%, #7A4C8C 80%, #5C2A34 100%);
}
.aqi-hero-card {
    border-radius: 12px;
    padding: 28px 32px;
    color: #FFFFFF;
}
.aqi-hero-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 64px;
    font-weight: 700;
    line-height: 1;
    font-variant-numeric: tabular-nums;
}
.aqi-hero-label {
    font-size: 15px;
    opacity: 0.85;
    margin-bottom: 6px;
}
.aqi-hero-hazard {
    font-size: 20px;
    font-weight: 600;
    margin-top: 10px;
}
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}
.aqi-hero-card {
    animation: fadeInUp 0.5s ease-out;
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 14px;
    margin-top: 8px;
}
.metric-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 12px rgba(31,41,51,0.07);
    border: 1px solid rgba(31,41,51,0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 24px rgba(31,41,51,0.14);
}
.metric-icon-circle {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 19px;
    margin-bottom: 12px;
}
.metric-card-label {
    font-size: 11.5px;
    color: #6B7785;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
    font-weight: 600;
}
.metric-card-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: #1F2933;
}
.metric-bar-track {
    width: 100%;
    height: 5px;
    background: rgba(31,41,51,0.07);
    border-radius: 3px;
    margin-top: 10px;
    overflow: hidden;
}
.metric-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.4s ease;
}
.metric-card-unit {
    font-size: 12px;
    color: #9AA5B1;
    margin-left: 4px;
    font-weight: 500;
}
.section-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px;
    font-weight: 600;
    color: #1F2933;
    margin: 30px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.aqi-hero-wrap {
    position: relative;
    border-radius: 20px;
    padding: 2px;
    overflow: hidden;
}
.aqi-hero-glow {
    position: absolute;
    top: -40%;
    right: -20%;
    width: 220px;
    height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0) 70%);
    pointer-events: none;
}
.health-card {
    border-radius: 14px;
    padding: 20px 24px;
    background: #FFFFFF;
    box-shadow: 0 2px 12px rgba(31,41,51,0.07);
    display: flex;
    align-items: center;
    gap: 14px;
    height: 100%;
}
.status-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    margin-bottom: 18px;
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.status-dot.live {
    background: #5B9C6E;
    animation: pulseDot 1.8s infinite;
}
.status-dot.cached {
    background: #C9A227;
}
@keyframes pulseDot {
    0% { box-shadow: 0 0 0 0 rgba(91,156,110,0.55); }
    70% { box-shadow: 0 0 0 7px rgba(91,156,110,0); }
    100% { box-shadow: 0 0 0 0 rgba(91,156,110,0); }
}
.legend-card {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 16px;
}
.legend-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.6;
    margin-bottom: 8px;
    font-weight: 600;
}
.legend-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12.5px;
    padding: 3px 0;
}
.legend-swatch {
    width: 10px;
    height: 10px;
    border-radius: 3px;
    flex-shrink: 0;
}
.tip-card {
    border-radius: 12px;
    padding: 14px 16px;
    background: rgba(255,255,255,0.06);
    border-left: 3px solid;
    font-size: 13px;
    line-height: 1.5;
    margin-top: 6px;
}
button[kind="primary"] {
    background: linear-gradient(135deg, #146B76 0%, #0B3D44 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(20,107,118,0.30);
}
button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #1F2933 !important;
    border: 1px solid rgba(31,41,51,0.12) !important;
}
div[data-testid="stButton"] button {
    border-radius: 999px !important;
    padding: 10px 18px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# ---------------------- Connect Hopsworks (safe) ----------------------
@st.cache_resource
def connect_hopsworks():
    """
    Tries to log into Hopsworks. Returns (project, fs) on success,
    or (None, None) if Hopsworks is unreachable, instead of crashing the app.
    """
    try:
        project = login(api_key_value=HOPSWORKS_API_KEY, host=HOPSWORKS_HOST)
        fs = project.get_feature_store()
        return project, fs
    except Exception as e:
        st.session_state["hopsworks_error"] = str(e)
        return None, None

project, fs = connect_hopsworks()
HOPSWORKS_UP = project is not None and fs is not None

_status_class = "live" if HOPSWORKS_UP else "cached"
_status_text = "Live data" if HOPSWORKS_UP else "Cached mode"
st.sidebar.markdown(
    f'<div class="status-badge"><span class="status-dot {_status_class}"></span>{_status_text}</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="legend-card">'
    '<div class="legend-title">AQI Scale</div>'
    '<div class="legend-row"><span class="legend-swatch" style="background:#5B9C6E;"></span>0–50 · Good</div>'
    '<div class="legend-row"><span class="legend-swatch" style="background:#C9A227;"></span>51–100 · Moderate</div>'
    '<div class="legend-row"><span class="legend-swatch" style="background:#D97F3D;"></span>101–150 · Sensitive Groups</div>'
    '<div class="legend-row"><span class="legend-swatch" style="background:#C1503E;"></span>151–200 · Unhealthy</div>'
    '<div class="legend-row"><span class="legend-swatch" style="background:#7A4C8C;"></span>201–300 · Very Unhealthy</div>'
    '<div class="legend-row"><span class="legend-swatch" style="background:#5C2A34;"></span>300+ · Hazardous</div>'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------- Fallback: load features from local CSV ----------------------
def load_features_fallback():
    """
    Loads the locally stored feature CSV when Hopsworks feature store
    is unavailable. Returns a DataFrame sorted by timestamp, or None
    if the CSV itself can't be found.
    """
    if not os.path.exists(FALLBACK_CSV_PATH):
        return None
    df = pd.read_csv(FALLBACK_CSV_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------- Fallback: load a previously cached model ----------------------
# Structure on disk:
#   cached_models/day1/xgboost/model.pkl
#   cached_models/day1/randomforest/model.pkl
#   cached_models/day1/ridge/model.pkl
#   ... same for day2, day3
# No scaler.pkl / features.json saved alongside these — just the raw model pkl.
MODEL_TYPE_PRIORITY = ["xgboost", "randomforest", "ridge"]


def load_cached_model(day_choice):
    """
    Looks for a model that was manually placed in cached_models/day{n}/{type}/model.pkl.
    Tries model types in priority order (xgboost -> randomforest -> ridge) and
    returns the first one that loads successfully.

    Since no features.json is saved, the expected input columns are pulled from
    the model itself (feature_names_in_ for sklearn-style models, or
    get_booster().feature_names for XGBoost). Returns (model, scaler, features,
    model_type) or (None, None, None, None) if nothing usable was found.
    """
    day_dir = os.path.join(CACHED_MODELS_DIR, f"day{day_choice}")

    for model_type in MODEL_TYPE_PRIORITY:
        type_dir = os.path.join(day_dir, model_type)
        if not os.path.isdir(type_dir):
            continue

        # Accept any .pkl file inside this folder, regardless of exact name
        pkl_files = [f for f in os.listdir(type_dir) if f.lower().endswith(".pkl")]
        if not pkl_files:
            continue

        model_path = os.path.join(type_dir, pkl_files[0])
        try:
            model = joblib.load(model_path)
        except Exception:
            continue

        # Try to recover the feature/column names the model expects
        features = None
        if hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)
        elif hasattr(model, "get_booster"):
            try:
                features = model.get_booster().feature_names
            except Exception:
                features = None

        # No scaler is available for these manually-cached models
        return model, None, features, model_type

    return None, None, None, None


def cache_model_locally(model_dir_downloaded, day_choice, model_type):
    """
    Copies a freshly-downloaded Hopsworks model into cached_models/day{n}/{type}/
    so it can be used as a fallback the next time Hopsworks is down.
    Safe to call every time a model is successfully downloaded.
    """
    try:
        target_dir = os.path.join(CACHED_MODELS_DIR, f"day{day_choice}", model_type)
        os.makedirs(target_dir, exist_ok=True)
        for fname in ["model.pkl", "scaler.pkl", "features.json"]:
            src = os.path.join(model_dir_downloaded, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(target_dir, fname))
    except Exception:
        # Caching is best-effort — never let this break the main app
        pass


# ============================================================== 
# ======================= FORECAST DASHBOARD =================== 
# ============================================================== 
if page == "Forecast Dashboard":

    FEATURE_GROUP_NAME = "aqi_karachi_features_final"
    FEATURE_GROUP_VERSION = 3

    if "day_choice" not in st.session_state:
        st.session_state.day_choice = 1

    today = datetime.today()
    st.sidebar.markdown(
        f"📅 Forecast reference date: {today.strftime('%A, %d %b %Y')}"
    )

    st.title("🏙️ Karachi AQI – 3 Day Forecast")
    st.markdown('<div class="aqi-scale-strip"></div>', unsafe_allow_html=True)

    day_labels = {
        i: f"Day {i} · {(today + timedelta(days=i-1)).strftime('%a, %d %b')}"
        for i in (1, 2, 3)
    }
    pill_cols = st.columns(3)
    for i, pcol in enumerate(pill_cols, start=1):
        with pcol:
            if st.button(
                day_labels[i],
                key=f"day_pill_{i}",
                use_container_width=True,
                type="primary" if st.session_state.day_choice == i else "secondary",
            ):
                st.session_state.day_choice = i
                st.rerun()

    day_choice = st.session_state.day_choice

    # ---------------------- Get feature data (Hopsworks -> CSV fallback) ----------------------
    df_recent = None
    using_fallback_data = False
    fallback_data_date = None

    if HOPSWORKS_UP:
        try:
            fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
            df_recent = fg.read().sort_values("timestamp").tail(3).reset_index(drop=True)
        except Exception:
            pass

    if df_recent is None or len(df_recent) == 0:
        using_fallback_data = True
        df_fallback = load_features_fallback()
        if df_fallback is None:
            st.error("❌ No data available: Hopsworks is down and no local fallback CSV was found.")
            st.stop()
        df_recent = df_fallback.tail(3).reset_index(drop=True)
        if "timestamp" in df_recent.columns:
            fallback_data_date = df_recent["timestamp"].max()

    if len(df_recent) < 3:
        X_latest = df_recent.tail(1)
    else:
        if day_choice == 1:
            X_latest = df_recent.iloc[-1:]
        elif day_choice == 2:
            X_latest = df_recent.iloc[-2:-1]
        else:
            X_latest = df_recent.iloc[-3:-2]

    X_latest = X_latest.fillna(0)

    if using_fallback_data:
        st.caption("· showing recent cached data")

    # ---------------------- Load model (Hopsworks -> cached local model fallback) ----------------------
    model = None
    scaler = None
    features = None
    using_fallback_model = False

    if HOPSWORKS_UP:
        try:
            mr = project.get_model_registry()
            day_tag = f"day{day_choice}"

            model_names_to_try = [
                f"aqi_ridge_{day_tag}",
                f"aqi_randomforest_{day_tag}",
                f"aqi_xgboost_{day_tag}",
            ]

            day_models = []
            for model_name in model_names_to_try:
                try:
                    m = mr.get_model(model_name, version=None)
                    day_models.append(m)
                except Exception:
                    continue

            if day_models:
                def get_rmse(m):
                    try:
                        if hasattr(m, 'training_metrics') and m.training_metrics:
                            return m.training_metrics.get("rmse", float('inf'))
                        return float('inf')
                    except Exception:
                        return float('inf')

                model_obj = min(day_models, key=get_rmse)

                model_dir = model_obj.download()
                model = joblib.load(f"{model_dir}/model.pkl")

                scaler_path = f"{model_dir}/scaler.pkl"
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)

                features = json.load(open(f"{model_dir}/features.json"))["columns"]

                # Figure out which model type this was (for the cache folder name)
                _model_type = next(
                    (t for t in MODEL_TYPE_PRIORITY if t in model_obj.name.lower()
                     or ("randomforest" if t == "randomforest" else t) in model_obj.name.lower()),
                    "ridge"
                )
                # Save a local copy for next time Hopsworks is down
                cache_model_locally(model_dir, day_choice, _model_type)

        except Exception:
            pass

    if model is None:
        # Either Hopsworks is fully down, or the registry call above failed
        model, scaler, features, cached_model_type = load_cached_model(day_choice)
        if model is None:
            st.error("❌ No model available: Hopsworks is unreachable and no cached model "
                      f"was found locally for day {day_choice}.")
            st.info(f"Place a model.pkl under cached_models/day{day_choice}/xgboost/ "
                    "(or randomforest/, ridge/) to enable fallback predictions.")
            st.stop()

    if features is None:
        # Could not recover feature names from the model itself — fall back to
        # whatever numeric columns are available in the fetched data.
        features = [c for c in X_latest.columns if c != "timestamp"]

    try:
        for f in features:
            if f not in X_latest.columns:
                X_latest[f] = 0

        X_input = X_latest[features]
        if scaler:
            X_input = scaler.transform(X_input)

        # ---------------------- Prediction ----------------------
        pred = float(np.clip(model.predict(X_input)[0], 0, 500))
        color, hazard = aqi_color_hazard(pred)

        HEALTH_TIPS = {
            "Good": "Air quality is great today — a perfect day for outdoor activities. 🌿",
            "Moderate": "Generally fine, but unusually sensitive people should consider reducing prolonged outdoor exertion.",
            "Unhealthy for Sensitive Groups": "Children, elderly, and people with respiratory conditions should limit prolonged outdoor exertion.",
            "Unhealthy": "Everyone may begin to experience health effects. Consider wearing a mask outdoors.",
            "Very Unhealthy": "Health alert: avoid outdoor exertion. Keep windows closed and use an air purifier if possible.",
            "Hazardous": "Emergency conditions. Stay indoors and avoid all outdoor physical activity.",
        }
        st.sidebar.markdown(
            f'<div class="tip-card" style="border-color:{color};">💡 {HEALTH_TIPS.get(hazard, "")}</div>',
            unsafe_allow_html=True
        )

        # ✅ AQI NUMBER + Health Impact
        dark_color = _darken(color, 0.7)
        col1, col2 = st.columns([1, 1])
        with col1:
            hero_html = (
                f'<div class="aqi-hero-card" style="background: linear-gradient(135deg, {color} 0%, {dark_color} 100%); position:relative; overflow:hidden;">'
                f'<div class="aqi-hero-glow"></div>'
                f'<div class="aqi-hero-label">Predicted AQI — Day {day_choice}</div>'
                f'<div class="aqi-hero-number">{pred:.0f}</div>'
                f'<div class="aqi-hero-hazard">{hazard}</div>'
                f'</div>'
            )
            st.markdown(hero_html, unsafe_allow_html=True)

        with col2:
            health_html = (
                f'<div class="health-card">'
                f'<div class="metric-icon-circle" style="background:{color}22; color:{color}; width:52px; height:52px; font-size:26px;">🚦</div>'
                f'<div><div class="metric-card-label">Health Impact</div>'
                f'<div class="metric-card-value" style="color:{color}; font-size:20px;">{hazard}</div></div>'
                f'</div>'
            )
            st.markdown(health_html, unsafe_allow_html=True)

        # ✅ MEASUREMENT CARDS: Pollutants + Weather
        st.markdown('<div class="section-heading">📊 Current Measurements</div>', unsafe_allow_html=True)

        pollutant_meta = {
            "pm2_5": ("🌫️", "#EB5757", "#FDECEC", 150),
            "pm10": ("💨", "#D97F3D", "#FCEFE3", 250),
            "ozone": ("☀️", "#C9A227", "#FBF5DE", 180),
            "carbon_monoxide": ("🚗", "#5B6D8C", "#EAEDF3", 10000),
            "nitrogen_dioxide": ("🏭", "#7A4C8C", "#F1E9F5", 200),
            "sulphur_dioxide": ("🌋", "#5C2A34", "#F2E4E6", 200),
        }
        weather_meta = {
            "temperature_2m": ("🌡️", "#D9603D", "#FCEBE4"),
            "relative_humidity_2m": ("💧", "#2B6777", "#E4EFF1"),
            "wind_speed_10m": ("🍃", "#5B9C6E", "#E9F3EC"),
            "surface_pressure": ("🧭", "#4A5568", "#EAECEE"),
        }

        card_parts = ['<div class="metric-grid">']
        for p, (icon, icol, ibg, ref_max) in pollutant_meta.items():
            if p in X_latest.columns:
                val = X_latest[p].values[0]
                pct = max(3, min(100, (val / ref_max) * 100))
                card_parts.append(
                    f'<div class="metric-card"><div class="metric-icon-circle" style="background:{ibg}; color:{icol};">{icon}</div>'
                    f'<div class="metric-card-label">{p.replace("_", " ").title()}</div>'
                    f'<div class="metric-card-value">{val:.1f}<span class="metric-card-unit">μg/m³</span></div>'
                    f'<div class="metric-bar-track"><div class="metric-bar-fill" style="width:{pct:.0f}%; background:{icol};"></div></div></div>'
                )
        for w, (icon, icol, ibg) in weather_meta.items():
            if w in X_latest.columns:
                unit = "°C" if "temperature" in w else "%" if "humidity" in w else "m/s" if "wind" in w else "hPa"
                card_parts.append(
                    f'<div class="metric-card"><div class="metric-icon-circle" style="background:{ibg}; color:{icol};">{icon}</div>'
                    f'<div class="metric-card-label">{w.replace("_", " ").title()}</div>'
                    f'<div class="metric-card-value">{X_latest[w].values[0]:.1f}<span class="metric-card-unit">{unit}</span></div></div>'
                )
        card_parts.append('</div>')
        st.markdown("".join(card_parts), unsafe_allow_html=True)

    except Exception as e:
        st.error("❌ Prediction failed. Please try again shortly.")

# ============================================================== 
# =========================== EDA ============================== 
# ============================================================== 
else:
    FEATURE_GROUP_NAME = "aqi_karachi_features_final"
    FEATURE_GROUP_VERSION = 3

    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown('<div class="aqi-scale-strip"></div>', unsafe_allow_html=True)

    df = None
    if HOPSWORKS_UP:
        try:
            fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
            df = fg.read()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception:
            pass

    if df is None:
        df = load_features_fallback()
        if df is None:
            cols = [
                "pm2_5","pm10","ozone","carbon_monoxide",
                "nitrogen_dioxide","sulphur_dioxide",
                "temperature","humidity","windspeed","aqi"
            ]
            df = pd.DataFrame([{c: 0 for c in cols} for _ in range(24)])
            df["timestamp"] = pd.date_range(end=datetime.now(), periods=24, freq="H")
        else:
            st.caption("· showing recent cached data")

    # AQI trend
    st.subheader("📈 AQI Trend")
    if "aqi" in df.columns:
        st.line_chart(df.set_index("timestamp")["aqi"])

    # Correlation
    st.subheader("🔥 Pollutants + Weather Correlation")
    numeric = df.select_dtypes(include=[np.number])
    st.write(numeric.corr().style.background_gradient(cmap="coolwarm"))

    # Distributions
    st.subheader("📊 Feature Distributions")
    features = [
        "pm2_5","pm10","ozone",
        "carbon_monoxide","nitrogen_dioxide",
        "temperature","humidity","windspeed"
    ]
    for f in features:
        if f in df.columns:
            st.write(f)
            st.bar_chart(df[f])

    st.caption("Developed by Muzna Siddiqui | 10Pearls | AQI Predictor (Hopsworks Integrated)")
