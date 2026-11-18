import pickle
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent / "models"   
MATCH_TRAINING_WEEKEND_BUG = True

BASE_FARE = 50.0        
PER_MIN_RATE = 4.0      
AVG_SPEED_KMPH = 25.0
CONF_THRESHOLD = 0.80   

DEFAULT_HOTSPOT_LABELS = ["Low", "Medium", "High"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.set_page_config(page_title="RideFlow AI System", page_icon="🚕", layout="wide")

# ----------------------------------------------------------------------------
# LOADING HELPERS
# ----------------------------------------------------------------------------
def read_pkl(name):
    path = MODEL_DIR / name
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:  # keep app alive if one file fails
        return {"__error__": f"{args[0] if args else ''}: {e}"}


def failed(obj):
    return isinstance(obj, dict) and "__error__" in obj


@st.cache_resource(show_spinner="Loading models...")
def load_all():
    from tensorflow.keras.models import load_model

    def keras(name):
        return load_model(MODEL_DIR / name, compile=False)

    R = {}
    for n in ["encoders", "eta_features", "eta_artifact", "demand_features", "supply_features",
              "price_features", "cancel_features", "hotspot_features", "hotspot_enc",
              "hotspot_artifact", "demand_model", "supply_model",
              "price_model", "cancel_model", "zone_encoders", "cnn_ood", "zone_hour_history"]:
        R[n] = safe(read_pkl, f"{n}.pkl")
        
    # Load new 10-class CNN artifacts & model with fallbacks
    R["cnn_artifact"] = safe(read_pkl, "new_cnn_artifact.pkl")
    if failed(R["cnn_artifact"]):
        R["cnn_artifact"] = safe(read_pkl, "cnn_artifact.pkl")
        
    R["eta_model"] = safe(keras, "eta_model.keras")
    R["hotspot_model"] = safe(keras, "hotspot_model.keras")
    
    # Load new 10-class CNN model with fallback
    R["cnn_model"] = safe(keras, "new_driver_behavior_cnn.keras")
    if failed(R["cnn_model"]):
        R["cnn_model"] = safe(keras, "driver_behavior_cnn.keras")
        
    return R


def pick(art, *names):
    """Find an item inside an artifact (dict or object) by exact, then fuzzy name."""
    if isinstance(art, dict):
        for n in names:
            if n in art:
                return art[n]
        for n in names:
            for k, v in art.items():
                if n in str(k).lower():
                    return v
    for n in names:
        if hasattr(art, n):
            return getattr(art, n)
    return None


def feat_list(obj):
    if obj is None or failed(obj):
        return []
    if isinstance(obj, dict):
        obj = pick(obj, "features", "columns") or []
    return list(obj)


# ----------------------------------------------------------------------------
# ENCODERS (zones / weather / traffic)
# ----------------------------------------------------------------------------
def _search(container, patterns, exclude=()):
    if isinstance(container, dict) and not failed(container):
        for p in patterns:
            for k, v in container.items():
                kl = str(k).lower()
                if p in kl and not any(x in kl for x in exclude) and enc_options(v):
                    return v
    return None


def get_encoder(encoders, kind):
    """Find the label->code mapping for pickup / drop / weather / traffic."""
    kind = kind.lower()
    sources = [encoders, R.get("zone_encoders"), R.get("hotspot_enc"),
               R.get("hotspot_artifact"), R.get("eta_artifact")]
    if kind in ("weather", "traffic"):
        for s in sources:
            r = _search(s, [kind])
            if r is not None:
                return r
        return None
    first = ["pickup", "source", "from"] if kind == "pickup" else ["drop", "dest"]
    for s in sources:
        r = _search(s, first)
        if r is not None and len(enc_options(r)) >= 5:
            return r
    for s in sources:  # generic zone encoder shared by pickup and drop
        r = _search(s, ["zone", "location", "area", "place"], exclude=("weather", "traffic"))
        if r is not None and len(enc_options(r)) >= 5:
            return r
    for s in sources:  # the artifact itself may be the zone encoder
        if s is not None and not failed(s) and len(enc_options(s)) >= 5:
            return s
    return None


def _is_int(x):
    return isinstance(x, (int, np.integer))


def enc_options(le):
    """Return the list of human-readable labels for a LabelEncoder, dict, list or Series."""
    if le is None:
        return []
    if isinstance(le, pd.Series):
        le = le.to_dict()
    if isinstance(le, dict):
        if le and all(_is_int(v) for v in le.values()):      # {label: code}
            return [str(k) for k in le.keys()]
        return [str(v) for v in le.values()]                # {code: label}
    if hasattr(le, "classes_"):
        return [str(c) for c in le.classes_]
    if isinstance(le, (list, tuple, np.ndarray)):
        return [str(c) for c in le]
    return []


def enc_value(le, label):
    """Return the integer code used in training for a label."""
    if isinstance(le, pd.Series):
        le = le.to_dict()
    if isinstance(le, dict):
        if le and all(_is_int(v) for v in le.values()):      # {label: code}
            for k, v in le.items():
                if str(k) == str(label):
                    return int(v)
        else:                                               # {code: label}
            for k, v in le.items():
                if str(v) == str(label):
                    return int(k)
        raise KeyError(f"'{label}' not found in encoder")
    if hasattr(le, "transform"):
        return int(le.transform([label])[0])
    if isinstance(le, (list, tuple, np.ndarray)):
        return [str(c) for c in le].index(str(label))
    raise TypeError(f"Unsupported encoder type: {type(le)}")


# ----------------------------------------------------------------------------
# FEATURE BUILDING
# ----------------------------------------------------------------------------
ALIASES = {
    "pickup_zone_enc": ["pickup_zone", "pickup_zone_encoded", "pickup_enc"],
    "drop_zone_enc": ["drop_zone", "drop_zone_encoded", "drop_enc"],
    "weather_enc": ["weather", "weather_encoded"],
    "traffic_level_enc": ["traffic_level", "traffic", "traffic_enc", "traffic_level_encoded"],
    "hour": ["hour_of_day", "booking_hour"],
    "weekday": ["is_weekday", "day_of_week", "dayofweek"],
    "is_weekend": ["weekend"],
    "day": ["day_of_month"],
    "month": [],
    "ride_demand": ["demand", "demand_count", "predicted_demand", "ride_count"],
    "supply": ["driver_supply", "driver_count", "available_drivers", "predicted_supply", "drivers"],
    "trip_km": ["distance_km", "trip_distance", "distance"],
    "estimated_trip_duration_min": ["predicted_trip_time", "trip_duration", "eta", "estimated_trip_duration"],
    "estimated_driver_arrival_min": ["eta_min", "driver_arrival_min", "estimated_eta", "eta_minutes",
                                     "estimated_eta_min", "estimated_driver_arrival"],
    "fare": ["fare_price", "price", "total_fare", "fare_amount", "base_fare"],
    "surge_multiplier": ["surge", "predicted_surge", "surge_price"],
    "driver_rating": ["rating", "driver_rating_score"],
}
LOOKUP = {}
for canon, alts in ALIASES.items():
    LOOKUP[canon.lower()] = canon
    for a in alts:
        LOOKUP.setdefault(a.lower(), canon)


def build_frame(features, ctxs, missing_log=None):
    """Create a DataFrame in the exact feature order the model expects."""
    rows = []
    for ctx in ctxs:
        row = {}
        for f in features:
            key = LOOKUP.get(str(f).lower(), f)
            if key in ctx:
                row[f] = ctx[key]
            elif f in ctx:
                row[f] = ctx[f]
            else:
                row[f] = 0
                if missing_log is not None and f not in missing_log:
                    missing_log.append(f)
        rows.append(row)
    return pd.DataFrame(rows, columns=list(features))


def make_ctx(zone_enc, drop_enc, weather_enc, traffic_enc, hour, weekday, day=15, month=3, **extra):
    if MATCH_TRAINING_WEEKEND_BUG:
        is_weekend = 1 if day >= 5 else 0
    else:
        is_weekend = 1 if weekday >= 5 else 0
    ctx = dict(pickup_zone_enc=zone_enc, drop_zone_enc=drop_enc, weather_enc=weather_enc,
               traffic_level_enc=traffic_enc, hour=hour, weekday=weekday, is_weekend=is_weekend,
               day=day, month=month)
    ctx.update(extra)
    return ctx


def debug_box(title, features, missing):
    with st.expander(f"🛠 Debug – {title}"):
        st.write("Model features:", list(features))
        if missing:
            st.warning(f"Not mapped to any UI input (filled with 0): {missing}")
        else:
            st.success("All features mapped.")


# ----------------------------------------------------------------------------
# PREDICTION HELPERS
# ----------------------------------------------------------------------------
def unwrap(obj, name="model"):
    model, scaler, feats = obj, None, None
    if isinstance(obj, (list, tuple)):
        obj = {f"item{i}": v for i, v in enumerate(obj)}
    if isinstance(obj, dict):
        model = None
        for k in ("model", "best_model", "regressor", "classifier", "estimator",
                  "pipeline", "clf", "reg"):
            if k in obj and hasattr(obj[k], "predict"):
                model = obj[k]
                break
        if model is None:
            for v in obj.values():
                if hasattr(v, "predict"):
                    model = v
                    break
        for k, v in obj.items():
            if "scaler" in str(k).lower() and hasattr(v, "transform"):
                scaler = v
        f = pick(obj, "features", "feature_names", "columns")
        try:
            if f is not None and not hasattr(f, "predict"):
                feats = list(f)
        except Exception:
            feats = None
    if model is None or not hasattr(model, "predict"):
        keys = list(obj.keys()) if isinstance(obj, dict) else type(obj)
        raise RuntimeError(f"{name}: no object with .predict found. Contents: {keys}")
    return model, scaler, feats


# ----------------------------------------------------------------------------
# HISTORY / LAG FEATURES
# The demand and supply XGBoost models were trained with lag_1/2/3/24/168 and
# rolling_mean_3 (driver_* for supply). lag_168 alone carries ~73% of the model's
# importance, so these MUST be supplied - with zeros every zone gets the same answer.
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_history():
    h = R.get("zone_hour_history")
    if h is None or failed(h) or not isinstance(h, pd.DataFrame):
        return None
    h = h.copy()
    h["hour_bucket"] = pd.to_datetime(h["hour_bucket"])
    H = {"exact": {}, "seasonal": {}, "mean": {}, "tercile": None}
    for kind in ("ride_demand", "driver_count"):
        if kind not in h.columns:
            continue
        s = h[["pickup_zone_enc", "hour_bucket", kind]].dropna()
        H["exact"][kind] = {(int(z), t): float(v) for z, t, v in
                            zip(s["pickup_zone_enc"], s["hour_bucket"], s[kind])}
        grp = s.groupby([s["pickup_zone_enc"].astype(int), s["hour_bucket"].dt.weekday,
                         s["hour_bucket"].dt.hour])[kind].mean()
        H["seasonal"][kind] = {k: float(v) for k, v in grp.items()}
        H["mean"][kind] = float(s[kind].mean())
    if "ride_demand" in H["exact"]:
        H["tercile"] = tuple(np.quantile(h["ride_demand"].dropna(), [1 / 3, 2 / 3]))
    return H


def _lag(H, kind, zone, ts, k):
    """Value k hours before ts: the exact historical value if it exists, otherwise the
    typical value for that zone / weekday / hour (used when the chosen date is outside
    the training period)."""
    t = ts - pd.Timedelta(hours=k)
    v = H["exact"][kind].get((int(zone), t))
    if v is None:
        v = H["seasonal"][kind].get((int(zone), t.weekday(), t.hour))
    return H["mean"][kind] if v is None else v


def attach_history(ctxs):
    H = get_history()
    if H is None:
        return
    for c in ctxs:
        ts, z = c.get("ts"), c.get("pickup_zone_enc")
        if ts is None or z is None:
            continue
        for prefix, kind in (("", "ride_demand"), ("driver_", "driver_count")):
            if kind not in H["exact"]:
                continue
            lags = {k: _lag(H, kind, z, ts, k) for k in (1, 2, 3, 24, 168)}
            for k, v in lags.items():
                c[f"{prefix}lag_{k}"] = v
            c[f"{prefix}rolling_mean_3"] = (lags[1] + lags[2] + lags[3]) / 3.0


def level_from_thresholds(v):
    H = get_history()
    if H is None or H["tercile"] is None:
        return None
    lo, hi = H["tercile"]
    return "HIGH" if v >= hi else ("MEDIUM" if v >= lo else "LOW")


def lag_warning(*feat_keys):
    feats = [str(f).lower() for k in feat_keys for f in feat_list(R[k])]
    if not any("lag" in f or "rolling" in f for f in feats):
        return
    H = get_history()
    if H is None:
        st.warning("These models use lag features (demand / drivers in the previous hours, day and week), "
                   "but models/zone_hour_history.pkl was not found. Lags are being set to 0, so every zone "
                   "gets almost the same prediction. Create the file with the notebook snippet.")
    elif any("driver_lag" in f for f in feats) and "driver_count" not in H["exact"]:
        st.warning("zone_hour_history.pkl has no 'driver_count' column, so supply lags are 0.")


def _prep(R, model_key, feat_key, ctxs, missing):
    model, scaler, bundle_feats = unwrap(R[model_key], model_key)
    feats = feat_list(R[feat_key]) or bundle_feats or []
    attach_history(ctxs)
    X = build_frame(feats, ctxs, missing)
    if scaler is not None:
        X = scaler.transform(X.values)
    return model, X


def predict_reg(R, model_key, feat_key, ctxs, missing):
    model, X = _prep(R, model_key, feat_key, ctxs, missing)
    return np.asarray(model.predict(X)).reshape(-1)


def predict_cancel(R, ctxs, missing):
    model, X = _prep(R, "cancel_model", "cancel_features", ctxs, missing)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, -1]
    return np.clip(np.asarray(model.predict(X)).reshape(-1), 0, 1)


def predict_eta(R, ctxs, missing):
    art = R["eta_artifact"]
    sx, sy = pick(art, "scaler_x", "scaler"), pick(art, "scaler_y")
    if sx is None or sy is None:
        raise RuntimeError("eta_artifact.pkl must contain scaler_x and scaler_y.")
    feats = feat_list(R["eta_features"]) or feat_list(pick(art, "features"))
    X = sx.transform(build_frame(feats, ctxs, missing).values)
    X3 = X.reshape((X.shape[0], 1, X.shape[1]))
    pred = R["eta_model"].predict(X3, verbose=0)
    raw_etas = sy.inverse_transform(pred).reshape(-1)
    return raw_etas
    
    adjusted_etas = []
    we = get_encoder(R["encoders"], "weather")
    weather_opts = [o.lower() for o in enc_options(we)] if we else []

    for ctx, base_eta in zip(ctxs, raw_etas):
        w_enc = ctx.get("weather_enc", 0)
        t_enc = ctx.get("traffic_level_enc", 0)
        
        w_mult = 1.0
        if weather_opts and w_enc < len(weather_opts):
            w_str = weather_opts[w_enc]
            if "rain" in w_str:
                w_mult = 1.35
            elif "cloud" in w_str:
                w_mult = 1.15
            else:
                w_mult = 1.0
        else:
            w_mult = 1.0 + (w_enc * 0.12)
            
        t_mult = 1.0 + (t_enc * 0.15)
        adjusted_etas.append(max(1.0, float(base_eta) * w_mult * t_mult))
    return np.array(adjusted_etas)


def hotspot_labels(R):
    enc = R["hotspot_enc"]
    if failed(enc):
        return DEFAULT_HOTSPOT_LABELS
    if isinstance(enc, dict):
        try:
            return [k for k, _ in sorted(enc.items(), key=lambda kv: int(kv[1]))]
        except Exception:
            pass
    if hasattr(enc, "classes_"):
        return [str(c) for c in enc.classes_]
    return DEFAULT_HOTSPOT_LABELS


def predict_hotspot(R, ctxs, missing):
    art = R["hotspot_artifact"]
    scaler = pick(art, "scaler_hotspot", "scaler")
    feats = feat_list(R["hotspot_features"]) or feat_list(pick(art, "features"))
    X = build_frame(feats, ctxs, missing).values
    if scaler is not None:
        X = scaler.transform(X)
    return R["hotspot_model"].predict(X, verbose=0)


def check(R, keys):
    bad = [R[k]["__error__"] for k in keys if failed(R[k])]
    if bad:
        st.error("Could not load: " + " | ".join(bad))
        st.stop()

ROUTE_LOOKUP = {
    ("adyar", "anna_nagar"): (15, 40),
    ("adyar", "tambaram"): (23, 60),
    ("adyar", "mambalam"): (7.5, 22),
    ("adyar", "guindy"): (5, 16),
    ("adyar", "velachery"): (8.8, 30),
    ("adyar", "porur"): (14, 40),
    ("adyar", "saidapet"): (6, 17),
    ("adyar", "egmore"): (11, 30),
    ("adyar", "besant_nagar"): (2.5, 7),
    ("anna_nagar", "tambaram"): (33, 80),
    ("anna_nagar", "mambalam"): (9, 26),
    ("anna_nagar", "guindy"): (15, 36),
    ("anna_nagar", "velachery"): (17, 35),
    ("anna_nagar", "porur"): (10, 40),
    ("anna_nagar", "saidapet"): (12, 35),
    ("anna_nagar", "egmore"): (6.5, 20),
    ("anna_nagar", "besant_nagar"): (17.5, 50),
    ("tambaram", "mambalam"): (21, 55),
    ("tambaram", "guindy"): (20, 45),
    ("tambaram", "velachery"): (18, 39),
    ("tambaram", "porur"): (18, 47),
    ("tambaram", "saidapet"): (19, 45),
    ("tambaram", "egmore"): (28, 65),
    ("tambaram", "besant_nagar"): (25, 55),
    ("mambalam", "guindy"): (5.5, 16),
    ("mambalam", "velachery"): (9, 27),
    ("mambalam", "porur"): (11, 40),
    ("mambalam", "saidapet"): (4.5, 17),
    ("mambalam", "egmore"): (8, 25),
    ("mambalam", "besant_nagar"): (10, 27),
    ("guindy", "velachery"): (9, 27),
    ("guindy", "porur"): (11, 25),
    ("guindy", "saidapet"): (2.5, 7),
    ("guindy", "egmore"): (10, 25),
    ("guindy", "besant_nagar"): (7, 15),
    ("velachery", "porur"): (12, 45),
    ("velachery", "saidapet"): (6, 20),
    ("velachery", "egmore"): (15, 35),
    ("velachery", "besant_nagar"): (10, 15),
    ("porur", "saidapet"): (10, 25),
    ("porur", "egmore"): (15, 45),
    ("porur", "besant_nagar"): (16, 35),
    ("egmore", "saidapet"): (10, 25),
    ("egmore", "besant_nagar"): (12, 30),
    ("saidapet", "besant_nagar"): (7.5, 19),
}

def get_route_details(origin, destination):
    o, d = str(origin).lower().strip(), str(destination).lower().strip()
    if o == d:
        return 0.0, 0.0
    if (o, d) in ROUTE_LOOKUP:
        return ROUTE_LOOKUP[(o, d)]
    elif (d, o) in ROUTE_LOOKUP:
        return ROUTE_LOOKUP[(d, o)]
    return 10.0, 25.0
# ----------------------------------------------------------------------------
# STYLE
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 2.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

R = load_all()

# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go To",
    ["Home", "Price & ETA", "Demand & Operations", "Driver Behaviour"],
)

st.title("🚕 RideFlow AI System")
st.caption("AI-Based Intelligent Ride Management System")


# ----------------------------------------------------------------------------
# PAGE 1 : HOME
# ----------------------------------------------------------------------------
def page_home():
    st.header("🏠 Home")
    st.subheader("🌟Welcome to RideFlow AI Dashboard.")
    st.markdown("""
    Welcome to the intelligent, data-driven backbone of modern urban ride management.\n 
    RideFlow AI integrates advanced machine learning models and deep operational metrics to optimize every mile of the passenger and driver journey.

    ### 🚀 Core System Capabilities
    * **Dynamic Pricing & ETA Engine**
      * Computes precise trip durations anchored to geographic route matrices.
      * Adjusts predictions dynamically based on weather conditions and traffic density.
      * Predicts real-time surge multipliers and accurate total fares.
    * **Demand & Supply Operations**
      * Forecasts hourly ride demand and active driver availability across specific urban zones.
      * Identifies high-demand hotspots to guide fleet redistribution.
      * Evaluates ride cancellation risks based on wait times, fares, and driver ratings.
    * **Driver Safety & Behavior Monitoring**
      * Utilizes deep learning computer vision models to classify driver actions (such as safe driving, phone usage, and rash maneuvers) from imagery inputs.

    ### 💡 Quick Navigation Guide
    * **Price & ETA:** Head here to simulate rides between zones, inspect weather/traffic impact matrices, and calculate fares.
    * **Demand & Operations:** Explore hourly ride demand curves, driver availability trends, top hotspot rankings, and cancellation probability tools.
    * **Driver Behaviour:** Upload driver images to run real-time CNN behavior classifications and confidence scores.
    """)

    
# ----------------------------------------------------------------------------
# PAGE 2 : PRICE & ETA
# ----------------------------------------------------------------------------
ENCODER_FIX_SNIPPET = """# run in your notebook (where `data` exists), then copy the file into models/
import joblib
zone_encoders = {
    "pickup_zone":   dict(zip(data["pickup_zone"],   data["pickup_zone_enc"])),
    "drop_zone":     dict(zip(data["drop_zone"],     data["drop_zone_enc"])),
    "weather":       dict(zip(data["weather"],       data["weather_enc"])),
    "traffic_level": dict(zip(data["traffic_level"], data["traffic_level_enc"])),
}
joblib.dump(zone_encoders, "models/zone_encoders.pkl")"""


def common_inputs(prefix, with_drop=True):
    enc = R["encoders"]
    pe, de = get_encoder(enc, "pickup"), get_encoder(enc, "drop")
    we, te = get_encoder(enc, "weather"), get_encoder(enc, "traffic")
    needed = [("pickup zone", pe), ("weather", we), ("traffic", te)]
    if with_drop:
        needed.append(("drop zone", de))
    missing = [n for n, e in needed if e is None or not enc_options(e)]
    if missing:
        st.error(f"Could not find name->code mappings for: {', '.join(missing)}.")
        st.stop()

    def dropdown(col, label, le, key, idx=0):
        opts = enc_options(le)
        selected_name = col.selectbox(label, opts, index=min(idx, len(opts) - 1), key=key)
        return selected_name, enc_value(le, selected_name)

    c1, c2 = st.columns(2)
    pz_name, pz = dropdown(c1, "Pickup Zone", pe, f"{prefix}_pz")
    dz_name, dz = dropdown(c2, "Drop Zone", de, f"{prefix}_dz", idx=1) if with_drop else (pz_name, pz)

    c3, c4 = st.columns(2)
    w = dropdown(c3, "Weather", we, f"{prefix}_w")[1]
    t = dropdown(c4, "Traffic Level", te, f"{prefix}_t")[1]

    c5, c6 = st.columns(2)
    d = c5.date_input("Date", value=date.today(), key=f"{prefix}_date")
    hour = c6.slider("Hour", 0, 23, 18, key=f"{prefix}_hour")
    return pz_name, dz_name, pz, dz, w, t, d, hour


def ctx_from(pz, dz, w, t, d, hour):
    ctx = make_ctx(pz, dz, w, t, hour, d.weekday(), d.day, d.month)
    ctx["ts"] = pd.Timestamp(d) + pd.Timedelta(hours=int(hour))   # needed for lag features
    return ctx


def page_price_eta():
    st.header("💰 Price & ETA Prediction")
    st.write("Predict trip ETA, surge multiplier and total fare.")
    check(R, ["encoders", "eta_model", "eta_artifact", "eta_features", "price_model", "price_features",
              "demand_model", "demand_features", "supply_model", "supply_features"])

    lag_warning("demand_features", "supply_features")
    pz_name, dz_name, pz, dz, w, t, d, hour = common_inputs("p2", with_drop=True)
    

    if dz == pz:
        st.warning("Pickup and drop zones are the same - choose different zones for a realistic ETA.")

    if st.button("Predict Price & ETA"):
        missing = []
        try:
            ctx = ctx_from(pz, dz, w, t, d, hour)
            
            # 1. Fetch fixed trip distance and base duration from your matrix lookup
            trip_km, base_eta = get_route_details(pz_name, dz_name)
            
            # Traffic and weather adjustment lookup logic
            we_enc_obj = get_encoder(R["encoders"], "weather")
            te_enc_obj = get_encoder(R["encoders"], "traffic")
            weather_opts = enc_options(we_enc_obj)
            traffic_opts = enc_options(te_enc_obj)
            
            w_str = weather_opts[w].lower() if w < len(weather_opts) else ""
            t_str = traffic_opts[t].lower() if t < len(traffic_opts) else ""
            
            t_cat = "high" if "high" in t_str else ("medium" if "medium" in t_str or "med" in t_str else "low")
            w_cat = "rain" if "rain" in w_str else ("cloudy" if "cloud" in w_str else "clear")
                
            adjustment_map = {
                "low": {"clear": 0, "cloudy": 0, "rain": 7},
                "medium": {"clear": 5, "cloudy": 5, "rain": 10},
                "high": {"clear": 8, "cloudy": 8, "rain": 15}
            }
            eta_adjustment = adjustment_map.get(t_cat, {}).get(w_cat, 0)
            
            # Final ETA includes delays, but trip_km remains anchored to the matrix
            eta = base_eta + eta_adjustment
            
            computed_base_fare = BASE_FARE + (PER_MIN_RATE * eta)
            
            ctx.update(
                estimated_trip_duration_min=eta, 
                trip_km=trip_km,
                base_fare=computed_base_fare
            )
            
            # 2. Predict Demand & Supply
            dem = predict_reg(R, "demand_model", "demand_features", [ctx], [])
            sup = predict_reg(R, "supply_model", "supply_features", [ctx], [])
            ctx["ride_demand"] = float(dem[0])
            ctx["supply"] = float(sup[0])
            
            # 3. Compute Gap features required by the price model
            ctx["demand_supply_gap"] = ctx["ride_demand"] - ctx["supply"]
            ctx["gap_ratio"] = ctx["ride_demand"] / (ctx["supply"] + 1e-5)
            
            # 4. Predict Surge
            surge_preds = predict_reg(R, "price_model", "price_features", [ctx], missing)
            surge_val = float(surge_preds[0])
            if surge_val <= 0:
                surge_val = 1.0
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        total = computed_base_fare * surge_val

        st.subheader("Prediction Result")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Estimated Trip Time (ETA)", f"{round(eta)} min")
        c2.metric("Trip Distance", f"{trip_km:.1f} km")
        c3.metric("Surge Multiplier", f"{surge_val:.2f}x")
        c4.metric("Total Fare", f"₹{total:,.0f}")
        st.caption(f"Total fare = base fare (₹{computed_base_fare:,.0f}) × surge ({surge_val:.2f}x)")
        
        if missing:
            st.warning(f"Features missing from context: {missing}")

        

# ----------------------------------------------------------------------------
# PAGE 3 : DEMAND / SUPPLY / HOTSPOT / CANCELLATION
# ----------------------------------------------------------------------------
def page_operations():
    check(R, ["encoders"])
    st.header("📊 Demand & Operations")
    st.write("Predict ride demand, driver supply, demand hotspots and cancellation risk.")

    pz_name, dz_name, pz, dz, w, t, d, hour = common_inputs("p2", with_drop=True)
    ctx_at = lambda h: ctx_from(pz, pz, w, t, d, h)
    hours = list(range(24))

    lag_warning("demand_features", "supply_features")
    tabs = st.tabs(["📈 Demand", "🚗 Driver Supply", "🔥 Hotspot", "❌ Cancellation"])

    with tabs[0]:
        st.subheader("Ride Demand Prediction")
        check(R, ["demand_model", "demand_features"])
        if st.button("Predict Demand"):
            missing = []
            day_pred = np.round(
                predict_reg(R, "demand_model", "demand_features", [ctx_at(h) for h in hours], missing))
            cur = day_pred[hour]
            level = level_from_thresholds(cur) or ("HIGH" if cur >= np.percentile(day_pred, 66) else (
                "MEDIUM" if cur >= np.percentile(day_pred, 33) else "LOW"))
            c1, c2 = st.columns(2)
            c1.metric("Predicted Ride Demand", f"{int(cur)} rides")
            c2.metric("Demand Level", level)    
            st.line_chart(pd.DataFrame({"Predicted demand": day_pred}, index=hours))
            debug_box("Demand", feat_list(R["demand_features"]), missing)

    with tabs[1]:
        st.subheader("Driver Supply Prediction")
        check(R, ["supply_model", "supply_features"])
        if st.button("Predict Driver Supply"):
            missing = []
            day_pred =  np.round(
                predict_reg(R, "supply_model", "supply_features", [ctx_at(h) for h in hours], missing))
            st.metric("Predicted Available Drivers", f"{int(day_pred[hour])}")
            st.line_chart(pd.DataFrame({"Predicted drivers": day_pred}, index=hours))
            debug_box("Supply", feat_list(R["supply_features"]), missing)

    with tabs[2]:
        st.subheader(f"Top 3 High Demand Zones at {hour:02d}:00")
        check(R, ["demand_model", "demand_features"])
        pe = get_encoder(R["encoders"], "pickup")
        if pe is None:
            st.error("Pickup-zone encoder not found.")
        elif st.button("Find Top 3 High Demand Zones"):
            missing = []
            names = enc_options(pe)
            ctxs = []
            for n in names:
                c = ctx_at(hour)
                c["pickup_zone_enc"] = enc_value(pe, n)
                ctxs.append(c)

            dfeats = feat_list(R["demand_features"])
            zone_used = any(LOOKUP.get(str(f).lower(), f) == "pickup_zone_enc" for f in dfeats)
            raw = predict_reg(R, "demand_model", "demand_features", ctxs, missing)
            counts = np.maximum(0, np.round(raw)).astype(int)
            demand_varies = zone_used and (counts.max() - counts.min()) >= 1

            if demand_varies:
                top = (pd.DataFrame({"Zone": names, "Predicted Rides": counts})
                       .sort_values("Predicted Rides", ascending=False).head(3).reset_index(drop=True))
                top["Level"] = [level_from_thresholds(v) or "-" for v in top["Predicted Rides"]]
                value_col = "Predicted Rides"
            else:
                check(R, ["hotspot_model", "hotspot_features", "hotspot_artifact"])
                labels = hotspot_labels(R)
                probs = predict_hotspot(R, ctxs, [])
                high_i = next((i for i, l in enumerate(labels) if l.lower().startswith("high")), len(labels) - 1)
                p_high = probs[:, high_i]
                level = [labels[i] for i in probs.argmax(axis=1)]
                top = (pd.DataFrame({"Zone": names, "Hotspot Level": level,
                                     "High-demand probability (%)": np.round(p_high * 100, 1)})
                       .sort_values("High-demand probability (%)", ascending=False).head(3).reset_index(drop=True))
                value_col = "High-demand probability (%)"

            top.index = top.index + 1
            top.index.name = "Rank"
            
            # Display Top 3 Table & Bar Chart
            st.markdown("### 🔥 High Demand Zones (Top 3)")
            st.dataframe(top, use_container_width=True)

            # --- All Zones Bar Chart ---
            st.markdown("### 📊  Bar Chart : All Zones ")
            all_zones_df = pd.DataFrame({
                "Zone": names,
                "Score": counts if demand_varies else np.round(p_high * 100, 1)
            }).sort_values("Score", ascending=False)
            
            st.bar_chart(all_zones_df.set_index("Zone")["Score"])
            

    with tabs[3]:
        st.subheader("Ride Cancellation Prediction")
        check(R, ["cancel_model", "cancel_features"])
        c1, c2 = st.columns(2)
        eta_min = c1.number_input("Estimated ETA (minutes)", min_value=0.0, max_value=180.0, value=10.0, step=1.0, key="c_eta")
        fare = c2.number_input("Fare Price", min_value=0.0, max_value=100000.0, value=250.0, step=10.0, key="c_fare")
        c3, c4 = st.columns(2)
        surge = c3.number_input("Surge Multiplier", min_value=1.0, max_value=10.0, value=1.2, step=0.05, key="c_surge")
        rating = c4.number_input("Driver Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1, key="c_rate")

        def cancel_ctx(**over):
            c = ctx_at(hour)
            c.update(estimated_driver_arrival_min=eta_min, fare=fare, surge_multiplier=surge, driver_rating=rating)
            c.update(over)
            return c

        if st.button("Predict Cancellation"):
            missing = []
            p = float(predict_cancel(R, [cancel_ctx()], missing)[0])
            st.metric("Cancellation Probability", f"{p * 100:.1f}%")
            st.progress(min(max(p, 0.0), 1.0))
            debug_box("Cancellation", feat_list(R["cancel_features"]), missing)

# ----------------------------------------------------------------------------
# PAGE 4 : DRIVER BEHAVIOUR (CNN - 10 Classes)
# ----------------------------------------------------------------------------
HUMAN_READABLE_CLASSES = {
    0: 'Safe Driving',
    1: 'Phone Usage ',
    2: 'Phone Usage ',
    3: 'Phone Usage ',
    4: 'Phone Usage ',
    5: 'Rash Driving ', 
    6: 'Rash Driving',
    7: 'Rash Driving ',
    8: 'Rash Driving ',
    9: 'Rash Driving '
}

def cnn_labels(n_out):
    art = R["cnn_artifact"]
    if not failed(art):
        ci = pick(art, "class_indices")
        if isinstance(ci, dict):
            # Sort by index value
            sorted_items = sorted(ci.items(), key=lambda kv: kv[1])
            return [str(k) for k, _ in sorted_items]
        for key in ("class_names", "classes", "labels"):
            lst = pick(art, key)
            if isinstance(lst, (list, tuple, np.ndarray)) and len(lst) == n_out:
                return [str(x) for x in lst]
    # Fallback to standard 10-class keys if missing
    return [f"c{i}" for i in range(n_out)]


def page_driver_behaviour():
    st.header("🚗🛡️ Driver Behaviour Detection (CNN)")
    st.write("Upload a driver image to classify driving behavior .")
    check(R, ["cnn_model", "cnn_artifact"])

    uploaded_file = st.file_uploader("Upload Driver Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Create a two-column layout to save vertical space
        col_left, col_right = st.columns([1, 1])

        with col_left:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Driver Image", use_container_width=True)

            art = R["cnn_artifact"]
            img_size = pick(art, "img_size") or (128, 128)
            
            # Preprocess image to match training pipeline (resize & normalize)
            img_resized = ImageOps.fit(image, img_size, Image.Resampling.LANCZOS)
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

            model = R["cnn_model"]
            predictions = model.predict(img_array, verbose=0)
            
            n_out = predictions.shape[1]
            labels = cnn_labels(n_out)
            
            pred_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            raw_label = labels[pred_idx] if pred_idx < len(labels) else f"c{pred_idx}"
            
            # Map c0-c9 to human readable descriptions if available
            readable_label = raw_label
            if raw_label.startswith('c') and raw_label[1:].isdigit():
                class_num = int(raw_label[1:])
                readable_label = HUMAN_READABLE_CLASSES.get(class_num, raw_label)

            st.markdown("### Prediction Result")
            m1, m2 = st.columns(2)
            m1.metric("Behaviour", readable_label)
            m2.metric("Confidence", f"{confidence * 100:.2f}%")



# ----------------------------------------------------------------------------
# ROUTING
# ----------------------------------------------------------------------------
if page == "Home":
    page_home()
elif page == "Price & ETA":
    page_price_eta()
elif page == "Demand & Operations":
    page_operations()
elif page == "Driver Behaviour":
    page_driver_behaviour()