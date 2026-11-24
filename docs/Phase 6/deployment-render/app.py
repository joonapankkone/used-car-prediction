"""
AIDA Pricer — Tiimi 2
Phase 6 Deployment — Render.com
CatBoost price prediction model
"""

import os
import json
import datetime
import requests
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

DATA_PATH     = Path(os.getenv("DATA_PATH", "./data"))
MODEL_PATH    = DATA_PATH / "models" / "catboost_price.cbm"
COUNTIES_JSON = DATA_PATH / "counties_lookup.json"
MAKES_JSON    = DATA_PATH / "makes_models.json"
STATS_JSON    = DATA_PATH / "brand_model_stats.json"

# ── Lazy-loaded globals ────────────────────────────────────────────────────────
_model             = None
_feature_names     = None
_counties_data     = None
_makes_models      = None   # {make: [model, ...]}
_brand_model_stats = None   # {"Toyota_Camry": {bm_avg_price, bm_avg_dom, bm_avg_km}}
_geoid_stats       = None   # {"06037": {county_avg_price, county_avg_dom}}
_brand_stats       = None   # {"Toyota": {brand_avg_price, brand_avg_dom}}
_global_fallbacks  = None
_gas_price_cache   = {"value": None, "fetched_at": None}


# ── Model ──────────────────────────────────────────────────────────────────────
def load_model():
    global _model, _feature_names
    if _model is None:
        from catboost import CatBoostRegressor
        m = CatBoostRegressor()
        m.load_model(str(MODEL_PATH))
        _model = m
        _feature_names = m.feature_names_
    return _model, _feature_names


# ── Counties ───────────────────────────────────────────────────────────────────
def load_counties():
    global _counties_data
    if _counties_data is not None:
        return _counties_data
    if COUNTIES_JSON.exists():
        with open(COUNTIES_JSON, "r", encoding="utf-8") as f:
            _counties_data = json.load(f)
    else:
        _counties_data = {"states": [], "counties": {}}
    return _counties_data


def get_county_coords(geoid: str):
    data = load_counties()
    statefp = geoid[:2] if len(geoid) >= 2 else ""
    for c in data.get("counties", {}).get(statefp, []):
        if c["geoid"] == geoid:
            return c["lat"], c["lon"]
    return 37.0, -95.0


# ── Makes / Models ─────────────────────────────────────────────────────────────
def load_makes_models():
    global _makes_models
    if _makes_models is not None:
        return _makes_models
    if MAKES_JSON.exists():
        with open(MAKES_JSON, "r", encoding="utf-8") as f:
            _makes_models = json.load(f)
    else:
        # Fallback staattinen lista jos JSON puuttuu
        _makes_models = {m: [] for m in [
            "Acura","Alfa Romeo","Audi","BMW","Buick","Cadillac","Chevrolet",
            "Chrysler","Dodge","Ferrari","Fiat","Ford","Genesis","GMC","Honda",
            "Hyundai","Infiniti","Jaguar","Jeep","Kia","Lamborghini","Land Rover",
            "Lexus","Lincoln","Maserati","Mazda","Mercedes-Benz","MINI","Mitsubishi",
            "Nissan","Porsche","Ram","Rolls-Royce","Subaru","Tesla","Toyota",
            "Volkswagen","Volvo",
        ]}
    return _makes_models


# ── Aggregates (korvaa train_slim.parquet) ────────────────────────────────────
def load_stats():
    global _brand_model_stats, _geoid_stats, _brand_stats, _global_fallbacks
    if _brand_model_stats is not None:
        return
    if STATS_JSON.exists():
        with open(STATS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        _brand_model_stats = data.get("brand_model_stats", {})
        _geoid_stats       = data.get("geoid_stats", {})
        _brand_stats       = data.get("brand_stats", {})
        _global_fallbacks  = data.get("global_fallbacks", {})
    else:
        print("[WARN] brand_model_stats.json puuttuu — käytetään fallback-arvoja")
        _brand_model_stats = {}
        _geoid_stats       = {}
        _brand_stats       = {}
        _global_fallbacks  = {
            "avg_price": 28000.0, "avg_dom": 76.0,
            "county_avg_price": 28000.0, "county_avg_dom": 76.0,
        }


def get_county_stats(geoid: str) -> dict:
    load_stats()
    key = str(geoid).zfill(5)
    if _geoid_stats and key in _geoid_stats:
        return _geoid_stats[key]
    return {
        "county_avg_price": _global_fallbacks.get("county_avg_price", 28000.0),
        "county_avg_dom":   _global_fallbacks.get("county_avg_dom",   76.0),
    }


def get_brand_model_avg_price(make: str, model: str) -> float:
    load_stats()
    key = f"{make}_{model}"
    if _brand_model_stats:
        if key in _brand_model_stats:
            return float(_brand_model_stats[key]["bm_avg_price"])
        # Fallback: merkkitason keskiarvo
        if _brand_stats and make in _brand_stats:
            return float(_brand_stats[make]["brand_avg_price"])
    return float(_global_fallbacks.get("avg_price", 28000.0)) if _global_fallbacks else 28000.0


def get_brand_model_avg_dom(make: str, model: str) -> float:
    load_stats()
    key = f"{make}_{model}"
    if _brand_model_stats:
        if key in _brand_model_stats:
            return float(_brand_model_stats[key]["bm_avg_dom"])
        if _brand_stats and make in _brand_stats:
            return float(_brand_stats[make]["brand_avg_dom"])
    return float(_global_fallbacks.get("avg_dom", 76.0)) if _global_fallbacks else 76.0


def get_brand_model_avg_km(make: str, model: str) -> float:
    load_stats()
    key = f"{make}_{model}"
    if _brand_model_stats and key in _brand_model_stats:
        return float(_brand_model_stats[key].get("bm_avg_km", 50000.0))
    return 50000.0


# ── Gas Price (EIA API, välimuistitus 1h) ─────────────────────────────────────
def fetch_gas_price() -> float:
    """Hakee tuoreimman Regular-bensiinin viikkohinnan EIA:lta ($/gallon)."""
    now = datetime.datetime.utcnow()
    cached = _gas_price_cache
    if (cached["value"] is not None and cached["fetched_at"] is not None
            and (now - cached["fetched_at"]).total_seconds() < 3600):
        return cached["value"]

    api_key = os.getenv("EIA_API_KEY", "")
    if not api_key:
        print("[WARN] EIA_API_KEY puuttuu — käytetään oletushintaa $3.20")
        return 3.20

    series_id = "PET.EMM_EPMR_PTE_NUS_DPG.W"
    url = f"https://api.eia.gov/v2/seriesid/{series_id}?api_key={api_key}"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        rows = data["response"]["data"]
        # Uusin hinta ensin
        rows_sorted = sorted(rows, key=lambda x: x["period"], reverse=True)
        price = float(rows_sorted[0]["value"])
        _gas_price_cache["value"] = price
        _gas_price_cache["fetched_at"] = now
        return price
    except Exception as e:
        print(f"[WARN] EIA haku epäonnistui: {e} — käytetään oletushintaa $3.20")
        return _gas_price_cache["value"] or 3.20


# ── Feature Engineering ────────────────────────────────────────────────────────

OPTION_KEYWORDS = {
    "opt_leather_seats":    ["leather"],
    "opt_sunroof":          ["sunroof", "moonroof"],
    "opt_navigation":       ["navigation", "nav system"],
    "opt_backup_camera":    ["backup camera", "rear camera"],
    "opt_heated_seats":     ["heated seats"],
    "opt_bluetooth":        ["bluetooth"],
    "opt_alloy_wheels":     ["alloy wheels"],
    "opt_blind_spot":       ["blind spot"],
    "opt_remote_start":     ["remote start"],
    "opt_android_auto":     ["android auto"],
    "opt_carplay":          ["carplay"],
    "opt_third_row":        ["third row"],
    "opt_adaptive_cruise":  ["adaptive cruise"],
    "opt_parking_sensors":  ["parking sensors"],
    "opt_lane_departure":   ["lane departure"],
    "opt_cross_traffic":    ["cross traffic"],
    "opt_tow_package":      ["tow package"],
    "opt_multi_zone":       ["multi zone", "dual zone", "tri zone"],
    "opt_premium_audio":    ["premium audio", "bose", "harman", "jbl"],
    "opt_cold_weather":     ["cold weather", "winter package"],
    "opt_heat_package":     ["heated steering", "seat heater"],
    "opt_off_road":         ["off road", "off-road", "rock crawl"],
}

ALL_OPT_FLAGS = list(OPTION_KEYWORDS.keys())


def parse_options(opts_input) -> dict:
    """
    opts_input voi olla:
      - merkkijono (legacy vapaa teksti)
      - JSON-taulukko avaimista ["opt_sunroof", "opt_navigation"]
    """
    result = {flag: 0 for flag in ALL_OPT_FLAGS}
    if not opts_input:
        return result

    # Yritä ensin JSON-taulukko
    if isinstance(opts_input, str) and opts_input.startswith("["):
        try:
            keys = json.loads(opts_input)
            for k in keys:
                if k in result:
                    result[k] = 1
            return result
        except Exception:
            pass

    # Fallback: vapaa teksti → avainsanahaku
    s = (opts_input if isinstance(opts_input, str) else str(opts_input)).lower()
    for flag, kws in OPTION_KEYWORDS.items():
        if any(kw in s for kw in kws):
            result[flag] = 1
    return result


def listed_date_features(date_str=None):
    try:
        d = (datetime.datetime.strptime(date_str, "%Y-%m-%d")
             if date_str else datetime.datetime.today())
    except Exception:
        d = datetime.datetime.today()
    m = d.month
    season = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2}.get(m, 3)
    weekday = d.weekday()
    return {
        "listed_month":   m,
        "listed_season":  season,
        "listed_weekday": weekday,
        "is_weekend":     int(weekday >= 5),
    }


AGE_BINS  = [0, 1, 3, 5, 8, 12, 20, 999]
AGE_LBLS  = [0, 1, 2, 3, 4,  5,  6]
MILE_BINS = [0, 10000, 25000, 50000, 75000, 100000, 150000, 9_999_999]
MILE_LBLS = [0, 1, 2, 3, 4, 5, 6]


def bin_value(val, bins, labels):
    for i, (lo, hi) in enumerate(zip(bins, bins[1:])):
        if lo <= val < hi:
            return labels[i]
    return labels[-1]


def trans_group(t):
    t = (t or "").lower()
    if "manual" in t:   return 1
    if "cvt" in t:      return 2
    if "electric" in t: return 3
    return 0


LUXURY_MAKES = {
    "Acura", "Alfa Romeo", "Audi", "BMW", "Cadillac", "Ferrari", "Genesis",
    "Infiniti", "Jaguar", "Lamborghini", "Land Rover", "Lexus", "Lincoln",
    "Maserati", "Mercedes-Benz", "Porsche", "Rolls-Royce", "Volvo",
}

# CatBoost hoitaa kaikkien kategoristen featureiden enkoodauksen sisäisesti .cbm-mallissa.
# Kategoriat välitetään alkuperäisinä string-arvoina — ei numeerisina enkoodauksina.
# Muutettavat numeroiset featuret (trans_group, age_bin jne.) ovat johdetut
# numeeriset featuret, eivät kategorisia — ne enkoodataan kuten trainingissa.


def build_feature_vector(data: dict, feature_names):
    make          = data.get("make_name", "")
    model         = data.get("model_name", "")
    year          = int(data.get("year", 2018))
    mileage       = float(data.get("mileage", 50000))
    horsepower    = float(data.get("horsepower", 200))
    engine_disp   = float(data.get("engine_displacement", 2.0))
    fuel_type     = data.get("fuel_type", "Gasoline")
    body_type     = data.get("body_type", "Sedan")
    wheel_system  = data.get("wheel_system", "FWD")
    engine_type   = data.get("engine_type", "I4")
    transmission  = data.get("transmission_display", "Automatic")
    major_options = data.get("major_options", "")
    listed_date   = data.get("listed_date", None)
    geoid         = str(data.get("geoid", "")).zfill(5) if data.get("geoid") else ""

    # Koordinaatit ja osavaltio
    lat, lon = get_county_coords(geoid)
    statefp  = geoid[:2] if len(geoid) >= 2 else "0"

    # Johdetut featuret
    cur_year = datetime.datetime.today().year
    car_age  = max(0, cur_year - year)
    is_ev    = 1 if fuel_type == "Electric" else 0
    luxury   = 1 if make in LUXURY_MAKES else 0

    # Gas price: sähköautolle 0, muille EIA reaaliajassa
    gas_price = 0.0 if is_ev else fetch_gas_price()

    # Aggregaatit
    cs              = get_county_stats(geoid)
    bm_avg_price    = get_brand_model_avg_price(make, model)
    bm_avg_dom      = get_brand_model_avg_dom(make, model)
    bm_avg_km       = get_brand_model_avg_km(make, model)

    # Päivämääräfeaturet
    dates = listed_date_features(listed_date)

    # Optiot
    opts = parse_options(major_options)

    # hp_per_liter ja power_density
    hp_per_liter  = horsepower / engine_disp if engine_disp > 0 else 0.0
    power_density = horsepower / max(car_age, 1)

    row = {
        # Perustiedot — numeeriset
        "year":                year,
        "mileage":             mileage,
        "horsepower":          horsepower,
        "engine_displacement": engine_disp,
        "latitude":            lat,
        "longitude":           lon,
        "gas_price":           gas_price,
        "is_ev":               is_ev,
        "mileage_missing":     0,
        "length_missing":      0,
        # Kategoriat stringeinä — CatBoost enkoodaa itse .cbm-mallissa
        # (sama kuin training: cat_features=cat_cols, data stringeinä)
        "make_name":           str(make),
        "model_name":          str(model),
        "body_type":           str(body_type),
        "fuel_type":           str(fuel_type),
        "wheel_system":        str(wheel_system),
        "engine_type":         str(engine_type),
        "trans_group":         str(transmission),
        # Päivämääräfeaturet
        "listed_month":        dates["listed_month"],
        "listed_season":       dates["listed_season"],
        "listed_weekday":      dates["listed_weekday"],
        "is_weekend":          dates["is_weekend"],
        # Johdetut numeeriset
        "car_age":             car_age,
        "age_bin":             bin_value(car_age, AGE_BINS, AGE_LBLS),
        "mileage_bin":         bin_value(mileage, MILE_BINS, MILE_LBLS),
        "km_per_year":         mileage / max(car_age, 1),
        "hp_per_liter":        hp_per_liter,
        "power_density":       power_density,
        # Aggregaatit
        # GEOID ja STATEFP olivat str-tyyppisiä traindatassa
        # → CatBoost enkoodaa ne kategorisina stringeinä
        "GEOID":               geoid if geoid else "00000",
        "STATEFP":             statefp if statefp else "00",
        "county_avg_price":    cs["county_avg_price"],
        "county_avg_dom":      cs["county_avg_dom"],
        "brand_model_avg_price": bm_avg_price,
        "model_avg_km":        bm_avg_km,
        "brand_avg_dom":       bm_avg_dom,
        "model_avg_dom":       bm_avg_dom,
        # Luxury-lippu
        "luxury_flag":         luxury,
        # Optiot
        "options_count":       sum(opts.values()),
        **opts,
    }

    # Kategoriat jotka CatBoost odottaa stringeinä — ei saa korvata 0:lla
    CAT_FEATURES = {
        "make_name", "model_name", "body_type", "fuel_type",
        "wheel_system", "engine_type", "GEOID", "STATEFP",
    }

    if feature_names:
        for f in feature_names:
            if f not in row:
                row[f] = "" if f in CAT_FEATURES else 0
        return pd.DataFrame([{f: row.get(f, "" if f in CAT_FEATURES else 0)
                               for f in feature_names}])
    return pd.DataFrame([row])


# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/counties/states")
def api_states():
    data = load_counties()
    return jsonify(sorted(data.get("states", []), key=lambda x: x["name"]))


@app.route("/api/counties/<statefp>")
def api_counties(statefp):
    data = load_counties()
    counties = data.get("counties", {}).get(statefp, [])
    return jsonify(sorted(counties, key=lambda x: x["name"]))


@app.route("/api/makes")
def get_makes():
    mm = load_makes_models()
    return jsonify(sorted(mm.keys()))


@app.route("/api/models/<make>")
def get_models(make):
    mm = load_makes_models()
    models = mm.get(make, [])
    return jsonify(sorted(models))


@app.route("/api/gas_price")
def api_gas_price():
    price = fetch_gas_price()
    return jsonify({"gas_price": price, "unit": "USD/gallon"})


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        model, feature_names = load_model()
        X    = build_feature_vector(data, feature_names)
        pred = max(1000.0, float(model.predict(X)[0]))
        mae  = 2495
        return jsonify({
            "predicted_price":  round(pred),
            "confidence_low":   round(max(500, pred - mae)),
            "confidence_high":  round(pred + mae),
            "mae":              mae,
            "currency":         "USD",
            "status":           "ok",
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status":          "ok" if MODEL_PATH.exists() else "model_missing",
        "model_exists":    MODEL_PATH.exists(),
        "counties_exists": COUNTIES_JSON.exists(),
        "makes_exists":    MAKES_JSON.exists(),
        "stats_exists":    STATS_JSON.exists(),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
