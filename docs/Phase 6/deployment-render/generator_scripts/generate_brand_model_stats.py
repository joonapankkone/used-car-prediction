"""
generate_brand_model_stats.py
==============================
Aja omalla koneella ennen deployta.

Lukee train_slim.parquet:sta aggregaatit jotka app.py tarvitsee
ennusteen tekemiseen. Kirjoittaa data/brand_model_stats.json.

Korvataan sillä train_slim.parquet Renderissä — tiedosto on pieni
(< 1 MB) eikä vaadi Git LFS:ää.

Sisältö:
  brand_model_stats  — merkki+malli -tason aggregaatit
  geoid_stats        — piirikunta (GEOID) -tason aggregaatit
  global_fallbacks   — fallback-arvot jos make/geoid ei löydy

Rakenne (brand_model_stats):
{
  "Toyota_Camry": {
    "bm_avg_price": 22500.0,
    "bm_avg_dom":   65.3,
    "bm_avg_km":    48200.0
  },
  ...
}

Rakenne (geoid_stats):
{
  "06037": {
    "county_avg_price": 31000.0,
    "county_avg_dom":   71.2
  },
  ...
}

Käyttö:
  python generate_brand_model_stats.py
  python generate_brand_model_stats.py --parquet /polku/train_slim.parquet --out /polku/brand_model_stats.json
"""

import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import polars as pl

load_dotenv()
DATA_PATH = Path(os.getenv("DATA_PATH"))

DEFAULT_PARQUET = DATA_PATH / "train.parquet"
DEFAULT_OUT     = DATA_PATH / "brand_model_stats.json"

# Sarakkeet joita tarvitaan — muut ohitetaan (säästää muistia)
NEEDED_COLS = [
    "make_name", "model_name", "price", "daysonmarket",
    "mileage", "GEOID", "county_avg_price", "county_avg_dom",
]


def safe_float(val) -> float:
    """Polars null / NaN → Python float, turvallisesti."""
    if val is None:
        return 0.0
    f = float(val)
    return 0.0 if (f != f) else f   # NaN-tarkistus


def generate(parquet_path: Path, out_path: Path) -> None:
    print(f"Luetaan: {parquet_path}")

    # Luetaan vain tarvittavat sarakkeet
    available = pl.read_parquet(parquet_path, n_rows=1).columns
    cols = [c for c in NEEDED_COLS if c in available]
    missing = set(NEEDED_COLS) - set(cols)
    if missing:
        print(f"[VAROITUS] Sarakkeita ei löydy parquetista: {missing}")

    df = pl.read_parquet(parquet_path, columns=cols)
    print(f"Rivejä: {df.height:,}")

    # ── 1. brand_model_stats ──────────────────────────────────────────────
    bm_agg = (
        df.with_columns(
            (pl.col("make_name") + "_" + pl.col("model_name")).alias("bm")
        )
        .group_by("bm")
        .agg([
            pl.col("price").mean().alias("bm_avg_price"),
            pl.col("daysonmarket").mean().alias("bm_avg_dom"),
            pl.col("mileage").mean().alias("bm_avg_km"),
        ])
        .sort("bm")
    )

    brand_model_stats: dict[str, dict] = {}
    for row in bm_agg.iter_rows(named=True):
        brand_model_stats[row["bm"]] = {
            "bm_avg_price": round(safe_float(row["bm_avg_price"]), 2),
            "bm_avg_dom":   round(safe_float(row["bm_avg_dom"]),   2),
            "bm_avg_km":    round(safe_float(row["bm_avg_km"]),    2),
        }
    print(f"brand_model_stats: {len(brand_model_stats)} yhdistelmää")

    # ── 2. geoid_stats ───────────────────────────────────────────────────
    geoid_stats: dict[str, dict] = {}
    if "GEOID" in cols and "county_avg_price" in cols:
        geo_agg = (
            df.group_by("GEOID")
              .agg([
                  pl.col("county_avg_price").mean(),
                  pl.col("county_avg_dom").mean(),
              ])
              .sort("GEOID")
        )
        for row in geo_agg.iter_rows(named=True):
            geoid_key = str(row["GEOID"]).zfill(5)
            geoid_stats[geoid_key] = {
                "county_avg_price": round(safe_float(row["county_avg_price"]), 2),
                "county_avg_dom":   round(safe_float(row["county_avg_dom"]),   2),
            }
        print(f"geoid_stats: {len(geoid_stats)} piirikuntatilastoa")
    else:
        print("[VAROITUS] GEOID/county_avg_price puuttuu — geoid_stats jää tyhjäksi")

    # ── 3. brand_avg_dom (merkkitason, ilman mallia) ─────────────────────
    #    Tätä käytetään fallbackina kun malli ei löydy listasta
    brand_agg = (
        df.group_by("make_name")
          .agg([
              pl.col("price").mean().alias("brand_avg_price"),
              pl.col("daysonmarket").mean().alias("brand_avg_dom"),
          ])
          .sort("make_name")
    )
    brand_stats: dict[str, dict] = {}
    for row in brand_agg.iter_rows(named=True):
        brand_stats[row["make_name"]] = {
            "brand_avg_price": round(safe_float(row["brand_avg_price"]), 2),
            "brand_avg_dom":   round(safe_float(row["brand_avg_dom"]),   2),
        }
    print(f"brand_stats: {len(brand_stats)} merkkiä")

    # ── 4. Global fallbacks ──────────────────────────────────────────────
    global_fallbacks = {
        "avg_price":       round(safe_float(df["price"].mean()), 2),
        "avg_dom":         round(safe_float(df["daysonmarket"].mean()), 2),
        "county_avg_price": 28000.0,
        "county_avg_dom":   76.0,
    }
    print(f"Global fallbacks: hinta={global_fallbacks['avg_price']}, dom={global_fallbacks['avg_dom']}")

    # ── Kirjoitetaan JSON ────────────────────────────────────────────────
    output = {
        "brand_model_stats": brand_model_stats,
        "geoid_stats":       geoid_stats,
        "brand_stats":       brand_stats,
        "global_fallbacks":  global_fallbacks,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024
    print(f"\nValmis → {out_path}  ({size_kb:.0f} KB)")
    print("Tämä tiedosto korvaa train_slim.parquet Renderissä.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--out",     type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    generate(args.parquet, args.out)
