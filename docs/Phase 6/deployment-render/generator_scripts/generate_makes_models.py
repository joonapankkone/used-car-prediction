"""
generate_makes_models.py
========================
Aja omalla koneella ennen deployta.

Lukee train_slim.parquet:sta kaikki make_name / model_name -kombinaatiot
ja kirjoittaa data/makes_models.json -tiedoston.

Rakenne:
{
  "Toyota": ["Camry", "Corolla", "RAV4", ...],
  "Ford":   ["F-150", "Mustang", ...],
  ...
}

Käyttö:
  python generate_makes_models.py
  python generate_makes_models.py --parquet /polku/train_slim.parquet --out /polku/makes_models.json
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
DEFAULT_OUT     = DATA_PATH / "makes_models.json"


def generate(parquet_path: Path, out_path: Path) -> None:
    print(f"Luetaan: {parquet_path}")
    df = pl.read_parquet(parquet_path, columns=["make_name", "model_name"])

    # Poistetaan tyhjät / null-arvot
    df = df.drop_nulls().filter(
        (pl.col("make_name").str.len_chars() > 0) &
        (pl.col("model_name").str.len_chars() > 0)
    )

    # Rakennetaan dict: merkki -> lista malleja (aakkosjärjestyksessä)
    result: dict[str, list[str]] = {}
    for make, group in (
        df.group_by("make_name")
          .agg(pl.col("model_name").unique().sort())
          .sort("make_name")
          .iter_rows()
    ):
        result[make] = group

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))

    total_models = sum(len(v) for v in result.values())
    print(f"Valmis: {len(result)} merkkiä, {total_models} mallia yhteensä")
    print(f"Kirjoitettu: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--out",     type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    generate(args.parquet, args.out)
