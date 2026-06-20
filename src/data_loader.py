"""
data_loader.py

Fetches the idealista18 academic dataset (Rey-Blanco, Arbues, Lopez & Paez, 2024,
Environment and Planning B, https://doi.org/10.1177/23998083241242844) and
produces a clean, analysis-ready listings table for Barcelona.

Source: https://github.com/paezha/idealista18 (ODbL-1.0 license)

Usage:
    python -m src.data_loader
or:
    from src.data_loader import load_barcelona_listings
    df = load_barcelona_listings()
"""

from __future__ import annotations

import os
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rdata

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

BASE_URL = "https://github.com/paezha/idealista18/raw/master/data/"
FILES_NEEDED = [
    "Barcelona_Sale.rda",
    "Barcelona_Polygons.rda",
    "Barcelona_POIS.rda",
]


def download_raw_files() -> None:
    """Download the .rda files from the idealista18 GitHub repo if not present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname in FILES_NEEDED:
        dest = RAW_DIR / fname
        if not dest.exists():
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(BASE_URL + fname, dest)
        else:
            print(f"Found cached {fname}, skipping download.")


def _read_rda(fname: str, **kwargs) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rdata.read_rda(str(RAW_DIR / fname), **kwargs)


def load_raw_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load the three raw Barcelona tables as pandas objects."""
    listings = _read_rda("Barcelona_Sale.rda")["Barcelona_Sale"]
    polygons = _read_rda("Barcelona_Polygons.rda", default_encoding="utf8")["Barcelona_Polygons"]
    pois = _read_rda("Barcelona_POIS.rda")["Barcelona_POIS"]
    return listings, polygons, pois


def _point_in_polygon_join(listings: pd.DataFrame, polygons: pd.DataFrame) -> pd.Series:
    """
    Assign each listing to a neighborhood by nearest-centroid matching.

    NOTE: the geometry column from idealista18 is a nested coordinate list,
    not a proper Shapely geometry. For a robust spatial join, install
    `shapely` + `geopandas` and do a real point-in-polygon test. As a fast,
    dependency-light approximation for the prototype, we assign each listing
    to the neighborhood whose centroid is closest in lat/lon space.
    This is documented here as a deliberate simplification - revisit with a
    real spatial join if neighborhood accuracy matters for your analysis.
    """
    def polygon_centroid(geom) -> tuple[float, float]:
        # geometry is a nested list of rings of [lon, lat] pairs
        pts = []

        def collect(g):
            if isinstance(g, (list, np.ndarray)):
                if len(g) > 0 and isinstance(g[0], (float, int, np.floating, np.integer)):
                    pts.append(g)
                else:
                    for sub in g:
                        collect(sub)

        collect(geom)
        arr = np.array(pts, dtype=float)
        return arr[:, 0].mean(), arr[:, 1].mean()

    centroids = polygons["geometry"].apply(polygon_centroid)
    cen_lon = centroids.apply(lambda t: t[0]).to_numpy()
    cen_lat = centroids.apply(lambda t: t[1]).to_numpy()
    names = polygons["LOCATIONNAME"].to_numpy()

    lon = listings["LONGITUDE"].to_numpy()
    lat = listings["LATITUDE"].to_numpy()

    assigned = []
    for lo, la in zip(lon, lat):
        d2 = (cen_lon - lo) ** 2 + (cen_lat - la) ** 2
        assigned.append(names[np.argmin(d2)])
    return pd.Series(assigned, index=listings.index, name="NEIGHBORHOOD")


def clean_listings(listings: pd.DataFrame, polygons: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning/filtering and feature engineering."""
    df = listings.copy()

    # Drop the most recent period only, since the same dwelling can appear in
    # multiple quarters of 2018 (price re-listed); keep the latest snapshot
    # per ASSETID so each property appears once.
    df = df.sort_values("PERIOD").drop_duplicates(subset="ASSETID", keep="last")

    # Drop rows with missing core fields
    df = df.dropna(subset=["PRICE", "CONSTRUCTEDAREA", "ROOMNUMBER", "LATITUDE", "LONGITUDE"])

    # Filter obvious outliers (documented choice: keep listings within
    # reasonable bounds for a residential-recommendation use case)
    df = df[(df["PRICE"] >= 30_000) & (df["PRICE"] <= 3_000_000)]
    df = df[(df["CONSTRUCTEDAREA"] >= 20) & (df["CONSTRUCTEDAREA"] <= 500)]

    df["PRICE_PER_SQM"] = df["PRICE"] / df["CONSTRUCTEDAREA"]

    df["NEIGHBORHOOD"] = _point_in_polygon_join(df, polygons)

    # Tidy item id
    df = df.rename(columns={"ASSETID": "item_id"})
    df = df.reset_index(drop=True)

    keep_cols = [
        "item_id", "PRICE", "PRICE_PER_SQM", "CONSTRUCTEDAREA", "ROOMNUMBER",
        "BATHNUMBER", "HASTERRACE", "HASLIFT", "HASAIRCONDITIONING",
        "HASPARKINGSPACE", "HASBOXROOM", "HASWARDROBE", "HASSWIMMINGPOOL",
        "HASDOORMAN", "HASGARDEN", "ISDUPLEX", "ISSTUDIO", "CONSTRUCTIONYEAR",
        "DISTANCE_TO_CITY_CENTER", "DISTANCE_TO_METRO", "DISTANCE_TO_DIAGONAL",
        "LATITUDE", "LONGITUDE", "NEIGHBORHOOD",
    ]
    return df[keep_cols]


def load_barcelona_listings(force_refresh: bool = False) -> pd.DataFrame:
    """
    Main entry point: returns a clean Barcelona listings DataFrame, one row
    per dwelling, ready to feed into the recommender modules.
    """
    processed_path = PROCESSED_DIR / "barcelona_listings.csv"
    if processed_path.exists() and not force_refresh:
        return pd.read_csv(processed_path)

    download_raw_files()
    listings, polygons, _pois = load_raw_tables()
    clean = clean_listings(listings, polygons)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(processed_path, index=False)
    return clean


if __name__ == "__main__":
    df = load_barcelona_listings(force_refresh=True)
    print(df.shape)
    print(df.head())
    print(df["NEIGHBORHOOD"].value_counts().head(10))
