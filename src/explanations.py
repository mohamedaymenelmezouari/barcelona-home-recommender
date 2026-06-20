"""
explanations.py

Per-method recommendation explanations. Senior recommender systems treat
explainability as a first-class concern (see Tintarev & Masthoff, 2007),
not just an evaluation metric. Each function below returns a short, human
readable string the UI can show alongside a recommended listing.
"""

from __future__ import annotations

import pandas as pd


def _format_price(p: float) -> str:
    return f"€{p:,.0f}"


def explain_popularity(listing: pd.Series, n_interactions: int) -> str:
    return f"Trending: {n_interactions} buyer interactions across the platform"


def explain_best_value(listing: pd.Series, neighborhood_rank: int, neighborhood: str) -> str:
    return f"#{neighborhood_rank} cheapest €/m² in {neighborhood}"


def explain_item_cf(listing: pd.Series, anchor_listing: pd.Series | None) -> str:
    if anchor_listing is None:
        return "Similar to listings you've engaged with"
    return (
        f"Buyers who saved {_format_price(anchor_listing['PRICE'])} in "
        f"{anchor_listing['NEIGHBORHOOD']} also engaged with this one"
    )


def explain_user_cf(listing: pd.Series, n_similar_users: int) -> str:
    return f"{n_similar_users} buyers with profiles like yours engaged with this listing"


def explain_content_based(listing: pd.Series, profile_summary: dict) -> str:
    parts = []
    price_diff = abs(listing["PRICE"] - profile_summary["avg_price"]) / max(profile_summary["avg_price"], 1)
    if price_diff < 0.15:
        parts.append(f"price near your average ({_format_price(profile_summary['avg_price'])})")
    size_diff = abs(listing["CONSTRUCTEDAREA"] - profile_summary["avg_size"]) / max(profile_summary["avg_size"], 1)
    if size_diff < 0.20:
        parts.append(f"~{profile_summary['avg_size']:.0f} m² like the ones you save")
    if listing["NEIGHBORHOOD"] in profile_summary["top_neighborhoods"]:
        parts.append(f"in {listing['NEIGHBORHOOD']}, a neighborhood you've engaged with")
    if not parts:
        parts.append("matches the feature profile of listings you've saved")
    return "Matches your profile: " + "; ".join(parts)


def explain_als(listing: pd.Series, score: float | None = None) -> str:
    if score is not None:
        return f"Latent-factor match score: {score:.2f}"
    return "Latent-factor model identified strong signal for your profile"


def build_user_profile_summary(user_id: str, interactions: pd.DataFrame, listings: pd.DataFrame) -> dict:
    """Aggregate the listings a user has engaged with into a compact profile."""
    user_events = interactions[
        (interactions["user_id"] == user_id) & (interactions["event_type"].isin(["save", "contact"]))
    ]
    if user_events.empty:
        user_events = interactions[interactions["user_id"] == user_id]
    engaged = listings[listings["item_id"].isin(user_events["item_id"])]
    if engaged.empty:
        return {"avg_price": 0.0, "avg_size": 0.0, "top_neighborhoods": set(), "n_engaged": 0}
    return {
        "avg_price": float(engaged["PRICE"].mean()),
        "avg_size": float(engaged["CONSTRUCTEDAREA"].mean()),
        "top_neighborhoods": set(engaged["NEIGHBORHOOD"].value_counts().head(3).index),
        "n_engaged": int(len(engaged)),
    }
