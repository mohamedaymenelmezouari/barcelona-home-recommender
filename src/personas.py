"""
personas.py

idealista18 has no user accounts or interaction history - it's a snapshot of
listings only. To make collaborative filtering and matrix factorization
possible, we generate a documented, transparent synthetic user layer:

  1. Define a handful of buyer PERSONAS (preference archetypes).
  2. Instantiate many individual synthetic users per persona, each with a
     small amount of noise around the persona's preference vector, so the
     interaction matrix has realistic within-persona variation.
  3. For each user, compute a fit score against every listing, then sample
     three tiers of implicit feedback with decreasing volume and increasing
     intent: VIEW -> SAVE -> CONTACT. This mirrors real product-analytics
     funnels (impression -> favorite -> lead) and gives weighted-implicit
     algorithms (e.g. ALS) a natural confidence signal.

This is clearly synthetic data and should be described as such in the
slide deck. It is NOT a substitute for real user behavior - it exists so
that every week's algorithm has something real to learn from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Interaction strength weights, used later by collaborative filtering /
# matrix factorization as implicit confidence weights.
EVENT_WEIGHTS = {"view": 1, "save": 3, "contact": 5}


@dataclass
class Persona:
    name: str
    price_range: tuple[float, float]          # (min, max) EUR
    size_range: tuple[float, float]           # (min, max) sqm
    min_rooms: int
    wants_central: bool                       # cares about distance to center
    wants_metro: bool                         # cares about distance to metro
    amenity_weights: dict[str, float] = field(default_factory=dict)
    weight: float = 1.0                       # relative population share

    def score(self, listings: pd.DataFrame, noise_rng: np.random.Generator) -> np.ndarray:
        """Fit score for this persona against every listing, in [0, ~1]."""
        price_fit = _range_fit(listings["PRICE"], *self.price_range)
        size_fit = _range_fit(listings["CONSTRUCTEDAREA"], *self.size_range)
        room_fit = (listings["ROOMNUMBER"] >= self.min_rooms).astype(float).to_numpy()

        center_fit = 1.0
        if self.wants_central:
            center_fit = 1.0 / (1.0 + listings["DISTANCE_TO_CITY_CENTER"])

        metro_fit = 1.0
        if self.wants_metro:
            metro_fit = 1.0 / (1.0 + listings["DISTANCE_TO_METRO"])

        amenity_fit = np.zeros(len(listings))
        for col, w in self.amenity_weights.items():
            amenity_fit = amenity_fit + w * listings[col].fillna(0).to_numpy()
        if self.amenity_weights:
            amenity_fit = amenity_fit / sum(self.amenity_weights.values())

        score = (
            0.35 * price_fit
            + 0.20 * size_fit
            + 0.15 * room_fit
            + 0.15 * np.asarray(center_fit)
            + 0.05 * np.asarray(metro_fit)
            + 0.10 * amenity_fit
        )
        # small amount of individual noise so users within a persona differ
        score = np.asarray(score, dtype=float) + noise_rng.normal(0, 0.05, size=len(listings))
        return np.clip(score, 0, None)


def _range_fit(series: pd.Series, lo: float, hi: float) -> np.ndarray:
    """1.0 inside [lo, hi], decaying smoothly outside it."""
    x = series.to_numpy(dtype=float)
    mid = (lo + hi) / 2
    half_width = max((hi - lo) / 2, 1e-6)
    z = np.abs(x - mid) / half_width
    return np.clip(1.0 - 0.5 * np.maximum(z - 1, 0), 0, 1)


PERSONAS: list[Persona] = [
    Persona(
        name="young_couple_central",
        price_range=(150_000, 350_000),
        size_range=(40, 75),
        min_rooms=1,
        wants_central=True,
        wants_metro=True,
        amenity_weights={"HASLIFT": 1.0, "HASAIRCONDITIONING": 0.7},
        weight=1.4,
    ),
    Persona(
        name="family_needs_space",
        price_range=(300_000, 650_000),
        size_range=(90, 160),
        min_rooms=3,
        wants_central=False,
        wants_metro=True,
        amenity_weights={"HASLIFT": 1.0, "HASTERRACE": 0.8, "HASPARKINGSPACE": 0.6},
        weight=1.2,
    ),
    Persona(
        name="investor_yield_focused",
        price_range=(100_000, 250_000),
        size_range=(30, 70),
        min_rooms=1,
        wants_central=True,
        wants_metro=True,
        amenity_weights={"HASLIFT": 0.5},
        weight=0.9,
    ),
    Persona(
        name="retiree_downsizing",
        price_range=(200_000, 450_000),
        size_range=(50, 90),
        min_rooms=2,
        wants_central=True,
        wants_metro=False,
        amenity_weights={"HASLIFT": 1.5, "HASDOORMAN": 0.6},
        weight=0.8,
    ),
    Persona(
        name="luxury_buyer",
        price_range=(700_000, 2_500_000),
        size_range=(120, 350),
        min_rooms=3,
        wants_central=True,
        wants_metro=False,
        amenity_weights={"HASSWIMMINGPOOL": 1.0, "HASDOORMAN": 0.8, "HASGARDEN": 0.6},
        weight=0.4,
    ),
    Persona(
        name="budget_first_buyer",
        price_range=(80_000, 180_000),
        size_range=(35, 65),
        min_rooms=1,
        wants_central=False,
        wants_metro=True,
        amenity_weights={"HASLIFT": 0.6},
        weight=1.3,
    ),
    Persona(
        name="remote_worker_needs_room",
        price_range=(180_000, 380_000),
        size_range=(60, 110),
        min_rooms=2,
        wants_central=False,
        wants_metro=False,
        amenity_weights={"HASAIRCONDITIONING": 0.8, "HASTERRACE": 0.5},
        weight=1.0,
    ),
    Persona(
        name="student_shared_flat",
        price_range=(60_000, 140_000),
        size_range=(25, 55),
        min_rooms=1,
        wants_central=False,
        wants_metro=True,
        amenity_weights={},
        weight=1.0,
    ),
]


def generate_users(personas: list[Persona], n_users: int, seed: int = 42) -> pd.DataFrame:
    """Sample n_users total, distributed across personas by their weight."""
    rng = np.random.default_rng(seed)
    weights = np.array([p.weight for p in personas])
    probs = weights / weights.sum()
    persona_choice = rng.choice(len(personas), size=n_users, p=probs)

    users = pd.DataFrame({
        "user_id": [f"U{idx:05d}" for idx in range(n_users)],
        "persona": [personas[i].name for i in persona_choice],
    })
    return users


def generate_interactions(
    listings: pd.DataFrame,
    users: pd.DataFrame,
    personas: list[Persona],
    browse_size: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each user: sample `browse_size` listings as their "browsing session"
    (weighted slightly toward their persona's fit, to mimic search/filter
    behavior rather than uniform random browsing), then probabilistically
    promote some of those views into saves and contacts based on fit score.
    """
    rng = np.random.default_rng(seed)
    persona_lookup = {p.name: p for p in personas}
    n_items = len(listings)
    item_ids = listings["item_id"].to_numpy()

    records = []
    for _, user in users.iterrows():
        persona = persona_lookup[user["persona"]]
        fit = persona.score(listings, rng)

        # Simulate a filtered search: most real users don't browse the whole
        # catalog uniformly, they apply price/size/location filters first.
        # We approximate that by restricting the candidate pool to the
        # top-fit items, with a small slice of random exploration mixed in
        # so the data isn't perfectly clean.
        pool_size = min(3000, n_items)
        top_idx = np.argsort(fit)[-pool_size:]
        explore_size = max(1, int(0.15 * browse_size))
        explore_idx = rng.choice(n_items, size=explore_size, replace=False)
        pool_idx = np.unique(np.concatenate([top_idx, explore_idx]))

        pool_fit = fit[pool_idx]
        pool_probs = pool_fit / (pool_fit.sum() + 1e-9)
        chosen = rng.choice(len(pool_idx), size=min(browse_size, len(pool_idx)), replace=False, p=pool_probs)
        viewed_idx = pool_idx[chosen]

        viewed_fit = fit[viewed_idx]
        # normalize within the session for save/contact probability
        if viewed_fit.max() > 0:
            rel_fit = viewed_fit / viewed_fit.max()
        else:
            rel_fit = viewed_fit

        save_mask = rng.random(len(viewed_idx)) < (0.35 * rel_fit)
        contact_mask = save_mask & (rng.random(len(viewed_idx)) < (0.25 * rel_fit))

        for j, idx in enumerate(viewed_idx):
            records.append((user["user_id"], item_ids[idx], "view", EVENT_WEIGHTS["view"]))
            if save_mask[j]:
                records.append((user["user_id"], item_ids[idx], "save", EVENT_WEIGHTS["save"]))
            if contact_mask[j]:
                records.append((user["user_id"], item_ids[idx], "contact", EVENT_WEIGHTS["contact"]))

    interactions = pd.DataFrame(records, columns=["user_id", "item_id", "event_type", "weight"])
    return interactions


def build_synthetic_data(
    listings: pd.DataFrame,
    n_users: int = 1200,
    browse_size: int = 200,
    seed: int = 42,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    users = generate_users(PERSONAS, n_users=n_users, seed=seed)
    interactions = generate_interactions(listings, users, PERSONAS, browse_size=browse_size, seed=seed)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        users.to_csv(PROCESSED_DIR / "users.csv", index=False)
        interactions.to_csv(PROCESSED_DIR / "interactions.csv", index=False)

    return users, interactions


if __name__ == "__main__":
    from src.data_loader import load_barcelona_listings

    listings = load_barcelona_listings()
    users, interactions = build_synthetic_data(listings)

    print("Users:", users.shape)
    print(users["persona"].value_counts())
    print("\nInteractions:", interactions.shape)
    print(interactions["event_type"].value_counts())
    print("\nAvg interactions per user:", interactions.groupby("user_id").size().mean().round(1))
    print("Avg interactions per item:", interactions.groupby("item_id").size().mean().round(2))
