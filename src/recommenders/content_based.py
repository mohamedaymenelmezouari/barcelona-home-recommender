"""
content_based.py - Week 5

Content-based filtering using the listing features themselves - no
interaction data required. idealista18 has no text/description field, so
this is feature-vector similarity (price, size, rooms, amenities,
location), not NLP/TF-IDF based.

Two use cases are supported:
  - similar_items(item_id, k):  "more like this" for a single listing -
    works even for a brand-new listing with zero interactions (handles the
    item cold-start problem that CF cannot).
  - recommend(user_id, interactions, k): builds a user preference vector by
    weighted-averaging the features of everything they've saved/contacted,
    then ranks the rest of the catalog by similarity to that vector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = [
    "PRICE", "PRICE_PER_SQM", "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER",
    "CONSTRUCTIONYEAR", "DISTANCE_TO_CITY_CENTER", "DISTANCE_TO_METRO",
]
BINARY_FEATURES = [
    "HASTERRACE", "HASLIFT", "HASAIRCONDITIONING", "HASPARKINGSPACE",
    "HASBOXROOM", "HASWARDROBE", "HASSWIMMINGPOOL", "HASDOORMAN",
    "HASGARDEN", "ISDUPLEX", "ISSTUDIO",
]


class ContentBasedRecommender:
    def __init__(self, listings: pd.DataFrame):
        self.listings = listings.reset_index(drop=True)
        self.item_ids = self.listings["item_id"].to_numpy()
        self.item_idx = {it: i for i, it in enumerate(self.item_ids)}

        numeric = self.listings[NUMERIC_FEATURES].copy()
        numeric["CONSTRUCTIONYEAR"] = numeric["CONSTRUCTIONYEAR"].fillna(numeric["CONSTRUCTIONYEAR"].median())

        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric)

        # Binary amenity features are already 0/1 - scale them down a bit
        # relative to the standardized numeric features so price/size/location
        # (the things buyers usually filter on first) dominate the similarity,
        # with amenities as a secondary signal. Weight is a documented,
        # tunable design choice - worth a sentence in the slide deck.
        binary = self.listings[BINARY_FEATURES].fillna(0).to_numpy()
        amenity_weight = 0.5

        self.feature_matrix = np.hstack([numeric_scaled, amenity_weight * binary])

    def similar_items(self, item_id: str, k: int = 10) -> list[str]:
        """'More like this' for a single listing - works with zero interaction history."""
        if item_id not in self.item_idx:
            return []
        idx = self.item_idx[item_id]
        sims = cosine_similarity(self.feature_matrix[idx : idx + 1], self.feature_matrix).flatten()
        order = np.argsort(-sims)
        recs = [self.item_ids[i] for i in order if i != idx][:k]
        return recs

    def fit_user_profiles(self, interactions: pd.DataFrame) -> None:
        """
        Precompute every user's profile vector and seen-item set ONCE.
        Call this before recommend() when evaluating many users - looping
        recommend() with a raw interactions DataFrame re-filters it on every
        call, which is O(n_users x n_interactions) and far too slow for
        leave-one-out evaluation over thousands of users.
        """
        valid = interactions[interactions["item_id"].isin(self.item_idx)]
        self.user_profiles: dict[str, np.ndarray] = {}
        self.seen_items: dict[str, set] = {}

        for user_id, grp in valid.groupby("user_id"):
            idxs = grp["item_id"].map(self.item_idx).to_numpy()
            weights = grp["weight"].to_numpy(dtype=float)
            vecs = self.feature_matrix[idxs]
            self.user_profiles[user_id] = np.average(vecs, axis=0, weights=weights)
            self.seen_items[user_id] = set(grp["item_id"])

    def recommend(self, user_id: str, k: int = 10) -> list[str]:
        """Requires fit_user_profiles() to have been called first."""
        profile = getattr(self, "user_profiles", {}).get(user_id)
        if profile is None:
            return []
        sims = cosine_similarity(profile.reshape(1, -1), self.feature_matrix).flatten()
        seen = self.seen_items.get(user_id, set())
        order = np.argsort(-sims)
        recs = []
        for i in order:
            item_id = self.item_ids[i]
            if item_id in seen:
                continue
            recs.append(item_id)
            if len(recs) >= k:
                break
        return recs


if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.data_loader import load_barcelona_listings
    from src.evaluation import leave_one_out_split, precision_recall_at_k, ndcg_at_k
    from src.recommenders.non_personalized import PopularityRecommender
    from src.recommenders.collaborative import ItemBasedCF, UserBasedCF

    listings = load_barcelona_listings()
    interactions = pd.read_csv("data/processed/interactions.csv")
    train, test = leave_one_out_split(interactions)

    t0 = time.time()
    cb = ContentBasedRecommender(listings)
    cb_fit_time = time.time() - t0

    # sanity check: "more like this" for a sample listing should look similar
    sample_item = listings.iloc[0]
    print("Sample listing:", sample_item[["PRICE", "CONSTRUCTEDAREA", "NEIGHBORHOOD"]].to_dict())
    similar = cb.similar_items(sample_item["item_id"], k=5)
    print("Similar listings:")
    print(listings[listings["item_id"].isin(similar)][["PRICE", "CONSTRUCTEDAREA", "NEIGHBORHOOD"]])
    print()

    t0 = time.time()
    cb.fit_user_profiles(train)
    cb_profile_time = time.time() - t0

    results = {}
    pop = PopularityRecommender(train)
    seen_by_user = train.groupby("user_id")["item_id"].apply(set).to_dict()
    results["Popularity"] = precision_recall_at_k(
        lambda u, k: pop.recommend(k, exclude=seen_by_user.get(u, set())), test, k=10
    )

    item_cf = ItemBasedCF(train)
    results["Item-based CF"] = precision_recall_at_k(item_cf.recommend, test, k=10)

    user_cf = UserBasedCF(train)
    results["User-based CF"] = precision_recall_at_k(user_cf.recommend, test, k=10)

    results["Content-based"] = precision_recall_at_k(cb.recommend, test, k=10)

    print("\nPrecision/Recall@10 comparison:")
    for name, m in results.items():
        print(f"  {name:20s} recall={m['recall_at_k']:.4f}")
    print(f"\nContent-based: feature matrix build {cb_fit_time:.3f}s, "
          f"user profile precompute {cb_profile_time:.3f}s (no interaction "
          f"data needed for the feature matrix itself - only for personalizing to a user)")
