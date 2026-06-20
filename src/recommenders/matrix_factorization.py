"""
matrix_factorization.py - Week 6

Alternating Least Squares (ALS) matrix factorization on the same implicit
weight matrix used by collaborative.py (view=1, save=3, contact=5, max per
user-item pair). Reuses _BaseCF for matrix construction so all methods sit
on identical data - any difference in results is down to the algorithm,
not the data prep.

ALS learns latent factors for every user and item; a user's score for an
item is the dot product of their factor vectors. Unlike neighborhood-based
CF, this doesn't require computing a full similarity matrix at all -
that's the central scalability argument for matrix factorization at
real-world scale, even though our catalog here is small enough that
neighborhood CF is already fast.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.recommenders.collaborative import _BaseCF  # noqa: E402


class MatrixFactorizationRecommender(_BaseCF):
    def __init__(
        self,
        interactions: pd.DataFrame,
        factors: int = 32,
        regularization: float = 0.05,
        iterations: int = 20,
        seed: int = 42,
    ):
        super().__init__(interactions)
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=seed,
        )
        t0 = time.time()
        # implicit expects a user-item matrix of confidence weights, already
        # what self.user_item is (max weight per user-item pair)
        self.model.fit(self.user_item, show_progress=False)
        self.fit_time_sec = time.time() - t0

    def recommend(self, user_id: str, k: int = 10) -> list[str]:
        if user_id not in self.user_idx:
            return []
        u = self.user_idx[user_id]
        ids, _scores = self.model.recommend(
            u, self.user_item[u], N=k, filter_already_liked_items=True
        )
        return [self.item_ids[i] for i in ids]


if __name__ == "__main__":
    from src.data_loader import load_barcelona_listings
    from src.evaluation import leave_one_out_split, precision_recall_at_k, ndcg_at_k
    from src.recommenders.collaborative import ItemBasedCF, UserBasedCF
    from src.recommenders.content_based import ContentBasedRecommender
    from src.recommenders.non_personalized import PopularityRecommender

    listings = load_barcelona_listings()
    interactions = pd.read_csv("data/processed/interactions.csv")
    train, test = leave_one_out_split(interactions)

    results = {}
    timings = {}

    pop = PopularityRecommender(train)
    seen_by_user = train.groupby("user_id")["item_id"].apply(set).to_dict()
    results["Popularity"] = precision_recall_at_k(
        lambda u, k: pop.recommend(k, exclude=seen_by_user.get(u, set())), test, k=10
    )
    timings["Popularity"] = None

    item_cf = ItemBasedCF(train)
    results["Item-based CF"] = precision_recall_at_k(item_cf.recommend, test, k=10)
    timings["Item-based CF"] = item_cf.fit_time_sec

    user_cf = UserBasedCF(train)
    results["User-based CF"] = precision_recall_at_k(user_cf.recommend, test, k=10)
    timings["User-based CF"] = user_cf.fit_time_sec

    cb = ContentBasedRecommender(listings)
    cb.fit_user_profiles(train)
    results["Content-based"] = precision_recall_at_k(cb.recommend, test, k=10)
    timings["Content-based"] = None  # see content_based.py for its own timing breakdown

    mf = MatrixFactorizationRecommender(train)
    results["Matrix Factorization (ALS)"] = precision_recall_at_k(mf.recommend, test, k=10)
    timings["Matrix Factorization (ALS)"] = mf.fit_time_sec

    print("Method                        recall@10   ndcg@10   fit_time_sec")
    for name in results:
        ndcg = ndcg_at_k(
            {
                "Popularity": lambda u, k: pop.recommend(k, exclude=seen_by_user.get(u, set())),
                "Item-based CF": item_cf.recommend,
                "User-based CF": user_cf.recommend,
                "Content-based": cb.recommend,
                "Matrix Factorization (ALS)": mf.recommend,
            }[name],
            test,
            k=10,
        )
        t = timings[name]
        t_str = f"{t:.3f}" if t is not None else "n/a"
        print(f"{name:30s} {results[name]['recall_at_k']:.4f}     {ndcg:.4f}    {t_str}")
