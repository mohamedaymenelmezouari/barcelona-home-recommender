"""
collaborative.py - Week 4

Item-based and user-based collaborative filtering on the synthetic
interaction matrix (data/processed/interactions.csv).

Both use cosine similarity on the sparse user-item weight matrix
(weight: view=1, save=3, contact=5, we take the MAX weight observed per
user-item pair, since a "contact" implies the view/save already happened;
summing would double-count the same underlying interest).

Deliberately built side-by-side so fit_time_sec can be compared directly -
see the scalability note in README / the __main__ block below. With this
dataset's shape (1,200 synthetic users x ~38.7k interacted-with items),
the usual "item-based scales better than user-based" rule of thumb is
inverted: the user-user similarity matrix is far smaller to compute here.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


class _BaseCF:
    def __init__(self, interactions: pd.DataFrame):
        # Collapse repeated events on the same (user, item) pair into a
        # single strength value: the strongest signal observed.
        agg = interactions.groupby(["user_id", "item_id"])["weight"].max().reset_index()

        self.user_ids = agg["user_id"].unique()
        self.item_ids = agg["item_id"].unique()
        self.user_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item_idx = {it: i for i, it in enumerate(self.item_ids)}

        rows = agg["user_id"].map(self.user_idx).to_numpy()
        cols = agg["item_id"].map(self.item_idx).to_numpy()
        vals = agg["weight"].astype(float).to_numpy()

        self.user_item = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(len(self.user_ids), len(self.item_ids))
        )
        self.fit_time_sec: float | None = None

    def _seen_items(self, user_id: str) -> set:
        if user_id not in self.user_idx:
            return set()
        row = self.user_item[self.user_idx[user_id]]
        return set(row.indices)

    def _rank_and_filter(self, scores: np.ndarray, seen: set, k: int) -> list[str]:
        order = np.argsort(-scores)
        recs = []
        for idx in order:
            if idx in seen:
                continue
            recs.append(self.item_ids[idx])
            if len(recs) >= k:
                break
        return recs


class ItemBasedCF(_BaseCF):
    """'Buyers who liked this listing also liked...', similarity between
    items based on which users interacted with both."""

    def __init__(self, interactions: pd.DataFrame):
        super().__init__(interactions)
        t0 = time.time()
        item_user = self.user_item.T.tocsr()  # items x users
        self.item_sim = cosine_similarity(item_user, dense_output=False)  # items x items, sparse
        self.fit_time_sec = time.time() - t0

    def recommend(self, user_id: str, k: int = 10) -> list[str]:
        if user_id not in self.user_idx:
            return []
        user_vec = self.user_item[self.user_idx[user_id]]  # 1 x n_items
        scores = np.asarray((user_vec @ self.item_sim).todense()).flatten()
        return self._rank_and_filter(scores, self._seen_items(user_id), k)


class UserBasedCF(_BaseCF):
    """Find similar buyers, recommend what they liked that this user hasn't seen."""

    def __init__(self, interactions: pd.DataFrame):
        super().__init__(interactions)
        t0 = time.time()
        self.user_sim = cosine_similarity(self.user_item, dense_output=False)  # users x users, sparse
        self.fit_time_sec = time.time() - t0

    def recommend(self, user_id: str, k: int = 10) -> list[str]:
        if user_id not in self.user_idx:
            return []
        sim_row = self.user_sim[self.user_idx[user_id]]  # 1 x n_users
        scores = np.asarray((sim_row @ self.user_item).todense()).flatten()
        return self._rank_and_filter(scores, self._seen_items(user_id), k)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.evaluation import leave_one_out_split, precision_recall_at_k, ndcg_at_k
    from src.recommenders.non_personalized import PopularityRecommender

    interactions = pd.read_csv("data/processed/interactions.csv")
    train, test = leave_one_out_split(interactions)

    print(f"Matrix shape (train): {train.user_id.nunique()} users x {train.item_id.nunique()} items")
    print()

    results = {}

    pop = PopularityRecommender(train)
    seen_by_user = train.groupby("user_id")["item_id"].apply(set).to_dict()
    pop_recommend = lambda u, k: pop.recommend(k, exclude=seen_by_user.get(u, set()))
    results["Popularity (baseline)"] = {
        **precision_recall_at_k(pop_recommend, test, k=10),
        "ndcg_at_10": ndcg_at_k(pop_recommend, test, k=10),
        "fit_time_sec": None,
    }

    item_cf = ItemBasedCF(train)
    results["Item-based CF"] = {
        **precision_recall_at_k(item_cf.recommend, test, k=10),
        "ndcg_at_10": ndcg_at_k(item_cf.recommend, test, k=10),
        "fit_time_sec": round(item_cf.fit_time_sec, 3),
    }

    user_cf = UserBasedCF(train)
    results["User-based CF"] = {
        **precision_recall_at_k(user_cf.recommend, test, k=10),
        "ndcg_at_10": ndcg_at_k(user_cf.recommend, test, k=10),
        "fit_time_sec": round(user_cf.fit_time_sec, 3),
    }

    print(pd.DataFrame(results).T)
