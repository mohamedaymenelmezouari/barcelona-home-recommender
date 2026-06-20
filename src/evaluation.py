"""
evaluation.py

Basic evaluation from week one (per the brief: "use Agile development, basic
evaluation from the beginning"), extended over the following weeks with
diversity, novelty, and bias analysis as more methods come online.

Train/test split strategy: for each user, hold out their single highest-
weight interaction (their "best" save/contact) as the test item, train on
everything else. This is a standard leave-one-out protocol for implicit
feedback recommenders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def leave_one_out_split(interactions: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    test_rows = []
    train_idx = []

    for user_id, grp in interactions.groupby("user_id"):
        # prefer holding out a "save" or "contact" event as the test target,
        # since that is the recommender's actual goal (drive saves/leads)
        candidates = grp[grp["event_type"].isin(["save", "contact"])]
        if len(candidates) == 0:
            train_idx.extend(grp.index.tolist())
            continue
        test_row = candidates.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0]
        test_rows.append(test_row)
        # IMPORTANT: remove ALL rows for this (user, item) pair from train,
        # not just the sampled one. Otherwise a leftover "view" row for the
        # same item keeps it marked as "seen", and any recommender that
        # correctly excludes already-seen items can never recommend it -
        # making the held-out item structurally unrecoverable regardless of
        # how good the model is.
        same_item_mask = grp["item_id"] == test_row["item_id"]
        train_idx.extend(grp.index[~same_item_mask].tolist())

    test_df = pd.DataFrame(test_rows)
    train_df = interactions.loc[train_idx]
    return train_df, test_df


def precision_recall_at_k(
    recommend_fn,
    test_df: pd.DataFrame,
    k: int = 10,
) -> dict[str, float]:
    """
    recommend_fn(user_id, k) -> list[item_id]
    Computes Precision@k and Recall@k against the single held-out item per
    user (so Recall@k is equivalent to Hit Rate@k in this single-item setup).
    """
    hits = 0
    n_users = 0
    for _, row in test_df.iterrows():
        recs = recommend_fn(row["user_id"], k)
        n_users += 1
        if row["item_id"] in recs:
            hits += 1

    hit_rate = hits / max(n_users, 1)
    precision = hit_rate / k  # only one relevant item per user in this setup
    return {"precision_at_k": precision, "recall_at_k": hit_rate, "n_evaluated": n_users}


def ndcg_at_k(recommend_fn, test_df: pd.DataFrame, k: int = 10) -> float:
    scores = []
    for _, row in test_df.iterrows():
        recs = recommend_fn(row["user_id"], k)
        if row["item_id"] in recs:
            rank = recs.index(row["item_id"]) + 1
            scores.append(1.0 / np.log2(rank + 1))
        else:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0


def catalog_coverage(all_recs: list[list[str]], n_items_total: int) -> float:
    """Fraction of the catalog that ever appears in any recommendation list."""
    recommended = set()
    for recs in all_recs:
        recommended.update(recs)
    return len(recommended) / n_items_total


def novelty_at_k(all_recs: list[list[str]], interactions: pd.DataFrame) -> float:
    """
    Average inverse popularity of recommended items (higher = more novel).
    Popularity is the number of distinct users who interacted with an item.
    """
    pop = interactions.groupby("item_id")["user_id"].nunique()
    n_users = interactions["user_id"].nunique()

    scores = []
    for recs in all_recs:
        for item in recs:
            p = pop.get(item, 1) / n_users
            scores.append(-np.log2(p + 1e-9))
    return float(np.mean(scores)) if scores else 0.0


def intra_list_diversity(all_recs: list[list[str]], listings: pd.DataFrame, feature_cols: list[str]) -> float:
    """
    Average pairwise distance between recommended items within each list,
    using normalized feature vectors. Higher = more diverse recommendations.
    """
    feats = listings.set_index("item_id")[feature_cols]
    feats = (feats - feats.mean()) / (feats.std() + 1e-9)

    diversities = []
    for recs in all_recs:
        vecs = feats.loc[[i for i in recs if i in feats.index]].to_numpy()
        if len(vecs) < 2:
            continue
        dists = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                dists.append(np.linalg.norm(vecs[i] - vecs[j]))
        diversities.append(np.mean(dists))
    return float(np.mean(diversities)) if diversities else 0.0


def neighborhood_bias_report(all_recs: list[list[str]], listings: pd.DataFrame) -> pd.Series:
    """
    How concentrated are recommendations across neighborhoods, vs. how listings
    are actually distributed in the catalog? A method that only ever
    recommends 3 prime neighborhoods out of 69 has a geographic bias problem
    worth discussing in the slide deck.
    """
    neigh_lookup = listings.set_index("item_id")["NEIGHBORHOOD"]
    rec_neighborhoods = []
    for recs in all_recs:
        rec_neighborhoods.extend(neigh_lookup.reindex(recs).dropna().tolist())
    rec_dist = pd.Series(rec_neighborhoods).value_counts(normalize=True)
    catalog_dist = listings["NEIGHBORHOOD"].value_counts(normalize=True)
    return pd.DataFrame({"rec_share": rec_dist, "catalog_share": catalog_dist}).fillna(0)


if __name__ == "__main__":
    from src.recommenders.non_personalized import PopularityRecommender

    interactions = pd.read_csv("data/processed/interactions.csv")
    train, test = leave_one_out_split(interactions)
    print("Train:", train.shape, "Test:", test.shape)

    pop = PopularityRecommender(train)

    def recommend_fn(user_id, k):
        return pop.recommend(k=k)

    metrics = precision_recall_at_k(recommend_fn, test, k=10)
    print("Popularity baseline @10:", metrics)
    print("NDCG@10:", ndcg_at_k(recommend_fn, test, k=10))
