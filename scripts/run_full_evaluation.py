"""
run_full_evaluation.py - Week 7

Runs every recommender built so far (Popularity, Item-CF, User-CF,
Content-based, ALS matrix factorization) through the full evaluation
suite: accuracy (Precision/Recall/NDCG@10) AND the "accuracy is not
enough" metrics the brief explicitly asks for - catalog coverage, novelty,
intra-list diversity, and neighborhood (geographic) bias.

Usage:
    python scripts/run_full_evaluation.py

Writes results to data/processed/evaluation_results.csv and prints a
summary table - use this directly in the slide deck's "method comparison"
section.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_barcelona_listings
from src.evaluation import (
    catalog_coverage,
    intra_list_diversity,
    leave_one_out_split,
    ndcg_at_k,
    neighborhood_bias_report,
    novelty_at_k,
    precision_recall_at_k,
)
from src.recommenders.collaborative import ItemBasedCF, UserBasedCF
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.matrix_factorization import MatrixFactorizationRecommender
from src.recommenders.non_personalized import PopularityRecommender

DIVERSITY_FEATURES = ["PRICE", "CONSTRUCTEDAREA", "ROOMNUMBER", "DISTANCE_TO_CITY_CENTER"]
K = 10


def neighborhood_bias_score(all_recs: list[list[str]], listings: pd.DataFrame) -> float:
    """Total variation distance between recommended-neighborhood share and
    catalog-neighborhood share. 0 = recommendations mirror the catalog's
    geographic spread exactly. Higher = more concentrated in fewer areas
    than the catalog actually is."""
    report = neighborhood_bias_report(all_recs, listings)
    return float((report["rec_share"] - report["catalog_share"]).abs().sum() / 2)


def evaluate_method(name, recommend_fn, all_users, test_df, listings, train_interactions, n_items_total):
    t0 = time.time()
    all_recs = [recommend_fn(u, K) for u in all_users]
    gen_time = time.time() - t0

    acc = precision_recall_at_k(recommend_fn, test_df, k=K)
    ndcg = ndcg_at_k(recommend_fn, test_df, k=K)
    coverage = catalog_coverage(all_recs, n_items_total)
    novelty = novelty_at_k(all_recs, train_interactions)
    diversity = intra_list_diversity(all_recs, listings, DIVERSITY_FEATURES)
    bias = neighborhood_bias_score(all_recs, listings)

    return {
        "method": name,
        "recall_at_10": acc["recall_at_k"],
        "ndcg_at_10": ndcg,
        "catalog_coverage": coverage,
        "novelty": novelty,
        "intra_list_diversity": diversity,
        "neighborhood_bias": bias,
        "rec_gen_time_sec": round(gen_time, 2),
    }


def main():
    listings = load_barcelona_listings()
    interactions = pd.read_csv("data/processed/interactions.csv")
    train, test = leave_one_out_split(interactions)
    n_items_total = listings["item_id"].nunique()
    all_users = train["user_id"].unique().tolist()

    seen_by_user = train.groupby("user_id")["item_id"].apply(set).to_dict()

    rows = []

    pop = PopularityRecommender(train)
    rows.append(evaluate_method(
        "Popularity",
        lambda u, k: pop.recommend(k, exclude=seen_by_user.get(u, set())),
        all_users, test, listings, train, n_items_total,
    ))

    item_cf = ItemBasedCF(train)
    rows.append(evaluate_method(
        "Item-based CF", item_cf.recommend, all_users, test, listings, train, n_items_total,
    ))

    user_cf = UserBasedCF(train)
    rows.append(evaluate_method(
        "User-based CF", user_cf.recommend, all_users, test, listings, train, n_items_total,
    ))

    cb = ContentBasedRecommender(listings)
    cb.fit_user_profiles(train)
    rows.append(evaluate_method(
        "Content-based", cb.recommend, all_users, test, listings, train, n_items_total,
    ))

    mf = MatrixFactorizationRecommender(train)
    rows.append(evaluate_method(
        "Matrix Factorization (ALS)", mf.recommend, all_users, test, listings, train, n_items_total,
    ))

    results = pd.DataFrame(rows).set_index("method")
    pd.set_option("display.width", 120)
    print(results.round(4))

    out_path = Path("data/processed/evaluation_results.csv")
    results.to_csv(out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
