"""
non_personalized.py

Week 3 of the syllabus: non-personalized recommenders. These don't use any
per-user signal - they're the baseline every later method should beat.
"""

from __future__ import annotations

import pandas as pd


class PopularityRecommender:
    """Recommends the most-saved/contacted listings overall."""

    def __init__(self, interactions: pd.DataFrame):
        score = interactions.groupby("item_id")["weight"].sum()
        self.ranking = score.sort_values(ascending=False)

    def recommend(self, k: int = 10, exclude: set | None = None) -> list[str]:
        items = self.ranking.index.tolist()
        if exclude:
            items = [i for i in items if i not in exclude]
        return items[:k]


class BestValueRecommender:
    """
    Recommends listings with the best price-per-sqm within each neighborhood
    - a simple, explainable non-personalized heuristic that doesn't depend
    on any interaction data at all (useful for brand-new listings / cold start).
    """

    def __init__(self, listings: pd.DataFrame):
        self.listings = listings

    def recommend(self, neighborhood: str | None = None, k: int = 10) -> list[str]:
        df = self.listings
        if neighborhood:
            df = df[df["NEIGHBORHOOD"] == neighborhood]
        return df.sort_values("PRICE_PER_SQM").head(k)["item_id"].tolist()


class TrendingRecommender:
    """Recommends listings with the most recent/most concentrated interaction
    activity - a stand-in for "trending now" since idealista18 is a single
    snapshot rather than a live feed. Uses interaction *count* (not just
    weight) as a proxy for current attention."""

    def __init__(self, interactions: pd.DataFrame):
        counts = interactions.groupby("item_id").size()
        self.ranking = counts.sort_values(ascending=False)

    def recommend(self, k: int = 10) -> list[str]:
        return self.ranking.index[:k].tolist()


if __name__ == "__main__":
    from src.data_loader import load_barcelona_listings

    listings = load_barcelona_listings()
    interactions = pd.read_csv("data/processed/interactions.csv")

    pop = PopularityRecommender(interactions)
    print("Top 5 most popular listings:", pop.recommend(5))

    best_value = BestValueRecommender(listings)
    print("Top 5 best value in Sant Antoni:", best_value.recommend("Sant Antoni", 5))
