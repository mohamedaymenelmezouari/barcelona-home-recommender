"""
build_dataset.py

One-command pipeline: download idealista18 -> clean listings -> generate
synthetic users + interactions. Run this whenever you want a fresh dataset
(e.g. after changing persona definitions).

Usage:
    python scripts/build_dataset.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_barcelona_listings
from src.personas import build_synthetic_data

if __name__ == "__main__":
    print("Step 1/2: loading and cleaning Barcelona listings...")
    listings = load_barcelona_listings(force_refresh=True)
    print(f"  -> {len(listings)} listings ready.")

    print("Step 2/2: generating synthetic users and interactions...")
    users, interactions = build_synthetic_data(listings, n_users=1200)
    print(f"  -> {len(users)} users, {len(interactions)} interaction events.")

    print("\nDone. Files written to data/processed/:")
    print("  - barcelona_listings.csv")
    print("  - users.csv")
    print("  - interactions.csv")
