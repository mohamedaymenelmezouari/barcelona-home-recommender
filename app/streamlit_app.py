"""
Barcelona Home Recommender prototype.

A multi-view Streamlit application demonstrating five recommendation
techniques on the idealista18 Barcelona dataset:

    Browse   - paginated catalog with filters and inline save / similar-items
    For You  - personalized recommendations with per-method explanations
    Compare  - side-by-side comparison of all methods for the same buyer

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_barcelona_listings
from src.explanations import (
    build_user_profile_summary,
    explain_als,
    explain_best_value,
    explain_content_based,
    explain_item_cf,
    explain_popularity,
    explain_user_cf,
)
from src.recommenders.collaborative import ItemBasedCF, UserBasedCF
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.matrix_factorization import MatrixFactorizationRecommender
from src.recommenders.non_personalized import BestValueRecommender, PopularityRecommender

# ----------------------------------------------------------------------------
# Page configuration and custom styling
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Barcelona Home Recommender",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Trim Streamlit default top padding */
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px;}

    /* Header band */
    .app-header {
        background: linear-gradient(135deg, #1B263B 0%, #2A3F5F 100%);
        color: #FFFFFF;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.25rem;
    }
    .app-header h1 {color: #FFFFFF; font-size: 1.65rem; margin: 0; font-weight: 700;}
    .app-header p {color: #C8D3E0; margin: 0.35rem 0 0 0; font-size: 0.9rem;}

    /* KPI strip */
    .kpi-row {display: flex; gap: 0.75rem; margin-bottom: 1.25rem;}
    .kpi {
        flex: 1; background: #F7F6F4; border-radius: 10px; padding: 0.85rem 1rem;
        border: 1px solid #E5E7EB;
    }
    .kpi-label {color: #6B7280; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;}
    .kpi-value {color: #1B263B; font-size: 1.45rem; font-weight: 700; line-height: 1.2; margin-top: 0.1rem;}
    .kpi-sub {color: #6B7280; font-size: 0.75rem; margin-top: 0.15rem;}

    /* Listing card */
    .listing-card {
        background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px;
        padding: 0.85rem 1rem; margin-bottom: 0.6rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .listing-card.recommended {border-left: 3px solid #C1502E;}
    .listing-card .price {font-size: 1.25rem; font-weight: 700; color: #1B263B;}
    .listing-card .meta {color: #4B5563; font-size: 0.9rem; margin-top: 0.15rem;}
    .listing-card .neighborhood {color: #1C7293; font-weight: 600;}
    .listing-card .why {
        background: #FBF1ED; color: #8C3A1D; padding: 0.4rem 0.7rem;
        border-radius: 6px; font-size: 0.8rem; margin-top: 0.5rem;
        display: inline-block;
    }
    .listing-card .badges {margin-top: 0.4rem;}
    .badge {
        display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px;
        font-size: 0.7rem; background: #EEF2F7; color: #1B263B;
        margin-right: 0.3rem; margin-top: 0.2rem;
    }

    /* Method theming chips on Compare view */
    .method-chip {
        display: inline-block; padding: 0.3rem 0.7rem; border-radius: 999px;
        font-weight: 600; font-size: 0.85rem; color: #FFFFFF;
    }

    /* Suppress excessive Streamlit chrome */
    [data-testid="stSidebarNavItems"] {display: none;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

METHOD_COLORS = {
    "Popularity": "#6B7280",
    "Best value": "#1C7293",
    "Item-based CF": "#C1502E",
    "User-based CF": "#E0A458",
    "Content-based": "#2C5F2D",
    "Matrix Factorization (ALS)": "#5B3F87",
}

PERSONA_LABELS = {
    "young_couple_central": "Young couple, central",
    "family_needs_space": "Family, needs space",
    "investor_yield_focused": "Investor, yield focused",
    "retiree_downsizing": "Retiree, downsizing",
    "luxury_buyer": "Luxury buyer",
    "budget_first_buyer": "First-time buyer, budget",
    "remote_worker_needs_room": "Remote worker, needs room",
    "student_shared_flat": "Student, shared flat",
}


# ----------------------------------------------------------------------------
# Cached data and model loading
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset...")
def get_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    listings = load_barcelona_listings()
    interactions = pd.read_csv("data/processed/interactions.csv")
    users = pd.read_csv("data/processed/users.csv")
    return listings, interactions, users


@st.cache_resource(show_spinner="Fitting recommender models (first run only, ~5 seconds)...")
def load_models(_listings: pd.DataFrame, _interactions: pd.DataFrame) -> dict:
    """Fit every recommender once at startup and report individual fit times."""
    timings: dict[str, float] = {}

    t = time.time(); pop = PopularityRecommender(_interactions); timings["Popularity"] = time.time() - t
    t = time.time(); bv = BestValueRecommender(_listings); timings["Best value"] = time.time() - t
    icf = ItemBasedCF(_interactions); timings["Item-based CF"] = icf.fit_time_sec or 0.0
    ucf = UserBasedCF(_interactions); timings["User-based CF"] = ucf.fit_time_sec or 0.0
    t = time.time(); cb = ContentBasedRecommender(_listings); cb.fit_user_profiles(_interactions); timings["Content-based"] = time.time() - t
    mf = MatrixFactorizationRecommender(_interactions); timings["Matrix Factorization (ALS)"] = mf.fit_time_sec or 0.0

    seen_by_user = _interactions.groupby("user_id")["item_id"].apply(set).to_dict()
    item_popularity = _interactions.groupby("item_id").size().to_dict()
    bv_rank_in_neighborhood: dict[str, int] = {}
    for nb, grp in _listings.sort_values("PRICE_PER_SQM").groupby("NEIGHBORHOOD", sort=False):
        for rank, item_id in enumerate(grp["item_id"].tolist(), start=1):
            bv_rank_in_neighborhood[item_id] = rank

    return {
        "pop": pop, "best_value": bv, "item_cf": icf, "user_cf": ucf,
        "cb": cb, "mf": mf,
        "seen_by_user": seen_by_user, "item_popularity": item_popularity,
        "bv_rank": bv_rank_in_neighborhood, "timings": timings,
    }


# ----------------------------------------------------------------------------
# Recommendation dispatcher with explanations
# ----------------------------------------------------------------------------

def get_recommendations(
    method: str, user_id: str, models: dict, listings: pd.DataFrame,
    interactions: pd.DataFrame, neighborhood: str | None, k: int,
) -> tuple[list[str], dict[str, str]]:
    """Return ranked list of item_ids and a dict mapping item_id to explanation."""
    saved_session = st.session_state.get("saved_items", set())
    excluded = saved_session

    if method == "Popularity":
        baseline_exclude = models["seen_by_user"].get(user_id, set()) | excluded
        recs = models["pop"].recommend(k=k, exclude=baseline_exclude)
        explanations = {r: explain_popularity(None, models["item_popularity"].get(r, 0)) for r in recs}
        return recs, explanations

    if method == "Best value":
        recs = models["best_value"].recommend(neighborhood=neighborhood, k=k)
        listings_by_id = listings.set_index("item_id")
        explanations = {}
        for r in recs:
            row = listings_by_id.loc[r]
            explanations[r] = explain_best_value(row, models["bv_rank"].get(r, 0), row["NEIGHBORHOOD"])
        return recs, explanations

    # personalized methods
    if method == "Item-based CF":
        raw = models["item_cf"].recommend(user_id, k=k * 2)
    elif method == "User-based CF":
        raw = models["user_cf"].recommend(user_id, k=k * 2)
    elif method == "Content-based":
        raw = models["cb"].recommend(user_id, k=k * 2)
    elif method == "Matrix Factorization (ALS)":
        raw = models["mf"].recommend(user_id, k=k * 2)
    else:
        return [], {}

    recs = [r for r in raw if r not in saved_session][:k]
    listings_by_id = listings.set_index("item_id")

    if method == "Item-based CF":
        user_saves = interactions[
            (interactions["user_id"] == user_id) & (interactions["event_type"].isin(["save", "contact"]))
        ]
        anchor = None
        if not user_saves.empty and user_saves["item_id"].iloc[0] in listings_by_id.index:
            anchor = listings_by_id.loc[user_saves["item_id"].iloc[0]]
        explanations = {r: explain_item_cf(listings_by_id.loc[r], anchor) for r in recs}
    elif method == "User-based CF":
        n_similar = 25  # this is the cosine-similarity neighborhood size implicit in user-CF
        explanations = {r: explain_user_cf(listings_by_id.loc[r], n_similar) for r in recs}
    elif method == "Content-based":
        profile = build_user_profile_summary(user_id, interactions, listings)
        explanations = {r: explain_content_based(listings_by_id.loc[r], profile) for r in recs}
    else:  # ALS
        explanations = {r: explain_als(listings_by_id.loc[r]) for r in recs}

    return recs, explanations


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------

def render_header(listings: pd.DataFrame, interactions: pd.DataFrame, users: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="app-header">
            <h1>🏠 Barcelona Home Recommender</h1>
            <p>Five recommender techniques compared on real Barcelona listings.
            Built on the idealista18 academic dataset (Rey-Blanco et al., 2024).</p>
        </div>
        """, unsafe_allow_html=True,
    )

    kpis = [
        ("Listings", f"{len(listings):,}", "Barcelona, 2018"),
        ("Neighborhoods", f"{listings['NEIGHBORHOOD'].nunique()}", "across the city"),
        ("Synthetic buyers", f"{len(users):,}", "8 persona archetypes"),
        ("Interactions", f"{len(interactions):,}", "views, saves, contacts"),
    ]
    chunks = ['<div class="kpi-row">']
    for label, value, sub in kpis:
        chunks.append(
            f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>'
        )
    chunks.append("</div>")
    st.markdown("".join(chunks), unsafe_allow_html=True)


def amenity_badges(row: pd.Series) -> str:
    badges = []
    if row.get("HASLIFT"): badges.append("Lift")
    if row.get("HASTERRACE"): badges.append("Terrace")
    if row.get("HASAIRCONDITIONING"): badges.append("A/C")
    if row.get("HASPARKINGSPACE"): badges.append("Parking")
    if row.get("HASSWIMMINGPOOL"): badges.append("Pool")
    return "".join(f'<span class="badge">{b}</span>' for b in badges)


def render_listing_card(
    row: pd.Series, recommended: bool = False, explanation: str | None = None,
) -> str:
    cls = "listing-card recommended" if recommended else "listing-card"
    rooms = int(row["ROOMNUMBER"])
    baths = int(row.get("BATHNUMBER", 0))
    return (
        f'<div class="{cls}">'
        f'<div class="price">€{row["PRICE"]:,.0f}</div>'
        f'<div class="meta">{row["CONSTRUCTEDAREA"]:.0f} m² · {rooms} rooms · {baths} bath · '
        f'<span class="neighborhood">{row["NEIGHBORHOOD"]}</span></div>'
        f'<div class="meta" style="color: #6B7280; font-size: 0.8rem;">'
        f'€{row["PRICE_PER_SQM"]:.0f}/m² · {row["DISTANCE_TO_CITY_CENTER"]:.1f} km to center'
        f'</div>'
        f'<div class="badges">{amenity_badges(row)}</div>'
        + (f'<div class="why">{explanation}</div>' if explanation else "")
        + "</div>"
    )


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

def render_sidebar(users: pd.DataFrame, listings: pd.DataFrame, models: dict) -> dict:
    with st.sidebar:
        st.subheader("Identity")
        personas_sorted = sorted(users["persona"].unique(), key=lambda p: PERSONA_LABELS.get(p, p))
        persona = st.selectbox(
            "Buyer persona", personas_sorted,
            format_func=lambda p: PERSONA_LABELS.get(p, p),
        )
        persona_users = users[users["persona"] == persona]["user_id"].tolist()
        user_id = st.selectbox("Synthetic user", persona_users[:20], index=0)

        st.subheader("Filters")
        neighborhoods = ["All neighborhoods"] + sorted(listings["NEIGHBORHOOD"].dropna().unique())
        neighborhood = st.selectbox("Neighborhood", neighborhoods)
        price_max = st.slider("Max price (€)", 50_000, 2_500_000, 800_000, step=25_000, format="€%d")
        min_rooms = st.slider("Min rooms", 0, 6, 0)

        st.subheader("Session")
        saved = st.session_state.get("saved_items", set())
        st.metric("Listings saved this session", len(saved))
        if saved and st.button("Clear saved", use_container_width=True):
            st.session_state.saved_items = set()
            st.rerun()

        with st.expander("Model fit times", expanded=False):
            for name, t in models["timings"].items():
                st.caption(f"{name}: {t:.2f}s")

    return {
        "user_id": user_id, "persona": persona,
        "neighborhood": None if neighborhood == "All neighborhoods" else neighborhood,
        "price_max": price_max, "min_rooms": min_rooms,
    }


def apply_filters(listings: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    df = listings[listings["PRICE"] <= ctx["price_max"]]
    if ctx["neighborhood"]:
        df = df[df["NEIGHBORHOOD"] == ctx["neighborhood"]]
    if ctx["min_rooms"] > 0:
        df = df[df["ROOMNUMBER"] >= ctx["min_rooms"]]
    return df


# ----------------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------------

def view_browse(listings: pd.DataFrame, ctx: dict, models: dict) -> None:
    filtered = apply_filters(listings, ctx)

    head_left, head_right = st.columns([3, 1])
    with head_left:
        st.subheader(f"{len(filtered):,} listings match your filters")
    with head_right:
        sort_choice = st.selectbox(
            "Sort by",
            ["Price (low to high)", "Price (high to low)", "€/m² (best value)", "Largest first", "Most popular"],
            label_visibility="collapsed",
        )

    if sort_choice == "Price (low to high)":
        filtered = filtered.sort_values("PRICE")
    elif sort_choice == "Price (high to low)":
        filtered = filtered.sort_values("PRICE", ascending=False)
    elif sort_choice == "€/m² (best value)":
        filtered = filtered.sort_values("PRICE_PER_SQM")
    elif sort_choice == "Largest first":
        filtered = filtered.sort_values("CONSTRUCTEDAREA", ascending=False)
    elif sort_choice == "Most popular":
        filtered = filtered.assign(_pop=filtered["item_id"].map(models["item_popularity"]).fillna(0))
        filtered = filtered.sort_values("_pop", ascending=False).drop(columns="_pop")

    page_size = 10
    n_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    page = st.session_state.get("browse_page", 1)
    page = max(1, min(page, n_pages))

    start = (page - 1) * page_size
    page_rows = filtered.iloc[start : start + page_size]

    if page_rows.empty:
        st.info("No listings match the current filters. Try widening them in the sidebar.")
        return

    for _, row in page_rows.iterrows():
        col_card, col_actions = st.columns([5, 1])
        with col_card:
            st.markdown(render_listing_card(row), unsafe_allow_html=True)
        with col_actions:
            item_id = row["item_id"]
            saved = item_id in st.session_state.get("saved_items", set())
            label = "✓ Saved" if saved else "♡ Save"
            if st.button(label, key=f"save_{item_id}", use_container_width=True):
                st.session_state.setdefault("saved_items", set())
                if saved:
                    st.session_state.saved_items.discard(item_id)
                else:
                    st.session_state.saved_items.add(item_id)
                st.rerun()
            if st.button("Similar", key=f"sim_{item_id}", use_container_width=True):
                st.session_state.similar_anchor = item_id
                st.session_state.view = "For you"
                st.rerun()

    # Pagination
    pc1, pc2, pc3 = st.columns([1, 2, 1])
    with pc1:
        if st.button("◀ Previous", disabled=page <= 1, use_container_width=True):
            st.session_state.browse_page = page - 1
            st.rerun()
    with pc2:
        st.markdown(
            f"<div style='text-align: center; padding-top: 0.5rem;'>"
            f"Page <b>{page}</b> of {n_pages}</div>",
            unsafe_allow_html=True,
        )
    with pc3:
        if st.button("Next ▶", disabled=page >= n_pages, use_container_width=True):
            st.session_state.browse_page = page + 1
            st.rerun()


def view_for_you(
    listings: pd.DataFrame, interactions: pd.DataFrame, ctx: dict, models: dict,
) -> None:
    if "similar_anchor" in st.session_state:
        anchor_id = st.session_state.pop("similar_anchor")
        listings_by_id = listings.set_index("item_id")
        if anchor_id in listings_by_id.index:
            anchor = listings_by_id.loc[anchor_id]
            st.subheader("More like this")
            st.markdown(render_listing_card(anchor, recommended=False, explanation="Anchor listing"),
                        unsafe_allow_html=True)
            sims = models["cb"].similar_items(anchor_id, k=8)
            sim_rows = listings_by_id.loc[[s for s in sims if s in listings_by_id.index]]
            st.markdown("**Top 8 most similar listings (content-based):**")
            for _, row in sim_rows.iterrows():
                st.markdown(
                    render_listing_card(row, recommended=True,
                                        explanation="Matches anchor on price, size, location and amenities"),
                    unsafe_allow_html=True,
                )
            st.divider()

    method = st.radio(
        "Recommendation method",
        list(METHOD_COLORS.keys()),
        index=2, horizontal=True,
    )
    color = METHOD_COLORS[method]
    st.markdown(
        f'<div class="method-chip" style="background: {color}">{method}</div>',
        unsafe_allow_html=True,
    )

    k = st.slider("How many recommendations?", 5, 20, 10)

    with st.spinner(f"Generating recommendations with {method}..."):
        t0 = time.time()
        rec_ids, explanations = get_recommendations(
            method, ctx["user_id"], models, listings, interactions, ctx["neighborhood"], k=k,
        )
        gen_ms = (time.time() - t0) * 1000

    profile = build_user_profile_summary(ctx["user_id"], interactions, listings)
    info_cols = st.columns(3)
    info_cols[0].metric("Generated in", f"{gen_ms:.0f} ms")
    info_cols[1].metric("Listings engaged with", profile["n_engaged"])
    info_cols[2].metric("Avg saved price", f"€{profile['avg_price']:,.0f}" if profile["n_engaged"] else ", ")

    if not rec_ids:
        st.warning(
            "No recommendations available. This typically means the user has "
            "no interaction history yet (cold start). Try a different method or user.",
        )
        return

    listings_by_id = listings.set_index("item_id")
    valid_ids = [r for r in rec_ids if r in listings_by_id.index]
    for item_id in valid_ids:
        row = listings_by_id.loc[item_id]
        col_card, col_actions = st.columns([5, 1])
        with col_card:
            st.markdown(
                render_listing_card(row, recommended=True, explanation=explanations.get(item_id)),
                unsafe_allow_html=True,
            )
        with col_actions:
            saved = item_id in st.session_state.get("saved_items", set())
            if st.button("✓ Saved" if saved else "♡ Save", key=f"rec_save_{item_id}",
                         use_container_width=True):
                st.session_state.setdefault("saved_items", set())
                if saved:
                    st.session_state.saved_items.discard(item_id)
                else:
                    st.session_state.saved_items.add(item_id)
                st.rerun()


def view_compare(
    listings: pd.DataFrame, interactions: pd.DataFrame, ctx: dict, models: dict,
) -> None:
    st.subheader("Side-by-side method comparison")
    st.caption(
        "Top recommendation from each method for the selected buyer, so you "
        "can see how the algorithms diverge on the same input."
    )
    methods = ["Item-based CF", "User-based CF", "Content-based", "Matrix Factorization (ALS)"]
    listings_by_id = listings.set_index("item_id")

    cols = st.columns(len(methods))
    for col, method in zip(cols, methods):
        color = METHOD_COLORS[method]
        with col:
            st.markdown(
                f'<div class="method-chip" style="background: {color}">{method}</div>',
                unsafe_allow_html=True,
            )
            t0 = time.time()
            rec_ids, explanations = get_recommendations(
                method, ctx["user_id"], models, listings, interactions, ctx["neighborhood"], k=3,
            )
            gen_ms = (time.time() - t0) * 1000
            st.caption(f"Generated in {gen_ms:.0f} ms")
            for item_id in rec_ids:
                if item_id in listings_by_id.index:
                    st.markdown(
                        render_listing_card(
                            listings_by_id.loc[item_id], recommended=True,
                            explanation=explanations.get(item_id),
                        ),
                        unsafe_allow_html=True,
                    )

    st.divider()
    st.subheader("Quantitative comparison (offline evaluation)")
    eval_path = Path("data/processed/evaluation_results.csv")
    if eval_path.exists():
        eval_df = pd.read_csv(eval_path)
        eval_df.columns = ["Method", "Recall@10", "NDCG@10", "Coverage", "Novelty",
                           "Diversity", "Neighborhood bias", "Gen time (s)"]
        st.dataframe(
            eval_df.style.format({
                "Recall@10": "{:.4f}", "NDCG@10": "{:.4f}", "Coverage": "{:.2%}",
                "Novelty": "{:.2f}", "Diversity": "{:.2f}", "Neighborhood bias": "{:.2f}",
                "Gen time (s)": "{:.2f}",
            }).background_gradient(
                subset=["Recall@10", "NDCG@10"], cmap="Greens",
            ),
            hide_index=True, use_container_width=True,
        )
        st.caption(
            "Leave-one-out evaluation across all 1,200 synthetic buyers, k=10. "
            "Coverage = share of catalog ever recommended. Neighborhood bias = "
            "how concentrated geographically the recommendations are vs. the catalog."
        )
    else:
        st.info("Run `python scripts/run_full_evaluation.py` to populate this table.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    listings, interactions, users = get_data()
    models = load_models(listings, interactions)

    st.session_state.setdefault("saved_items", set())
    st.session_state.setdefault("view", "Browse")

    render_header(listings, interactions, users)
    ctx = render_sidebar(users, listings, models)

    view = st.radio(
        "View",
        ["Browse", "For you", "Compare"],
        index=["Browse", "For you", "Compare"].index(st.session_state.view),
        horizontal=True, label_visibility="collapsed",
    )
    st.session_state.view = view

    if view == "Browse":
        view_browse(listings, ctx, models)
    elif view == "For you":
        view_for_you(listings, interactions, ctx, models)
    else:
        view_compare(listings, interactions, ctx, models)


if __name__ == "__main__":
    main()
