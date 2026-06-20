"""
Barcelona Home Recommender — professional prototype.

Three views on the idealista18 Barcelona dataset:
    Browse   — paginated catalog with filters, sort, save, and similar-items
    For You  — personalized recommendations with user profile and explanations
    Compare  — side-by-side method comparison with offline evaluation table

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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Barcelona Home Recommender",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ── Layout ── */
.block-container {
    padding-top: 1rem;
    padding-bottom: 4rem;
    max-width: 1380px;
}

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #0F1C2E 0%, #1B3A5C 60%, #22527A 100%);
    color: #fff;
    padding: 1.6rem 2rem 1.4rem;
    border-radius: 14px;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.18);
}
.app-header-icon {
    font-size: 2.6rem;
    line-height: 1;
    flex-shrink: 0;
}
.app-header h1 {
    color: #fff;
    font-size: 1.75rem;
    margin: 0 0 0.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.app-header p {
    color: #A8C4DE;
    margin: 0;
    font-size: 0.88rem;
    line-height: 1.5;
}
.app-header .tag {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: #D4E8F7;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    margin-top: 0.5rem;
    margin-right: 0.4rem;
}

/* ── KPI strip ── */
.kpi-row {display: flex; gap: 0.8rem; margin-bottom: 1.4rem;}
.kpi {
    flex: 1;
    background: #fff;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    border: 1px solid #E5E9EF;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    position: relative;
    overflow: hidden;
}
.kpi::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1B3A5C, #C1502E);
    border-radius: 12px 12px 0 0;
}
.kpi-icon {font-size: 1.3rem; margin-bottom: 0.35rem; display: block;}
.kpi-label {color: #6B7280; font-size: 0.73rem; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600;}
.kpi-value {color: #0F1C2E; font-size: 1.5rem; font-weight: 800; line-height: 1.1; margin-top: 0.15rem;}
.kpi-sub {color: #9CA3AF; font-size: 0.73rem; margin-top: 0.2rem;}

/* ── Section title ── */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0F1C2E;
    margin: 0 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.section-title .count {
    background: #EEF2F7;
    color: #4B5563;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.1rem 0.5rem;
    border-radius: 999px;
}

/* ── Listing card ── */
.listing-card {
    background: #fff;
    border: 1px solid #E5E9EF;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.65rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: box-shadow 0.15s ease;
    position: relative;
}
.listing-card:hover {box-shadow: 0 4px 12px rgba(0,0,0,0.08);}
.listing-card.recommended {
    border-left: 3px solid #C1502E;
    background: linear-gradient(to right, #FEFAF9 0%, #fff 6%);
}
.listing-card .card-top {display: flex; justify-content: space-between; align-items: flex-start;}
.listing-card .price {font-size: 1.3rem; font-weight: 800; color: #0F1C2E; letter-spacing: -0.01em;}
.listing-card .price-psm {
    font-size: 0.78rem;
    color: #fff;
    background: #1B3A5C;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    font-weight: 600;
    white-space: nowrap;
}
.listing-card .meta-row {color: #4B5563; font-size: 0.88rem; margin-top: 0.3rem;}
.listing-card .neighborhood {
    color: #1B3A5C;
    font-weight: 700;
    background: #EEF5FB;
    padding: 0.1rem 0.45rem;
    border-radius: 5px;
    font-size: 0.82rem;
}
.listing-card .distance {color: #9CA3AF; font-size: 0.78rem; margin-top: 0.2rem;}
.listing-card .why {
    background: #FBF1ED;
    color: #7B3020;
    padding: 0.4rem 0.65rem;
    border-radius: 7px;
    font-size: 0.79rem;
    margin-top: 0.55rem;
    line-height: 1.45;
    border-left: 2px solid #C1502E;
}
.listing-card .why::before {content: "Why this listing  ·  "; font-weight: 700;}
.listing-card .badges {margin-top: 0.45rem; display: flex; flex-wrap: wrap; gap: 0.3rem;}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.2rem;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    font-size: 0.71rem;
    font-weight: 600;
    background: #F3F4F6;
    color: #374151;
    border: 1px solid #E5E7EB;
}

/* ── User profile card ── */
.profile-card {
    background: linear-gradient(135deg, #0F1C2E 0%, #1B3A5C 100%);
    color: #fff;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    display: flex;
    gap: 1.2rem;
    align-items: center;
}
.profile-avatar {
    width: 48px; height: 48px;
    border-radius: 50%;
    background: rgba(255,255,255,0.15);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0;
}
.profile-info .label {color: #A8C4DE; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600;}
.profile-info .name {color: #fff; font-size: 0.95rem; font-weight: 700; margin-top: 0.1rem;}
.profile-info .sub {color: #C8DCEA; font-size: 0.78rem; margin-top: 0.2rem;}
.profile-stat {text-align: center; flex: 1;}
.profile-stat .stat-val {color: #fff; font-size: 1.1rem; font-weight: 800;}
.profile-stat .stat-lbl {color: #A8C4DE; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;}
.profile-divider {width: 1px; background: rgba(255,255,255,0.15); align-self: stretch;}

/* ── Method chip ── */
.method-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.82rem;
    color: #fff;
    letter-spacing: 0.01em;
    margin-bottom: 0.5rem;
}

/* ── Method description ── */
.method-desc {
    background: #F8FAFC;
    border: 1px solid #E5E9EF;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    font-size: 0.8rem;
    color: #4B5563;
    margin-bottom: 0.75rem;
    line-height: 1.5;
}

/* ── Tabs ── */
.tab-bar {
    display: flex;
    gap: 0;
    background: #F1F3F7;
    border-radius: 10px;
    padding: 3px;
    margin-bottom: 1.2rem;
}
.tab-btn {
    flex: 1;
    text-align: center;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-size: 0.88rem;
    font-weight: 600;
    cursor: pointer;
    color: #6B7280;
    border: none;
    background: transparent;
    transition: all 0.15s;
}
.tab-btn.active {
    background: #fff;
    color: #0F1C2E;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
}

/* ── Pagination ── */
.page-info {
    text-align: center;
    color: #6B7280;
    font-size: 0.85rem;
    padding-top: 0.5rem;
    font-weight: 500;
}

/* ── Eval table ── */
.eval-header {
    font-size: 1.0rem;
    font-weight: 700;
    color: #0F1C2E;
    margin: 1.2rem 0 0.3rem;
}
.eval-caption {
    font-size: 0.78rem;
    color: #6B7280;
    line-height: 1.5;
    margin-bottom: 0.75rem;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    color: #9CA3AF;
    font-size: 0.76rem;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid #F1F3F7;
    margin-top: 2rem;
}
.app-footer strong {color: #4B5563;}

/* ── Misc ── */
[data-testid="stSidebarNavItems"] {display: none;}
footer {visibility: hidden;}
.stRadio > div {gap: 0.2rem;}
div[data-testid="stMetric"] {
    background: #F8FAFC;
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    border: 1px solid #E5E9EF;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "Popularity":                "#6B7280",
    "Best value":                "#1B3A5C",
    "Item-based CF":             "#C1502E",
    "User-based CF":             "#D97706",
    "Content-based":             "#2C6E2F",
    "Matrix Factorization (ALS)":"#5B3F87",
}

METHOD_ICONS = {
    "Popularity":                "📊",
    "Best value":                "💎",
    "Item-based CF":             "🔗",
    "User-based CF":             "👥",
    "Content-based":             "🏷️",
    "Matrix Factorization (ALS)":"⚡",
}

METHOD_DESC = {
    "Popularity": "Recommends the listings most interacted with across all users. Fast and simple — no personalization.",
    "Best value": "Ranks listings by price per m² within your chosen neighborhood. No interaction data needed.",
    "Item-based CF": "Finds listings similar to ones you've already saved or contacted, based on co-interaction patterns.",
    "User-based CF": "Identifies buyers with similar tastes and surfaces what they liked but you haven't seen.",
    "Content-based": "Matches listings to your interaction history using property features (price, size, location, amenities).",
    "Matrix Factorization (ALS)": "Learns latent preference factors from the full interaction matrix. Best speed-accuracy trade-off.",
}

PERSONA_LABELS = {
    "young_couple_central":     "Young couple — central",
    "family_needs_space":       "Family — needs space",
    "investor_yield_focused":   "Investor — yield focused",
    "retiree_downsizing":       "Retiree — downsizing",
    "luxury_buyer":             "Luxury buyer",
    "budget_first_buyer":       "First-time buyer — budget",
    "remote_worker_needs_room": "Remote worker — needs room",
    "student_shared_flat":      "Student — shared flat",
}

PERSONA_ICONS = {
    "young_couple_central":     "💑",
    "family_needs_space":       "👨‍👩‍👧‍👦",
    "investor_yield_focused":   "📈",
    "retiree_downsizing":       "🏡",
    "luxury_buyer":             "✨",
    "budget_first_buyer":       "🔑",
    "remote_worker_needs_room": "💻",
    "student_shared_flat":      "🎓",
}

# ---------------------------------------------------------------------------
# Data + model loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset…")
def get_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    listings = load_barcelona_listings()
    interactions = pd.read_csv("data/processed/interactions.csv")
    users = pd.read_csv("data/processed/users.csv")
    return listings, interactions, users


@st.cache_resource(show_spinner="Fitting recommender models (first run only, ~5 s)…")
def load_models(_listings: pd.DataFrame, _interactions: pd.DataFrame) -> dict:
    timings: dict[str, float] = {}

    t = time.time(); pop = PopularityRecommender(_interactions);            timings["Popularity"] = time.time() - t
    t = time.time(); bv  = BestValueRecommender(_listings);                 timings["Best value"] = time.time() - t
    icf = ItemBasedCF(_interactions);                                        timings["Item-based CF"] = icf.fit_time_sec or 0.0
    ucf = UserBasedCF(_interactions);                                        timings["User-based CF"] = ucf.fit_time_sec or 0.0
    t = time.time(); cb  = ContentBasedRecommender(_listings);
    cb.fit_user_profiles(_interactions);                                     timings["Content-based"] = time.time() - t
    mf  = MatrixFactorizationRecommender(_interactions);                     timings["Matrix Factorization (ALS)"] = mf.fit_time_sec or 0.0

    seen_by_user    = _interactions.groupby("user_id")["item_id"].apply(set).to_dict()
    item_popularity = _interactions.groupby("item_id").size().to_dict()
    bv_rank_in_nb: dict[str, int] = {}
    for _nb, grp in _listings.sort_values("PRICE_PER_SQM").groupby("NEIGHBORHOOD", sort=False):
        for rank, item_id in enumerate(grp["item_id"].tolist(), start=1):
            bv_rank_in_nb[item_id] = rank

    return {
        "pop": pop, "best_value": bv, "item_cf": icf, "user_cf": ucf,
        "cb": cb, "mf": mf,
        "seen_by_user": seen_by_user, "item_popularity": item_popularity,
        "bv_rank": bv_rank_in_nb, "timings": timings,
    }


# ---------------------------------------------------------------------------
# Recommendation dispatcher
# ---------------------------------------------------------------------------

def get_recommendations(
    method: str,
    user_id: str,
    models: dict,
    listings: pd.DataFrame,
    interactions: pd.DataFrame,
    neighborhood: str | None,
    k: int,
) -> tuple[list[str], dict[str, str]]:
    saved_session = st.session_state.get("saved_items", set())
    excluded = saved_session

    if method == "Popularity":
        excl = models["seen_by_user"].get(user_id, set()) | excluded
        recs = models["pop"].recommend(k=k, exclude=excl)
        exps = {r: explain_popularity(None, models["item_popularity"].get(r, 0)) for r in recs}
        return recs, exps

    if method == "Best value":
        recs = models["best_value"].recommend(neighborhood=neighborhood, k=k)
        lbi  = listings.set_index("item_id")
        exps = {}
        for r in recs:
            row = lbi.loc[r]
            exps[r] = explain_best_value(row, models["bv_rank"].get(r, 0), row["NEIGHBORHOOD"])
        return recs, exps

    # personalized
    raw: list[str]
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
    lbi  = listings.set_index("item_id")

    if method == "Item-based CF":
        user_saves = interactions[
            (interactions["user_id"] == user_id) &
            (interactions["event_type"].isin(["save", "contact"]))
        ]
        anchor = None
        if not user_saves.empty and user_saves["item_id"].iloc[0] in lbi.index:
            anchor = lbi.loc[user_saves["item_id"].iloc[0]]
        exps = {r: explain_item_cf(lbi.loc[r], anchor) for r in recs}
    elif method == "User-based CF":
        exps = {r: explain_user_cf(lbi.loc[r], 25) for r in recs}
    elif method == "Content-based":
        profile = build_user_profile_summary(user_id, interactions, listings)
        exps = {r: explain_content_based(lbi.loc[r], profile) for r in recs}
    else:
        exps = {r: explain_als(lbi.loc[r]) for r in recs}

    return recs, exps


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_header(listings: pd.DataFrame, interactions: pd.DataFrame, users: pd.DataFrame) -> None:
    tags = "".join(
        f'<span class="tag">{t}</span>'
        for t in ["idealista18 dataset", "Barcelona 2018", "5 algorithms", "Esade RecSys"]
    )
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-header-icon">🏠</div>
            <div>
                <h1>Barcelona Home Recommender</h1>
                <p>Five recommender techniques compared side-by-side on real Barcelona property listings.</p>
                <div style="margin-top:0.5rem">{tags}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpis = [
        ("🏘️", "Listings", f"{len(listings):,}", "Barcelona, 2018"),
        ("🗺️", "Neighborhoods", f"{listings['NEIGHBORHOOD'].nunique()}", "across the city"),
        ("👤", "Synthetic buyers", f"{len(users):,}", "8 persona archetypes"),
        ("🔁", "Interactions", f"{len(interactions):,}", "views · saves · contacts"),
    ]
    chunks = ['<div class="kpi-row">']
    for icon, label, value, sub in kpis:
        chunks.append(
            f'<div class="kpi">'
            f'<span class="kpi-icon">{icon}</span>'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'</div>'
        )
    chunks.append("</div>")
    st.markdown("".join(chunks), unsafe_allow_html=True)


def amenity_badges(row: pd.Series) -> str:
    amenities = [
        ("HASLIFT",           "🛗 Lift"),
        ("HASTERRACE",        "🌿 Terrace"),
        ("HASAIRCONDITIONING", "❄️ A/C"),
        ("HASPARKINGSPACE",   "🚗 Parking"),
        ("HASSWIMMINGPOOL",   "🏊 Pool"),
    ]
    return "".join(
        f'<span class="badge">{label}</span>'
        for key, label in amenities
        if row.get(key)
    )


def render_listing_card(
    row: pd.Series,
    recommended: bool = False,
    explanation: str | None = None,
) -> str:
    cls     = "listing-card recommended" if recommended else "listing-card"
    rooms   = int(row["ROOMNUMBER"])
    baths   = int(row.get("BATHNUMBER", 0))
    dist    = row["DISTANCE_TO_CITY_CENTER"]
    psm     = row["PRICE_PER_SQM"]
    area    = row["CONSTRUCTEDAREA"]
    price   = row["PRICE"]
    nb      = row["NEIGHBORHOOD"]

    why_html = f'<div class="why">{explanation}</div>' if explanation else ""
    badges   = amenity_badges(row)

    return f"""
    <div class="{cls}">
        <div class="card-top">
            <div class="price">€{price:,.0f}</div>
            <div class="price-psm">€{psm:.0f}/m²</div>
        </div>
        <div class="meta-row">
            {area:.0f} m² &nbsp;·&nbsp; {rooms} rooms &nbsp;·&nbsp; {baths} bath
            &nbsp;&nbsp;<span class="neighborhood">{nb}</span>
        </div>
        <div class="distance">📍 {dist:.1f} km to city centre</div>
        <div class="badges">{badges}</div>
        {why_html}
    </div>
    """


def render_footer() -> None:
    st.markdown(
        """
        <div class="app-footer">
            <strong>Barcelona Home Recommender</strong> &nbsp;·&nbsp;
            Built by <strong>Mohamed Aymen Elmezouari</strong> &nbsp;·&nbsp;
            Esade Recommender Systems &nbsp;·&nbsp;
            Dataset: <em>idealista18</em> (Rey-Blanco et al., 2024, ODbL-1.0)
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(users: pd.DataFrame, listings: pd.DataFrame, models: dict) -> dict:
    with st.sidebar:
        st.markdown("### 👤 Buyer identity")
        personas_sorted = sorted(users["persona"].unique(), key=lambda p: PERSONA_LABELS.get(p, p))
        persona = st.selectbox(
            "Persona",
            personas_sorted,
            format_func=lambda p: f"{PERSONA_ICONS.get(p,'')}  {PERSONA_LABELS.get(p, p)}",
        )
        persona_users = users[users["persona"] == persona]["user_id"].tolist()
        user_id = st.selectbox("Synthetic user", persona_users[:20], index=0)

        st.divider()
        st.markdown("### 🔍 Filters")
        neighborhoods = ["All neighborhoods"] + sorted(listings["NEIGHBORHOOD"].dropna().unique())
        neighborhood = st.selectbox("Neighborhood", neighborhoods)
        price_max = st.slider(
            "Max price (€)", 50_000, 2_500_000, 800_000, step=25_000, format="€%d",
        )
        min_rooms = st.slider("Min rooms", 0, 6, 0)

        st.divider()
        st.markdown("### 🔖 Session")
        saved = st.session_state.get("saved_items", set())
        col1, col2 = st.columns(2)
        col1.metric("Saved", len(saved))
        if saved:
            if col2.button("Clear", use_container_width=True):
                st.session_state.saved_items = set()
                st.rerun()

        with st.expander("Model fit times", expanded=False):
            for name, t in models["timings"].items():
                icon = METHOD_ICONS.get(name, "")
                st.caption(f"{icon} {name}: **{t:.2f} s**")

    return {
        "user_id":      user_id,
        "persona":      persona,
        "neighborhood": None if neighborhood == "All neighborhoods" else neighborhood,
        "price_max":    price_max,
        "min_rooms":    min_rooms,
    }


def apply_filters(listings: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    df = listings[listings["PRICE"] <= ctx["price_max"]]
    if ctx["neighborhood"]:
        df = df[df["NEIGHBORHOOD"] == ctx["neighborhood"]]
    if ctx["min_rooms"] > 0:
        df = df[df["ROOMNUMBER"] >= ctx["min_rooms"]]
    return df


# ---------------------------------------------------------------------------
# Browse view
# ---------------------------------------------------------------------------

def view_browse(listings: pd.DataFrame, ctx: dict, models: dict) -> None:
    filtered = apply_filters(listings, ctx)

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            f'<div class="section-title">Catalog '
            f'<span class="count">{len(filtered):,} listings</span></div>',
            unsafe_allow_html=True,
        )
    with c2:
        sort_choice = st.selectbox(
            "Sort",
            ["Price ↑", "Price ↓", "€/m² best value", "Largest first", "Most popular"],
            label_visibility="collapsed",
        )

    if sort_choice == "Price ↑":
        filtered = filtered.sort_values("PRICE")
    elif sort_choice == "Price ↓":
        filtered = filtered.sort_values("PRICE", ascending=False)
    elif sort_choice == "€/m² best value":
        filtered = filtered.sort_values("PRICE_PER_SQM")
    elif sort_choice == "Largest first":
        filtered = filtered.sort_values("CONSTRUCTEDAREA", ascending=False)
    else:
        filtered = (
            filtered
            .assign(_pop=filtered["item_id"].map(models["item_popularity"]).fillna(0))
            .sort_values("_pop", ascending=False)
            .drop(columns="_pop")
        )

    page_size = 10
    n_pages   = max(1, (len(filtered) + page_size - 1) // page_size)
    page      = max(1, min(st.session_state.get("browse_page", 1), n_pages))

    page_rows = filtered.iloc[(page - 1) * page_size : page * page_size]

    if page_rows.empty:
        st.info("No listings match the current filters. Try widening them in the sidebar.")
        return

    for _, row in page_rows.iterrows():
        col_card, col_actions = st.columns([5, 1])
        with col_card:
            st.markdown(render_listing_card(row), unsafe_allow_html=True)
        with col_actions:
            item_id = row["item_id"]
            saved   = item_id in st.session_state.get("saved_items", set())
            if st.button("✓ Saved" if saved else "♡ Save", key=f"save_{item_id}", use_container_width=True):
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

    pc1, pc2, pc3 = st.columns([1, 2, 1])
    with pc1:
        if st.button("◀ Prev", disabled=page <= 1, use_container_width=True):
            st.session_state.browse_page = page - 1
            st.rerun()
    with pc2:
        st.markdown(
            f'<div class="page-info">Page <b>{page}</b> of {n_pages}</div>',
            unsafe_allow_html=True,
        )
    with pc3:
        if st.button("Next ▶", disabled=page >= n_pages, use_container_width=True):
            st.session_state.browse_page = page + 1
            st.rerun()


# ---------------------------------------------------------------------------
# For You view
# ---------------------------------------------------------------------------

def view_for_you(
    listings: pd.DataFrame,
    interactions: pd.DataFrame,
    ctx: dict,
    models: dict,
) -> None:
    # ── Similar-items anchor ──────────────────────────────────────────────
    if "similar_anchor" in st.session_state:
        anchor_id = st.session_state.pop("similar_anchor")
        lbi = listings.set_index("item_id")
        if anchor_id in lbi.index:
            anchor = lbi.loc[anchor_id]
            st.markdown('<div class="section-title">📌 Similar listings</div>', unsafe_allow_html=True)
            st.markdown(render_listing_card(anchor, recommended=False, explanation="Anchor listing"), unsafe_allow_html=True)
            sims     = models["cb"].similar_items(anchor_id, k=8)
            sim_rows = lbi.loc[[s for s in sims if s in lbi.index]]
            st.caption("Top 8 most similar listings by content features:")
            for _, row in sim_rows.iterrows():
                st.markdown(
                    render_listing_card(row, recommended=True, explanation="Matches anchor on price, size, location and amenities"),
                    unsafe_allow_html=True,
                )
            st.divider()

    # ── User profile card ─────────────────────────────────────────────────
    profile  = build_user_profile_summary(ctx["user_id"], interactions, listings)
    persona  = ctx["persona"]
    p_icon   = PERSONA_ICONS.get(persona, "👤")
    p_label  = PERSONA_LABELS.get(persona, persona)
    avg_p    = f"€{profile['avg_price']:,.0f}" if profile["n_engaged"] else "—"

    st.markdown(
        f"""
        <div class="profile-card">
            <div class="profile-avatar">{p_icon}</div>
            <div class="profile-info" style="flex:2">
                <div class="label">Buyer profile</div>
                <div class="name">{ctx["user_id"]}</div>
                <div class="sub">{p_label}</div>
            </div>
            <div class="profile-divider"></div>
            <div class="profile-stat">
                <div class="stat-val">{profile["n_engaged"]}</div>
                <div class="stat-lbl">Engaged</div>
            </div>
            <div class="profile-divider"></div>
            <div class="profile-stat">
                <div class="stat-val">{avg_p}</div>
                <div class="stat-lbl">Avg save price</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Method selector ───────────────────────────────────────────────────
    method = st.radio(
        "Method",
        list(METHOD_COLORS.keys()),
        index=2,
        horizontal=True,
        label_visibility="collapsed",
    )
    color = METHOD_COLORS[method]
    icon  = METHOD_ICONS[method]
    st.markdown(
        f'<div class="method-chip" style="background:{color}">{icon} {method}</div>'
        f'<div class="method-desc">{METHOD_DESC[method]}</div>',
        unsafe_allow_html=True,
    )

    k = st.slider("Number of recommendations", 5, 20, 10)

    # ── Generate ──────────────────────────────────────────────────────────
    with st.spinner(f"Generating with {method}…"):
        t0 = time.time()
        rec_ids, explanations = get_recommendations(
            method, ctx["user_id"], models, listings, interactions,
            ctx["neighborhood"], k=k,
        )
        gen_ms = (time.time() - t0) * 1000

    m1, m2, m3 = st.columns(3)
    m1.metric("Generated in", f"{gen_ms:.0f} ms")
    m2.metric("Listings engaged", profile["n_engaged"])
    m3.metric("Avg saved price", avg_p)

    if not rec_ids:
        st.warning(
            "No recommendations available. The selected user may have no interaction history "
            "(cold-start problem). Try switching method or selecting a different user."
        )
        return

    st.markdown(
        f'<div class="section-title" style="margin-top:1rem">Recommendations '
        f'<span class="count">{len(rec_ids)}</span></div>',
        unsafe_allow_html=True,
    )

    lbi = listings.set_index("item_id")
    for item_id in [r for r in rec_ids if r in lbi.index]:
        row = lbi.loc[item_id]
        col_card, col_act = st.columns([5, 1])
        with col_card:
            st.markdown(
                render_listing_card(row, recommended=True, explanation=explanations.get(item_id)),
                unsafe_allow_html=True,
            )
        with col_act:
            saved = item_id in st.session_state.get("saved_items", set())
            if st.button("✓ Saved" if saved else "♡ Save", key=f"rec_save_{item_id}",
                         use_container_width=True):
                st.session_state.setdefault("saved_items", set())
                if saved:
                    st.session_state.saved_items.discard(item_id)
                else:
                    st.session_state.saved_items.add(item_id)
                st.rerun()


# ---------------------------------------------------------------------------
# Compare view
# ---------------------------------------------------------------------------

def view_compare(
    listings: pd.DataFrame,
    interactions: pd.DataFrame,
    ctx: dict,
    models: dict,
) -> None:
    st.markdown(
        '<div class="section-title">Side-by-side method comparison</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Top-3 recommendations from each personalized method for the selected buyer. "
        "Use this view to see how the algorithms diverge on identical input."
    )

    methods = ["Item-based CF", "User-based CF", "Content-based", "Matrix Factorization (ALS)"]
    lbi     = listings.set_index("item_id")
    cols    = st.columns(len(methods))

    for col, method in zip(cols, methods):
        color = METHOD_COLORS[method]
        icon  = METHOD_ICONS[method]
        with col:
            st.markdown(
                f'<div class="method-chip" style="background:{color}">{icon} {method}</div>'
                f'<div class="method-desc" style="font-size:0.73rem">{METHOD_DESC[method]}</div>',
                unsafe_allow_html=True,
            )
            t0 = time.time()
            rec_ids, explanations = get_recommendations(
                method, ctx["user_id"], models, listings, interactions,
                ctx["neighborhood"], k=3,
            )
            gen_ms = (time.time() - t0) * 1000
            st.caption(f"⚡ {gen_ms:.0f} ms")
            if not rec_ids:
                st.info("No recommendations (cold start).")
            for item_id in rec_ids:
                if item_id in lbi.index:
                    st.markdown(
                        render_listing_card(
                            lbi.loc[item_id],
                            recommended=True,
                            explanation=explanations.get(item_id),
                        ),
                        unsafe_allow_html=True,
                    )

    # ── Offline evaluation table ──────────────────────────────────────────
    st.divider()
    st.markdown('<div class="eval-header">📊 Offline evaluation — all methods at k=10</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="eval-caption">'
        'Leave-one-out across all 1,200 synthetic buyers. '
        'Coverage = share of the 46.7k catalog ever recommended. '
        'Novelty = avg inverse popularity (higher = less mainstream). '
        'Diversity = avg pairwise feature distance within a list. '
        'Neighborhood bias = total variation distance from catalog distribution (0 = mirrors catalog, 1 = maximally concentrated).'
        '</div>',
        unsafe_allow_html=True,
    )

    eval_path = Path("data/processed/evaluation_results.csv")
    if eval_path.exists():
        eval_df = pd.read_csv(eval_path)
        eval_df.columns = [
            "Method", "Recall@10", "NDCG@10", "Coverage",
            "Novelty", "Diversity", "Neighborhood bias", "Gen time (s)",
        ]
        st.dataframe(
            eval_df.style
            .format({
                "Recall@10":        "{:.4f}",
                "NDCG@10":          "{:.4f}",
                "Coverage":         "{:.2%}",
                "Novelty":          "{:.2f}",
                "Diversity":        "{:.2f}",
                "Neighborhood bias":"{:.2f}",
                "Gen time (s)":     "{:.2f}",
            })
            .background_gradient(subset=["Recall@10", "NDCG@10"], cmap="Greens")
            .background_gradient(subset=["Coverage"],               cmap="Blues")
            .background_gradient(subset=["Diversity"],              cmap="Purples"),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Run `python scripts/run_full_evaluation.py` to populate the evaluation table.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.view = view

    st.divider()

    if view == "Browse":
        view_browse(listings, ctx, models)
    elif view == "For you":
        view_for_you(listings, interactions, ctx, models)
    else:
        view_compare(listings, interactions, ctx, models)

    render_footer()


if __name__ == "__main__":
    main()
