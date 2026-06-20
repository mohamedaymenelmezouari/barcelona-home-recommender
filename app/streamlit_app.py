"""
Barcelona Home Recommender

Three-view Streamlit prototype on the idealista18 Barcelona dataset:
    Browse   — paginated catalog with filters, sorting, save, and similar-items
    For You  — personalized recommendations with per-method explanations
    Compare  — side-by-side method comparison with offline evaluation results

Usage:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

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
# Styles
# ---------------------------------------------------------------------------

STYLES = """
<style>
/* Layout */
.block-container {
    padding-top: 1.25rem;
    padding-bottom: 4rem;
    max-width: 1380px;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, #0F1C2E 0%, #1B3A5C 100%);
    padding: 1.5rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
}
.app-header h1 {
    color: #fff;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 0.3rem;
    letter-spacing: -0.02em;
}
.app-header p {
    color: #A8C4DE;
    font-size: 0.85rem;
    margin: 0;
}

/* KPI strip */
.kpi-row { display: flex; gap: 0.75rem; margin-bottom: 1.5rem; }
.kpi {
    flex: 1;
    background: #fff;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 0.9rem 1rem;
}
.kpi-label {
    color: #64748B;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}
.kpi-value {
    color: #0F1C2E;
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1.1;
    margin-top: 0.2rem;
}
.kpi-sub { color: #94A3B8; font-size: 0.72rem; margin-top: 0.15rem; }

/* Listing card */
.card {
    background: #fff;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 0.95rem 1.1rem;
    margin-bottom: 0.6rem;
}
.card.rec { border-left: 3px solid #C1502E; }
.card-row { display: flex; justify-content: space-between; align-items: flex-start; }
.price { font-size: 1.25rem; font-weight: 700; color: #0F1C2E; }
.psm-tag {
    font-size: 0.75rem;
    background: #0F1C2E;
    color: #fff;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    font-weight: 600;
}
.meta { color: #475569; font-size: 0.85rem; margin-top: 0.3rem; }
.nb {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.1rem 0.45rem;
    border-radius: 4px;
}
.dist { color: #94A3B8; font-size: 0.75rem; margin-top: 0.2rem; }
.badges { display: flex; flex-wrap: wrap; gap: 0.25rem; margin-top: 0.4rem; }
.badge {
    font-size: 0.68rem;
    font-weight: 500;
    background: #F1F5F9;
    color: #475569;
    border: 1px solid #E2E8F0;
    padding: 0.1rem 0.45rem;
    border-radius: 3px;
}
.reason {
    margin-top: 0.55rem;
    padding: 0.4rem 0.65rem;
    background: #FEF3EE;
    border-left: 2px solid #C1502E;
    color: #7C2D12;
    font-size: 0.78rem;
    border-radius: 0 4px 4px 0;
}

/* Profile card */
.profile {
    background: #0F1C2E;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    display: flex;
    gap: 2rem;
    align-items: center;
    margin-bottom: 1.1rem;
}
.profile-id { color: #fff; font-size: 0.95rem; font-weight: 700; }
.profile-sub { color: #94A3B8; font-size: 0.78rem; margin-top: 0.15rem; }
.profile-stat .val { color: #fff; font-size: 1.05rem; font-weight: 700; }
.profile-stat .lbl { color: #64748B; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; }
.vr { width: 1px; background: #1E3A5C; align-self: stretch; }

/* Method chip */
.method-tag {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.8rem;
    color: #fff;
    margin-bottom: 0.5rem;
}

/* Method description */
.method-info {
    font-size: 0.78rem;
    color: #64748B;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.9rem;
    line-height: 1.5;
}

/* Pagination */
.page-info {
    text-align: center;
    color: #64748B;
    font-size: 0.82rem;
    padding-top: 0.55rem;
}

/* Eval table caption */
.tbl-caption {
    font-size: 0.76rem;
    color: #64748B;
    line-height: 1.5;
    margin-bottom: 0.6rem;
}

/* Footer */
.footer {
    margin-top: 2.5rem;
    padding-top: 1.25rem;
    border-top: 1px solid #E2E8F0;
    text-align: center;
    font-size: 0.74rem;
    color: #94A3B8;
}
.footer strong { color: #64748B; }

/* Hide Streamlit chrome */
[data-testid="stSidebarNavItems"] { display: none; }
footer { visibility: hidden; }
div[data-testid="stMetric"] {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "Popularity":                "#64748B",
    "Best value":                "#1B3A5C",
    "Item-based CF":             "#C1502E",
    "User-based CF":             "#B45309",
    "Content-based":             "#166534",
    "Matrix Factorization (ALS)":"#5B3F87",
}

METHOD_DESC = {
    "Popularity":
        "Recommends the most-interacted listings across all users. No personalization.",
    "Best value":
        "Ranks listings by price per m² within the selected neighborhood. No interaction data required.",
    "Item-based CF":
        "Finds listings co-interacted with items you already saved or contacted.",
    "User-based CF":
        "Identifies buyers with similar interaction patterns and surfaces their preferred listings.",
    "Content-based":
        "Matches listings to your history using property features: price, size, location, amenities.",
    "Matrix Factorization (ALS)":
        "Decomposes the interaction matrix into latent factors. Fastest inference at comparable accuracy.",
}

PERSONA_LABELS = {
    "young_couple_central":     "Young couple, central",
    "family_needs_space":       "Family, needs space",
    "investor_yield_focused":   "Investor, yield focused",
    "retiree_downsizing":       "Retiree, downsizing",
    "luxury_buyer":             "Luxury buyer",
    "budget_first_buyer":       "First-time buyer, budget",
    "remote_worker_needs_room": "Remote worker, needs room",
    "student_shared_flat":      "Student, shared flat",
}

# ---------------------------------------------------------------------------
# Data and model loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset...")
def get_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    listings     = load_barcelona_listings()
    interactions = pd.read_csv("data/processed/interactions.csv")
    users        = pd.read_csv("data/processed/users.csv")
    return listings, interactions, users


@st.cache_resource(show_spinner="Fitting models (first run only)...")
def load_models(_listings: pd.DataFrame, _interactions: pd.DataFrame) -> dict:
    timings: dict[str, float] = {}

    t = time.time(); pop = PopularityRecommender(_interactions);        timings["Popularity"]                = time.time() - t
    t = time.time(); bv  = BestValueRecommender(_listings);             timings["Best value"]                = time.time() - t
    icf = ItemBasedCF(_interactions);                                    timings["Item-based CF"]             = icf.fit_time_sec or 0.0
    ucf = UserBasedCF(_interactions);                                    timings["User-based CF"]             = ucf.fit_time_sec or 0.0
    t = time.time(); cb  = ContentBasedRecommender(_listings)
    cb.fit_user_profiles(_interactions);                                 timings["Content-based"]             = time.time() - t
    mf  = MatrixFactorizationRecommender(_interactions);                 timings["Matrix Factorization (ALS)"]= mf.fit_time_sec or 0.0

    seen_by_user    = _interactions.groupby("user_id")["item_id"].apply(set).to_dict()
    item_popularity = _interactions.groupby("item_id").size().to_dict()

    bv_rank: dict[str, int] = {}
    for _nb, grp in _listings.sort_values("PRICE_PER_SQM").groupby("NEIGHBORHOOD", sort=False):
        for rank, iid in enumerate(grp["item_id"].tolist(), start=1):
            bv_rank[iid] = rank

    return {
        "pop": pop, "best_value": bv, "item_cf": icf, "user_cf": ucf,
        "cb": cb, "mf": mf,
        "seen_by_user": seen_by_user,
        "item_popularity": item_popularity,
        "bv_rank": bv_rank,
        "timings": timings,
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
    saved = st.session_state.get("saved_items", set())

    if method == "Popularity":
        excl = models["seen_by_user"].get(user_id, set()) | saved
        recs = models["pop"].recommend(k=k, exclude=excl)
        exps = {r: explain_popularity(None, models["item_popularity"].get(r, 0)) for r in recs}
        return recs, exps

    if method == "Best value":
        recs = models["best_value"].recommend(neighborhood=neighborhood, k=k)
        lbi  = listings.set_index("item_id")
        exps = {r: explain_best_value(lbi.loc[r], models["bv_rank"].get(r, 0), lbi.loc[r]["NEIGHBORHOOD"]) for r in recs}
        return recs, exps

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

    recs = [r for r in raw if r not in saved][:k]
    lbi  = listings.set_index("item_id")

    if method == "Item-based CF":
        user_saves = interactions[
            (interactions["user_id"] == user_id) &
            (interactions["event_type"].isin(["save", "contact"]))
        ]
        anchor = lbi.loc[user_saves["item_id"].iloc[0]] if (
            not user_saves.empty and user_saves["item_id"].iloc[0] in lbi.index
        ) else None
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
    st.markdown(
        """
        <div class="app-header">
            <h1>Barcelona Home Recommender</h1>
            <p>Five recommender algorithms compared on the idealista18 Barcelona dataset
            (Rey-Blanco et al., 2024, Environment and Planning B).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpis = [
        ("Listings",         f"{len(listings):,}",                   "Barcelona, 2018"),
        ("Neighborhoods",    f"{listings['NEIGHBORHOOD'].nunique()}", "across the city"),
        ("Synthetic buyers", f"{len(users):,}",                      "8 persona types"),
        ("Interactions",     f"{len(interactions):,}",               "views, saves, contacts"),
    ]
    parts = ["<div class='kpi-row'>"]
    for label, value, sub in kpis:
        parts.append(
            f"<div class='kpi'>"
            f"<div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div>"
            f"<div class='kpi-sub'>{sub}</div>"
            f"</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def amenity_badges(row: pd.Series) -> str:
    fields = [
        ("HASLIFT",            "Lift"),
        ("HASTERRACE",         "Terrace"),
        ("HASAIRCONDITIONING",  "A/C"),
        ("HASPARKINGSPACE",    "Parking"),
        ("HASSWIMMINGPOOL",    "Pool"),
        ("HASDOORMAN",         "Doorman"),
        ("HASGARDEN",          "Garden"),
    ]
    return "".join(
        f'<span class="badge">{label}</span>'
        for key, label in fields
        if row.get(key)
    )


def render_card(row: pd.Series, recommended: bool = False, explanation: str | None = None) -> str:
    cls   = "card rec" if recommended else "card"
    rooms = int(row["ROOMNUMBER"])
    baths = int(row.get("BATHNUMBER", 0))
    area  = row["CONSTRUCTEDAREA"]
    psm   = row["PRICE_PER_SQM"]
    dist  = row["DISTANCE_TO_CITY_CENTER"]
    nb    = row["NEIGHBORHOOD"]
    price = row["PRICE"]

    reason_html = f'<div class="reason">{explanation}</div>' if explanation else ""

    return (
        f'<div class="{cls}">'
        f'<div class="card-row">'
        f'<div class="price">&#8364;{price:,.0f}</div>'
        f'<div class="psm-tag">&#8364;{psm:.0f}/m&#178;</div>'
        f'</div>'
        f'<div class="meta">{area:.0f} m&#178; &middot; {rooms} rooms &middot; {baths} bath'
        f' &nbsp; <span class="nb">{nb}</span></div>'
        f'<div class="dist">{dist:.1f} km to city centre</div>'
        f'<div class="badges">{amenity_badges(row)}</div>'
        f'{reason_html}'
        f'</div>'
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer">
            <strong>Barcelona Home Recommender</strong> &nbsp;&middot;&nbsp;
            Mohamed Aymen Elmezouari &nbsp;&middot;&nbsp;
            Esade Recommender Systems &nbsp;&middot;&nbsp;
            Dataset: idealista18 (Rey-Blanco et al., 2024) &mdash; ODbL 1.0
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(users: pd.DataFrame, listings: pd.DataFrame, models: dict) -> dict:
    with st.sidebar:
        st.subheader("Buyer")
        personas = sorted(users["persona"].unique(), key=lambda p: PERSONA_LABELS.get(p, p))
        persona  = st.selectbox(
            "Persona",
            personas,
            format_func=lambda p: PERSONA_LABELS.get(p, p),
        )
        persona_users = users[users["persona"] == persona]["user_id"].tolist()
        user_id = st.selectbox("User", persona_users[:20], index=0)

        st.subheader("Filters")
        neighborhoods = ["All neighborhoods"] + sorted(listings["NEIGHBORHOOD"].dropna().unique())
        neighborhood  = st.selectbox("Neighborhood", neighborhoods)
        price_max     = st.slider("Max price (€)", 50_000, 2_500_000, 800_000, step=25_000, format="€%d")
        min_rooms     = st.slider("Min rooms", 0, 6, 0)

        st.subheader("Session")
        saved = st.session_state.get("saved_items", set())
        c1, c2 = st.columns(2)
        c1.metric("Saved", len(saved))
        if saved and c2.button("Clear", use_container_width=True):
            st.session_state.saved_items = set()
            st.rerun()

        with st.expander("Model fit times"):
            for name, t in models["timings"].items():
                st.caption(f"{name}: {t:.2f} s")

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
        st.markdown(f"**{len(filtered):,} listings** match your filters")
    with c2:
        sort_by = st.selectbox(
            "Sort",
            ["Price (low to high)", "Price (high to low)", "Best value (€/m²)", "Largest first", "Most popular"],
            label_visibility="collapsed",
        )

    sort_map = {
        "Price (low to high)":   ("PRICE", True),
        "Price (high to low)":   ("PRICE", False),
        "Best value (€/m²)":     ("PRICE_PER_SQM", True),
        "Largest first":         ("CONSTRUCTEDAREA", False),
    }
    if sort_by in sort_map:
        col, asc = sort_map[sort_by]
        filtered = filtered.sort_values(col, ascending=asc)
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
        st.info("No listings match the current filters.")
        return

    for _, row in page_rows.iterrows():
        col_card, col_act = st.columns([5, 1])
        with col_card:
            st.markdown(render_card(row), unsafe_allow_html=True)
        with col_act:
            item_id = row["item_id"]
            is_saved = item_id in st.session_state.get("saved_items", set())
            if st.button("Saved" if is_saved else "Save", key=f"save_{item_id}", use_container_width=True):
                st.session_state.setdefault("saved_items", set())
                if is_saved:
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
        if st.button("Previous", disabled=page <= 1, use_container_width=True):
            st.session_state.browse_page = page - 1
            st.rerun()
    with pc2:
        st.markdown(f'<div class="page-info">Page {page} of {n_pages}</div>', unsafe_allow_html=True)
    with pc3:
        if st.button("Next", disabled=page >= n_pages, use_container_width=True):
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
    # Similar-items anchor
    if "similar_anchor" in st.session_state:
        anchor_id = st.session_state.pop("similar_anchor")
        lbi = listings.set_index("item_id")
        if anchor_id in lbi.index:
            st.subheader("Similar listings")
            st.markdown(render_card(lbi.loc[anchor_id], explanation="Anchor listing"), unsafe_allow_html=True)
            sims = models["cb"].similar_items(anchor_id, k=8)
            valid_sims = [s for s in sims if s in lbi.index]
            st.caption(f"Top {len(valid_sims)} most similar listings by content features:")
            for sid in valid_sims:
                st.markdown(
                    render_card(lbi.loc[sid], recommended=True,
                                explanation="Similar to anchor on price, size, location and amenities"),
                    unsafe_allow_html=True,
                )
            st.divider()

    # User profile
    profile = build_user_profile_summary(ctx["user_id"], interactions, listings)
    avg_p   = f"€{profile['avg_price']:,.0f}" if profile["n_engaged"] else "n/a"

    st.markdown(
        f"""
        <div class="profile">
            <div>
                <div class="profile-id">{ctx["user_id"]}</div>
                <div class="profile-sub">{PERSONA_LABELS.get(ctx["persona"], ctx["persona"])}</div>
            </div>
            <div class="vr"></div>
            <div class="profile-stat">
                <div class="val">{profile["n_engaged"]}</div>
                <div class="lbl">Listings engaged</div>
            </div>
            <div class="vr"></div>
            <div class="profile-stat">
                <div class="val">{avg_p}</div>
                <div class="lbl">Avg saved price</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Method selector
    method = st.radio(
        "Method",
        list(METHOD_COLORS.keys()),
        index=2,
        horizontal=True,
        label_visibility="collapsed",
    )
    color = METHOD_COLORS[method]
    st.markdown(
        f'<div class="method-tag" style="background:{color}">{method}</div>'
        f'<div class="method-info">{METHOD_DESC[method]}</div>',
        unsafe_allow_html=True,
    )

    k = st.slider("Recommendations to show", 5, 20, 10)

    with st.spinner(f"Running {method}..."):
        t0 = time.time()
        rec_ids, explanations = get_recommendations(
            method, ctx["user_id"], models, listings, interactions,
            ctx["neighborhood"], k=k,
        )
        gen_ms = (time.time() - t0) * 1000

    col1, col2, col3 = st.columns(3)
    col1.metric("Generation time", f"{gen_ms:.0f} ms")
    col2.metric("Listings engaged", profile["n_engaged"])
    col3.metric("Avg saved price", avg_p)

    if not rec_ids:
        st.warning(
            "No recommendations available for this user. "
            "They may have no interaction history (cold-start). "
            "Try a different method or select another user."
        )
        return

    st.markdown(f"**{len(rec_ids)} recommendations**")
    lbi = listings.set_index("item_id")
    for item_id in [r for r in rec_ids if r in lbi.index]:
        col_card, col_act = st.columns([5, 1])
        with col_card:
            st.markdown(
                render_card(lbi.loc[item_id], recommended=True, explanation=explanations.get(item_id)),
                unsafe_allow_html=True,
            )
        with col_act:
            is_saved = item_id in st.session_state.get("saved_items", set())
            if st.button("Saved" if is_saved else "Save", key=f"rec_save_{item_id}", use_container_width=True):
                st.session_state.setdefault("saved_items", set())
                if is_saved:
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
    st.subheader("Method comparison")
    st.caption(
        "Top 3 recommendations from each personalized method for the selected user. "
        "Generation time shown per method."
    )

    methods = ["Item-based CF", "User-based CF", "Content-based", "Matrix Factorization (ALS)"]
    lbi     = listings.set_index("item_id")
    cols    = st.columns(len(methods))

    for col, method in zip(cols, methods):
        color = METHOD_COLORS[method]
        with col:
            st.markdown(
                f'<div class="method-tag" style="background:{color}">{method}</div>'
                f'<div class="method-info" style="font-size:0.73rem">{METHOD_DESC[method]}</div>',
                unsafe_allow_html=True,
            )
            t0 = time.time()
            rec_ids, explanations = get_recommendations(
                method, ctx["user_id"], models, listings, interactions,
                ctx["neighborhood"], k=3,
            )
            gen_ms = (time.time() - t0) * 1000
            st.caption(f"{gen_ms:.0f} ms")

            if not rec_ids:
                st.info("No recommendations (cold-start).")
            for item_id in rec_ids:
                if item_id in lbi.index:
                    st.markdown(
                        render_card(lbi.loc[item_id], recommended=True, explanation=explanations.get(item_id)),
                        unsafe_allow_html=True,
                    )

    # Offline evaluation table
    st.divider()
    st.subheader("Offline evaluation")
    st.markdown(
        '<div class="tbl-caption">'
        "Leave-one-out evaluation across all 1,200 synthetic buyers at k=10. "
        "Coverage: share of the 46,676-listing catalog ever recommended. "
        "Novelty: mean inverse popularity of recommended items (higher = more niche). "
        "Diversity: mean pairwise feature distance within a recommendation list. "
        "Neighborhood bias: total variation distance between recommended and catalog neighborhood distributions "
        "(0 mirrors the catalog, 1 is maximally concentrated)."
        "</div>",
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
                "Recall@10":         "{:.4f}",
                "NDCG@10":           "{:.4f}",
                "Coverage":          "{:.2%}",
                "Novelty":           "{:.2f}",
                "Diversity":         "{:.2f}",
                "Neighborhood bias": "{:.2f}",
                "Gen time (s)":      "{:.2f}",
            })
            .background_gradient(subset=["Recall@10", "NDCG@10"], cmap="Greens")
            .background_gradient(subset=["Coverage"],               cmap="Blues")
            .background_gradient(subset=["Diversity"],              cmap="Purples"),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Run `python scripts/run_full_evaluation.py` to generate the evaluation table.")


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
