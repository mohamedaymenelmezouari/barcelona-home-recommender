# Barcelona Home Recommender

A real-estate recommender systems prototype built for the Esade Recommender
Systems class. Five techniques are implemented end to end on the same
dataset, compared on accuracy plus diversity, novelty, and bias, and
exposed through a multi-view Streamlit application.

## Domain and dataset

Real estate, Barcelona. Listings come from **idealista18** (Rey-Blanco,
Arbués, López & Páez, 2024, *Environment and Planning B*,
[doi.org/10.1177/23998083241242844](https://doi.org/10.1177/23998083241242844)),
an academically published, ODbL-1.0-licensed dataset of 2018 Idealista
listings for Madrid, Barcelona, and Valencia. The Barcelona subset is
used here: roughly 61k raw listings, 46,676 after deduplication and
cleaning, with price, size, rooms, amenities, distance to city
center and metro, and neighborhood (68 Barcelona neighborhoods).

Source repository: [github.com/paezha/idealista18](https://github.com/paezha/idealista18)

## Important design decision: the synthetic user layer

idealista18 has no users or interaction history. It is a snapshot of
listings only. Real estate is fundamentally a low repeat-purchase
domain: a property is bought once, so there is no natural collaborative
signal the way there is with movies or songs.

To make collaborative filtering and matrix factorization meaningful, this
project generates a **documented, transparent synthetic interaction
layer** (`src/personas.py`):

1. Eight buyer personas are defined (young couple, family, investor,
   retiree, luxury buyer, budget first-time buyer, remote worker,
   student), each with a preference vector over price, size, rooms,
   location, and amenities.
2. About 1,200 individual synthetic users are sampled from these
   personas with individual noise, so users within a persona are not
   identical clones.
3. Each user gets a simulated browsing session: a search-filtered
   candidate pool weighted toward their persona's fit, plus a small
   slice of random exploration. Views are probabilistically promoted to
   saves and contacts, producing a three-tier implicit feedback signal
   (view=1, save=3, contact=5) suitable for weighted-implicit algorithms
   such as ALS.

This synthetic layer is clearly stated as synthetic in the slide deck. It
exists purely so every algorithm has signal to learn from. The listings
themselves and all listing features are real.

We validated that personas produce sensible, well-separated behavior.
For example, `luxury_buyer` saves listings averaging around €820k, while
`student_shared_flat` averages around €127k. The full per-persona
breakdown is on slide 5 of the deck.

## Project structure

```
recsys-project/
├── data/
│   ├── raw/                # downloaded .rda files (gitignored, regenerated on demand)
│   └── processed/          # barcelona_listings.csv, users.csv, interactions.csv, evaluation_results.csv
├── src/
│   ├── data_loader.py      # fetches and cleans idealista18 into a listings table
│   ├── personas.py         # synthetic user + interaction generator
│   ├── explanations.py     # per-method "why this listing" copy for the UI
│   ├── evaluation.py       # precision/recall/NDCG, coverage, novelty, diversity, bias
│   └── recommenders/
│       ├── non_personalized.py     # popularity, best-value, trending
│       ├── collaborative.py        # item-based and user-based CF
│       ├── content_based.py        # feature-vector similarity, with similar_items() for cold start
│       └── matrix_factorization.py # ALS via the implicit library
├── app/
│   └── streamlit_app.py    # three-view prototype: Browse, For You, Compare
├── scripts/
│   ├── build_dataset.py        # one command: rebuild data + interactions from scratch
│   └── run_full_evaluation.py  # runs the full method comparison and writes evaluation_results.csv
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python scripts/build_dataset.py        # downloads data, builds listings and interactions
python scripts/run_full_evaluation.py  # writes the comparison results table
streamlit run app/streamlit_app.py     # launches the prototype UI
```

First-run model fitting takes about five seconds; subsequent runs use
cached models thanks to `st.cache_resource`.

## The Streamlit prototype

The app has three views, switchable from a single top-level toggle:

* **Browse.** Paginated catalog with sorting (price low/high, best
  €/m², largest first, most popular), neighborhood and price filters,
  and inline save and "Similar" actions. The Similar action exposes the
  content-based `similar_items()` helper, which works for any listing
  including those with zero interaction history.
* **For You.** Personalized recommendations from the selected method,
  with **per-method explanations** under each card (popularity shows
  total interactions, item-based CF references one of the user's prior
  saves, user-based CF cites the size of the similar-user neighborhood,
  content-based names the matching feature axes, and ALS reports its
  latent-factor signal). Includes a one-line live diagnostic: generation
  time in milliseconds, number of listings the user has engaged with so
  far, and their average save price.
* **Compare.** Side-by-side top-3 recommendations from all four
  personalized methods for the selected buyer, plus the offline
  evaluation results table at the bottom. This is the view most aligned
  with the project brief's "compare methods" goal.

A persona dropdown in the sidebar lets the grader log in as any of the
1,200 synthetic users. Saved listings persist across the session and
feed back into the recommendation logic via session-state exclusion.

## Method comparison (full evaluation)

Leave-one-out across all 1,200 synthetic buyers at k=10.

| Method | Recall@10 | NDCG@10 | Coverage | Novelty | Diversity | Nbhd. bias | Rec-gen time |
|---|---|---|---|---|---|---|---|
| Popularity | 0.0000 | 0.0000 | 0.03% | 5.09 | 1.41 | 0.77 | 4.91s |
| Item-based CF | 0.0025 | 0.0008 | 12.6% | 6.62 | 1.58 | 0.19 | 3.14s |
| User-based CF | 0.0050 | 0.0015 | 3.3% | 5.78 | 1.47 | 0.33 | 2.46s |
| Content-based | 0.0025 | 0.0008 | 1.6% | 6.75 | 0.35 | 0.45 | 4.61s |
| Matrix Factorization (ALS) | 0.0042 | 0.0020 | 3.7% | 5.95 | 1.61 | 0.29 | 0.51s |

Coverage is the share of the 46.7k catalog ever recommended. Novelty
is the average inverse popularity of recommended items (higher means
less mainstream). Intra-list diversity is the average pairwise feature
distance within a recommendation list. Neighborhood bias is the total
variation distance between recommended-neighborhood distribution and
catalog-neighborhood distribution (0 mirrors the catalog, 1 is maximally
concentrated).

Absolute recall numbers look small because every method is being asked
to find one specific held-out listing out of about 38.7k candidates at
k=10, which is a deliberately strict test. The meaningful comparison is
between methods, not against absolute thresholds.

### Three findings worth presenting

1. **Popularity is a filter bubble by construction.** Covering 0.03% of
   the catalog means it surfaces essentially the same dozen listings to
   every user, and it has by far the worst neighborhood bias of the
   five methods (0.77), concentrating recommendations in a handful of
   prime districts. It "works" only in the narrow sense of not crashing.
2. **Content-based trades diversity for novelty.** Lowest intra-list
   diversity (0.35 versus about 1.5 for every other method), highest
   novelty score. This is the textbook content-based trade-off:
   optimizing for similarity to a user's known preferences produces
   tight, repetitive, individually unusual recommendation sets, a
   personalized filter bubble distinct from popularity's
   non-personalized one.
3. **ALS wins on production speed.** Generates recommendations 5 to 9
   times faster than every other method (0.51s versus 2.46 to 4.91s
   per pass over 1,200 users), with accuracy on par with user-based CF.
   This is inference cost, separate from fit-time cost, and matters more
   at production scale where fitting happens offline but recommending
   happens on every page load.

## Technical challenge worth keeping in the deck

Early versions of the leave-one-out evaluation split removed only the
single sampled test row (a "save" event for example), but left other
event rows for the same (user, item) pair in train (such as the "view"
that preceded the save). Any recommender that correctly excludes
already-seen items then found that item structurally unrecommendable.
Both collaborative filtering methods scored exactly 0.0 Recall@10 at
every k, identical to the popularity baseline, even though the
underlying models were fine.

The fix: remove all rows for the held-out (user, item) pair from train,
not just the sampled one. After the fix, both CF methods clearly beat
popularity at every k. This is a useful cautionary example of how an
evaluation bug can look identical to a modeling failure, and is covered
on slide 10 of the deck.

## Known limitations

* Interactions are simulated, not observed. Documented above and
  throughout the deck.
* Neighborhood assignment uses nearest-centroid matching rather than a
  true point-in-polygon spatial join (documented in `data_loader.py`),
  which is fine for a prototype but worth flagging as a simplification.
* Listings are a single 2018 snapshot (four quarterly snapshots
  collapsed into one row per property), so anything framed as
  "Trending" is a proxy based on synthetic interaction volume, not real
  recency.

## Roadmap status

| Week | Deliverable | Status |
|---|---|---|
| 1 | UI prototype | Done. Three-view Streamlit app with per-method explanations and side-by-side comparison. |
| 2 | Setup, dataset, EDA, preprocessing | Done. `data_loader.py` produces the 46.7k cleaned listings table. |
| 2.5 | Synthetic interaction layer | Done. `personas.py` produces 1,200 users and roughly 330k interaction events. |
| 3 | Non-personalized | Done. Popularity, BestValue, Trending. |
| 4 | Collaborative filtering | Done. Item-based and user-based CF. |
| 5 | Content-based filtering | Done. Feature-vector similarity with `similar_items()` for cold start. |
| 6 | Matrix factorization | Done. ALS via the `implicit` library. |
| 7 | Full evaluation | Done. `run_full_evaluation.py` produces the comparison table on accuracy, coverage, novelty, diversity, and neighborhood bias. |
