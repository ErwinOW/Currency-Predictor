# Currency Predictor

A machine-learning currency forecasting platform that combines historical
exchange rates with macroeconomic, commodity, market, and news-sentiment
data to estimate short-term currency movements and presents the results
through an interactive web dashboard.

Full project spec: [docs/project_rundown.md](docs/project_rundown.md).
This is a portfolio project — see §29 of the spec for limitations and
intent. Not a trading system.

**First target:** MYR/IDR only, 4–6 week build. Read as "1 MYR = ~4,379 IDR" — MYR is the base (the asset being priced), IDR is the quote (the currency the price is expressed in).

## Status

The data pipeline (ingestion → PostgreSQL → features) and a working model
ladder with walk-forward backtesting are done. Next up is either another
data source (commodities, news sentiment) or XGBoost, then the API/UI.

- [x] Project scaffold
- [x] PostgreSQL running (Docker Compose)
- [x] Exchange-rate ingestion working end-to-end (2,988 daily MYR/IDR rows loaded, 2014-12-31 → present)
- [x] Feature engineering (9 price-based features, 26,771 values in the `features` table)
- [x] Interest-rate data (Malaysia OPR via BNM, Indonesia via FRED) + daily alignment (3 more features, 8,964 values)
- [x] Walk-forward backtest harness + naive baselines (`python ml/backtest.py`)
- [x] Moving average + linear regression (in `ml/models.py`)
- [x] Random forest (in `ml/models.py`, `RF_FEATURE_COLUMNS` — first model to beat naive on every metric)
- [x] Tests for feature engineering, backtest scoring, time alignment, random forest, XGBoost, and the API (`pytest` — 21 of 24 passing, 3 are the API routes below still in progress)
- [x] Commodity data (Brent + WTI via FRED, Malaysian palm oil via Yahoo Finance) + daily alignment (6 more features, 17,925 values)
- [x] XGBoost (in `ml/models.py` — best model so far on MAE and RMSE)
- [x] Live prediction generation (`python ml/generate_prediction.py` — trains on all data, writes to `predictions`, confidence + interval derived from real backtest history, not arbitrary numbers)
- [~] FastAPI (`api/main.py` — `/currencies` and `/prediction/{pair}` done; `/historical`, `/indicators`, `/model-performance` in progress)
- [ ] News-sentiment data (§4.9)
- [ ] React dashboard
- [ ] Automation (scheduler)

### Current model results (walk-forward backtest, 2022–2026)

| Model | MAE (IDR) | RMSE (IDR) | Directional accuracy |
|---|---|---|---|
| Naive (tomorrow = today) | 10.11 | 13.78 | n/a |
| Linear regression | 10.12 | 13.81 | 51.7% |
| Random forest | 10.08 | 13.77 | **52.2%** |
| **XGBoost** | **10.07** | **13.76** | 52.0% |
| Moving average (7-day) | 16.97 | 22.76 | 51.3% |
| Naive momentum | 14.65 | 19.88 | 48.5% |

XGBoost is currently the best model on error (MAE/RMSE), random forest
edges it slightly on direction — both only marginally ahead of just
guessing "no change." Comparing the two models' `feature_importances_` is
more informative than either number alone: XGBoost assigns **exactly
zero** importance to all three interest-rate features (it never once
splits on them), while random forest gave them small but nonzero weight.
That difference comes from how each algorithm is built — boosting fits
each new tree to what's still unexplained after the stronger features
already did their work, so a feature with nothing left to add gets
skipped entirely; a random forest's per-tree random feature sampling
means even a weak feature gets picked in *some* trees by chance. The two
models also disagree on palm oil: XGBoost ranks it 5th out of 14
features, random forest ranks it near last — a concrete example of two
model families extracting different signal from identical data.

Daily FX is still close to a random walk overall; meaningfully beating
that likely needs either a longer prediction horizon (interest rates
would matter more there) or a data source with more day-to-day signal,
like news sentiment.

## Setup

### 1. Python environment

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Environment variables

`.env` is already created with local dev defaults (gitignored). Adjust if
needed — see `.env.example` for the full list of variables.

### 3. Database (PostgreSQL via Docker)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).

> On this machine Docker Desktop installed to
> `%LOCALAPPDATA%\Programs\DockerDesktop` (user-local, not `Program Files`)
> and its `resources\bin` folder isn't on PATH by default. If `docker` isn't
> found, add it to PATH or call it with the full path — the credential
> helper (`docker-credential-desktop.exe`) also lives there and is needed
> for `docker compose up` to pull images.

```bash
docker compose up -d
```

Then apply the schema:

```bash
docker exec -i currency_predictor_db psql -U currency_app -d currency_predictor < db/schema.sql
```

### 4. Run the first ingestion script

```bash
python ingestion/fetch_exchange_rates.py --start 2015-01-01
```

This fetches daily MYR/IDR rates, saves the raw response to
`data/raw/exchange_rates/`, validates it, and upserts it into the
`exchange_rates` table in Postgres.

### 5. Run feature engineering

```bash
python etl/feature_engineering.py
```

Reads `exchange_rates`, computes 9 price-based features (returns, moving
averages, volatility, momentum — see the docstring in
`etl/feature_engineering.py` for the full list and definitions), and
upserts them into the `features` table, long-form (one row per
date/feature). Every feature is backward-looking only — no future
information leaks into a given day's row (§10).

### 6. Fetch interest-rate data and align it

Needs a free `FRED_API_KEY` in `.env` — see `.env.example` for where to get
one. Malaysia's OPR needs no key (Bank Negara Malaysia's public API).

```bash
python ingestion/fetch_interest_rates.py --start-year 2014
python etl/interest_rate_features.py
```

The first script pulls the raw rate history into `interest_rates` (a
handful of rows per year — real events, not one row per day). The second
turns that into 3 daily features (`interest_rate_malaysia`,
`interest_rate_indonesia`, `interest_rate_differential`) using
`pandas.merge_asof`, forward-filling each day from the most recent known
rate — see the docstring in `etl/interest_rate_features.py` for why this
approach can't leak future rate decisions backward.

### 7. Fetch commodity prices and align them

Reuses the same `FRED_API_KEY` from step 6. No key needed for palm oil
(Yahoo Finance's public chart endpoint).

```bash
python ingestion/fetch_commodity_prices.py --start-year 2014
python etl/commodity_features.py
```

Brent and WTI crude come from FRED; Malaysian palm oil comes from Yahoo
Finance (ticker `CPO=F`) since FRED only has it monthly. Produces 6 daily
features (`{brent,wti,palm_oil}_price` and `_return_1d` for each) via the
same `pandas.merge_asof` alignment as interest rates — now pulled out into
`etl/time_alignment.py` since two different scripts needed the identical
logic. See `ingestion/fetch_commodity_prices.py`'s docstring for two real
data quirks it handles: WTI's genuine negative price in April 2020, and
Yahoo silently returning monthly-resolution data for long date ranges
unless you pass explicit start/end timestamps.

### 8. Run the tests

```bash
pytest
```

Checks the feature math against hand-calculated examples (e.g. that a
100 → 110 move really computes as a 10% return) and that the no-leakage
rule actually holds (changing tomorrow's price must not change today's
features), plus the backtest scoring and random-forest functions.

### 9. Run the model backtest

```bash
python ml/backtest.py
```

Trains and walk-forward-tests every model (naive, moving average, linear
regression, random forest, XGBoost) on the same folds and prints
MAE/RMSE/directional accuracy per year and overall — see "Current model
results" above for the latest numbers.

### 10. Generate today's live prediction

```bash
python ml/generate_prediction.py
```

Trains XGBoost on every day of history available (unlike the backtest,
which deliberately holds years back), predicts the next trading day's
close, and writes a row to `predictions`. Per §14, the confidence value
and prediction range are both derived from the model's actual walk-forward
track record (see the docstring in `ml/generate_prediction.py`), not
invented numbers.

### 11. Run the API

```bash
uvicorn api.main:app --reload
```

Then open http://127.0.0.1:8000/docs for the interactive API docs, built
automatically from the route type hints and `api/schemas.py`. Run
`ingestion/fetch_exchange_rates.py`, `ml/generate_prediction.py`, and
`ml/backtest.py` at least once first, since the routes read what those
scripts produce.

## Project layout

```
db/          connection helper + schema.sql
ingestion/   scripts that pull from external APIs into Postgres (§6)
etl/         cleaning, time alignment, feature engineering (§9–11)
api/         FastAPI app (§18)
frontend/    React dashboard (§19)
data/raw/    raw API responses, kept for reproducibility (§8)
data/processed/  engineered feature tables
models/      trained model artifacts
docs/        project rundown + any design notes
tests/       pytest tests for etl/ and ml/
```

## Learning notes

This project doubles as a hands-on way to learn data engineering and ML,
not just a finished deliverable — this section is a running log of what
each step actually taught, kept for the portfolio and as a personal
reference.

**Data engineering**
- Pulling data from real APIs and handling their quirks: pagination-free
  REST calls, a genuine duplicate-row bug in Bank Negara Malaysia's own
  API (caught by validation, not assumed away), Yahoo Finance silently
  coarsening a long date range to monthly resolution unless given
  explicit start/end timestamps instead of a relative range
- Idempotent loading with `INSERT ... ON CONFLICT ... DO UPDATE` (upserts)
  so re-running an ingestion script is always safe
- Keeping raw API responses on disk separately from the processed
  database, for reproducibility
- Secrets management: API keys in a gitignored `.env`, never in code or
  in git history
- Running a local service (Postgres) in Docker instead of installing it
  system-wide
- Refactoring once, not before: `forward_fill_to_daily` and the
  load/save boilerplate around the `features` table were written once
  for interest rates, then pulled into shared modules
  (`etl/time_alignment.py`, `etl/features_common.py`) only once a second
  caller (commodities) actually needed the same logic
- Validation ranges have to fit the data, not the other way around: WTI
  crude genuinely went negative in April 2020 (a real, famous event, not
  an error) — a single positive-only sanity check across all commodities
  would have silently discarded real data

**Time-series / ML fundamentals**
- The no-data-leakage rule (§10): a feature for day *t* may only use
  information available by day *t*. Enforced with `shift()`/`rolling()`
  (price features) and `pandas.merge_asof(direction="backward")`
  (interest-rate alignment) — and *proven*, not just asserted, with a
  dedicated test in each case that mutates a future value and checks
  nothing earlier changes
- Walk-forward validation instead of a random train/test split, because
  shuffling time-series data lets the model "see the future" during
  training
- Predicting *returns* rather than raw price levels, so the model's
  target stays in a similar range across years
- A model ladder that starts from a trivial baseline (naive: "tomorrow =
  today") specifically so every fancier model has a real number to beat
- Overfitting, made concrete: an unconstrained decision tree can
  memorize individual training rows; `max_depth` and `min_samples_leaf`
  are direct dials against that risk
- `feature_importances_` as a way to check *why* a model performs the
  way it does, not just *how well* — used three times now: to notice the
  interest-rate features barely mattered for a 1-day-ahead target, to
  confirm oil's daily returns mattered noticeably more than either
  interest-rate feature, and to compare random forest against XGBoost on
  the identical feature set
- Bagging vs. boosting: a random forest trains many trees independently
  (each on a random subset) and averages them; XGBoost trains trees
  sequentially, each one correcting the previous trees' errors. Made
  concrete by comparing their `feature_importances_` side by side -
  XGBoost gave three features exactly zero importance (never split on
  them once its stronger features already covered that ground), while
  random forest's per-tree random sampling gave every feature at least a
  small, nonzero score by chance
- A "production" model differs from a backtested one in what it's allowed
  to train on: the backtest deliberately withholds years to score the
  model honestly, but the live predictor (`ml/generate_prediction.py`)
  retrains on everything available, since there's no future left to
  withhold once you actually want tomorrow's number
- An honest confidence value, not an invented one (§14): an empirical
  prediction interval built from a model's own pooled walk-forward
  errors (their 5th/95th percentile) sidesteps assuming errors are
  bell-curve-shaped, and "confidence" reported as the model's actual
  historical directional accuracy (~52%) rather than a reassuring-looking
  made-up number

**Python, learned by debugging real errors**
- Code after a `raise` never executes in that function
- Positional arguments can't follow keyword arguments in a call
  (`f(x=1, y)` is a `SyntaxError`)
- `NameError` means a name was used with no value ever assigned to it
- `thing.method = (...)` overwrites the method itself; `thing.method(...)`
  calls it — the parentheses aren't optional
- Reading a traceback from the bottom up, and reading sklearn's own error
  messages (e.g. "feature names unseen at fit time" naming the exact
  columns involved) as a debugging shortcut instead of guessing

**Process**
- Writing tests as an acceptance target *before* the implementation is
  correct (each new script's tests were written and confirmed failing
  first, then made to pass)
- Small, focused commits with a "why," not just a "what," in the message
