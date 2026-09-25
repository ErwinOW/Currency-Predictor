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
- [x] Tests for feature engineering, backtest scoring, interest-rate alignment, and the random forest (`pytest` — 14 passing)
- [ ] Models: XGBoost
- [ ] Commodity and/or news-sentiment data (§4.7, §4.9)
- [ ] FastAPI
- [ ] React dashboard
- [ ] Automation (scheduler)

### Current model results (walk-forward backtest, 2022–2026)

| Model | MAE (IDR) | RMSE (IDR) | Directional accuracy |
|---|---|---|---|
| Naive (tomorrow = today) | 10.11 | 13.78 | n/a |
| Linear regression | 10.12 | 13.81 | 51.7% |
| **Random forest** | **10.09** | **13.76** | **52.8%** |
| Moving average (7-day) | 16.97 | 22.76 | 51.3% |
| Naive momentum | 14.65 | 19.88 | 48.5% |

Random forest is currently the best model, but only marginally ahead of
just guessing "no change." Its `feature_importances_` show why: the
interest-rate features are ~4% of its total importance combined — OPR/BI
rates only change a few times a year, so there's little new signal in them
for a 1-day-ahead prediction. Daily FX moves are close to a random walk;
beating that meaningfully likely needs a data source that actually moves
day-to-day, which is why commodities or news sentiment are the natural
next step rather than a fancier model on the same inputs.

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

### 7. Run the tests

```bash
pytest
```

Checks the feature math against hand-calculated examples (e.g. that a
100 → 110 move really computes as a 10% return) and that the no-leakage
rule actually holds (changing tomorrow's price must not change today's
features), plus the backtest scoring and random-forest functions.

### 8. Run the model backtest

```bash
python ml/backtest.py
```

Trains and walk-forward-tests every model (naive, moving average, linear
regression, random forest) on the same folds and prints MAE/RMSE/
directional accuracy per year and overall — see "Current model results"
above for the latest numbers.

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
  API (caught by validation, not assumed away)
- Idempotent loading with `INSERT ... ON CONFLICT ... DO UPDATE` (upserts)
  so re-running an ingestion script is always safe
- Keeping raw API responses on disk separately from the processed
  database, for reproducibility
- Secrets management: API keys in a gitignored `.env`, never in code or
  in git history
- Running a local service (Postgres) in Docker instead of installing it
  system-wide

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
  way it does, not just *how well* — used to notice the interest-rate
  features barely mattered for a 1-day-ahead target

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
