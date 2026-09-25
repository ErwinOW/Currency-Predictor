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

Building the walking skeleton: exchange-rate ingestion → PostgreSQL, before
widening to more data sources.

- [x] Project scaffold
- [x] PostgreSQL running (Docker Compose)
- [x] Exchange-rate ingestion working end-to-end (2,988 daily MYR/IDR rows loaded, 2014-12-31 → present)
- [x] Feature engineering (9 price-based features, 26,771 values in the `features` table)
- [x] Walk-forward backtest harness + naive baselines (`python ml/backtest.py`)
- [x] Moving average + linear regression (in `ml/models.py`)
- [ ] Models: random forest → XGBoost
- [ ] Backtesting
- [ ] FastAPI
- [ ] React dashboard
- [x] Tests for feature engineering and backtest scoring (`pytest`)
- [ ] Automation (scheduler)

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

### 6. Run the tests

```bash
pytest
```

Checks the feature math against hand-calculated examples (e.g. that a
100 → 110 move really computes as a 10% return) and that the no-leakage
rule actually holds (changing tomorrow's price must not change today's
features), plus the backtest scoring functions.

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
