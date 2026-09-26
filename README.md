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

End-to-end and working: data pipeline (ingestion → PostgreSQL → features),
a full model ladder with walk-forward backtesting, a live prediction
generator, a FastAPI backend, a React dashboard consuming it, and a
scheduled automation layer running the whole thing unattended. What's
left is deployment - see the checklist below.

- [x] Project scaffold
- [x] PostgreSQL running (Docker Compose)
- [x] Exchange-rate ingestion working end-to-end (2,988 daily MYR/IDR rows loaded, 2014-12-31 → present)
- [x] Feature engineering (9 price-based features, 26,771 values in the `features` table)
- [x] Interest-rate data (Malaysia OPR via BNM, Indonesia via FRED) + daily alignment (3 more features, 8,964 values)
- [x] Walk-forward backtest harness + naive baselines (`python ml/backtest.py`)
- [x] Moving average + linear regression (in `ml/models.py`)
- [x] Random forest (in `ml/models.py`, `RF_FEATURE_COLUMNS` — first model to beat naive on every metric)
- [x] Tests for feature engineering, backtest scoring, time alignment, random forest, XGBoost, the API, and the automation pipeline (`pytest` — 28 passing)
- [x] News-sentiment data (GDELT bulk files, weekly samples 2015-present) + daily alignment (2 more features, 5,940 values)
- [x] Commodity data (Brent + WTI via FRED, Malaysian palm oil via Yahoo Finance) + daily alignment (6 more features, 17,925 values)
- [x] XGBoost (in `ml/models.py` — best model so far on MAE and RMSE)
- [x] Live prediction generation (`python ml/generate_prediction.py` — trains on all data, writes to `predictions`, confidence + interval derived from real backtest history, not arbitrary numbers)
- [x] FastAPI (`api/main.py` — `/currencies`, `/prediction/{pair}`, `/historical/{pair}`, `/indicators/{pair}`, `/model-performance`)
- [x] React dashboard (`frontend/` — Vite + React + Tailwind + Chart.js; Currency Overview, Historical Chart, Economic Factors, Model Performance)
- [x] Automation (`automation/scheduler.py` — APScheduler, daily + weekly jobs, per-step error isolation and logging to `logs/pipeline.log`)

### Current model results (walk-forward backtest, 2022–2026)

| Model | MAE (IDR) | RMSE (IDR) | Directional accuracy |
|---|---|---|---|
| Naive (tomorrow = today) | 10.17 | 13.88 | n/a |
| Linear regression | 10.19 | 13.91 | 51.6% |
| **XGBoost** | **10.15** | **13.86** | 52.6% |
| Random forest | 10.15 | 13.86 | **52.7%** |
| Moving average (7-day) | 17.13 | 22.96 | 51.2% |
| Naive momentum | 14.72 | 19.99 | 48.6% |

XGBoost and random forest are essentially tied for best, both only
marginally ahead of just guessing "no change." Comparing the two models'
`feature_importances_` is more informative than either number alone:
XGBoost assigns **exactly zero** importance to all three interest-rate
features (it never once splits on them), while random forest gave them
small but nonzero weight. That difference comes from how each algorithm
is built — boosting fits each new tree to what's still unexplained after
the stronger features already did their work, so a feature with nothing
left to add gets skipped entirely; a random forest's per-tree random
feature sampling means even a weak feature gets picked in *some* trees by
chance. The two models also disagree on palm oil: XGBoost ranks it
mid-table, random forest ranks it near last — a concrete example of two
model families extracting different signal from identical data.

News sentiment (§4.9) earns a real, if modest, place in both models:
XGBoost ranks it above every interest-rate feature (~6.6% importance vs
~2% combined for all three rate features), landing just behind the price
and commodity features. Random forest ranks it lower but still above two
of the three interest-rate features. That's a genuinely interesting result given sentiment here is
*weekly*-resolution (see the setup step below for why), coarser than the
daily commodity data — it still carries more signal than data that only
changes a handful of times a year.

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

### 8. Fetch news-sentiment data and align it

No API key needed, but this one takes a while — roughly 25-30 minutes,
since it downloads ~600 files sequentially.

```bash
python ingestion/fetch_news_sentiment.py --start 2015-02-19
python etl/sentiment_features.py
```

Pulls from GDELT's bulk Global Knowledge Graph files (not its search API —
see the docstring in `ingestion/fetch_news_sentiment.py` for why: the
search API only covers 2017-present and was unusably rate-limited from
this environment, while the bulk files are unthrottled static downloads
covering full history back to 2015). Each bulk file is a 15-minute,
~10 MB dump of ALL global news, so this samples just **one file per week**
rather than all ~96/day — a real, explicit tradeoff documented in the
script: this produces a *weekly*-resolution feature, not a true daily
aggregate, forward-filled to daily the same way as interest rates.

### 9. Run the tests

```bash
pytest
```

Checks the feature math against hand-calculated examples (e.g. that a
100 → 110 move really computes as a 10% return) and that the no-leakage
rule actually holds (changing tomorrow's price must not change today's
features), plus the backtest scoring and random-forest functions.

### 10. Run the model backtest

```bash
python ml/backtest.py
```

Trains and walk-forward-tests every model (naive, moving average, linear
regression, random forest, XGBoost) on the same folds and prints
MAE/RMSE/directional accuracy per year and overall — see "Current model
results" above for the latest numbers.

### 11. Generate today's live prediction

```bash
python ml/generate_prediction.py
```

Trains XGBoost on every day of history available (unlike the backtest,
which deliberately holds years back), predicts the next trading day's
close, and writes a row to `predictions`. Per §14, the confidence value
and prediction range are both derived from the model's actual walk-forward
track record (see the docstring in `ml/generate_prediction.py`), not
invented numbers.

### 12. Run the API

```bash
uvicorn api.main:app --reload
```

Then open http://127.0.0.1:8000/docs for the interactive API docs, built
automatically from the route type hints and `api/schemas.py`. Run
`ingestion/fetch_exchange_rates.py`, `ml/generate_prediction.py`, and
`ml/backtest.py` at least once first, since the routes read what those
scripts produce.

### 13. Run the frontend

Requires [Node.js](https://nodejs.org) (LTS) and the API running (step 12).

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (typically http://localhost:5173). The dev
server proxies `/api/*` to the FastAPI backend (see
`frontend/vite.config.js`), so no CORS setup is needed.

### 14. Run the automation layer

```bash
python -m automation.pipeline daily    # run once, immediately
python -m automation.pipeline weekly   # run once, immediately
python -m automation.scheduler         # run forever, on a schedule
```

`scheduler.py` runs `daily_job()` (exchange rates → price features →
commodities → commodity features → a fresh live prediction) every day at
06:00 UTC, and `weekly_job()` (interest rates, news sentiment, and a full
backtest — all three change slowly enough that daily reruns would be
wasted work) every Monday at 07:00 UTC. Each step is isolated: one
source failing is logged and skipped rather than taking down the rest of
the run — see `logs/pipeline.log` (gitignored) for a running record in
the format §22 of the spec describes, or `automation/pipeline.py`'s
`run_step()`.

## Project layout

```
db/          connection helper + schema.sql
ingestion/   scripts that pull from external APIs into Postgres (§6)
etl/         cleaning, time alignment, feature engineering (§9–11)
api/         FastAPI app (§18)
frontend/    React dashboard (§19)
automation/  scheduled pipeline jobs (§21) + monitoring/logging (§22)
data/raw/    raw API responses, kept for reproducibility (§8)
data/processed/  engineered feature tables
models/      trained model artifacts
docs/        project rundown + any design notes
tests/       pytest tests for etl/ and ml/
```

## Deployment

Three separate free services (§23), plus GitHub Actions for scheduling:

```
GitHub ──push──▶ Render (API) ◀──FRONTEND_URL── Vercel (frontend)
  │                   │
  │ schedule          │ DATABASE_URL
  ▼                   ▼
GitHub Actions ───▶ Supabase (Postgres)
  (DATABASE_URL, FRED_API_KEY secrets)
```

Render's free Postgres auto-deletes after 30 days, a bad fit for a
project meant to stay up — Supabase's free Postgres doesn't expire, so
the database and the backend are two different providers here rather
than Render managing both.

### 1. Database (Supabase)

1. Create a free project at [supabase.com](https://supabase.com) — note
   the database password you set, you'll need it in the connection string.
2. Get the connection string: Project Settings → Database → Connection
   string (URI format). It looks like
   `postgresql://postgres:[password]@[host]:5432/postgres`.
3. Apply the schema:
   ```bash
   psql "<your Supabase connection string>" -f db/schema.sql
   ```
4. Populate it once with historical data — point `.env`'s `DATABASE_URL`
   at Supabase temporarily and run steps 4–11 from Setup above (exchange
   rates through the live prediction), or just steps 4, 6, 7, 8 for a
   faster minimal dataset without full news-sentiment history.

### 2. Backend (Render)

1. Push this repo to GitHub (already done, if you're reading this from
   the repo).
2. [render.com](https://render.com) → New → Blueprint → connect this
   repo. Render reads `render.yaml` and creates the web service
   automatically.
3. Before the first deploy succeeds, set the two required env vars in
   the Render dashboard (Environment tab): `DATABASE_URL` (Supabase's
   connection string from step 1) and `FRED_API_KEY`.
4. Once deployed, note the service URL (e.g.
   `https://currency-predictor-api.onrender.com`) — needed in step 3.

Free-tier note: Render's free web services spin down after 15 minutes of
inactivity and take ~30-50 seconds to wake back up on the next request —
normal for a free portfolio deployment, not a bug.

### 3. Frontend (Vercel)

1. [vercel.com](https://vercel.com) → New Project → import this repo.
2. Set the project's root directory to `frontend/` (Vercel auto-detects
   the Vite build settings from there).
3. Add an environment variable: `VITE_API_URL` = the Render URL from
   step 2 (e.g. `https://currency-predictor-api.onrender.com`).
4. Deploy. Note the resulting URL (e.g. `https://your-app.vercel.app`).

### 4. Close the loop: CORS

Go back to Render (step 2) and add one more env var: `FRONTEND_URL` = the
Vercel URL from step 3. Redeploy the backend so `api/main.py`'s CORS
middleware picks it up — without this, the deployed frontend can reach
the API directly (e.g. via curl) but the *browser* will block it.

### 5. Automation (GitHub Actions)

In the GitHub repo: Settings → Secrets and variables → Actions → add
`DATABASE_URL` and `FRED_API_KEY` as repository secrets. The two
workflows in `.github/workflows/` then run automatically on their own
schedule (see `automation/pipeline.py` for what each does) — no server
needs to stay running just to fire a job once a day or once a week.

## Learning notes

This project is also a record of learning to build a full application by
**directing an AI coding agent (Claude)** rather than hand-writing most
of the code — the skill being practiced is scoping work, making the real
decisions, and verifying what comes back, not typing every line. Kept as
a running log for the portfolio and as a personal reference.

**Directing AI effectively**
- Starting from a full written spec (`docs/project_rundown.md`) rather
  than piecemeal requests — a clear upfront brief meant the agent could
  make consistent, well-reasoned calls across dozens of files without
  re-explaining the goal every time
- "Walking skeleton first": directing the agent to build one thin slice
  end-to-end (fetch → database → query back out) before widening to more
  data sources — the same discipline applies whether a human or an AI is
  doing the typing, and it surfaced real problems (a Docker PATH issue, a
  Windows SSL quirk) while there was still only one thing to debug
- Treating real forks in the road as decisions to make deliberately, not
  ones to let the agent make silently: which data source to trust, which
  hosting provider fits a portfolio project's actual constraints (a free
  database that doesn't expire vs. one that does), sequencing backend
  before frontend so the UI shows real data from the start instead of
  disposable mock data
- Being explicit about who does what: an ambiguous "let's do the next
  step" once resulted in the agent writing code the user actually wanted
  to write themselves — a concrete lesson that clear task ownership has
  to be stated, not assumed, when directing an AI collaborator

**Verifying AI's work, not just trusting it**
- Asking for a real test suite and actually running it, rather than
  accepting "this should work" — caught a real bug (`forward_fill_to_daily`
  crashing on a `DatetimeIndex` input) before it ever touched the database
- Spot-checking the agent's output against ground truth: a computed
  return checked by hand against raw prices, a feature value checked
  against the exact API response it came from, an API endpoint hit
  directly with `curl` instead of trusting the dashboard alone
- Re-reading AI-authored code turned up real mistakes worth catching: a
  stray non-English character dropped into a docstring, a duplicated
  block pasted twice, a generated cache file that should have been
  gitignored but wasn't at first
- Several genuine problems only surfaced by actually *running* things,
  not by reading a plan: Docker/Node.js installed but not on this
  machine's PATH, a free API's rate limit that no amount of waiting could
  clear, a dev server that silently stalled the first time new
  dependencies were added. An agent's plan can be sound and the real
  environment can still disagree with it

**What AI made possible at this scope**
- One project spanning Python/pandas/PostgreSQL, scikit-learn/XGBoost,
  FastAPI, React/Vite/Tailwind, Docker, a scheduled automation layer, and
  a three-service cloud deployment — a breadth of tooling that would
  normally take much longer to become independently proficient in across
  every layer first. The leverage isn't skipping understanding; it's
  reviewing and directing across a full stack instead of mastering one
  layer at a time before starting the next
- Two different ways of working with the same agent, used deliberately at
  different points: writing code by hand with the agent as a reviewer and
  guide (to actually learn the mechanics of a new library), and directing
  the agent to implement directly with a clear explanation afterward (to
  move faster once the goal was to ship a working system) — recognizing
  which mode fits the moment, and saying so, rather than defaulting to one

**Honesty as a collaboration norm**
- An agent that surfaces its own limitations is more useful than one that
  hides them: a feature explicitly documented as *weekly*-resolution
  rather than presented as a true daily aggregate, a live prediction's
  confidence value reported as an honest ~52% (barely better than a coin
  flip) instead of a reassuring invented number, a hosting tradeoff
  (a free database that auto-deletes after 30 days) flagged before it
  became a problem rather than after
- The same standard applied back the other way: asking the agent to
  explain *why* behind every change, not just *what* changed, made it
  possible to actually evaluate its decisions instead of rubber-stamping
  a diff

**What got built, and why (technical reference)**

The sections below are the concrete engineering decisions behind the
system above — reviewed and directed rather than hand-typed, but still
the reasoning a portfolio reviewer (or future me) would want to see.

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
- A "free" API can have two very different faces: GDELT's own search API
  (rate-limited hard enough from this environment to be unusable no
  matter how long between requests) vs. its bulk static files (fully
  reliable, unthrottled, ~10 MB per 15-minute snapshot). When one path to
  a data source is a wall, it's worth checking whether the same provider
  exposes the data a different way before giving up or switching sources
  entirely
- Explicit, honest sampling tradeoffs beat silently doing less than
  advertised: downloading *every* 15-minute GDELT file back to 2015 would
  mean ~400,000 files (multiple terabytes) — not practical. Sampling one
  file per week instead is a real resolution downgrade (weekly, not
  daily), and that's stated plainly in the code and docs rather than
  presented as equivalent to a true daily aggregate

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

**Python, from the one stretch of hand-typed code (the random forest and
XGBoost model classes)**
- Code after a `raise` never executes in that function
- Positional arguments can't follow keyword arguments in a call
  (`f(x=1, y)` is a `SyntaxError`)
- `NameError` means a name was used with no value ever assigned to it
- `thing.method = (...)` overwrites the method itself; `thing.method(...)`
  calls it — the parentheses aren't optional
- Reading a traceback from the bottom up, and reading sklearn's own error
  messages (e.g. "feature names unseen at fit time" naming the exact
  columns involved) as a debugging shortcut instead of guessing

**Automation / operations**
- Different data sources justify different update cadences: daily jobs
  for things that genuinely change daily (exchange rates, commodities),
  a separate weekly job for things that don't (interest rates, this
  project's weekly-sampled sentiment, and model retraining) - running
  everything on the same daily schedule would just waste time re-fetching
  unchanged data
- Isolating each pipeline step in its own try/except so one failing data
  source (a dead API, a network blip) gets logged and skipped instead of
  silently taking down every step after it - a small pattern
  (`run_step()`) that turns "the whole pipeline crashed" into "one line
  in the log said FAILED, everything else still ran"
- Making a CLI script's `main()` safely callable from *other* Python
  code, not just the command line: `def main(argv=None)` plus
  `parser.parse_args(argv)` means calling `main(argv=[])` from automation
  code uses the script's own defaults, regardless of whatever the calling
  process's real `sys.argv` happens to contain
- A "full historical backfill" script and an "incremental daily update"
  script don't have to be two separate files: `fetch_news_sentiment.py`
  defaults to resuming from the last stored sample, but `--start` still
  overrides it for a one-time full backfill

**Process**
- Writing tests as an acceptance target *before* the implementation is
  correct (each new script's tests were written and confirmed failing
  first, then made to pass)
- Small, focused commits with a "why," not just a "what," in the message
