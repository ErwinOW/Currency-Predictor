-- Currency Predictor — core schema
-- Only the tables needed for the walking-skeleton step are filled in below.
-- Others are stubbed per the project rundown (docs/project_rundown.md §7) and
-- will be fleshed out as each data source comes online.

CREATE TABLE IF NOT EXISTS exchange_rates (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    currency_pair   VARCHAR(10) NOT NULL,
    open            NUMERIC(12, 6),
    high            NUMERIC(12, 6),
    low             NUMERIC(12, 6),
    close           NUMERIC(12, 6) NOT NULL,
    volume          NUMERIC(20, 2),
    source          VARCHAR(50) NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (date, currency_pair)
);

-- Stubs for later steps (§7, §25) — created now so the schema stays visible
-- as one source of truth, populated when we build each ingestion script.

CREATE TABLE IF NOT EXISTS economic_indicators (
    id          SERIAL PRIMARY KEY,
    date        DATE NOT NULL,
    country     VARCHAR(50) NOT NULL,
    indicator   VARCHAR(100) NOT NULL,
    value       NUMERIC(20, 6),
    frequency   VARCHAR(20) NOT NULL,
    UNIQUE (date, country, indicator)
);

CREATE TABLE IF NOT EXISTS interest_rates (
    id          SERIAL PRIMARY KEY,
    date        DATE NOT NULL,
    country     VARCHAR(50) NOT NULL,
    rate        NUMERIC(8, 4) NOT NULL,
    UNIQUE (date, country)
);

CREATE TABLE IF NOT EXISTS commodities (
    id          SERIAL PRIMARY KEY,
    date        DATE NOT NULL,
    commodity   VARCHAR(50) NOT NULL,
    price       NUMERIC(14, 4) NOT NULL,
    UNIQUE (date, commodity)
);

CREATE TABLE IF NOT EXISTS market_data (
    id          SERIAL PRIMARY KEY,
    date        DATE NOT NULL,
    symbol      VARCHAR(20) NOT NULL,
    value       NUMERIC(20, 6) NOT NULL,
    UNIQUE (date, symbol)
);

CREATE TABLE IF NOT EXISTS news (
    id          SERIAL PRIMARY KEY,
    date        DATE NOT NULL,
    source      VARCHAR(100),
    headline    TEXT,
    url         TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sentiment (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    source          VARCHAR(100),
    sentiment_score NUMERIC(6, 4),
    article_count   INTEGER,
    UNIQUE (date, source)
);

CREATE TABLE IF NOT EXISTS events (
    id          SERIAL PRIMARY KEY,
    date        DATE NOT NULL,
    category    VARCHAR(50),
    description TEXT,
    impact_score NUMERIC(6, 4)
);

CREATE TABLE IF NOT EXISTS features (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    currency_pair   VARCHAR(10) NOT NULL,
    feature_name    VARCHAR(100) NOT NULL,
    feature_value   NUMERIC(20, 8),
    UNIQUE (date, currency_pair, feature_name)
);

CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT now(),
    currency_pair   VARCHAR(10) NOT NULL,
    horizon         VARCHAR(20) NOT NULL,
    predicted_rate  NUMERIC(12, 6),
    lower_bound     NUMERIC(12, 6),
    upper_bound     NUMERIC(12, 6),
    direction       VARCHAR(10),
    confidence      NUMERIC(5, 4),
    model_version   VARCHAR(50)
);
