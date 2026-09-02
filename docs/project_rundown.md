# Currency Prediction System --- Project Rundown

## 1. Project Goal

Build an end-to-end currency prediction system that estimates short-term
exchange-rate movements using:

-   Historical exchange-rate data
-   Interest-rate decisions
-   Inflation
-   GDP and economic indicators
-   Central-bank policies
-   Trade balances
-   Commodity prices
-   Market data
-   Political/economic events
-   News and market sentiment

The objective is **not** to achieve perfect or production-grade
financial prediction accuracy. The objective is to build a complete,
working **Data Engineering + Machine Learning + Web application** that
demonstrates how these different sources can be collected, processed,
combined, modeled, evaluated, and displayed.

A suitable project description:

> A machine-learning currency forecasting platform that combines
> historical exchange rates with macroeconomic, commodity, market, and
> news-sentiment data to estimate short-term currency movements and
> present the results through an interactive web dashboard.

------------------------------------------------------------------------

# 2. Recommended Scope

The recommended version is a **portfolio-grade project**, rather than a
full enterprise/production system.

### Target development time

Approximately **4--6 weeks part-time** with substantial assistance from
Claude AI.

A simplified MVP could be completed in approximately **2--3 weeks**.

An advanced production-style version could take **2--4+ months**.

The recommended target is the **4--6 week version**.

------------------------------------------------------------------------

# 3. High-Level Architecture

``` text
                         DATA SOURCES
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
   Exchange Rates       Economic Data       Market Data
          |                   |                   |
          +-------------------+-------------------+
                              |
                              v
                       Python ETL Pipeline
                              |
                              v
                         PostgreSQL
                              |
                              v
                  Data Cleaning & Validation
                              |
                              v
                       Time Alignment
                              |
                              v
                     Feature Engineering
                              |
                 +------------+------------+
                 |                         |
                 v                         v
          Historical Data          News / Sentiment
                 |                         |
                 +------------+------------+
                              |
                              v
                         ML MODEL
                    (XGBoost / Baselines)
                              |
                              v
                     Backtesting & Evaluation
                              |
                              v
                         FastAPI
                              |
                              v
                       React Web UI
                              |
                              v
                  Prediction + Analytics
```

------------------------------------------------------------------------

# 4. Data Sources

The system should combine multiple categories of information.

## 4.1 Exchange Rate Data

Primary market data.

Example:

-   USD/MYR
-   EUR/USD
-   USD/IDR
-   USD/SGD

For the first version, start with **one currency pair**, preferably
USD/MYR.

Potential fields:

-   Date
-   Open
-   High
-   Low
-   Close
-   Volume, if available

Derived features:

-   Daily return
-   7-day return
-   30-day return
-   Moving averages
-   Historical volatility
-   Momentum

------------------------------------------------------------------------

## 4.2 Interest Rates

Examples:

-   US Federal Funds Rate
-   Malaysia OPR
-   Interest-rate differential
-   Central-bank rate changes

For USD/MYR:

``` text
Interest Rate Differential =
US Interest Rate - Malaysia OPR
```

The model can use both the absolute rates and their differences.

------------------------------------------------------------------------

## 4.3 Inflation

Examples:

-   US CPI
-   Malaysia CPI
-   Inflation rate
-   Inflation differential

Potential features:

``` text
US inflation
Malaysia inflation
US-Malaysia inflation differential
Change in inflation
```

Inflation data is generally monthly, so it needs to be aligned with
daily exchange-rate data.

------------------------------------------------------------------------

## 4.4 GDP and Economic Indicators

Possible indicators:

-   GDP growth
-   Unemployment
-   PMI
-   Retail sales
-   Industrial production
-   Consumer confidence

Important complication:

``` text
Exchange rate  -> Daily
Oil price      -> Daily
CPI            -> Monthly
GDP            -> Quarterly
Interest rate  -> Event-based
```

The ETL pipeline must align these different frequencies without
introducing future information.

------------------------------------------------------------------------

## 4.5 Central-Bank Policies

Potential inputs:

-   Interest-rate decisions
-   Monetary policy statements
-   Quantitative easing/tightening
-   Forward guidance
-   Policy announcements

For the first version, these can be simplified into event-based
features.

Example:

``` text
rate_change = +0.25
rate_change =  0
rate_change = -0.25
```

A later version could use NLP to analyze policy statements.

------------------------------------------------------------------------

## 4.6 Trade and External-Sector Data

Potential inputs:

-   Trade balance
-   Current account
-   Export growth
-   Import growth
-   Foreign-exchange reserves

These can help capture external demand and capital-flow conditions.

------------------------------------------------------------------------

## 4.7 Commodity Prices

Especially useful for currencies influenced by commodity
exports/imports.

Potential inputs:

-   Brent crude
-   WTI crude
-   Palm oil
-   Gold
-   Natural gas

For USD/MYR, oil and palm oil are particularly interesting features to
experiment with.

------------------------------------------------------------------------

## 4.8 Market Data

Potential inputs:

-   S&P 500
-   KLCI
-   VIX
-   US Treasury yields
-   Malaysian bond yields
-   Other relevant stock-market indices

These can provide information about:

-   Risk appetite
-   Market stress
-   Capital flows
-   Interest-rate expectations

------------------------------------------------------------------------

## 4.9 News and Sentiment

News can be converted into numerical features.

Example:

``` text
News article
     |
     v
NLP sentiment model
     |
     v
Sentiment score
     |
     +----> Positive
     +----> Neutral
     +----> Negative
```

Example daily aggregate:

``` text
Date        Sentiment
2026-08-20  +0.32
2026-08-21  -0.15
2026-08-22  -0.71
```

Potential additional features:

-   Number of relevant articles
-   Average sentiment
-   Positive article count
-   Negative article count
-   News-event category
-   News intensity

------------------------------------------------------------------------

# 5. Unexpected News

Unexpected events are difficult because they cannot be known beforehand.

Instead of trying to directly predict "unexpected news", the system can
detect and quantify events after they occur.

Possible approach:

``` text
News
 |
 v
NLP
 |
 v
Event classification
 |
 +---- Economic
 +---- Political
 +---- Monetary
 +---- Trade
 +---- Geopolitical
 |
 v
Event impact score
```

A more advanced implementation could compare an economic announcement
with the market's prior expectation and calculate a "surprise" feature.

This is optional for the first version.

------------------------------------------------------------------------

# 6. Data Ingestion Layer

Python will be responsible for retrieving data from APIs and other
sources.

Possible ingestion methods:

-   REST APIs
-   CSV files
-   Database connectors
-   Web scraping when necessary
-   News APIs

Recommended approach:

``` text
API
 |
 v
Python ingestion script
 |
 v
Validate data
 |
 v
Save raw data
 |
 v
Transform
 |
 v
PostgreSQL
```

The ingestion scripts should be reusable and scheduled rather than
manually run every time.

------------------------------------------------------------------------

# 7. Data Storage

## Recommended Database: PostgreSQL

A simple schema could contain:

``` text
exchange_rates
economic_indicators
interest_rates
commodities
market_data
news
sentiment
events
features
predictions
```

Example:

### exchange_rates

``` text
id
date
currency_pair
open
high
low
close
```

### economic_indicators

``` text
id
date
country
indicator
value
frequency
```

### sentiment

``` text
id
date
source
sentiment_score
article_count
```

### predictions

``` text
id
timestamp
currency_pair
horizon
predicted_rate
lower_bound
upper_bound
direction
confidence
model_version
```

------------------------------------------------------------------------

# 8. Raw Data Storage

In addition to PostgreSQL, raw files can be retained for
reproducibility.

Example:

``` text
data/
├── raw/
│   ├── exchange_rates/
│   ├── economic/
│   ├── commodities/
│   ├── market/
│   └── news/
│
├── processed/
│   └── features/
│
└── models/
```

For a simple project, local storage is sufficient.

For a cloud version, object storage such as Amazon S3 can be used later.

------------------------------------------------------------------------

# 9. Data Cleaning

The system needs to handle:

-   Missing values
-   Duplicate records
-   Incorrect dates
-   Outliers
-   Different units
-   Different currencies
-   Different frequencies
-   API inconsistencies

Example:

``` text
Raw data
   |
   v
Remove duplicates
   |
   v
Validate dates
   |
   v
Handle missing values
   |
   v
Detect outliers
   |
   v
Normalize formats
```

------------------------------------------------------------------------

# 10. Time Alignment

This is one of the most important parts of the project.

Different data sources are released at different frequencies.

Example:

``` text
Exchange rate -> Daily
Oil            -> Daily
CPI            -> Monthly
GDP            -> Quarterly
Interest rate  -> Event
News           -> Continuous
```

The model's daily dataset needs to know what information was actually
available on each day.

### Critical rule

**Never use information that was released after the prediction
timestamp.**

This prevents **data leakage**.

Example:

If predicting August 25:

``` text
Allowed:
Information available by August 24

Not allowed:
Information released on August 26
```

------------------------------------------------------------------------

# 11. Feature Engineering

Potential features include:

## Price features

-   Previous close
-   1-day return
-   7-day return
-   30-day return
-   Moving average
-   Volatility

## Interest-rate features

-   US rate
-   Malaysia rate
-   Rate differential
-   Recent rate change

## Inflation features

-   US CPI
-   Malaysia CPI
-   Inflation differential
-   Change in CPI

## Commodity features

-   Oil price
-   Palm oil price
-   Gold price
-   Commodity returns

## Market features

-   S&P 500 return
-   KLCI return
-   VIX
-   Treasury yields

## News features

-   Sentiment score
-   Article count
-   Negative-news ratio
-   Positive-news ratio

## Event features

-   Interest-rate decision
-   Election/political event
-   Major economic announcement
-   Policy announcement

------------------------------------------------------------------------

# 12. Model Strategy

Do not immediately jump to deep learning.

Start with simple baselines, then progressively increase complexity.

## Model 1 --- Naive Baseline

Example:

``` text
Tomorrow's price = Today's price
```

This provides a reference point.

------------------------------------------------------------------------

## Model 2 --- Moving Average

Use recent historical prices.

------------------------------------------------------------------------

## Model 3 --- Linear Regression

Use engineered features to predict:

``` text
next-day return
```

------------------------------------------------------------------------

## Model 4 --- Random Forest

A useful tree-based baseline for tabular data.

------------------------------------------------------------------------

## Model 5 --- XGBoost

Recommended primary model for the portfolio version.

Inputs:

``` text
Historical prices
+
Interest rates
+
Inflation
+
GDP
+
Commodities
+
Market data
+
News sentiment
+
Events
```

Output:

``` text
Expected future return
```

or:

``` text
Expected exchange rate
```

------------------------------------------------------------------------

## Model 6 --- LSTM / GRU (Optional)

Deep-learning models can be added later to model sequential behavior.

For example:

``` text
Previous 30 days
       |
       v
      LSTM
       |
       v
Next-day prediction
```

Do not make LSTM mandatory.

The project is already strong without it.

------------------------------------------------------------------------

# 13. Predicting Returns vs Prices

It is often more useful to predict **returns** rather than directly
predicting the raw exchange rate.

Example:

``` text
Current USD/MYR = 4.18

Model:
Expected return = +0.72%

Prediction:
USD/MYR ≈ 4.21
```

The system can still display the predicted exchange rate to the user.

------------------------------------------------------------------------

# 14. Prediction Output

The UI should not simply show one number.

Recommended output:

``` text
Current Rate:
4.18

Predicted Rate:
4.21

Expected Movement:
+0.72%

Direction:
Bullish

Expected Range:
4.17 – 4.25

Confidence:
64%
```

Important:

The confidence value should be based on an actual statistical/model
methodology rather than being an arbitrary number.

------------------------------------------------------------------------

# 15. Backtesting

The system needs historical testing.

Do NOT randomly split time-series data.

Instead use walk-forward validation.

Example:

``` text
Training          Testing
2015 -------- 2021 | 2022

Training               Testing
2015 ------------- 2022 | 2023

Training                    Testing
2015 ------------------ 2023 | 2024
```

This better simulates how the system would have behaved in the real
world.

------------------------------------------------------------------------

# 16. Evaluation Metrics

Useful metrics:

### MAE

Mean Absolute Error.

Measures average prediction error.

### RMSE

Penalizes larger errors more heavily.

### Directional Accuracy

Measures whether the model correctly predicts:

``` text
UP
or
DOWN
```

This can be especially useful for currency movement.

### Additional metrics

-   Precision/recall for direction classification
-   Hit rate
-   Prediction interval coverage
-   Stability across different time periods

------------------------------------------------------------------------

# 17. Model Comparison

The dashboard could show:

``` text
Model              Directional Accuracy
------------------------------------------------
Naive Baseline          51%
Linear Regression       53%
Random Forest           57%
XGBoost                 62%
LSTM                    59%
```

These numbers are examples only and must be replaced with actual
results.

This makes the project much more credible because it demonstrates that
the model was actually evaluated.

------------------------------------------------------------------------

# 18. Backend

## Recommended: FastAPI

Python backend exposing REST endpoints.

Example:

``` text
GET /currencies
GET /currency/USD-MYR
GET /historical/USD-MYR
GET /prediction/USD-MYR
GET /indicators/USD-MYR
GET /sentiment/USD-MYR
GET /model-performance
```

The backend handles:

``` text
Frontend request
      |
      v
FastAPI
      |
      +---- PostgreSQL
      |
      +---- ML model
      |
      v
JSON response
```

------------------------------------------------------------------------

# 19. Frontend

## Recommended: React + Tailwind CSS

The web application should display the system's results clearly.

Main sections:

### Currency Overview

``` text
USD/MYR

Current:       4.18
Prediction:    4.21
Movement:      +0.72%
Direction:     Bullish
Confidence:    64%
```

### Historical Chart

Display:

``` text
Historical exchange rate
          +
Model prediction
          +
Prediction range
```

### Economic Factors

Example:

``` text
US Rate              4.50%
Malaysia OPR         3.00%
Oil                  $72.30
Inflation             2.1%
Sentiment             +0.34
```

### Model Performance

Show comparison between models.

### Insights

Explain important factors contributing to the prediction.

------------------------------------------------------------------------

# 20. Suggested UI

A dashboard could have:

``` text
+--------------------------------------------------+
| Currency AI                         USD/MYR      |
+--------------------------------------------------+
|                                                  |
| Current Rate        Prediction                   |
| 4.18                4.21                         |
|                     +0.72%                       |
|                                                  |
|        Historical + Prediction Chart             |
|                                                  |
+--------------------------------------------------+
| Economic Factors                                 |
|                                                  |
| US Rate          4.50%           Positive        |
| Malaysia OPR     3.00%           Neutral         |
| Oil              $72.30          Positive        |
| Inflation         2.1%           Neutral         |
| Sentiment        +0.34           Positive        |
|                                                  |
+--------------------------------------------------+
| Model Performance                                |
| XGBoost             62% directional accuracy    |
| Random Forest       57%                         |
| Baseline            51%                         |
+--------------------------------------------------+
```

------------------------------------------------------------------------

# 21. Automation

Use a scheduler to automate the pipeline.

Possible tools:

-   Cron
-   APScheduler
-   Airflow

For this project, **Cron or APScheduler is enough initially**.

Example:

``` text
Every day
   |
   v
Fetch new data
   |
   v
Validate
   |
   v
Store in PostgreSQL
   |
   v
Generate features
   |
   v
Generate prediction
   |
   v
Update dashboard
```

Model retraining can occur weekly or whenever enough new data becomes
available.

------------------------------------------------------------------------

# 22. Monitoring

Keep basic logs for:

-   Data ingestion failures
-   Missing data
-   API failures
-   Model errors
-   Prediction generation failures

Example:

``` text
2026-08-25 08:00
Exchange API: SUCCESS
Economic API: SUCCESS
News API: SUCCESS
Database: SUCCESS
Feature generation: SUCCESS
Model prediction: SUCCESS
```

Advanced monitoring is optional.

------------------------------------------------------------------------

# 23. Recommended Technology Stack

## Data Engineering

-   Python
-   pandas
-   NumPy
-   requests
-   PostgreSQL

## Machine Learning

-   scikit-learn
-   XGBoost
-   Optional: PyTorch for LSTM/GRU

## NLP

-   A pretrained sentiment-analysis model
-   News API

## Backend

-   FastAPI
-   Pydantic

## Frontend

-   React
-   Tailwind CSS
-   Chart.js or another React-compatible chart library

## Automation

-   Cron / APScheduler initially
-   Airflow optionally

## Version Control

-   Git
-   GitHub

## Deployment

Recommended simple architecture:

``` text
GitHub
   |
   +---- Backend -> Render / Railway
   |
   +---- PostgreSQL -> Managed PostgreSQL
   |
   +---- Frontend -> Vercel / Netlify
```

AWS can be introduced later if desired.

------------------------------------------------------------------------

# 24. What NOT to Build Initially

The original full architecture contains many enterprise-level
components.

Do not make these mandatory:

-   Kubernetes
-   Kafka
-   Distributed computing
-   Complex feature stores
-   Full model registry
-   Real-time streaming infrastructure
-   Transformer forecasting models
-   Complicated cloud architecture
-   Enterprise-grade authentication
-   Advanced MLOps

They increase complexity significantly without necessarily improving the
portfolio value proportionally.

------------------------------------------------------------------------

# 25. Recommended Final Architecture

The practical portfolio version should be:

``` text
+----------------------------------------------------------+
|                       DATA SOURCES                       |
|                                                          |
| Exchange | Economic | Interest | Commodities | Markets  |
|                    News / Sentiment                       |
+----------------------------+-----------------------------+
                             |
                             v
+----------------------------------------------------------+
|                    PYTHON ETL PIPELINE                   |
|                                                          |
| Ingestion -> Validation -> Cleaning -> Time Alignment   |
+----------------------------+-----------------------------+
                             |
                             v
+----------------------------------------------------------+
|                       POSTGRESQL                         |
|                                                          |
| Exchange | Economic | Commodities | Markets | News      |
| Features | Predictions                                  |
+----------------------------+-----------------------------+
                             |
                             v
+----------------------------------------------------------+
|                  FEATURE ENGINEERING                     |
|                                                          |
| Returns | Moving Average | Volatility | Rate Differential|
| Inflation | Commodity | Market | Sentiment | Events      |
+----------------------------+-----------------------------+
                             |
                             v
+----------------------------------------------------------+
|                       ML MODELS                          |
|                                                          |
| Naive -> Linear Regression -> Random Forest -> XGBoost  |
|                                Optional LSTM             |
+----------------------------+-----------------------------+
                             |
                             v
+----------------------------------------------------------+
|                 BACKTESTING & EVALUATION                 |
|                                                          |
| Walk-forward validation | MAE | RMSE | Directional Acc.|
+----------------------------+-----------------------------+
                             |
                             v
+----------------------------------------------------------+
|                       FASTAPI                            |
|                                                          |
| REST API -> Predictions -> Historical Data -> Indicators|
+----------------------------+-----------------------------+
                             |
                             v
+----------------------------------------------------------+
|                     REACT WEB UI                         |
|                                                          |
| Charts | Prediction | Confidence | Factors | Performance|
+----------------------------------------------------------+
```

------------------------------------------------------------------------

# 26. Using Claude AI

Claude can significantly reduce implementation time.

Good uses:

-   Generate boilerplate
-   Write ETL scripts
-   Generate SQL schemas
-   Debug errors
-   Create FastAPI endpoints
-   Create React components
-   Write tests
-   Refactor code
-   Explain unfamiliar code
-   Generate documentation

However, manually verify:

-   Data leakage
-   Time-series splitting
-   Date handling
-   API data quality
-   Feature calculations
-   Model evaluation
-   Financial assumptions

The objective should be to **use AI as a development assistant**, not
blindly generate the entire project.

------------------------------------------------------------------------

# 27. Project Success Criteria

The project is successful if it can:

1.  Retrieve historical currency data.
2.  Retrieve multiple economic/market variables.
3.  Store the data in PostgreSQL.
4.  Clean and align the datasets.
5.  Generate meaningful features.
6.  Train at least one ML model.
7.  Perform historical backtesting.
8.  Generate a future prediction.
9.  Incorporate news/sentiment.
10. Expose predictions through an API.
11. Display results through a web dashboard.
12. Run the data pipeline automatically.
13. Document the architecture and methodology.

Perfect prediction accuracy is **not** a requirement.

------------------------------------------------------------------------

# 28. Portfolio Positioning

This project demonstrates skills across several areas:

### Data Engineering

-   API ingestion
-   ETL
-   Data cleaning
-   SQL
-   PostgreSQL
-   Scheduling
-   Data modeling

### Machine Learning

-   Feature engineering
-   Regression
-   Gradient boosting
-   Time-series validation
-   Model evaluation

### AI / NLP

-   Sentiment analysis
-   Event classification
-   Text-to-feature conversion

### Software Engineering

-   Python
-   REST APIs
-   FastAPI
-   React
-   Git
-   Testing

### Deployment

-   Backend deployment
-   Database deployment
-   Frontend deployment

This makes it particularly suitable as a portfolio project demonstrating
a **CS → Software/Data Engineering → AI** progression.

------------------------------------------------------------------------

# 29. Important Limitations

Currency markets are highly complex and affected by information that may
be impossible to predict in advance.

Therefore:

-   The model will not reliably predict every market movement.
-   Unexpected events can cause large errors.
-   Historical relationships can change.
-   Data quality can strongly affect predictions.
-   Sentiment models can misunderstand context.
-   Macroeconomic data is often revised after initial publication.
-   Correlation does not necessarily mean causation.

The project should be presented as an **experimental
forecasting/analytics system**, not a guaranteed trading system or
financial-advice tool.

------------------------------------------------------------------------

# 30. Recommended Final Project

For the best balance between complexity, time, and portfolio value:

``` text
                 USD/MYR
                    |
                    v
       +-------------------------+
       | Historical Exchange Rate|
       +-------------------------+
                    |
                    |
 +------------------+------------------+
 |                  |                  |
 v                  v                  v
Interest        Economic           Commodity
Rates           Indicators          Prices
 |                  |                  |
 +------------------+------------------+
                    |
                    v
               Market Data
                    |
                    v
             News + Sentiment
                    |
                    v
             Python ETL Pipeline
                    |
                    v
               PostgreSQL
                    |
                    v
          Feature Engineering
                    |
                    v
                 XGBoost
                    |
                    v
         Walk-Forward Backtest
                    |
                    v
                FastAPI
                    |
                    v
              React Dashboard
                    |
                    v
        Prediction + Explanation
```

This is the version to build first.

The system should prioritize **completeness, correctness of the
pipeline, reproducibility, good engineering, and a polished UI** over
trying to achieve extremely high prediction accuracy.
