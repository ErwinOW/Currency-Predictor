# Currency Predictor - Frontend

React + Vite + Tailwind CSS dashboard (§19-20). Consumes the FastAPI
backend in `../api/` - see the root [README](../README.md) for the full
project.

## Setup

```bash
npm install
npm run dev
```

Requires the API running separately (`uvicorn api.main:app --reload` from
the project root) - the dev server proxies `/api/*` to
`http://127.0.0.1:8000` (see `vite.config.js`), so the browser only ever
talks to one origin and the backend needs no CORS configuration.

## Structure

```
src/
  api.js                     fetch wrapper for the backend endpoints
  App.jsx                    fetches all data, lays out the panels
  components/
    CurrencyOverview.jsx      current/predicted rate, direction, confidence
    HistoricalChart.jsx       Chart.js line chart + the next-day prediction
    EconomicFactors.jsx       latest interest-rate/commodity values
    ModelPerformance.jsx      backtest results table (naive vs XGBoost etc.)
```
