// Thin wrapper around the FastAPI backend (api/main.py). Requests go to
// /api/*, which vite.config.js proxies to http://127.0.0.1:8000 during
// development - so the browser only ever talks to one origin, and the
// backend never needs CORS configuration.

const BASE = "/api";

async function getJSON(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const getPrediction = (pair) => getJSON(`/prediction/${pair}`);
export const getHistorical = (pair) => getJSON(`/historical/${pair}`);
export const getIndicators = (pair) => getJSON(`/indicators/${pair}`);
export const getModelPerformance = () => getJSON("/model-performance");
