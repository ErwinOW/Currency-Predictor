// Thin wrapper around the FastAPI backend (api/main.py).
//
// In development, BASE is "/api", which vite.config.js proxies to
// http://127.0.0.1:8000 - the browser only ever talks to one origin, so
// no CORS setup is needed locally.
//
// In production the frontend and backend are two separately deployed
// services (e.g. Vercel + Render) with no proxy between them, so BASE
// must be the backend's real URL instead - set VITE_API_URL as an
// environment variable in the frontend's hosting dashboard (Vite only
// exposes env vars prefixed VITE_, and only ones present at BUILD time,
// not runtime - see the deployment README). The backend's own CORS
// middleware (api/main.py) must then explicitly allow this frontend's
// origin via its FRONTEND_URL environment variable.
const BASE = import.meta.env.VITE_API_URL || "/api";

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
