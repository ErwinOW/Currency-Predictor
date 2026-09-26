import { useEffect, useState } from "react";
import { getHistorical, getIndicators, getModelPerformance, getPrediction } from "./api";
import CurrencyOverview from "./components/CurrencyOverview";
import HistoricalChart from "./components/HistoricalChart";
import EconomicFactors from "./components/EconomicFactors";
import ModelPerformance from "./components/ModelPerformance";

const PAIR = "MYR-IDR";

export default function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    Promise.all([getPrediction(PAIR), getHistorical(PAIR), getIndicators(PAIR), getModelPerformance()])
      .then(([prediction, historical, indicators, models]) => {
        if (!cancelled) setData({ prediction, historical, indicators, models });
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <div className="mx-auto max-w-2xl p-8">
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-6 text-rose-800">
          <p className="font-semibold">Couldn&apos;t load the dashboard.</p>
          <p className="mt-1 text-sm">{error}</p>
          <p className="mt-3 text-sm">
            Make sure the API is running (<code>uvicorn api.main:app --reload</code>) and that
            you&apos;ve run the ingestion, feature, and prediction scripts at least once.
          </p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex min-h-screen items-center justify-center text-slate-500">
        Loading&hellip;
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="mx-auto max-w-5xl space-y-6 p-6">
        <CurrencyOverview prediction={data.prediction} />
        <HistoricalChart historical={data.historical} prediction={data.prediction} />
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <EconomicFactors indicators={data.indicators} />
          <ModelPerformance models={data.models} />
        </div>
        <p className="pb-4 text-center text-xs text-slate-400">
          Experimental forecasting system for portfolio/educational purposes &mdash; not
          financial advice or a trading signal. See docs/project_rundown.md §29.
        </p>
      </div>
    </div>
  );
}
