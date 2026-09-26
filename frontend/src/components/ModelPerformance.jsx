// "Model Performance" panel (§17/§19/§20): shows the walk-forward backtest
// results weren't just assumed - the naive baseline is shown alongside the
// fancier models so a reader can see how much they actually improved on it.

const MODEL_LABELS = {
  naive: "Naive Baseline",
  naive_momentum: "Naive Momentum",
  moving_avg_7: "Moving Average (7d)",
  linear_regression: "Linear Regression",
  random_forest: "Random Forest",
  xgboost: "XGBoost",
};

function formatPct(value) {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
}

export default function ModelPerformance({ models }) {
  const sorted = [...models].sort((a, b) => a.mae - b.mae);
  const bestMae = Math.min(...models.map((m) => m.mae));

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="text-base font-semibold text-slate-900">Model Performance</h2>
      <p className="text-sm text-slate-500">Walk-forward backtest, 2022&ndash;present</p>

      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="text-slate-500">
              <th className="py-2 pr-4 font-medium">Model</th>
              <th className="py-2 pr-4 font-medium">MAE</th>
              <th className="py-2 pr-4 font-medium">RMSE</th>
              <th className="py-2 font-medium">Directional Accuracy</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {sorted.map((model) => (
              <tr key={model.model} className={model.mae === bestMae ? "bg-indigo-50/60" : undefined}>
                <td className="py-2 pr-4 font-medium text-slate-900">
                  {MODEL_LABELS[model.model] ?? model.model}
                </td>
                <td className="py-2 pr-4 text-slate-700">{model.mae.toFixed(2)}</td>
                <td className="py-2 pr-4 text-slate-700">{model.rmse.toFixed(2)}</td>
                <td className="py-2 text-slate-700">{formatPct(model.directional_accuracy)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
