// "Currency Overview" panel (project_rundown.md §19/§20): current rate,
// predicted rate, expected movement, direction, and confidence - the
// headline numbers of the dashboard.

const DIRECTION_STYLES = {
  Bullish: "bg-emerald-100 text-emerald-700",
  Bearish: "bg-rose-100 text-rose-700",
  Neutral: "bg-slate-100 text-slate-700",
};

function formatRate(value) {
  return value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export default function CurrencyOverview({ prediction }) {
  const {
    currency_pair,
    as_of_date,
    current_rate,
    predicted_rate,
    expected_movement_pct,
    direction,
    lower_bound,
    upper_bound,
    confidence,
  } = prediction;

  const movementSign = expected_movement_pct >= 0 ? "+" : "";
  const movementColor = expected_movement_pct >= 0 ? "text-emerald-600" : "text-rose-600";

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h1 className="text-lg font-semibold text-slate-900">Currency AI</h1>
        <span className="text-sm text-slate-500">
          {currency_pair} &middot; as of {as_of_date}
        </span>
      </div>

      <div className="mt-6 grid grid-cols-1 gap-6 sm:grid-cols-2">
        <div>
          <p className="text-sm text-slate-500">Current Rate</p>
          <p className="text-4xl font-bold text-slate-900">{formatRate(current_rate)}</p>
        </div>
        <div>
          <p className="text-sm text-slate-500">Predicted Rate</p>
          <p className="text-4xl font-bold text-slate-900">{formatRate(predicted_rate)}</p>
          <p className={`mt-1 text-sm font-medium ${movementColor}`}>
            {movementSign}
            {expected_movement_pct.toFixed(2)}%
          </p>
        </div>
      </div>

      <div className="mt-6 flex flex-wrap items-center gap-4">
        <span className={`rounded-full px-3 py-1 text-sm font-medium ${DIRECTION_STYLES[direction] ?? DIRECTION_STYLES.Neutral}`}>
          {direction}
        </span>
        <span className="text-sm text-slate-600">
          Expected range: {formatRate(lower_bound)} &ndash; {formatRate(upper_bound)}
        </span>
        <span className="text-sm text-slate-600">
          Confidence: {(confidence * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
}
