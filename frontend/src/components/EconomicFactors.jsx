// "Economic Factors" panel (§19/§20): the latest value of each indicator
// feature (interest rates, commodity prices) feeding the model.

function formatValue(name, value) {
  if (name.includes("OPR") || name.includes("Interest Rate")) {
    return `${value.toFixed(2)}%`;
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

export default function EconomicFactors({ indicators }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="text-base font-semibold text-slate-900">Economic Factors</h2>
      <dl className="mt-4 divide-y divide-slate-100">
        {indicators.map((indicator) => (
          <div key={indicator.name} className="flex items-center justify-between py-2">
            <dt className="text-sm text-slate-600">{indicator.name}</dt>
            <dd className="text-sm font-medium text-slate-900">{formatValue(indicator.name, indicator.value)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
