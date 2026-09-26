// "Historical Chart" panel (§19/§20): historical exchange rate, the model's
// prediction for the next day, and the prediction's expected range.
import { useMemo } from "react";
import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Tooltip);

// Showing every trading day since 2015 would cram ~2,900 points into one
// chart; the last year is enough to see recent behaviour clearly while
// keeping the prediction point visually meaningful at the edge.
const DAYS_SHOWN = 365;

export default function HistoricalChart({ historical, prediction }) {
  const data = useMemo(() => {
    const recent = historical.slice(-DAYS_SHOWN);
    const labels = [...recent.map((point) => point.date), prediction.as_of_date + " (predicted)"];

    const actual = recent.map((point) => point.close);
    // One extra point at the end (the prediction) with everything before
    // it left null, so Chart.js draws it as a separate, disconnected dot
    // rather than joining it straight into the historical line.
    const predictedSeries = [...recent.map(() => null), prediction.predicted_rate];
    const rangeUpper = [...recent.map(() => null), prediction.upper_bound];
    const rangeLower = [...recent.map(() => null), prediction.lower_bound];

    return {
      labels,
      datasets: [
        {
          label: "Historical close",
          data: actual,
          borderColor: "#0f172a",
          backgroundColor: "transparent",
          pointRadius: 0,
          borderWidth: 1.5,
          tension: 0.15,
        },
        {
          label: "Predicted range (upper)",
          data: rangeUpper,
          borderColor: "transparent",
          backgroundColor: "rgba(129, 140, 248, 0.25)",
          pointRadius: 0,
          fill: "+1",
        },
        {
          label: "Predicted range (lower)",
          data: rangeLower,
          borderColor: "transparent",
          backgroundColor: "transparent",
          pointRadius: 0,
          fill: false,
        },
        {
          label: "Prediction",
          data: predictedSeries,
          borderColor: "#4f46e5",
          backgroundColor: "#4f46e5",
          pointRadius: 5,
          pointStyle: "circle",
          showLine: false,
        },
      ],
    };
  }, [historical, prediction]);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { mode: "index", intersect: false },
    },
    scales: {
      x: {
        ticks: { maxTicksLimit: 8 },
        grid: { display: false },
      },
      y: {
        ticks: { callback: (value) => value.toLocaleString() },
      },
    },
  };

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="text-base font-semibold text-slate-900">Historical + Prediction</h2>
      <p className="text-sm text-slate-500">Last {DAYS_SHOWN} trading days, with the next-day prediction and its expected range</p>
      <div className="mt-4 h-80">
        <Line data={data} options={options} />
      </div>
    </div>
  );
}
