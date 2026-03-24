"use client";

import { useEffect, useMemo, useState } from "react";

import {
  ChargePlanSuggestion,
  FlightDayScore,
  ScoreBreakdown
} from "@/lib/contracts/schemas";
import { Badge } from "@/components/ui/badge";

type Props = {
  days: FlightDayScore[];
  breakdownByDate: Record<string, ScoreBreakdown>;
  chargePlan: ChargePlanSuggestion[];
  selectedDate?: string | null;
  onSelectedDateChange?: (date: string) => void;
};

type CalendarCell = {
  date: string;
  day: number;
  score: number;
  confidenceTier: "high" | "medium" | "low";
  weatherSummary: string;
} | null;

function scoreClass(score: number) {
  if (score >= 75) return "bg-emerald-100 border-emerald-300";
  if (score >= 60) return "bg-amber-100 border-amber-300";
  return "bg-rose-100 border-rose-300";
}

function rationaleFromBreakdown(breakdown: ScoreBreakdown | null) {
  if (!breakdown) return [];
  const bullets: string[] = [];
  if (breakdown.stress < 70) bullets.push("Projected battery wear is elevated for this mission.");
  if (breakdown.charging < 70) bullets.push("Charging timing and high-SOC dwell are critical for this date.");
  if (breakdown.thermal < 60) bullets.push("Temperature conditions add battery stress in the wear model.");
  if (breakdown.weather < 60) bullets.push("Weather matters here mostly because it increases modeled mission energy draw.");
  if (bullets.length === 0) bullets.push("Battery-wear conditions are relatively manageable for this date.");
  return bullets;
}

export function RecommendationCalendar({
  days,
  breakdownByDate,
  chargePlan,
  selectedDate,
  onSelectedDateChange
}: Props) {
  const [internalSelectedDate, setInternalSelectedDate] = useState<string | null>(
    days[0]?.date ?? null
  );
  const activeSelectedDate = selectedDate ?? internalSelectedDate;

  useEffect(() => {
    if (selectedDate !== undefined) {
      return;
    }
    if (!days.length) {
      setInternalSelectedDate(null);
      return;
    }
    setInternalSelectedDate((current) =>
      current && days.some((day) => day.date === current) ? current : days[0].date
    );
  }, [days, selectedDate]);

  const cells = useMemo<CalendarCell[]>(() => {
    if (days.length === 0) return [];
    const sorted = [...days].sort((a, b) => a.date.localeCompare(b.date));
    const firstDate = new Date(`${sorted[0].date}T00:00:00Z`);
    const startOffset = firstDate.getUTCDay();
    const leading = Array.from({ length: startOffset }).map(() => null);
    const mapped = sorted.map((day) => ({
      date: day.date,
      day: new Date(`${day.date}T00:00:00Z`).getUTCDate(),
      score: day.score,
      confidenceTier: day.confidenceTier,
      weatherSummary: day.weatherSummary
    }));
    return [...leading, ...mapped];
  }, [days]);

  const selected =
    activeSelectedDate && days.length
      ? days.find((day) => day.date === activeSelectedDate) ?? null
      : null;
  const breakdown = selected ? breakdownByDate[selected.date] : null;
  const plannedCharge = selected
    ? chargePlan.find((item) => item.date === selected.date) ?? null
    : null;
  const rationale = rationaleFromBreakdown(breakdown);

  function handleDateSelect(date: string) {
    onSelectedDateChange?.(date);
    if (selectedDate === undefined) {
      setInternalSelectedDate(date);
    }
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-7 gap-2 text-center text-xs text-muted">
        {["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].map((name) => (
          <p key={name}>{name}</p>
        ))}
      </div>
      <div className="grid grid-cols-7 gap-2">
        {cells.map((cell, index) =>
          cell ? (
            <button
              key={cell.date}
              type="button"
              onClick={() => handleDateSelect(cell.date)}
              className={`rounded-xl border p-2 text-left transition hover:brightness-105 ${scoreClass(
                cell.score
              )} ${
                activeSelectedDate === cell.date
                  ? "ring-2 ring-blue-400"
                  : "ring-0 ring-transparent"
              }`}
            >
              <p className="text-xs font-semibold text-slate-900">{cell.day}</p>
              <p className="text-[11px] text-slate-700">{cell.score.toFixed(0)}</p>
            </button>
          ) : (
            <div key={`blank-${index}`} />
          )
        )}
      </div>

      {selected ? (
        <div className="rounded-2xl border border-stone-200 bg-white/85 p-4">
          <div className="mb-2 flex items-center justify-between">
            <p className="font-semibold text-slate-900">Day Details: {selected.date}</p>
            <Badge
              tone={
                selected.confidenceTier === "high"
                  ? "ok"
                  : selected.confidenceTier === "medium"
                    ? "warn"
                    : "risk"
              }
            >
              {selected.confidenceTier}
            </Badge>
          </div>
          <p className="text-sm font-semibold text-blue-700">
            Score {selected.score.toFixed(1)}
          </p>
          <p className="text-xs text-muted">{selected.weatherSummary}</p>

          {breakdown ? (
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <p className="rounded-lg border border-stone-200 bg-stone-50 p-2">
                Weather: <span className="font-semibold">{breakdown.weather.toFixed(1)}</span>
              </p>
              <p className="rounded-lg border border-stone-200 bg-stone-50 p-2">
                Thermal: <span className="font-semibold">{breakdown.thermal.toFixed(1)}</span>
              </p>
              <p className="rounded-lg border border-stone-200 bg-stone-50 p-2">
                Stress: <span className="font-semibold">{breakdown.stress.toFixed(1)}</span>
              </p>
              <p className="rounded-lg border border-stone-200 bg-stone-50 p-2">
                Charging: <span className="font-semibold">{breakdown.charging.toFixed(1)}</span>
              </p>
            </div>
          ) : null}

          <div className="mt-3">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted">
              Why this recommendation
            </p>
            <ul className="mt-1 space-y-1 text-xs text-slate-700">
              {rationale.map((bullet) => (
                <li key={bullet}>- {bullet}</li>
              ))}
            </ul>
          </div>

          <div className="mt-3 rounded-xl border border-blue-100 bg-blue-50/70 p-3 text-xs">
            <p className="font-semibold text-slate-900">Charge timing guidance</p>
            {plannedCharge ? (
              <p className="text-slate-700">
                Charge to {plannedCharge.targetSoc}% between{" "}
                {new Date(plannedCharge.chargeWindowStart).toLocaleString()} and{" "}
                {new Date(plannedCharge.chargeWindowEnd).toLocaleString()}.
              </p>
            ) : (
              <p className="text-slate-700">
                No specific window for this day. Prefer charging close to departure.
              </p>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}
