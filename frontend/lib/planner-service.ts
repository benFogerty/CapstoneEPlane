import { addDays } from "date-fns";

import { AIRPORTS } from "@/lib/airports";
import {
  PlannerDayResult,
  PlannerRequest,
  PlannerResponseSchema,
  PlannerRulImpact,
  PlannerWearImpact,
  WeatherDay
} from "@/lib/contracts/schemas";
import { getLivePlanePayload } from "@/lib/live-plane-service";
import { getLiveScenarioPlannerPayload } from "@/lib/live-planner-service";
import { getWeatherPayload, summarizeWeatherDay } from "@/lib/weather-service";

function clamp(value: number, min = 0, max = 100) {
  return Math.max(min, Math.min(max, value));
}

function parseUtcDate(value: string) {
  return new Date(`${value}T00:00:00Z`);
}

function daysBetween(a: string, b: string) {
  return Math.round((parseUtcDate(a).getTime() - parseUtcDate(b).getTime()) / 86_400_000);
}

function enumerateDates(startDate: string, endDate: string) {
  const dates: string[] = [];
  for (
    let cursor = parseUtcDate(startDate);
    cursor.getTime() <= parseUtcDate(endDate).getTime();
    cursor = addDays(cursor, 1)
  ) {
    dates.push(cursor.toISOString().slice(0, 10));
  }
  return dates;
}

function confidenceFromWeather(
  request: PlannerRequest,
  weatherSource: "live" | "mixed" | "fallback" | "manual",
  date: string
): "high" | "medium" | "low" {
  if (request.weatherMode === "manual" || weatherSource === "manual") {
    return "medium";
  }
  if (weatherSource === "fallback") {
    return "low";
  }
  const offset = daysBetween(date, new Date().toISOString().slice(0, 10));
  if (offset <= 9) {
    return weatherSource === "live" ? "high" : "medium";
  }
  if (offset <= 21) {
    return "medium";
  }
  return "low";
}

function buildManualWeatherDays(
  request: PlannerRequest
): {
  source: "manual";
  days: WeatherDay[];
} {
  const manual = request.manualWeather ?? { tempC: 21, windKph: 12, precipMm: 0 };
  const days = enumerateDates(request.startDate, request.endDate).map((date) => ({
    date,
    tempMinC: manual.tempC - 3,
    tempMaxC: manual.tempC + 3,
    precipMm: manual.precipMm,
    windKph: manual.windKph,
    summary: summarizeWeatherDay(manual.tempC + 3, manual.precipMm, manual.windKph),
    confidenceTier: "medium" as const
  }));
  return { source: "manual" as const, days };
}

function buildPlannerDay(
  request: PlannerRequest,
  planeId: string,
  modelDay: {
    date: string;
    sortieCount: number;
    durationMin: number;
    missionSocSpanPct: number;
    chargeDurationHr: number;
    targetSoc: number;
    chargeWindowStart: string | null;
    chargeWindowEnd: string | null;
    expectedDeltaSoh: number;
    postFlightSocPct: number;
    reserveMarginPct: number;
    modelStressScore: number;
    chargingScore: number;
    feasible: boolean;
    summary: string;
  },
  weatherDay: WeatherDay,
  weatherSource: "live" | "mixed" | "fallback" | "manual"
): PlannerDayResult {
  const weatherPenalty =
    weatherDay.precipMm * 3.0 + Math.max(0, weatherDay.windKph - 18) * 1.35;
  const tempMid = (weatherDay.tempMinC + weatherDay.tempMaxC) / 2;
  const thermalPenalty =
    Math.abs(tempMid - 21) * 1.4 +
    Math.max(0, 4 - weatherDay.tempMinC) * 0.85 +
    Math.max(0, weatherDay.tempMaxC - 31) * 1.05;

  const breakdown = {
    weather: Number(clamp(100 - weatherPenalty * 2.2).toFixed(2)),
    thermal: Number(clamp(100 - thermalPenalty * 2.5).toFixed(2)),
    wear: Number(clamp(modelDay.modelStressScore).toFixed(2)),
    charging: Number(clamp(modelDay.chargingScore).toFixed(2))
  };

  const isPast = modelDay.date < new Date().toISOString().slice(0, 10);
  const infeasiblePenalty =
    !modelDay.feasible || modelDay.reserveMarginPct < 0 ? 28 : 0;
  const score = Number(
    clamp(
      breakdown.weather * 0.03 +
        breakdown.thermal * 0.03 +
        breakdown.wear * 0.72 +
        breakdown.charging * 0.22 -
        infeasiblePenalty -
        (isPast ? 30 : 0)
    ).toFixed(2)
  );

  let status: PlannerDayResult["status"] = "avoid";
  if (!modelDay.feasible || modelDay.reserveMarginPct < 0 || isPast) {
    status = "infeasible";
  } else if (score >= 78 && modelDay.expectedDeltaSoh > -0.16) {
    status = "recommended";
  } else if (score >= 60) {
    status = "watch";
  }

  const why = [
    `Expected SOH delta is ${modelDay.expectedDeltaSoh.toFixed(3)} for this mission profile.`,
    `Reserve margin after the mission is ${modelDay.reserveMarginPct.toFixed(1)}% SOC.`,
    `Mission draw is modeled at ${modelDay.missionSocSpanPct.toFixed(1)}% SOC over ${modelDay.durationMin.toFixed(0)} minutes.`,
    `Charge target is capped at ${modelDay.targetSoc.toFixed(0)}% with an estimated ${modelDay.chargeDurationHr.toFixed(2)} h charging session.`,
    "Temperature and waiting-time effects are intended to influence the rank mainly through modeled battery wear, not through a large standalone weather heuristic."
  ];
  if (weatherSource === "fallback") {
    why.push("Weather is using fallback modeling, so confidence is reduced.");
  }

  return {
    planeId,
    date: modelDay.date,
    score,
    status,
    confidenceTier: confidenceFromWeather(request, weatherSource, modelDay.date),
    feasible: Boolean(modelDay.feasible && modelDay.reserveMarginPct >= 0 && !isPast),
    summary: modelDay.summary,
    weatherSummary: weatherDay.summary,
    why,
    breakdown,
    expectedDeltaSoh: Number(modelDay.expectedDeltaSoh.toFixed(4)),
    postFlightSocPct: Number(modelDay.postFlightSocPct.toFixed(2)),
    reserveMarginPct: Number(modelDay.reserveMarginPct.toFixed(2)),
    durationMin: Number(modelDay.durationMin.toFixed(0)),
    missionSocSpanPct: Number(modelDay.missionSocSpanPct.toFixed(1)),
    chargeDurationHr: Number(modelDay.chargeDurationHr.toFixed(2)),
    targetSocPct: Number(modelDay.targetSoc.toFixed(0)),
    sortieCount: modelDay.sortieCount,
    chargeWindowStart: modelDay.chargeWindowStart,
    chargeWindowEnd: modelDay.chargeWindowEnd
  };
}

function buildRulImpact(
  planeId: string,
  baseline: {
    replacementDatePred: string;
    rulDaysPred: number;
    rulCyclesPred: number;
  },
  currentSoh: number,
  totalExpectedDelta: number,
  flightsPerDay: number
): PlannerRulImpact {
  const baselineCycles = Math.max(1, baseline.rulCyclesPred);
  const perCycleLoss = Math.max((currentSoh - 40) / baselineCycles, 0.01);
  const deltaCycles = Math.round(totalExpectedDelta / perCycleLoss);
  const plannedRulCycles = Math.max(0, baseline.rulCyclesPred + deltaCycles);
  const plannedRulDays = Math.max(
    0,
    Math.round(plannedRulCycles / Math.max(flightsPerDay, 0.1))
  );
  const plannedReplacementDate = addDays(
    parseUtcDate(new Date().toISOString().slice(0, 10)),
    plannedRulDays
  )
    .toISOString()
    .slice(0, 10);

  return {
    planeId,
    baselineReplacementDate: baseline.replacementDatePred,
    plannedReplacementDate,
    baselineRulDays: baseline.rulDaysPred,
    plannedRulDays,
    baselineRulCycles: baseline.rulCyclesPred,
    plannedRulCycles,
    deltaRulDays: plannedRulDays - baseline.rulDaysPred,
    deltaRulCycles: plannedRulCycles - baseline.rulCyclesPred
  };
}

export async function buildPlannerPayload(input: PlannerRequest) {
  const airport = AIRPORTS[input.baseAirport.toUpperCase()];
  if (!airport) {
    throw new Error(`Unsupported airport code ${input.baseAirport}`);
  }

  const weatherPayload =
    input.weatherMode === "manual"
      ? buildManualWeatherDays(input)
      : await (async () => {
          const weather = await getWeatherPayload(
            airport.icao,
            input.startDate,
            input.endDate
          );
          return {
            source: weather.mode as "live" | "mixed" | "fallback",
            days: weather.days
          };
        })();
  const weatherByDate = new Map(weatherPayload.days.map((day) => [day.date, day]));

  const planeResults = await Promise.all(
    input.planeIds.map(async (planeId) => {
      const [livePlane, scenario] = await Promise.all([
        getLivePlanePayload(planeId),
        getLiveScenarioPlannerPayload(planeId, input, weatherPayload.days)
      ]);
      const days = scenario.modelDays.map((modelDay) =>
        buildPlannerDay(
          input,
          planeId,
          modelDay,
          weatherByDate.get(modelDay.date) ??
            weatherPayload.days[0] ?? {
              date: modelDay.date,
              tempMinC: 15,
              tempMaxC: 21,
              precipMm: 0,
              windKph: 12,
              summary: "Fallback day",
              confidenceTier: "low"
            },
          weatherPayload.source
        )
      );
      return {
        planeId,
        livePlane,
        scenario,
        days
      };
    })
  );

  const days = planeResults
    .flatMap((result) => result.days)
    .sort((a, b) => a.date.localeCompare(b.date) || a.planeId.localeCompare(b.planeId));
  const recommendedDays = [...days]
    .filter((day) => day.status === "recommended")
    .sort((a, b) => b.score - a.score || a.date.localeCompare(b.date));
  const notRecommendedDays = [...days]
    .filter((day) => day.status !== "recommended")
    .sort((a, b) => a.date.localeCompare(b.date) || a.score - b.score);

  const expectedWear: PlannerWearImpact[] = days.map((day) => ({
    planeId: day.planeId,
    date: day.date,
    expectedDeltaSoh: day.expectedDeltaSoh,
    postFlightSocPct: day.postFlightSocPct,
    reserveMarginPct: day.reserveMarginPct
  }));

  const chargeWindows = days
    .filter(
      (day) =>
        day.chargeWindowStart !== null &&
        day.chargeWindowEnd !== null &&
        day.status !== "infeasible"
    )
    .map((day) => ({
      planeId: day.planeId,
      date: day.date,
      targetSocPct: day.targetSocPct,
      chargeWindowStart: day.chargeWindowStart,
      chargeWindowEnd: day.chargeWindowEnd,
      rationale:
        day.reserveMarginPct < 6
          ? "Reserve margin is tight, so charging discipline matters for this day."
          : "Charge near departure to reduce high-SOC dwell while preserving margin."
    }));

  const rulImpact = planeResults.map(({ planeId, livePlane, days: planeDays }) => {
    const totalExpectedDelta = planeDays.reduce(
      (sum, day) => sum + day.expectedDeltaSoh,
      0
    );
    return buildRulImpact(
      planeId,
      livePlane.prediction.forecast as {
        replacementDatePred: string;
        rulDaysPred: number;
        rulCyclesPred: number;
      },
      Number((livePlane.health as { sohCurrent?: number }).sohCurrent ?? 0),
      totalExpectedDelta,
      Number(livePlane.ops?.flightsPerDayRecent ?? 0.2)
    );
  });

  const warnings = [
    ...(weatherPayload.source === "fallback"
      ? [
          "Weather is running in fallback mode for part or all of this range, so date confidence is reduced."
        ]
      : []),
    ...(recommendedDays.length === 0
      ? ["No dates are currently recommended under the selected battery constraints."]
      : [])
  ];

  const assumptions = Array.from(
    new Set([
      "This recommendation engine is advisory and battery-life-first, not a dispatch optimizer.",
      "The score prioritizes projected wear from the battery model, then charging feasibility, then thermal and weather conditions.",
      `The recommendation assumes ${input.opsDemand?.sortiesPerDay ?? 1} planned flight${(input.opsDemand?.sortiesPerDay ?? 1) === 1 ? "" : "s"} for the selected aircraft and day.`,
      ...planeResults.flatMap((result) => result.scenario.assumptions)
    ])
  );

  return PlannerResponseSchema.parse({
    planner: {
      mode: input.mode,
      generatedAt: new Date().toISOString(),
      startDate: input.startDate,
      endDate: input.endDate,
      baseAirport: airport.icao,
      weatherMode: input.weatherMode,
      weatherSource: weatherPayload.source,
      days,
      recommendedDays,
      notRecommendedDays,
      chargeWindows,
      expectedWear,
      rulImpact,
      warnings,
      assumptions
    }
  });
}
