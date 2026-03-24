import { NextResponse } from "next/server";

import { RecommendationsResponseSchema } from "@/lib/contracts/schemas";
import { getLivePlanePayload } from "@/lib/live-plane-service";
import { buildPlannerPayload } from "@/lib/planner-service";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const revalidate = 0;

function normalizeMonth(value: string | null) {
  const candidate = value ?? new Date().toISOString().slice(0, 7);
  return /^\d{4}-\d{2}$/.test(candidate)
    ? candidate
    : new Date().toISOString().slice(0, 7);
}

function monthRange(month: string) {
  const [year, monthIndex] = month.split("-").map(Number);
  return {
    startDate: `${year}-${String(monthIndex).padStart(2, "0")}-01`,
    endDate: new Date(Date.UTC(year, monthIndex, 0)).toISOString().slice(0, 10)
  };
}

function airportCodeFromLabel(label: string | null | undefined) {
  return label?.slice(0, 4).toUpperCase() ?? "CYKF";
}

export async function GET(
  request: Request,
  { params }: { params: Promise<{ planeId: string }> }
) {
  const { planeId } = await params;
  const { searchParams } = new URL(request.url);
  const month = normalizeMonth(searchParams.get("month"));
  const live = await getLivePlanePayload(planeId);
  const { startDate, endDate } = monthRange(month);
  const airportCode = airportCodeFromLabel(
    (live.health as { lastFlight?: { departureAirport?: string | null } }).lastFlight
      ?.departureAirport
  );

  const plannerPayload = await buildPlannerPayload({
    mode: "single_plane",
    planeIds: [planeId],
    startDate,
    endDate,
    baseAirport: airportCode,
    missionTemplate: {
      durationMin: 45,
      routeDistanceKm: 90,
      reserveSocPct: 30,
      departureWindowStart: "08:00",
      departureWindowEnd: "10:30"
    },
    chargePolicy: {
      targetSocCapPct: 92,
      latestChargeFinishLeadHours: 1.5
    },
    opsDemand: {
      sortiesPerDay: 1
    },
    weatherMode: "forecast"
  });

  const { planner } = plannerPayload;
  const cards = [
    {
      id: "timing-best-day-live",
      type: "timing" as const,
      action:
        planner.recommendedDays[0] !== undefined
          ? `Prioritize ${planner.recommendedDays[0].date}; it is the strongest battery-aware operating window this month.`
          : "No days meet the current recommendation threshold this month.",
      confidence:
        planner.recommendedDays[0]?.confidenceTier === "high"
          ? 0.9
          : planner.recommendedDays[0]?.confidenceTier === "medium"
            ? 0.8
            : 0.7,
      why: planner.recommendedDays[0]
        ? planner.recommendedDays[0].why.slice(0, 3)
        : planner.warnings.slice(0, 3)
    },
    {
      id: "charge-window-live",
      type: "charging" as const,
      action:
        planner.chargeWindows[0] !== undefined
          ? `Target ${planner.chargeWindows[0].targetSocPct.toFixed(0)}% SOC inside the recommended charge window instead of holding a full pack early.`
          : "Use the recommended charge windows to avoid unnecessary high-SOC dwell.",
      confidence: 0.85,
      why: [
        "Charge timing is generated from the aircraft recommendation model rather than a static heuristic.",
        ...(planner.warnings.length ? [planner.warnings[0]] : []),
        ...(planner.assumptions.length ? [planner.assumptions[0]] : [])
      ].slice(0, 3)
    },
    {
      id: "avoid-stress-live",
      type:
        planner.notRecommendedDays.some((day) => day.status === "infeasible")
          ? ("dont" as const)
          : ("do" as const),
      action:
        planner.notRecommendedDays.some((day) => day.status === "infeasible")
          ? "Avoid days where reserve margin or charge timing becomes infeasible."
          : "Use the calendar to spread flights across the steadier lower-wear windows.",
      confidence: 0.8,
      why: [
        `${planner.notRecommendedDays.length} days in this range are not recommended under default battery rules.`,
        `${planner.recommendedDays.length} days remain recommended.`,
        `Weather source is ${planner.weatherSource}.`
      ]
    }
  ];

  const payload = RecommendationsResponseSchema.parse({
    recommendations: {
      planeId,
      month,
      generatedAt: planner.generatedAt,
      flightDayScores: planner.recommendedDays.slice(0, 10).map((day) => ({
        date: day.date,
        score: day.score,
        confidenceTier: day.confidenceTier,
        weatherSummary: day.weatherSummary
      })),
      calendarDays: planner.days.map((day) => ({
        date: day.date,
        score: day.score,
        confidenceTier: day.confidenceTier,
        weatherSummary: day.weatherSummary
      })),
      scoreBreakdownByDate: Object.fromEntries(
        planner.days.map((day) => [
          day.date,
          {
            weather: day.breakdown.weather,
            thermal: day.breakdown.thermal,
            stress: day.breakdown.wear,
            charging: day.breakdown.charging
          }
        ])
      ),
      learnAssumptionsRef: "single_plane_recommendation_v1_defaults",
      chargePlan: planner.chargeWindows.slice(0, 5).map((window) => ({
        date: window.date,
        targetSoc: window.targetSocPct,
        chargeWindowStart: window.chargeWindowStart ?? "",
        chargeWindowEnd: window.chargeWindowEnd ?? "",
        rationale: window.rationale
      })),
      cards
    }
  });

  return NextResponse.json(payload);
}
