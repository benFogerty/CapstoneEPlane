import { runPythonJson } from "@/lib/live-python";
import { PlannerRequest, WeatherDay } from "@/lib/contracts/schemas";

const LIVE_TTL_MS = 60_000;

type LivePlannerModelDay = {
  planeId: string;
  date: string;
  sortieCount: number;
  durationMin: number;
  missionSocSpanPct: number;
  reserveSocPct: number;
  targetSoc: number;
  chargeDurationHr: number;
  chargeWindowStart: string | null;
  chargeWindowEnd: string | null;
  expectedDeltaSoh: number;
  postFlightSocPct: number;
  reserveMarginPct: number;
  modelStressScore: number;
  chargingScore: number;
  feasible: boolean;
  summary: string;
};

type PlannerWeatherContext = Pick<
  WeatherDay,
  "date" | "tempMinC" | "tempMaxC" | "precipMm" | "windKph" | "confidenceTier"
>;

type LivePlannerPayload = {
  planeId: string;
  generatedAt: string;
  startDate: string;
  endDate: string;
  assumptions: string[];
  modelDays: LivePlannerModelDay[];
};

type CacheEntry = {
  expiresAt: number;
  value?: LivePlannerPayload;
  promise?: Promise<LivePlannerPayload>;
};

const plannerCache = new Map<string, CacheEntry>();

function cacheKey(
  planeId: string,
  request: PlannerRequest,
  weatherDays: PlannerWeatherContext[]
) {
  return JSON.stringify({
    planeId,
    mode: request.mode,
    startDate: request.startDate,
    endDate: request.endDate,
    missionTemplate: request.missionTemplate,
    chargePolicy: request.chargePolicy,
    opsDemand: request.opsDemand,
    weatherDays
  });
}

async function runLivePlannerScript(
  planeId: string,
  request: PlannerRequest,
  weatherDays: PlannerWeatherContext[]
): Promise<LivePlannerPayload> {
  const payload = await runPythonJson<LivePlannerPayload>("live_model_outputs.py", [
    "--plane-id",
    planeId,
    "--planner-json",
    JSON.stringify({
      mode: request.mode,
      startDate: request.startDate,
      endDate: request.endDate,
      missionTemplate: request.missionTemplate,
      chargePolicy: request.chargePolicy,
      opsDemand: request.opsDemand,
      weatherDays
    })
  ]);
  if (!payload || payload.planeId !== planeId) {
    throw new Error(`Unexpected planner payload for plane ${planeId}`);
  }
  return payload;
}

export async function getLiveScenarioPlannerPayload(
  planeId: string,
  request: PlannerRequest,
  weatherDays: PlannerWeatherContext[]
): Promise<LivePlannerPayload> {
  const key = cacheKey(planeId, request, weatherDays);
  const now = Date.now();
  const cached = plannerCache.get(key);
  if (cached?.value && cached.expiresAt > now) {
    return cached.value;
  }
  if (cached?.promise) {
    return cached.promise;
  }

  const promise = runLivePlannerScript(planeId, request, weatherDays)
    .then((value) => {
      plannerCache.set(key, {
        value,
        expiresAt: Date.now() + LIVE_TTL_MS
      });
      return value;
    })
    .catch((error) => {
      plannerCache.delete(key);
      throw error;
    });

  plannerCache.set(key, {
    expiresAt: now + LIVE_TTL_MS,
    promise
  });

  return promise;
}
