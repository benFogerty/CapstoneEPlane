import {
  ChargingCostResponseSchema,
  FlightsResponseSchema,
  LearnEvaluateRequestSchema,
  LearnEvaluateResponseSchema,
  GlossaryResponseSchema,
  LearnBaselineResponseSchema,
  MissionGameBaselineResponseSchema,
  MissionGameEvaluateResponseSchema,
  MissionGameInput,
  PlannerRequest,
  PlaneHealthResponseSchema,
  PlannerRequestSchema,
  PlannerResponseSchema,
  PlanesResponseSchema,
  PredictionsResponseSchema,
  RecommendationsResponseSchema,
  SohTrendResponseSchema,
  WeatherResponseSchema
} from "@/lib/contracts/schemas";

async function fetchAndParse<T>(
  endpoint: string,
  parser: { parse: (data: unknown) => T }
) {
  const response = await fetch(endpoint, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Request failed: ${endpoint}`);
  }
  const payload = await response.json();
  return parser.parse(payload);
}

async function postAndParse<T>(
  endpoint: string,
  body: unknown,
  parser: { parse: (data: unknown) => T }
) {
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body),
    cache: "no-store"
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${endpoint}`);
  }
  const payload = await response.json();
  return parser.parse(payload);
}

export function getPlanes() {
  return fetchAndParse("/api/v1/planes", PlanesResponseSchema);
}

export function getPlaneHealth(planeId: string) {
  return fetchAndParse(
    `/api/v1/planes/${planeId}/health`,
    PlaneHealthResponseSchema
  );
}

export function getPlaneTrend(planeId: string, window: "30d" | "90d" | "1y" | "full") {
  return fetchAndParse(
    `/api/v1/planes/${planeId}/soh-trend?window=${window}`,
    SohTrendResponseSchema
  );
}

export function getPlaneFlights(planeId: string, limit = 25) {
  return fetchAndParse(
    `/api/v1/planes/${planeId}/flights?limit=${limit}`,
    FlightsResponseSchema
  );
}

export function getPlanePrediction(planeId: string) {
  return fetchAndParse(
    `/api/v1/planes/${planeId}/predictions`,
    PredictionsResponseSchema
  );
}

export function getPlaneRecommendations(planeId: string, month: string) {
  return fetchAndParse(
    `/api/v1/planes/${planeId}/recommendations?month=${month}`,
    RecommendationsResponseSchema
  );
}

export function runPlanner(input: PlannerRequest) {
  const body = PlannerRequestSchema.parse(input);
  return postAndParse("/api/v1/planner", body, PlannerResponseSchema);
}

export function getWeather(airport: string, start: string, end: string) {
  return fetchAndParse(
    `/api/v1/weather?airport=${airport}&start=${start}&end=${end}`,
    WeatherResponseSchema
  );
}

export function getGlossary() {
  return fetchAndParse("/api/v1/glossary", GlossaryResponseSchema);
}

export function getLearnBaseline(planeId: string) {
  return fetchAndParse(
    `/api/v1/learn/baseline?planeId=${planeId}`,
    LearnBaselineResponseSchema
  );
}

export function evaluateLearnScenario(
  planeId: string,
  inputs: unknown
) {
  const body = LearnEvaluateRequestSchema.parse({ planeId, inputs });
  return postAndParse(
    "/api/v1/learn/evaluate",
    body,
    LearnEvaluateResponseSchema
  );
}

export function getChargingCost(airport: string, date: string, energyKwh: number) {
  return fetchAndParse(
    `/api/v1/charging-cost?airport=${airport}&date=${date}&energyKwh=${energyKwh}`,
    ChargingCostResponseSchema
  );
}

export function getMissionGameBaseline() {
  return fetchAndParse(
    "/api/v1/mission-game/baseline",
    MissionGameBaselineResponseSchema
  );
}

export function evaluateMissionGame(input: MissionGameInput) {
  return postAndParse(
    "/api/v1/mission-game/evaluate",
    input,
    MissionGameEvaluateResponseSchema
  );
}
