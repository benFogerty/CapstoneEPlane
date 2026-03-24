import { z } from "zod";

export const RiskBandSchema = z.enum(["low", "medium", "watch", "critical", "decline", "high"]);
export const ConfidenceTierSchema = z.enum(["high", "medium", "low"]);
export const HealthLabelSchema = z.enum(["healthy", "medium", "watch", "decline", "critical"]);
export const WeatherModeSchema = z.enum(["live", "mixed", "fallback"]);
export const ChargingCostSourceModeSchema = z.enum(["live", "fallback"]);
export const MissionGameStatusSchema = z.enum([
  "recommended",
  "caution",
  "not_recommended"
]);
export const MissionGameModeSchema = z.enum(["single", "fleet_compare"]);
export const PayloadLevelSchema = z.enum(["light", "medium", "heavy"]);
export const PlannerModeSchema = z.literal("single_plane");
export const PlannerDayStatusSchema = z.enum([
  "recommended",
  "watch",
  "avoid",
  "infeasible"
]);

export const PlaneSummarySchema = z.object({
  planeId: z.string(),
  registration: z.string(),
  aircraftType: z.string(),
  flightsCount: z.number(),
  chargingEventsCount: z.number(),
  sohCurrent: z.number(),
  sohTrend30: z.number(),
  riskBand: RiskBandSchema,
  updatedAt: z.string()
});

export const PlaneLiveHealthSchema = z.object({
  planeId: z.string(),
  updatedAt: z.string(),
  sohCurrent: z.number(),
  sohTrend30: z.number(),
  sohTrend90: z.number(),
  currentChargeSoc: z.number(),
  timeSinceLastFlightHours: z.number(),
  riskBand: RiskBandSchema,
  healthScore: z.number(),
  healthLabel: HealthLabelSchema,
  healthExplanation: z.string(),
  metricsExplainabilityVersion: z.string(),
  confidence: z.number(),
  pack: z.object({
    voltage: z.number(),
    current: z.number(),
    tempAvg: z.number(),
    soc: z.number()
  }),
  lastFlight: z.object({
    flightId: z.number(),
    eventDate: z.string(),
    route: z.string(),
    departureAirport: z.string().nullable(),
    destinationAirport: z.string().nullable(),
    durationMin: z.number().nullable(),
    eventType: z.string()
  })
});

export const SohTrendPointSchema = z.object({
  date: z.string(),
  soh: z.number(),
  source: z.enum(["proxy", "observed_norm", "blend"])
});

export const FlightEventSummarySchema = z.object({
  flightId: z.number(),
  eventDate: z.string(),
  eventType: z.string(),
  route: z.string().nullable(),
  departureAirport: z.string().nullable(),
  destinationAirport: z.string().nullable(),
  durationMin: z.number().nullable(),
  isChargingEvent: z.boolean(),
  isFlightEvent: z.boolean()
});

export const ReplacementForecastSchema = z.object({
  replacementDatePred: z.string(),
  rulDaysPred: z.number(),
  rulCyclesPred: z.number(),
  confidence: z.number()
});

export const ForecastCurvePointSchema = z.object({
  date: z.string(),
  soh: z.number()
});

export const SohPredictionSchema = z.object({
  planeId: z.string(),
  forecast: ReplacementForecastSchema,
  forecastCurve: z.array(ForecastCurvePointSchema),
  sohTargetBlend: z.number(),
  sohProxyPoh: z.number(),
  sohObservedNorm: z.number()
});

export const RecommendationCardSchema = z.object({
  id: z.string(),
  type: z.enum(["do", "dont", "timing", "charging"]),
  action: z.string(),
  confidence: z.number(),
  why: z.array(z.string())
});

export const FlightDayScoreSchema = z.object({
  date: z.string(),
  score: z.number(),
  confidenceTier: ConfidenceTierSchema,
  weatherSummary: z.string()
});

export const ScoreBreakdownSchema = z.object({
  weather: z.number(),
  thermal: z.number(),
  stress: z.number(),
  charging: z.number()
});

export const ChargePlanSuggestionSchema = z.object({
  date: z.string(),
  targetSoc: z.number(),
  chargeWindowStart: z.string(),
  chargeWindowEnd: z.string(),
  rationale: z.string()
});

export const PlaneRecommendationsSchema = z.object({
  planeId: z.string(),
  month: z.string(),
  generatedAt: z.string(),
  flightDayScores: z.array(FlightDayScoreSchema),
  calendarDays: z.array(FlightDayScoreSchema),
  scoreBreakdownByDate: z.record(ScoreBreakdownSchema),
  learnAssumptionsRef: z.string(),
  chargePlan: z.array(ChargePlanSuggestionSchema),
  cards: z.array(RecommendationCardSchema)
});

export const WeatherDaySchema = z.object({
  date: z.string(),
  tempMinC: z.number(),
  tempMaxC: z.number(),
  precipMm: z.number(),
  windKph: z.number(),
  summary: z.string(),
  confidenceTier: ConfidenceTierSchema
});

export const WeatherResponseSchema = z.object({
  airport: z.string(),
  start: z.string(),
  end: z.string(),
  mode: WeatherModeSchema,
  demoMode: z.boolean(),
  days: z.array(WeatherDaySchema)
});

export const GlossaryItemSchema = z.object({
  id: z.string(),
  term: z.string(),
  plainLanguage: z.string(),
  whyItMatters: z.string(),
  technicalDetail: z.string().optional()
});

export const GlossaryResponseSchema = z.object({
  version: z.string(),
  items: z.array(GlossaryItemSchema)
});

export const LearnInputsSchema = z.object({
  ambientTempC: z.number(),
  flightDurationMin: z.number(),
  expectedPowerKw: z.number(),
  windSeverity: z.number(),
  precipitationSeverity: z.number(),
  chargeTargetSoc: z.number(),
  chargeLeadHours: z.number(),
  highSocIdleHours: z.number(),
  flightsPerWeek: z.number(),
  thermalManagementQuality: z.number(),
  cellImbalanceSeverity: z.number(),
  socEstimatorUncertainty: z.number()
});

export const LearnOutputsSchema = z.object({
  sohImpactDelta: z.number(),
  healthScore: z.number(),
  healthLabel: HealthLabelSchema,
  rulDaysShift: z.number(),
  recommendationSummary: z.string()
});

export const LearnTrajectoryPointSchema = z.object({
  label: z.string(),
  baseline: z.number(),
  simulated: z.number()
});

export const LearnEvaluationSchema = z.object({
  planeId: z.string(),
  outputs: LearnOutputsSchema.extend({
    healthExplanation: z.string(),
    expectedRangeKm: z.number(),
    confidence: z.number(),
    topDrivers: z.array(z.string())
  }),
  trajectory: z.array(LearnTrajectoryPointSchema)
});

export const LearnBaselineSchema = z.object({
  planeId: z.string(),
  assumptionsVersion: z.string(),
  baselineInputs: LearnInputsSchema,
  baselineOutputs: LearnOutputsSchema
});

export const ChargingCostEstimateSchema = z.object({
  airport: z.string(),
  state: z.string(),
  costPerKwhUsd: z.number(),
  estimatedSessionCostUsd: z.number(),
  energyKwh: z.number(),
  sourceMode: ChargingCostSourceModeSchema,
  generatedAt: z.string()
});

export const WeatherInputSchema = z.object({
  tempC: z.number(),
  windKph: z.number(),
  precipMm: z.number()
});

export const PlannerMissionTemplateSchema = z.object({
  durationMin: z.number().min(10).max(300),
  socSpanPct: z.number().min(5).max(95).optional(),
  routeDistanceKm: z.number().min(10).max(500).optional(),
  reserveSocPct: z.number().min(10).max(60),
  departureWindowStart: z.string().regex(/^\d{2}:\d{2}$/),
  departureWindowEnd: z.string().regex(/^\d{2}:\d{2}$/)
});

export const PlannerChargePolicySchema = z.object({
  targetSocCapPct: z.number().min(50).max(100),
  latestChargeFinishLeadHours: z.number().min(0).max(24)
});

export const PlannerOpsDemandSchema = z.object({
  sortiesPerDay: z.number().int().min(1).max(12)
});

export const PlannerRequestSchema = z
  .object({
    mode: PlannerModeSchema,
    planeIds: z.array(z.string()).min(1).max(1),
    startDate: z.string(),
    endDate: z.string(),
    baseAirport: z.string().min(4).max(4),
    missionTemplate: PlannerMissionTemplateSchema,
    chargePolicy: PlannerChargePolicySchema,
    opsDemand: PlannerOpsDemandSchema.optional(),
    weatherMode: z.enum(["forecast", "manual"]),
    manualWeather: WeatherInputSchema.optional()
  })
  .superRefine((value, ctx) => {
    if (
      value.weatherMode === "manual" &&
      value.manualWeather === undefined
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["manualWeather"],
        message: "manualWeather is required when weatherMode is manual."
      });
    }
    if (value.endDate < value.startDate) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["endDate"],
        message: "endDate must be on or after startDate."
      });
    }
    if (
      value.missionTemplate.departureWindowEnd <=
      value.missionTemplate.departureWindowStart
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["missionTemplate", "departureWindowEnd"],
        message: "departureWindowEnd must be after departureWindowStart."
      });
    }
  });

export const PlannerDayBreakdownSchema = z.object({
  weather: z.number(),
  thermal: z.number(),
  wear: z.number(),
  charging: z.number()
});

export const PlannerDayResultSchema = z.object({
  planeId: z.string(),
  date: z.string(),
  score: z.number(),
  status: PlannerDayStatusSchema,
  confidenceTier: ConfidenceTierSchema,
  feasible: z.boolean(),
  summary: z.string(),
  weatherSummary: z.string(),
  why: z.array(z.string()),
  breakdown: PlannerDayBreakdownSchema,
  expectedDeltaSoh: z.number(),
  postFlightSocPct: z.number(),
  reserveMarginPct: z.number(),
  durationMin: z.number(),
  missionSocSpanPct: z.number(),
  chargeDurationHr: z.number(),
  targetSocPct: z.number(),
  sortieCount: z.number().int().min(1),
  chargeWindowStart: z.string().nullable(),
  chargeWindowEnd: z.string().nullable()
});

export const PlannerChargeWindowSchema = z.object({
  planeId: z.string(),
  date: z.string(),
  targetSocPct: z.number(),
  chargeWindowStart: z.string().nullable(),
  chargeWindowEnd: z.string().nullable(),
  rationale: z.string()
});

export const PlannerWearImpactSchema = z.object({
  planeId: z.string(),
  date: z.string(),
  expectedDeltaSoh: z.number(),
  postFlightSocPct: z.number(),
  reserveMarginPct: z.number()
});

export const PlannerRulImpactSchema = z.object({
  planeId: z.string(),
  baselineReplacementDate: z.string(),
  plannedReplacementDate: z.string(),
  baselineRulDays: z.number(),
  plannedRulDays: z.number(),
  baselineRulCycles: z.number(),
  plannedRulCycles: z.number(),
  deltaRulDays: z.number(),
  deltaRulCycles: z.number()
});

export const PlannerResponseSchema = z.object({
  planner: z.object({
    mode: PlannerModeSchema,
    generatedAt: z.string(),
    startDate: z.string(),
    endDate: z.string(),
    baseAirport: z.string(),
    weatherMode: z.enum(["forecast", "manual"]),
    weatherSource: z.enum(["live", "mixed", "fallback", "manual"]),
    days: z.array(PlannerDayResultSchema),
    recommendedDays: z.array(PlannerDayResultSchema),
    notRecommendedDays: z.array(PlannerDayResultSchema),
    chargeWindows: z.array(PlannerChargeWindowSchema),
    expectedWear: z.array(PlannerWearImpactSchema),
    rulImpact: z.array(PlannerRulImpactSchema),
    warnings: z.array(z.string()),
    assumptions: z.array(z.string())
  })
});

export const MissionGameInputSchema = z.object({
  mode: MissionGameModeSchema,
  planeIds: z.array(z.string()).min(1),
  date: z.string(),
  plannedDurationMin: z.number().min(10).max(300),
  routeDistanceKm: z.number().min(10).max(500),
  targetSoc: z.number().min(50).max(100),
  payloadLevel: PayloadLevelSchema,
  weatherMode: z.enum(["forecast", "manual"]),
  manualWeather: WeatherInputSchema.optional()
});

export const MissionGameBreakdownSchema = z.object({
  batteryImpact: z.number(),
  safetyConfidence: z.number(),
  costEfficiency: z.number()
});

export const MissionGameSuggestionSchema = z.object({
  action: z.string(),
  expectedScoreDelta: z.number()
});

export const MissionGamePlaneResultSchema = z.object({
  planeId: z.string(),
  registration: z.string(),
  overallScore: z.number(),
  status: MissionGameStatusSchema,
  breakdown: MissionGameBreakdownSchema,
  estimatedCostUsd: z.number(),
  estimatedBatteryImpact: z.number()
});

export const MissionGameResultSchema = z.object({
  overallScore: z.number(),
  status: MissionGameStatusSchema,
  breakdown: MissionGameBreakdownSchema,
  estimatedCostUsd: z.number(),
  estimatedBatteryImpact: z.number(),
  why: z.array(z.string()),
  suggestions: z.array(MissionGameSuggestionSchema),
  evaluatedAt: z.string(),
  perPlaneResults: z.array(MissionGamePlaneResultSchema).optional()
});

export const MissionGameBaselineSchema = z.object({
  defaults: z.object({
    date: z.string(),
    plannedDurationMin: z.number(),
    routeDistanceKm: z.number(),
    targetSoc: z.number(),
    payloadLevel: PayloadLevelSchema,
    weatherMode: z.enum(["forecast", "manual"]),
    manualWeather: WeatherInputSchema
  }),
  scoringWeights: z.object({
    battery: z.number(),
    safety: z.number(),
    cost: z.number(),
    version: z.string()
  })
});

export const PlanesResponseSchema = z.object({
  planes: z.array(PlaneSummarySchema)
});

export const PlaneHealthResponseSchema = z.object({
  health: PlaneLiveHealthSchema
});

export const SohTrendResponseSchema = z.object({
  planeId: z.string(),
  window: z.enum(["30d", "90d", "1y", "full"]),
  points: z.array(SohTrendPointSchema)
});

export const FlightsResponseSchema = z.object({
  planeId: z.string(),
  flights: z.array(FlightEventSummarySchema)
});

export const PredictionsResponseSchema = z.object({
  prediction: SohPredictionSchema
});

export const RecommendationsResponseSchema = z.object({
  recommendations: PlaneRecommendationsSchema
});

export const LearnBaselineResponseSchema = z.object({
  baseline: LearnBaselineSchema,
  evaluation: LearnEvaluationSchema.optional()
});

export const LearnEvaluateRequestSchema = z.object({
  planeId: z.string(),
  inputs: LearnInputsSchema
});

export const LearnEvaluateResponseSchema = z.object({
  evaluation: LearnEvaluationSchema
});

export const ChargingCostResponseSchema = z.object({
  estimate: ChargingCostEstimateSchema
});

export const MissionGameEvaluateResponseSchema = z.object({
  result: MissionGameResultSchema
});

export const MissionGameBaselineResponseSchema = z.object({
  baseline: MissionGameBaselineSchema
});

export type PlaneSummary = z.infer<typeof PlaneSummarySchema>;
export type PlaneLiveHealth = z.infer<typeof PlaneLiveHealthSchema>;
export type SohTrendPoint = z.infer<typeof SohTrendPointSchema>;
export type FlightEventSummary = z.infer<typeof FlightEventSummarySchema>;
export type ReplacementForecast = z.infer<typeof ReplacementForecastSchema>;
export type ForecastCurvePoint = z.infer<typeof ForecastCurvePointSchema>;
export type SohPrediction = z.infer<typeof SohPredictionSchema>;
export type RecommendationCard = z.infer<typeof RecommendationCardSchema>;
export type FlightDayScore = z.infer<typeof FlightDayScoreSchema>;
export type ScoreBreakdown = z.infer<typeof ScoreBreakdownSchema>;
export type ChargePlanSuggestion = z.infer<typeof ChargePlanSuggestionSchema>;
export type PlaneRecommendations = z.infer<typeof PlaneRecommendationsSchema>;
export type WeatherDay = z.infer<typeof WeatherDaySchema>;
export type GlossaryItem = z.infer<typeof GlossaryItemSchema>;
export type LearnInputs = z.infer<typeof LearnInputsSchema>;
export type LearnOutputs = z.infer<typeof LearnOutputsSchema>;
export type LearnTrajectoryPoint = z.infer<typeof LearnTrajectoryPointSchema>;
export type LearnEvaluation = z.infer<typeof LearnEvaluationSchema>;
export type LearnBaseline = z.infer<typeof LearnBaselineSchema>;
export type ChargingCostEstimate = z.infer<typeof ChargingCostEstimateSchema>;
export type PlannerMissionTemplate = z.infer<typeof PlannerMissionTemplateSchema>;
export type PlannerChargePolicy = z.infer<typeof PlannerChargePolicySchema>;
export type PlannerOpsDemand = z.infer<typeof PlannerOpsDemandSchema>;
export type PlannerRequest = z.infer<typeof PlannerRequestSchema>;
export type PlannerDayBreakdown = z.infer<typeof PlannerDayBreakdownSchema>;
export type PlannerDayResult = z.infer<typeof PlannerDayResultSchema>;
export type PlannerChargeWindow = z.infer<typeof PlannerChargeWindowSchema>;
export type PlannerWearImpact = z.infer<typeof PlannerWearImpactSchema>;
export type PlannerRulImpact = z.infer<typeof PlannerRulImpactSchema>;
export type MissionGameInput = z.infer<typeof MissionGameInputSchema>;
export type MissionGameResult = z.infer<typeof MissionGameResultSchema>;
export type MissionGamePlaneResult = z.infer<typeof MissionGamePlaneResultSchema>;
export type MissionGameBaseline = z.infer<typeof MissionGameBaselineSchema>;
