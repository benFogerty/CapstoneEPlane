import { NextResponse } from "next/server";

import {
  PlannerRequestSchema,
  PlannerResponseSchema
} from "@/lib/contracts/schemas";
import { buildPlannerPayload } from "@/lib/planner-service";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function POST(request: Request) {
  const body = await request.json();
  const input = PlannerRequestSchema.parse(body);
  const payload = await buildPlannerPayload(input);
  return NextResponse.json(PlannerResponseSchema.parse(payload));
}
