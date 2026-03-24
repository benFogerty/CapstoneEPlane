from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SCRIPT = REPO_ROOT / "frontend" / "scripts" / "live_model_outputs.py"


def run_planner(payload: dict) -> dict:
    raw = subprocess.check_output(
        [
            str(PYTHON),
            str(SCRIPT),
            "--plane-id",
            "166",
            "--planner-json",
            json.dumps(payload),
        ],
        text=True,
        cwd=REPO_ROOT,
    )
    return json.loads(raw)


class PlannerScenarioTests(unittest.TestCase):
    def test_longer_duration_increases_expected_wear(self) -> None:
        base_payload = {
            "mode": "single_plane",
            "startDate": "2026-03-18",
            "endDate": "2026-03-22",
            "missionTemplate": {
                "durationMin": 30,
                "reserveSocPct": 30,
                "departureWindowStart": "08:00",
                "departureWindowEnd": "10:00",
            },
            "chargePolicy": {
                "targetSocCapPct": 92,
                "latestChargeFinishLeadHours": 1.5,
            },
            "opsDemand": {"sortiesPerDay": 1},
        }
        longer_payload = {
            **base_payload,
            "missionTemplate": {
                **base_payload["missionTemplate"],
                "durationMin": 90,
            },
        }

        shorter = run_planner(base_payload)
        longer = run_planner(longer_payload)

        shorter_mean = sum(day["expectedDeltaSoh"] for day in shorter["modelDays"]) / len(
            shorter["modelDays"]
        )
        longer_mean = sum(day["expectedDeltaSoh"] for day in longer["modelDays"]) / len(
            longer["modelDays"]
        )

        self.assertLess(longer_mean, shorter_mean)

    def test_longer_route_distance_increases_modeled_soc_draw(self) -> None:
        base_payload = {
            "mode": "single_plane",
            "startDate": "2026-03-18",
            "endDate": "2026-03-18",
            "missionTemplate": {
                "durationMin": 25,
                "routeDistanceKm": 60,
                "reserveSocPct": 30,
                "departureWindowStart": "08:00",
                "departureWindowEnd": "10:00",
            },
            "chargePolicy": {
                "targetSocCapPct": 92,
                "latestChargeFinishLeadHours": 1.5,
            },
            "opsDemand": {"sortiesPerDay": 1},
        }
        longer_route_payload = {
            **base_payload,
            "missionTemplate": {
                **base_payload["missionTemplate"],
                "routeDistanceKm": 160,
            },
        }

        shorter = run_planner(base_payload)
        longer = run_planner(longer_route_payload)

        self.assertGreater(
            longer["modelDays"][0]["missionSocSpanPct"],
            shorter["modelDays"][0]["missionSocSpanPct"],
        )
        self.assertLess(
            longer["modelDays"][0]["expectedDeltaSoh"],
            shorter["modelDays"][0]["expectedDeltaSoh"],
        )

    def test_low_soc_cap_and_high_reserve_create_infeasible_days(self) -> None:
        payload = {
            "mode": "single_plane",
            "startDate": "2026-03-18",
            "endDate": "2026-03-20",
            "missionTemplate": {
                "durationMin": 90,
                "reserveSocPct": 45,
                "departureWindowStart": "08:00",
                "departureWindowEnd": "09:00",
            },
            "chargePolicy": {
                "targetSocCapPct": 70,
                "latestChargeFinishLeadHours": 2.0,
            },
            "opsDemand": {"sortiesPerDay": 2},
        }

        result = run_planner(payload)

        self.assertTrue(any(not day["feasible"] for day in result["modelDays"]))
        self.assertTrue(
            any(
                day["reserveMarginPct"] < 0 or "reserve" in day["summary"].lower()
                for day in result["modelDays"]
            )
        )

    def test_hot_windy_weather_increases_modeled_wear(self) -> None:
        base_payload = {
            "mode": "single_plane",
            "startDate": "2026-03-18",
            "endDate": "2026-03-18",
            "missionTemplate": {
                "durationMin": 45,
                "routeDistanceKm": 90,
                "reserveSocPct": 30,
                "departureWindowStart": "08:00",
                "departureWindowEnd": "10:00",
            },
            "chargePolicy": {
                "targetSocCapPct": 92,
                "latestChargeFinishLeadHours": 1.5,
            },
            "opsDemand": {"sortiesPerDay": 1},
        }
        mild_payload = {
            **base_payload,
            "weatherDays": [
                {
                    "date": "2026-03-18",
                    "tempMinC": 15,
                    "tempMaxC": 22,
                    "precipMm": 0,
                    "windKph": 10,
                    "confidenceTier": "high",
                }
            ],
        }
        harsh_payload = {
            **base_payload,
            "weatherDays": [
                {
                    "date": "2026-03-18",
                    "tempMinC": 29,
                    "tempMaxC": 38,
                    "precipMm": 4,
                    "windKph": 34,
                    "confidenceTier": "high",
                }
            ],
        }

        mild = run_planner(mild_payload)
        harsh = run_planner(harsh_payload)

        self.assertLess(
            harsh["modelDays"][0]["expectedDeltaSoh"],
            mild["modelDays"][0]["expectedDeltaSoh"],
        )
        self.assertLess(
            harsh["modelDays"][0]["chargingScore"],
            mild["modelDays"][0]["chargingScore"],
        )

    def test_existing_pack_charge_above_soc_cap_stays_feasible(self) -> None:
        payload = {
            "mode": "single_plane",
            "startDate": "2026-03-18",
            "endDate": "2026-03-18",
            "missionTemplate": {
                "durationMin": 45,
                "routeDistanceKm": 90,
                "reserveSocPct": 30,
                "departureWindowStart": "08:00",
                "departureWindowEnd": "10:30",
            },
            "chargePolicy": {
                "targetSocCapPct": 92,
                "latestChargeFinishLeadHours": 1.5,
            },
            "opsDemand": {"sortiesPerDay": 1},
            "weatherDays": [
                {
                    "date": "2026-03-18",
                    "tempMinC": -1,
                    "tempMaxC": 7,
                    "precipMm": 0,
                    "windKph": 12,
                    "confidenceTier": "high",
                }
            ],
        }

        result = run_planner(payload)
        day = result["modelDays"][0]

        self.assertTrue(day["feasible"])
        self.assertNotIn("target soc cap is too low", day["summary"].lower())

    def test_waiting_longer_at_high_soc_increases_storage_wear(self) -> None:
        payload = {
            "mode": "single_plane",
            "startDate": "2026-03-18",
            "endDate": "2026-03-22",
            "missionTemplate": {
                "durationMin": 45,
                "routeDistanceKm": 90,
                "reserveSocPct": 30,
                "departureWindowStart": "08:00",
                "departureWindowEnd": "10:30",
            },
            "chargePolicy": {
                "targetSocCapPct": 92,
                "latestChargeFinishLeadHours": 1.5,
            },
            "opsDemand": {"sortiesPerDay": 1},
            "weatherDays": [
                {
                    "date": date,
                    "tempMinC": 8,
                    "tempMaxC": 18,
                    "precipMm": 0,
                    "windKph": 10,
                    "confidenceTier": "high",
                }
                for date in [
                    "2026-03-18",
                    "2026-03-19",
                    "2026-03-20",
                    "2026-03-21",
                    "2026-03-22",
                ]
            ],
        }

        result = run_planner(payload)
        first_day = result["modelDays"][0]
        last_day = result["modelDays"][-1]

        self.assertLess(last_day["expectedDeltaSoh"], first_day["expectedDeltaSoh"])
        self.assertLess(last_day["chargingScore"], first_day["chargingScore"])

    def test_prediction_mode_still_returns_forecast_curve(self) -> None:
        raw = subprocess.check_output(
            [str(PYTHON), str(SCRIPT), "--plane-id", "166"],
            text=True,
            cwd=REPO_ROOT,
        )
        payload = json.loads(raw)

        self.assertIn("prediction", payload)
        self.assertIn("forecastCurve", payload["prediction"])
        self.assertGreater(len(payload["prediction"]["forecastCurve"]), 10)


if __name__ == "__main__":
    unittest.main()
