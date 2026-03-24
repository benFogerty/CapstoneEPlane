from __future__ import annotations

import argparse
import json
import math
import warnings
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from live_plane_data import (
    _build_last_flight,
    _estimate_flights_per_day,
    _latest_charge_soc,
    _load_latent,
    _load_manifest,
    _safe_float,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "ml_workspace" / "soh_forecast" / "output" / "models" / "elastic_net.joblib"
RATED_CAPACITY_AH = 29.0
REPLACEMENT_THRESHOLD_SOH = 40.0


@dataclass
class PlaneProfile:
    plane_id: str
    current_soh: float
    current_soc: float
    flights_per_day_recent: float
    cadence_confidence: float
    mission_soc_span_pct: float
    mission_duration_hr: float
    charge_target_soc_pct: float
    charge_rate_pct_per_hr: float
    charge_to_flight_delay_hr: float
    reserve_soc_pct: float
    cumulative_efc: float
    cumulative_flight_hours: float
    avg_temp_c: float
    max_temp_c: float
    rms_current_a: float
    peak_c_rate: float
    initial_time: datetime
    latest_time: datetime
    last_flight: dict[str, Any]
    registration: str
    aircraft_type: str


@dataclass
class SimulationState:
    current_soh: float
    current_soc: float
    cumulative_efc: float
    cumulative_flight_hours: float
    current_time: datetime
    latest_observed_time: datetime
    previous_core: dict[str, float] = field(default_factory=dict)


def _load_feature_model() -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(DEFAULT_MODEL_PATH)


def _recent_numeric_median(series: pd.Series, default: float) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return float(default)
    return float(np.nanmedian(values.to_numpy(dtype=float)))


def _arrhenius_feature(temp_c: float) -> float:
    return float(math.exp((temp_c + 273.15) / 48.0))


def _clip(value: float, low: float, high: float) -> float:
    return float(min(high, max(low, value)))


def _recent_charge_delay_hours(latent_df: pd.DataFrame) -> float:
    ordered = latent_df.sort_values("event_datetime").copy()
    delays: list[float] = []
    for idx in range(len(ordered) - 1):
        current = ordered.iloc[idx]
        nxt = ordered.iloc[idx + 1]
        current_type = str(current.get("event_type", "")).lower()
        next_type = str(nxt.get("event_type", "")).lower()
        if "charge" not in current_type or next_type != "flight":
            continue
        start = current.get("event_datetime")
        end = nxt.get("event_datetime")
        duration_s = _safe_float(current.get("event_duration_s", np.nan), 0.0)
        if pd.isna(start) or pd.isna(end):
            continue
        charge_end = pd.Timestamp(start) + pd.Timedelta(seconds=max(duration_s, 0.0))
        delay_hr = (pd.Timestamp(end) - charge_end).total_seconds() / 3600.0
        if math.isfinite(delay_hr) and 0.0 <= delay_hr <= 24.0:
            delays.append(delay_hr)
    if not delays:
        return 1.5
    return float(np.nanmedian(np.array(delays, dtype=float)))


def _derive_plane_profile(plane_id: str) -> PlaneProfile:
    repo_root = PROJECT_ROOT
    manifest_df = _load_manifest(repo_root, plane_id)
    latent_df = _load_latent(repo_root, plane_id)
    latest = latent_df.sort_values("event_datetime").iloc[-1]
    current_soh = _safe_float(latest.get("latent_soh_filter_pct", np.nan), 0.0)
    current_soc = _latest_charge_soc(latent_df, latest)
    flights_per_day_recent, cadence_confidence = _estimate_flights_per_day(manifest_df)

    flights = latent_df.loc[latent_df["event_type"].eq("flight")].copy().tail(25)
    charges = latent_df.loc[
        latent_df["event_type"].astype(str).str.lower().str.contains("charge", na=False)
    ].copy().tail(25)

    flight_spans = (
        pd.to_numeric(flights.get("soc_max_pct"), errors="coerce")
        - pd.to_numeric(flights.get("soc_min_pct"), errors="coerce")
    ).abs()
    charge_spans = (
        pd.to_numeric(charges.get("soc_max_pct"), errors="coerce")
        - pd.to_numeric(charges.get("soc_min_pct"), errors="coerce")
    ).abs()
    mission_soc_span_pct = _clip(_recent_numeric_median(flight_spans, 48.0), 18.0, 70.0)
    mission_duration_hr = _clip(
        _recent_numeric_median(flights.get("event_duration_s"), 45.0 * 60.0) / 3600.0,
        0.35,
        2.0,
    )
    charge_target_soc_pct = _clip(
        _recent_numeric_median(charges.get("soc_max_pct"), max(current_soc + 10.0, 84.0)),
        76.0,
        96.0,
    )
    reserve_soc_pct = 30.0
    minimum_target = mission_soc_span_pct + reserve_soc_pct + 6.0
    charge_target_soc_pct = min(98.0, max(charge_target_soc_pct, minimum_target))
    charge_rate_pct_per_hr = _clip(
        _recent_numeric_median(
            charge_spans
            / (
                pd.to_numeric(charges.get("event_duration_s"), errors="coerce").replace(0, np.nan)
                / 3600.0
            ),
            36.0,
        ),
        12.0,
        90.0,
    )
    charge_to_flight_delay_hr = _clip(_recent_charge_delay_hours(latent_df), 0.5, 12.0)
    cumulative_efc = float(
        (
            (
                pd.to_numeric(latent_df.get("soc_max_pct"), errors="coerce")
                - pd.to_numeric(latent_df.get("soc_min_pct"), errors="coerce")
            )
            .abs()
            .fillna(0.0)
            / 100.0
        ).sum()
    )

    manifest_flights = manifest_df.loc[manifest_df["is_flight_event"] == 1].copy()
    flight_hours = _recent_numeric_median(
        manifest_flights.get("detail_duration"),
        mission_duration_hr * 60.0,
    ) / 60.0
    cumulative_flight_hours = max(
        flight_hours * max(len(manifest_flights), 1),
        float(
            (
                pd.to_numeric(flights.get("event_duration_s"), errors="coerce").fillna(0.0) / 3600.0
            ).sum()
        ),
    )

    avg_temp_c = _clip(_recent_numeric_median(latent_df.get("avg_cell_temp_mean_c"), 21.0), -10.0, 45.0)
    max_temp_c = _clip(
        _recent_numeric_median(latent_df.get("avg_cell_temp_max_c"), avg_temp_c + 2.5),
        avg_temp_c,
        55.0,
    )
    rms_current_a = _clip(_recent_numeric_median(latent_df.get("current_abs_mean_a"), 37.5), 8.0, 90.0)
    peak_current_a = _recent_numeric_median(latent_df.get("p95_abs_current_a"), rms_current_a * 1.35)
    peak_c_rate = _clip(max(peak_current_a / RATED_CAPACITY_AH, rms_current_a / RATED_CAPACITY_AH), 0.25, 3.0)
    latest_time = pd.Timestamp(latest["event_datetime"]).to_pydatetime().astimezone(timezone.utc)
    initial_time = max(datetime.now(timezone.utc), latest_time)
    last_flight = _build_last_flight(manifest_df, latent_df)
    registration = (
        str(manifest_df["registration"].dropna().iloc[0])
        if manifest_df["registration"].notna().any()
        else f"Plane {plane_id}"
    )
    aircraft_type = (
        str(manifest_df["aircraft_type"].dropna().iloc[0])
        if manifest_df["aircraft_type"].notna().any()
        else "Unknown aircraft"
    )

    return PlaneProfile(
        plane_id=str(plane_id),
        current_soh=float(current_soh),
        current_soc=float(current_soc),
        flights_per_day_recent=float(flights_per_day_recent),
        cadence_confidence=float(cadence_confidence),
        mission_soc_span_pct=float(mission_soc_span_pct),
        mission_duration_hr=float(mission_duration_hr),
        charge_target_soc_pct=float(charge_target_soc_pct),
        charge_rate_pct_per_hr=float(charge_rate_pct_per_hr),
        charge_to_flight_delay_hr=float(charge_to_flight_delay_hr),
        reserve_soc_pct=float(reserve_soc_pct),
        cumulative_efc=float(cumulative_efc),
        cumulative_flight_hours=float(cumulative_flight_hours),
        avg_temp_c=float(avg_temp_c),
        max_temp_c=float(max_temp_c),
        rms_current_a=float(rms_current_a),
        peak_c_rate=float(peak_c_rate),
        initial_time=initial_time,
        latest_time=latest_time,
        last_flight=last_flight,
        registration=registration,
        aircraft_type=aircraft_type,
    )


def _base_state(profile: PlaneProfile) -> SimulationState:
    return SimulationState(
        current_soh=profile.current_soh,
        current_soc=profile.current_soc,
        cumulative_efc=profile.cumulative_efc,
        cumulative_flight_hours=profile.cumulative_flight_hours,
        current_time=profile.initial_time,
        latest_observed_time=profile.latest_time,
        previous_core={},
    )


def _build_feature_row(
    bundle: dict[str, Any],
    state: SimulationState,
    profile: PlaneProfile,
    *,
    event_type: str,
    delta_soc_pct: float,
    duration_hr: float,
    event_time: datetime,
    avg_temp_c: float,
    max_temp_c: float,
    charge_delay_hr: float,
) -> tuple[np.ndarray, dict[str, float], float]:
    feature_names: list[str] = bundle["feature_names"]
    medians: dict[str, float] = bundle["feature_medians"]
    idle_hours = max((event_time - state.current_time).total_seconds() / 3600.0, 0.5)
    mission_efc = abs(delta_soc_pct) / 100.0
    cumulative_efc = state.cumulative_efc + mission_efc
    cumulative_flight_hours = state.cumulative_flight_hours + (duration_hr if event_type == "mission" else 0.0)
    rms_current_a = profile.rms_current_a * (1.05 if event_type == "charge" else 1.0)
    rms_c_rate = rms_current_a / RATED_CAPACITY_AH
    peak_c_rate = profile.peak_c_rate * (1.08 if event_type == "charge" else 1.0)
    temp_rise_proxy_c = max(1.0, max_temp_c - avg_temp_c)
    time_above_40c_min = max(0.0, duration_hr * 60.0 * (max_temp_c - 40.0) / 6.0)
    soc_weighted_idle_hours = idle_hours * max(state.current_soc, 10.0) / 100.0
    internal_resistance_ohm = float(medians.get("internal_resistance_ohm", 0.37)) + max(
        0.0, (100.0 - state.current_soh) * 0.0012
    )
    voltage_sag_v = internal_resistance_ohm * rms_current_a
    coulombic_efficiency = _clip(0.999 - mission_efc * 0.0015, 0.97, 0.9995)
    core = {
        "cumulative_efc": cumulative_efc,
        "cumulative_flight_hours": cumulative_flight_hours,
        "delta_soc_pct": abs(delta_soc_pct),
        "peak_c_rate": peak_c_rate,
        "rms_current_a": rms_current_a,
        "rms_c_rate": rms_c_rate,
        "mission_efc": mission_efc,
        "dod_pct": abs(delta_soc_pct),
        "max_temp_c": max_temp_c,
        "avg_temp_c": avg_temp_c,
        "time_above_40c_min": time_above_40c_min,
        "arrhenius_temp": _arrhenius_feature(avg_temp_c),
        "temp_rise_proxy_c": temp_rise_proxy_c,
        "idle_hours": idle_hours,
        "soc_weighted_idle_hours": soc_weighted_idle_hours,
        "charge_to_flight_delay_hours": charge_delay_hr if event_type == "mission" else 0.0,
        "self_discharge_rate_pct_per_hr": float(medians.get("self_discharge_rate_pct_per_hr", 0.12)),
        "internal_resistance_ohm": internal_resistance_ohm,
        "voltage_sag_v": voltage_sag_v,
        "voltage_sag_eff_wh_per_v": float(medians.get("voltage_sag_eff_wh_per_v", 477.7)),
        "coulombic_efficiency": coulombic_efficiency,
        "event_type_mission": 1.0 if event_type == "mission" else 0.0,
    }

    row: dict[str, float] = {}
    for name in feature_names:
        if name in core:
            row[name] = float(core[name])
            continue
        if name.endswith("_lag1"):
            base_name = name[:-5]
            row[name] = float(state.previous_core.get(base_name, core.get(base_name, medians.get(name, 0.0))))
            continue
        if name.endswith("_diff"):
            base_name = name[:-5]
            previous = float(state.previous_core.get(base_name, core.get(base_name, medians.get(name, 0.0))))
            current = float(core.get(base_name, medians.get(base_name, previous)))
            row[name] = current - previous
            continue
        if name == "ir_x_peak_c_rate":
            row[name] = core["internal_resistance_ohm"] * core["peak_c_rate"]
            continue
        if name == "temp_rise_x_rms_c_rate":
            row[name] = core["temp_rise_proxy_c"] * core["rms_c_rate"]
            continue
        if name == "arrhenius_x_time_above":
            row[name] = core["arrhenius_temp"] * core["time_above_40c_min"]
            continue
        if name == "idle_x_soc_weighted":
            row[name] = core["idle_hours"] * core["soc_weighted_idle_hours"]
            continue
        if name == "efc_x_dod":
            row[name] = core["cumulative_efc"] * core["dod_pct"]
            continue
        row[name] = float(medians.get(name, 0.0))

    vector = np.array([float(row[name]) for name in feature_names], dtype=float)
    return vector, core, idle_hours


def _simulate_event(
    model_payload: dict[str, Any],
    state: SimulationState,
    profile: PlaneProfile,
    *,
    event_type: str,
    delta_soc_pct: float,
    duration_hr: float,
    event_time: datetime,
    avg_temp_c: float,
    max_temp_c: float,
    charge_delay_hr: float,
) -> dict[str, Any]:
    vector, core, idle_hours = _build_feature_row(
        model_payload["bundle"],
        state,
        profile,
        event_type=event_type,
        delta_soc_pct=delta_soc_pct,
        duration_hr=duration_hr,
        event_time=event_time,
        avg_temp_c=avg_temp_c,
        max_temp_c=max_temp_c,
        charge_delay_hr=charge_delay_hr,
    )
    scaler = model_payload["scaler"]
    model = model_payload["model"]
    feature_names: list[str] = model_payload["bundle"]["feature_names"]
    frame = pd.DataFrame([vector], columns=feature_names)
    raw_pred = float(model.predict(scaler.transform(frame))[0])
    baseline_wear = max(0.004, abs(delta_soc_pct) / 100.0 * (0.055 if event_type == "mission" else 0.032))
    pred_delta = -max(abs(raw_pred), baseline_wear)
    pred_delta = _clip(pred_delta, -0.18, -0.003)

    soc_before = state.current_soc
    soh_before = state.current_soh
    if event_type == "charge":
        state.current_soc = _clip(state.current_soc + abs(delta_soc_pct), 0.0, 100.0)
    else:
        state.current_soc = _clip(state.current_soc - abs(delta_soc_pct), 0.0, 100.0)
    state.current_soh = max(0.0, state.current_soh + pred_delta)
    state.cumulative_efc = core["cumulative_efc"]
    state.cumulative_flight_hours = core["cumulative_flight_hours"]
    state.current_time = event_time + timedelta(hours=duration_hr)
    state.previous_core = {
        key: float(value)
        for key, value in core.items()
        if key != "event_type_mission"
    }

    return {
        "eventType": event_type,
        "predDeltaSoh": pred_delta,
        "idleHours": idle_hours,
        "socStartPct": soc_before,
        "socEndPct": state.current_soc,
        "sohStartPct": soh_before,
        "sohEndPct": state.current_soh,
    }


def _forecast_points(profile: PlaneProfile, model_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str, int, int]:
    state = _base_state(profile)
    now = datetime.now(timezone.utc)
    points = [{"date": profile.latest_time.date().isoformat(), "soh": float(profile.current_soh)}]
    if profile.latest_time.date() != now.date():
        points.append({"date": now.date().isoformat(), "soh": float(profile.current_soh)})

    flights = 0
    gap_hours_between_flights = max(6.0, 24.0 / max(profile.flights_per_day_recent, 0.2))
    replacement_date = now.date().isoformat()

    for _ in range(320):
        next_flight_time = state.current_time + timedelta(hours=gap_hours_between_flights)
        target_soc = _clip(profile.charge_target_soc_pct, 72.0, 98.0)
        if state.current_soc < target_soc - 0.5:
            charge_needed = target_soc - state.current_soc
            charge_duration_hr = charge_needed / max(profile.charge_rate_pct_per_hr, 5.0)
            charge_end = next_flight_time - timedelta(hours=profile.charge_to_flight_delay_hr)
            if charge_end <= state.current_time:
                charge_end = state.current_time + timedelta(hours=1.0)
                next_flight_time = charge_end + timedelta(hours=profile.charge_to_flight_delay_hr)
            charge_start = charge_end - timedelta(hours=charge_duration_hr)
            if charge_start < state.current_time:
                charge_start = state.current_time
                charge_end = charge_start + timedelta(hours=charge_duration_hr)
                next_flight_time = charge_end + timedelta(hours=profile.charge_to_flight_delay_hr)
            _simulate_event(
                model_payload,
                state,
                profile,
                event_type="charge",
                delta_soc_pct=charge_needed,
                duration_hr=charge_duration_hr,
                event_time=charge_start,
                avg_temp_c=profile.avg_temp_c + 1.0,
                max_temp_c=profile.max_temp_c + 1.0,
                charge_delay_hr=0.0,
            )

        _simulate_event(
            model_payload,
            state,
            profile,
            event_type="mission",
            delta_soc_pct=profile.mission_soc_span_pct,
            duration_hr=profile.mission_duration_hr,
            event_time=next_flight_time,
            avg_temp_c=profile.avg_temp_c,
            max_temp_c=profile.max_temp_c,
            charge_delay_hr=profile.charge_to_flight_delay_hr,
        )
        flights += 1
        flight_date = next_flight_time.date().isoformat()
        if points[-1]["date"] == flight_date:
            points[-1]["soh"] = float(state.current_soh)
        else:
            points.append({"date": flight_date, "soh": float(state.current_soh)})

        if state.current_soh <= REPLACEMENT_THRESHOLD_SOH:
            replacement_date = flight_date
            break
        if next_flight_time > now + timedelta(days=730):
            replacement_date = flight_date
            break

    rul_days = max((datetime.fromisoformat(f"{replacement_date}T00:00:00+00:00").date() - now.date()).days, 0)
    return points, replacement_date, rul_days, flights


def _month_dates(month_value: str) -> list[str]:
    year, month = [int(part) for part in month_value.split("-")]
    return [
        datetime(year, month, day, tzinfo=timezone.utc).date().isoformat()
        for day in range(1, monthrange(year, month)[1] + 1)
    ]


def _daterange(start_date: str, end_date: str) -> list[str]:
    start = datetime.fromisoformat(f"{start_date}T00:00:00+00:00").date()
    end = datetime.fromisoformat(f"{end_date}T00:00:00+00:00").date()
    dates: list[str] = []
    cursor = start
    while cursor <= end:
        dates.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return dates


def _time_on_date(date_iso: str, hhmm: str) -> datetime:
    hour, minute = [int(part) for part in hhmm.split(":")]
    return datetime.fromisoformat(f"{date_iso}T00:00:00+00:00") + timedelta(
        hours=hour,
        minutes=minute,
    )


def _resolve_sortie_count(ops_demand: dict[str, Any] | None) -> int:
    requested = int(_safe_float((ops_demand or {}).get("sortiesPerDay"), 1))
    return max(1, requested)


def _resolve_duration_hr(
    profile: PlaneProfile,
    mission_template: dict[str, Any],
    sortie_count: int,
) -> float:
    duration_min = _safe_float(
        mission_template.get("durationMin"),
        profile.mission_duration_hr * 60.0,
    )
    duration_hr = _clip(duration_min / 60.0, 0.2, 4.0)
    return duration_hr * max(1, sortie_count)


def _resolve_mission_soc_span_pct(
    profile: PlaneProfile,
    mission_template: dict[str, Any],
    sortie_count: int,
) -> float:
    explicit_soc = mission_template.get("socSpanPct")
    if explicit_soc is not None:
        return _clip(
            _safe_float(explicit_soc, profile.mission_soc_span_pct) * max(1, sortie_count),
            8.0,
            92.0,
        )

    duration_min = _safe_float(
        mission_template.get("durationMin"),
        profile.mission_duration_hr * 60.0,
    )
    baseline_duration_min = max(profile.mission_duration_hr * 60.0, 20.0)
    duration_scaled = profile.mission_soc_span_pct * duration_min / baseline_duration_min

    route_distance_km = mission_template.get("routeDistanceKm")
    if route_distance_km is not None:
        route_scaled = _safe_float(route_distance_km, duration_min * 2.1) / 3.0
        base_span = max(duration_scaled, route_scaled)
    else:
        base_span = duration_scaled

    base_span = _clip(base_span, 8.0, 68.0)
    return _clip(base_span * max(1, sortie_count), 8.0, 92.0)


def _build_scenario_day(
    plane_id: str,
    date_iso: str,
    profile: PlaneProfile,
    model_payload: dict[str, Any],
    mission_template: dict[str, Any],
    charge_policy: dict[str, Any],
    ops_demand: dict[str, Any] | None,
    weather_day: dict[str, Any] | None,
) -> dict[str, Any]:
    sortie_count = _resolve_sortie_count(ops_demand)
    mission_duration_hr = _resolve_duration_hr(profile, mission_template, sortie_count)
    mission_soc_span_pct = _resolve_mission_soc_span_pct(profile, mission_template, sortie_count)
    reserve_soc_pct = _clip(_safe_float(mission_template.get("reserveSocPct"), profile.reserve_soc_pct), 10.0, 60.0)
    target_soc_cap = _clip(
        _safe_float(charge_policy.get("targetSocCapPct"), profile.charge_target_soc_pct),
        50.0,
        100.0,
    )
    latest_charge_finish_lead_hours = _clip(
        _safe_float(charge_policy.get("latestChargeFinishLeadHours"), profile.charge_to_flight_delay_hr),
        0.0,
        24.0,
    )
    ambient_temp_min_c = _clip(
        _safe_float((weather_day or {}).get("tempMinC"), profile.avg_temp_c - 4.0),
        -25.0,
        45.0,
    )
    ambient_temp_max_c = _clip(
        _safe_float((weather_day or {}).get("tempMaxC"), profile.max_temp_c),
        ambient_temp_min_c,
        55.0,
    )
    ambient_temp_avg_c = (ambient_temp_min_c + ambient_temp_max_c) / 2.0
    wind_kph = max(0.0, _safe_float((weather_day or {}).get("windKph"), 12.0))
    precip_mm = max(0.0, _safe_float((weather_day or {}).get("precipMm"), 0.0))
    departure_window_start = str(mission_template.get("departureWindowStart", "08:00"))
    departure_window_end = str(mission_template.get("departureWindowEnd", "10:00"))

    window_start = _time_on_date(date_iso, departure_window_start)
    window_end = _time_on_date(date_iso, departure_window_end)
    flight_time = max(window_end, window_start + timedelta(minutes=15))
    state = _base_state(profile)
    target_soc = min(target_soc_cap, 98.0)
    thermal_charge_rate_factor = _clip(
        1.0
        - max(0.0, 8.0 - ambient_temp_avg_c) * 0.018
        - max(0.0, ambient_temp_avg_c - 30.0) * 0.015,
        0.68,
        1.02,
    )
    effective_charge_rate_pct_per_hr = max(
        profile.charge_rate_pct_per_hr * thermal_charge_rate_factor,
        5.0,
    )
    weather_adjusted_soc_span_pct = mission_soc_span_pct * (
        1.0 + max(0.0, wind_kph - 18.0) * 0.0035 + precip_mm * 0.004
    )
    weather_adjusted_soc_span_pct = _clip(weather_adjusted_soc_span_pct, 8.0, 95.0)
    minimum_required_target = weather_adjusted_soc_span_pct + reserve_soc_pct + 2.0
    preflight_soc_pct = max(state.current_soc, target_soc)
    charge_needed = max(0.0, target_soc - state.current_soc)
    charge_duration_hr = charge_needed / effective_charge_rate_pct_per_hr
    charge_end = flight_time - timedelta(hours=latest_charge_finish_lead_hours)
    charge_start = charge_end - timedelta(hours=charge_duration_hr)
    preflight_idle_hours = max((flight_time - profile.initial_time).total_seconds() / 3600.0, 0.0)

    today_iso = datetime.now(timezone.utc).date().isoformat()
    date_is_past = date_iso < today_iso
    reserve_target_feasible = preflight_soc_pct >= minimum_required_target
    charge_window_feasible = charge_needed <= 0.5 or (
        charge_end > profile.initial_time and charge_start >= profile.initial_time
    )

    charge_delta = 0.0
    charge_window_start_iso: str | None = None
    charge_window_end_iso: str | None = None
    if charge_needed > 0.5:
        charge_window_start_iso = charge_start.replace(microsecond=0).isoformat()
        charge_window_end_iso = charge_end.replace(microsecond=0).isoformat()

    if charge_needed > 0.5 and charge_window_feasible:
        charge_event = _simulate_event(
            model_payload,
            state,
            profile,
            event_type="charge",
            delta_soc_pct=charge_needed,
            duration_hr=charge_duration_hr,
            event_time=charge_start,
            avg_temp_c=ambient_temp_avg_c + 1.0,
            max_temp_c=min(ambient_temp_max_c + 2.0, 58.0),
            charge_delay_hr=0.0,
        )
        charge_delta = float(charge_event["predDeltaSoh"])

    # If the aircraft is already full or nearly full, waiting until a later day
    # should look worse than flying sooner, especially in warmer conditions.
    high_soc_dwell_hours = (
        preflight_idle_hours
        if charge_needed <= 0.5
        else max(latest_charge_finish_lead_hours, 0.0)
    )
    dwell_soc_pct = state.current_soc if charge_needed <= 0.5 else target_soc
    dwell_soc_factor = max(dwell_soc_pct - 80.0, 0.0) / 20.0
    dwell_temp_factor = (
        1.0
        + max(0.0, ambient_temp_avg_c - 20.0) * 0.04
        + max(0.0, ambient_temp_max_c - 30.0) * 0.06
    )
    calendar_dwell_delta = -min(
        high_soc_dwell_hours * dwell_soc_factor * dwell_temp_factor * 0.00022,
        0.03,
    )
    flight_event = _simulate_event(
        model_payload,
        state,
        profile,
        event_type="mission",
        delta_soc_pct=weather_adjusted_soc_span_pct,
        duration_hr=mission_duration_hr,
        event_time=max(flight_time, state.current_time + timedelta(hours=0.25)),
        avg_temp_c=ambient_temp_avg_c,
        max_temp_c=ambient_temp_max_c,
        charge_delay_hr=latest_charge_finish_lead_hours,
    )

    total_delta = charge_delta + calendar_dwell_delta + float(flight_event["predDeltaSoh"])
    reserve_margin_pct = state.current_soc - reserve_soc_pct
    raw_penalty = (
        abs(total_delta) * 220.0
        + max(0.0, 0.0 - reserve_margin_pct) * 8.0
        + max(0.0, target_soc - 85.0) * 1.35
        + max(0.0, charge_duration_hr - 1.5) * 8.0
        + max(0.0, wind_kph - 22.0) * 0.8
        + precip_mm * 2.2
        + abs(calendar_dwell_delta) * 320.0
        + (25.0 if not charge_window_feasible and charge_needed > 0.5 else 0.0)
        + (40.0 if not reserve_target_feasible else 0.0)
        + (30.0 if date_is_past else 0.0)
    )
    model_stress_score = _clip(100.0 - raw_penalty, 5.0, 100.0)
    charging_penalty = (
        max(0.0, target_soc - 85.0) * 1.8
        + max(0.0, charge_duration_hr - 1.5) * 18.0
        + max(0.0, 8.0 - ambient_temp_avg_c) * 0.8
        + max(0.0, ambient_temp_avg_c - 30.0) * 0.7
        + abs(calendar_dwell_delta) * 260.0
        + (30.0 if not charge_window_feasible and charge_needed > 0.5 else 0.0)
        + (35.0 if not reserve_target_feasible else 0.0)
    )
    charging_score = _clip(100.0 - charging_penalty, 5.0, 100.0)

    if date_is_past:
        summary = "Date is in the past for advisory planning."
    elif not reserve_target_feasible:
        summary = "Target SOC cap is too low to preserve mission reserve."
    elif not charge_window_feasible and charge_needed > 0.5:
        summary = "Charge window cannot satisfy this mission before departure."
    elif reserve_margin_pct < 0:
        summary = "Reserve SOC would be violated for this mission profile."
    elif calendar_dwell_delta <= -0.012:
        summary = "Waiting at high SOC before departure adds measurable storage wear."
    elif total_delta <= -0.18:
        summary = "Model projects a heavier degradation hit for this operating window."
    else:
        summary = "Model projects a manageable degradation profile for this mission."

    return {
        "planeId": plane_id,
        "date": date_iso,
        "sortieCount": int(sortie_count),
        "durationMin": float(mission_duration_hr * 60.0),
        "missionSocSpanPct": float(weather_adjusted_soc_span_pct),
        "reserveSocPct": float(reserve_soc_pct),
        "targetSoc": float(target_soc),
        "chargeDurationHr": float(max(charge_duration_hr, 0.0)),
        "chargeWindowStart": charge_window_start_iso,
        "chargeWindowEnd": charge_window_end_iso,
        "expectedDeltaSoh": float(total_delta),
        "postFlightSocPct": float(state.current_soc),
        "reserveMarginPct": float(reserve_margin_pct),
        "modelStressScore": float(model_stress_score),
        "chargingScore": float(charging_score),
        "feasible": bool(
            (not date_is_past)
            and reserve_target_feasible
            and charge_window_feasible
            and reserve_margin_pct >= 0.0
        ),
        "summary": summary,
    }


def _build_scenario_planner_payload(request_payload: dict[str, Any]) -> dict[str, Any]:
    plane_id = str(request_payload["planeId"])
    start_date = str(request_payload["startDate"])
    end_date = str(request_payload["endDate"])
    mission_template = dict(request_payload.get("missionTemplate", {}))
    charge_policy = dict(request_payload.get("chargePolicy", {}))
    ops_demand = request_payload.get("opsDemand")
    weather_days = request_payload.get("weatherDays")
    weather_by_date = {
        str(item.get("date")): item
        for item in (weather_days if isinstance(weather_days, list) else [])
        if isinstance(item, dict) and item.get("date") is not None
    }

    profile = _derive_plane_profile(plane_id)
    model_payload = _load_feature_model()
    model_days = [
        _build_scenario_day(
            plane_id=plane_id,
            date_iso=date_iso,
            profile=profile,
            model_payload=model_payload,
            mission_template=mission_template,
            charge_policy=charge_policy,
            ops_demand=ops_demand if isinstance(ops_demand, dict) else None,
            weather_day=weather_by_date.get(date_iso),
        )
        for date_iso in _daterange(start_date, end_date)
    ]

    return {
        "planeId": plane_id,
        "generatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "startDate": start_date,
        "endDate": end_date,
        "assumptions": [
            "This recommendation engine is advisory and battery-life-first, not a dispatch optimizer.",
            "Daily forecast temperatures are folded into the simulated charge and mission wear estimate.",
            "Waiting longer at high SOC before departure adds an explicit storage-wear penalty.",
            "Flights per day are modeled as repeated missions on the same aircraft and day.",
        ],
        "modelDays": model_days,
    }


def _build_month_model_payload(
    plane_id: str,
    month_value: str,
    profile: PlaneProfile,
    model_payload: dict[str, Any],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    raw_days: list[dict[str, Any]] = []
    target_soc = _clip(profile.charge_target_soc_pct, 72.0, 98.0)

    for date_iso in _month_dates(month_value):
        flight_time = datetime.fromisoformat(f"{date_iso}T14:00:00+00:00")
        day_offset = (flight_time.date() - now.date()).days
        state = _base_state(profile)
        charge_needed = max(0.0, target_soc - state.current_soc)
        charge_duration_hr = charge_needed / max(profile.charge_rate_pct_per_hr, 5.0)
        charge_end = flight_time - timedelta(hours=profile.charge_to_flight_delay_hr)
        charge_start = charge_end - timedelta(hours=charge_duration_hr)

        charge_delta = 0.0
        if charge_needed > 0.5:
            charge_event = _simulate_event(
                model_payload,
                state,
                profile,
                event_type="charge",
                delta_soc_pct=charge_needed,
                duration_hr=charge_duration_hr,
                event_time=max(charge_start, state.current_time),
                avg_temp_c=profile.avg_temp_c + 1.0,
                max_temp_c=profile.max_temp_c + 1.0,
                charge_delay_hr=0.0,
            )
            charge_delta = float(charge_event["predDeltaSoh"])

        flight_event = _simulate_event(
            model_payload,
            state,
            profile,
            event_type="mission",
            delta_soc_pct=profile.mission_soc_span_pct,
            duration_hr=profile.mission_duration_hr,
            event_time=max(flight_time, state.current_time + timedelta(hours=0.25)),
            avg_temp_c=profile.avg_temp_c,
            max_temp_c=profile.max_temp_c,
            charge_delay_hr=profile.charge_to_flight_delay_hr,
        )

        total_delta = charge_delta + float(flight_event["predDeltaSoh"])
        reserve_margin_pct = state.current_soc - profile.reserve_soc_pct
        raw_penalty = (
            abs(total_delta) * 180.0
            + max(0.0, 4.0 - reserve_margin_pct) * 5.0
            + max(0.0, target_soc - 88.0) * 0.9
            + max(0.0, charge_duration_hr - 1.5) * 8.0
            + max(0.0, -day_offset) * 3.0
        )
        charging_penalty = (
            max(0.0, target_soc - 85.0) * 1.8
            + max(0.0, charge_duration_hr - 1.5) * 18.0
            + (28.0 if reserve_margin_pct < 0 else 0.0)
        )
        raw_days.append(
            {
                "date": date_iso,
                "expectedDeltaSoh": float(total_delta),
                "postFlightSocPct": float(state.current_soc),
                "reserveMarginPct": float(reserve_margin_pct),
                "chargeWindowStart": max(charge_start, profile.initial_time).replace(microsecond=0).isoformat(),
                "chargeWindowEnd": max(charge_end, profile.initial_time).replace(microsecond=0).isoformat(),
                "chargeDurationHr": float(max(charge_duration_hr, 0.0)),
                "targetSoc": float(target_soc),
                "rawPenalty": float(raw_penalty),
                "chargingPenalty": float(charging_penalty),
                "dayOffset": int(day_offset),
            }
        )

    raw_values = [item["rawPenalty"] for item in raw_days if item["dayOffset"] >= 0]
    penalty_min = min(raw_values) if raw_values else 0.0
    penalty_span = (max(raw_values) - penalty_min) if len(raw_values) > 1 else 0.0

    model_days = []
    for item in raw_days:
        if penalty_span > 1e-6:
            scaled_penalty = (item["rawPenalty"] - penalty_min) / penalty_span
            stress_score = 100.0 - scaled_penalty * 42.0
        else:
            stress_score = 100.0 - item["rawPenalty"] * 0.9
        stress_score = _clip(stress_score, 28.0, 100.0)
        charging_score = _clip(100.0 - item["chargingPenalty"], 25.0, 100.0)

        if item["reserveMarginPct"] < 0:
            summary = "Reserve SOC would be violated for this mission profile."
        elif item["expectedDeltaSoh"] <= -0.18:
            summary = "Model projects a heavier degradation hit for this operating window."
        elif item["chargeDurationHr"] > 2.5:
            summary = "Longer charging dwell is needed before departure."
        else:
            summary = "Model projects a manageable degradation profile for this mission."

        model_days.append(
            {
                "date": item["date"],
                "modelStressScore": float(stress_score),
                "chargingScore": float(charging_score),
                "expectedDeltaSoh": float(item["expectedDeltaSoh"]),
                "postFlightSocPct": float(item["postFlightSocPct"]),
                "reserveMarginPct": float(item["reserveMarginPct"]),
                "targetSoc": float(item["targetSoc"]),
                "chargeWindowStart": item["chargeWindowStart"],
                "chargeWindowEnd": item["chargeWindowEnd"],
                "summary": summary,
            }
        )

    return {
        "planeId": plane_id,
        "month": month_value,
        "generatedAt": now.replace(microsecond=0).isoformat(),
        "modelDays": model_days,
    }


def _build_prediction_payload(plane_id: str) -> dict[str, Any]:
    profile = _derive_plane_profile(plane_id)
    model_payload = _load_feature_model()
    curve, replacement_date, rul_days, rul_cycles = _forecast_points(profile, model_payload)
    confidence = _clip(0.62 + profile.cadence_confidence * 0.32, 0.55, 0.9)
    return {
        "prediction": {
            "planeId": plane_id,
            "forecast": {
                "replacementDatePred": replacement_date,
                "rulDaysPred": int(rul_days),
                "rulCyclesPred": int(rul_cycles),
                "confidence": float(confidence),
            },
            "forecastCurve": curve,
            "sohTargetBlend": float(profile.current_soh),
            "sohProxyPoh": float(profile.current_soh),
            "sohObservedNorm": float(profile.current_soh),
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit model-backed forecast and recommendation payloads.")
    parser.add_argument("--plane-id", required=True, help="Plane id to load")
    parser.add_argument("--month", help="Emit model-backed recommendation payload for YYYY-MM")
    parser.add_argument(
        "--planner-json",
        help="Emit scenario planner payload from a JSON request body",
    )
    args = parser.parse_args()

    if args.planner_json:
        request_payload = json.loads(str(args.planner_json))
        request_payload["planeId"] = str(args.plane_id)
        payload = _build_scenario_planner_payload(request_payload)
    elif args.month:
        profile = _derive_plane_profile(str(args.plane_id))
        model_payload = _load_feature_model()
        payload = _build_month_model_payload(str(args.plane_id), str(args.month), profile, model_payload)
    else:
        payload = _build_prediction_payload(str(args.plane_id))
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
