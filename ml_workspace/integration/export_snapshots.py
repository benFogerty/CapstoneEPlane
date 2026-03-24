from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SnapshotConfig:
    plane_ids: list[str]
    output_dir: Path
    latent_root: Path
    forecast_root: Path
    eol_soh: float = 0.0
    min_trend_points: int = 10
    rul_fallback_per_day: float = -0.01
    rul_fallback_per_flight: float = -0.02
    rul_horizons: tuple[int, ...] = (1, 5, 10, 15, 20)
    model_recent_rows: int = 40
    cadence_window_days: int = 90
    default_flights_per_day: float = 0.2


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return parsed


def _parse_horizons(raw: str) -> tuple[int, ...]:
    values = []
    for piece in str(raw).split(","):
        token = piece.strip()
        if not token:
            continue
        try:
            parsed = int(token)
        except ValueError:
            continue
        if parsed > 0:
            values.append(parsed)
    if not values:
        return (1, 5, 10, 15, 20)
    return tuple(dict.fromkeys(values))


def _parse_args() -> SnapshotConfig:
    parser = argparse.ArgumentParser(description="Export frontend snapshot JSONs from ML outputs.")
    parser.add_argument("--plane-ids", default="166", help="Comma-separated plane ids")
    parser.add_argument(
        "--output-dir",
        default="frontend/public/snapshots",
        help="Frontend snapshots output directory",
    )
    parser.add_argument(
        "--latent-root",
        default="ml_workspace/latent_soh/output",
        help="Latent SOH output root",
    )
    parser.add_argument(
        "--forecast-root",
        default="ml_workspace/soh_forecast/output/multihorizon_runner_plane_166",
        help="Root directory containing best_models_by_horizon.json and horizon prediction CSVs",
    )
    parser.add_argument("--eol-soh", type=float, default=0.0, help="SOH threshold for RUL/replacement")
    parser.add_argument("--min-trend-points", type=int, default=10, help="Minimum points for trends")
    parser.add_argument(
        "--rul-horizons",
        default="1,5,10,15,20",
        help="Preferred forecast horizons for RUL estimation",
    )
    parser.add_argument(
        "--model-recent-rows",
        type=int,
        default=40,
        help="Recent rows per horizon/model used for per-flight degradation estimate",
    )
    parser.add_argument(
        "--cadence-window-days",
        type=int,
        default=90,
        help="Window used to estimate flights/day for replacement date",
    )
    parser.add_argument(
        "--default-flights-per-day",
        type=float,
        default=0.2,
        help="Fallback flights/day when cadence is sparse",
    )
    args = parser.parse_args()
    return SnapshotConfig(
        plane_ids=[pid.strip() for pid in str(args.plane_ids).split(",") if pid.strip()],
        output_dir=Path(args.output_dir),
        latent_root=Path(args.latent_root),
        forecast_root=Path(args.forecast_root),
        eol_soh=float(args.eol_soh),
        min_trend_points=int(args.min_trend_points),
        rul_horizons=_parse_horizons(args.rul_horizons),
        model_recent_rows=int(args.model_recent_rows),
        cadence_window_days=int(args.cadence_window_days),
        default_flights_per_day=float(args.default_flights_per_day),
    )


def _load_latent_table(latent_root: Path, plane_id: str) -> pd.DataFrame:
    table_path = latent_root / f"plane_{plane_id}" / "latent_soh_event_table.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing latent SOH table: {table_path}")
    df = pd.read_csv(table_path, parse_dates=["event_datetime"])
    df["plane_id"] = df["plane_id"].astype(str)
    return df


def _latest_row(df: pd.DataFrame) -> pd.Series:
    ordered = df.sort_values("event_datetime")
    return ordered.iloc[-1]


def _risk_band(soh: float) -> str:
    if soh >= 80:
        return "low"
    if soh >= 40:
        return "medium"
    if soh >= 20:
        return "watch"
    return "critical"


def _health_label(soh: float) -> tuple[str, str]:
    if soh >= 80:
        return "healthy", "Battery condition is healthy with low degradation signals."
    if soh >= 40:
        return "medium", "Battery condition is in the medium band; monitor stress and charging behavior."
    if soh >= 20:
        return "watch", "Battery condition is in the watch band; reduce stress and plan maintenance."
    return "critical", "Battery condition is degraded; plan maintenance or replacement."


def _trend_points(
    df: pd.DataFrame,
    soh_col: str,
    source: str = "blend",
) -> list[dict[str, object]]:
    working = df.copy()
    working["date"] = working["event_datetime"].dt.date
    if soh_col not in working.columns:
        raise KeyError(f"Missing SOH column for trend points: {soh_col}")
    grouped = working.groupby("date", as_index=False)[soh_col].median()
    points = []
    for _, row in grouped.iterrows():
        points.append(
            {
                "date": row["date"].isoformat(),
                "soh": float(row[soh_col]),
                "source": source,
            }
        )
    return points


def _window_delta(df: pd.DataFrame, days: int) -> float:
    if df.empty:
        return 0.0
    cutoff = df["event_datetime"].max() - pd.Timedelta(days=days)
    window = df.loc[df["event_datetime"] >= cutoff]
    if len(window) < 2:
        return 0.0
    return float(window["latent_soh_filter_pct"].iloc[-1] - window["latent_soh_filter_pct"].iloc[0])


def _is_charge_event(value: object) -> bool:
    return "charge" in str(value).strip().lower()


def _latest_charge_soc(df: pd.DataFrame, fallback_row: pd.Series) -> float:
    charge_df = df.loc[df["event_type"].map(_is_charge_event)].sort_values("event_datetime")
    if not charge_df.empty:
        charge_row = charge_df.iloc[-1]
        for col in ("soc_max_pct", "soc_mean_pct", "soc_min_pct"):
            if col in charge_row.index:
                soc_val = _safe_float(charge_row.get(col, np.nan), np.nan)
                if np.isfinite(soc_val):
                    return float(np.clip(soc_val, 0.0, 100.0))

    for col in ("soc_mean_pct", "soc_max_pct", "soc_min_pct"):
        if col in fallback_row.index:
            soc_val = _safe_float(fallback_row.get(col, np.nan), np.nan)
            if np.isfinite(soc_val):
                return float(np.clip(soc_val, 0.0, 100.0))
    return 0.0


def _recent_flight_decay_per_flight(df: pd.DataFrame, config: SnapshotConfig) -> float:
    flight_df = df.loc[df["event_type"].eq("flight")].sort_values("event_datetime")
    if flight_df.empty:
        return config.rul_fallback_per_flight
    flight_df["next_soh"] = flight_df["latent_soh_filter_pct"].shift(-1)
    flight_df["delta"] = flight_df["next_soh"] - flight_df["latent_soh_filter_pct"]
    recent = flight_df.dropna(subset=["delta"]).tail(30)
    if recent.empty:
        return config.rul_fallback_per_flight
    deltas = pd.to_numeric(recent["delta"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if deltas.empty:
        return config.rul_fallback_per_flight
    negative = deltas.loc[deltas < 0]
    candidate = float(np.nanmedian(negative if len(negative) >= 3 else deltas))
    if candidate >= 0:
        return config.rul_fallback_per_flight
    return candidate


def _load_best_models(config: SnapshotConfig) -> dict[str, dict[str, object]]:
    best_path = config.forecast_root / "best_models_by_horizon.json"
    if not best_path.exists():
        return {}
    try:
        payload = json.loads(best_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def _estimate_model_per_flight(
    plane_id: str,
    config: SnapshotConfig,
) -> tuple[float | None, float]:
    best_models = _load_best_models(config)
    if not best_models:
        return None, 0.55

    estimates: list[tuple[float, float]] = []
    for horizon in config.rul_horizons:
        target_name = f"latent_flight_{horizon}"
        best = best_models.get(target_name)
        if not best:
            continue
        model_col = str(best.get("model", "")).strip()
        if not model_col:
            continue

        pred_path = config.forecast_root / target_name / f"{target_name}_predictions.csv"
        if not pred_path.exists():
            continue

        pred_df = pd.read_csv(pred_path, parse_dates=["event_datetime"])
        if model_col not in pred_df.columns:
            continue
        pred_df["plane_id"] = pred_df["plane_id"].astype(str)
        plane_df = pred_df.loc[
            pred_df["plane_id"].eq(str(plane_id))
            & pred_df[model_col].notna()
            & pred_df["latent_soh_filter_pct"].notna()
        ].copy()
        if plane_df.empty:
            continue

        selected = plane_df
        for split_name in ("holdout", "final_test", "train_dev", "test", "valid", "train"):
            split_df = plane_df.loc[plane_df["split"].eq(split_name)].copy()
            if len(split_df) >= 5:
                selected = split_df
                break

        selected = selected.sort_values("event_datetime").tail(max(5, config.model_recent_rows))
        delta = pd.to_numeric(selected[model_col], errors="coerce") - pd.to_numeric(
            selected["latent_soh_filter_pct"], errors="coerce"
        )
        per_flight = (delta / float(horizon)).replace([np.inf, -np.inf], np.nan).dropna()
        if per_flight.empty:
            continue

        negative = per_flight.loc[per_flight < 0]
        estimate = float(np.nanmedian(negative if len(negative) >= 3 else per_flight))
        if estimate >= 0:
            continue

        metrics = best.get("metrics", {})
        delta_mae = np.nan
        if isinstance(metrics, dict):
            delta_mae = _safe_float(
                metrics.get(
                    "final_test_delta_mae",
                    metrics.get("backtest_mean_delta_mae", metrics.get("delta_mae", np.nan)),
                ),
                np.nan,
            )
        if np.isfinite(delta_mae) and delta_mae > 0:
            weight = float(1.0 / delta_mae)
        else:
            weight = 1.0
        estimates.append((estimate, weight))

    if not estimates:
        return None, 0.55

    values = np.array([item[0] for item in estimates], dtype=float)
    weights = np.array([item[1] for item in estimates], dtype=float)
    model_per_flight = float(np.average(values, weights=weights))
    confidence = float(min(0.92, 0.64 + 0.06 * len(estimates)))
    return model_per_flight, confidence


def _estimate_flights_per_day(df: pd.DataFrame, config: SnapshotConfig) -> tuple[float, float]:
    flights = df.loc[df["event_type"].eq("flight")].sort_values("event_datetime")
    if flights.empty:
        return config.default_flights_per_day, 0.55

    end_ts = flights["event_datetime"].max()
    start_cutoff = end_ts - pd.Timedelta(days=config.cadence_window_days)
    window = flights.loc[flights["event_datetime"] >= start_cutoff].copy()
    if len(window) < 5:
        window = flights.tail(10).copy()
    if len(window) < 2:
        return config.default_flights_per_day, 0.55

    span_days = (window["event_datetime"].iloc[-1] - window["event_datetime"].iloc[0]).total_seconds() / 86_400.0
    span_days = max(1.0, span_days)
    flights_per_day = float(len(window) / span_days)
    if not np.isfinite(flights_per_day) or flights_per_day <= 0:
        return config.default_flights_per_day, 0.55

    flights_per_day = float(np.clip(flights_per_day, 0.02, 10.0))
    confidence = float(min(0.9, 0.56 + 0.025 * len(window)))
    return flights_per_day, confidence


def _estimate_rul(
    df: pd.DataFrame,
    plane_id: str,
    current_soh: float,
    eol_soh: float,
    config: SnapshotConfig,
) -> tuple[int, int, float]:
    if current_soh <= eol_soh:
        return 0, 0, 0.8

    model_per_flight, model_conf = _estimate_model_per_flight(plane_id=plane_id, config=config)
    if model_per_flight is None or model_per_flight >= 0:
        model_per_flight = _recent_flight_decay_per_flight(df, config)
        model_conf = 0.62

    if model_per_flight >= 0:
        model_per_flight = config.rul_fallback_per_flight
        model_conf = min(model_conf, 0.58)

    rul_cycles = int(max(0.0, (current_soh - eol_soh) / abs(model_per_flight)))
    flights_per_day, cadence_conf = _estimate_flights_per_day(df, config)
    rul_days = int(max(0.0, np.ceil(rul_cycles / max(flights_per_day, 1e-6))))
    confidence = float(np.clip(0.5 * model_conf + 0.5 * cadence_conf, 0.55, 0.95))
    return rul_days, rul_cycles, confidence


def _build_health_snapshot(df: pd.DataFrame, plane_id: str, config: SnapshotConfig) -> dict[str, object]:
    latest = _latest_row(df)
    current_soh = _safe_float(latest.get("latent_soh_filter_pct", np.nan), 0.0)
    observed_soh = _safe_float(latest.get("observed_soh_pct", np.nan), current_soh)
    current_charge_soc = _latest_charge_soc(df, latest)
    trend_30 = _safe_float(_window_delta(df, 30), 0.0)
    trend_90 = _safe_float(_window_delta(df, 90), 0.0)

    risk_band = _risk_band(current_soh)
    label, explanation = _health_label(current_soh)
    now_iso = datetime.now(timezone.utc).isoformat()

    last_flight = df.loc[df["event_type"].eq("flight")].sort_values("event_datetime").tail(1)
    if last_flight.empty:
        last_flight_payload = {
            "flightId": 0,
            "eventDate": latest["event_datetime"].date().isoformat(),
            "route": "Unknown route",
            "departureAirport": None,
            "destinationAirport": None,
            "durationMin": None,
            "eventType": "unknown",
        }
    else:
        row = last_flight.iloc[0]
        last_flight_payload = {
            "flightId": int(row.get("flight_id", 0)),
            "eventDate": row["event_datetime"].date().isoformat(),
            "route": "Unknown route",
            "departureAirport": None,
            "destinationAirport": None,
            "durationMin": _safe_float(row.get("event_duration_s", np.nan), np.nan) / 60.0
            if not pd.isna(row.get("event_duration_s"))
            else None,
            "eventType": str(row.get("event_type", "flight")),
        }

    health = {
        "planeId": plane_id,
        "updatedAt": now_iso,
        "sohCurrent": _safe_float(current_soh, 0.0),
        "sohTrend30": _safe_float(trend_30, 0.0),
        "sohTrend90": _safe_float(trend_90, 0.0),
        "riskBand": risk_band,
        "healthScore": _safe_float(current_soh, 0.0),
        "healthLabel": label,
        "healthExplanation": explanation,
        "metricsExplainabilityVersion": "v1",
        "confidence": 0.8,
        "currentChargeSoc": current_charge_soc,
        "pack": {
            "voltage": _safe_float(latest.get("voltage_mean_v", 0.0), 0.0),
            "current": _safe_float(latest.get("current_abs_mean_a", 0.0), 0.0),
            "tempAvg": _safe_float(latest.get("avg_cell_temp_mean_c", 0.0), 0.0),
            "soc": current_charge_soc,
        },
        "lastFlight": last_flight_payload,
    }

    rul_days, rul_cycles, conf = _estimate_rul(df, plane_id, current_soh, config.eol_soh, config)
    replacement_date = (datetime.now(timezone.utc) + timedelta(days=rul_days)).date().isoformat()
    prediction = {
        "planeId": plane_id,
        "forecast": {
            "replacementDatePred": replacement_date,
            "rulDaysPred": int(rul_days),
            "rulCyclesPred": int(rul_cycles),
            "confidence": _safe_float(conf, 0.6),
        },
        "sohTargetBlend": _safe_float(current_soh, 0.0),
        "sohProxyPoh": _safe_float(latest.get("latent_soh_smooth_pct", observed_soh), observed_soh),
        "sohObservedNorm": _safe_float(observed_soh, 0.0),
    }

    return {"health": health, "prediction": prediction}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def export_plane_snapshots(plane_id: str, config: SnapshotConfig) -> None:
    df = _load_latent_table(config.latent_root, plane_id)
    if df.empty:
        raise RuntimeError(f"No latent SOH rows for plane {plane_id}")

    kpis_payload = _build_health_snapshot(df, plane_id, config)
    trend_payload = {
        "planeId": plane_id,
        "points": _trend_points(df, soh_col="latent_soh_filter_pct", source="blend"),
    }
    history_soh_col = (
        "latent_soh_smooth_pct" if "latent_soh_smooth_pct" in df.columns else "latent_soh_filter_pct"
    )
    history_payload = {
        "planeId": plane_id,
        "points": _trend_points(df, soh_col=history_soh_col, source="blend"),
    }

    _write_json(config.output_dir / f"plane_{plane_id}_kpis.json", kpis_payload)
    _write_json(config.output_dir / f"plane_{plane_id}_soh_trend.json", trend_payload)
    _write_json(config.output_dir / f"plane_{plane_id}_soh_history.json", history_payload)


def main() -> None:
    config = _parse_args()
    for plane_id in config.plane_ids:
        export_plane_snapshots(plane_id, config)


if __name__ == "__main__":
    main()
