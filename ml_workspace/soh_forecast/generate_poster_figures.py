from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "ml_workspace").exists() and (candidate / "data").exists():
            return candidate
    raise RuntimeError("Could not locate repo root")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
FIG_DIR = REPO_ROOT / "report_figures"
FIG_DIR.mkdir(exist_ok=True)


def save_latent_smoothing_and_estimation() -> Path:
    latent_path = REPO_ROOT / "ml_workspace" / "latent_soh" / "output" / "plane_166" / "latent_soh_event_table.csv"
    df = pd.read_csv(latent_path, parse_dates=["event_datetime"])
    df = df.loc[df["battery_id"].eq(1)].sort_values("event_datetime").copy()
    flight_df = df.loc[df["event_type"].eq("flight")].copy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    if "flight_index" in flight_df.columns:
        x = flight_df["flight_index"]
    elif "cumulative_flight_count" in flight_df.columns:
        x = flight_df["cumulative_flight_count"]
    else:
        x = pd.Series(range(len(flight_df)), index=flight_df.index, name="flight_index")

    axes[0].plot(
        x,
        flight_df["observed_soh_pct"],
        color="#94a3b8",
        linewidth=1.0,
        alpha=0.85,
        label="Observed SOH (flight events)",
    )
    axes[0].plot(
        x,
        flight_df["latent_soh_filter_pct"],
        color="#1d4ed8",
        linewidth=1.8,
        label="Causal latent SOH",
    )
    axes[0].plot(
        x,
        flight_df["latent_soh_smooth_pct"],
        color="#0f766e",
        linewidth=2.0,
        label="RTS smoothed latent SOH",
    )
    axes[0].set_title("Plane 166 Battery 1: Flight-Only Observed vs Latent SOH")
    axes[0].set_ylabel("SOH (%)")
    axes[0].legend(loc="best")

    axes[1].scatter(
        x,
        flight_df["measurement_sigma_pct"],
        s=14,
        alpha=0.75,
        color="#2563eb",
        label="Flight-event measurement sigma",
    )
    axes[1].plot(
        x,
        flight_df["measurement_sigma_pct"],
        color="#111827",
        linewidth=1.0,
        alpha=0.55,
    )
    axes[1].set_title("Condition-Aware Measurement Uncertainty on Flight Events")
    axes[1].set_ylabel("Measurement sigma (%)")
    axes[1].set_xlabel("Flight index")
    axes[1].legend(loc="best")

    plt.tight_layout()
    out = FIG_DIR / "poster_latent_smoothing_estimation.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def save_short_term_forecast() -> Path:
    base = REPO_ROOT / "ml_workspace" / "soh_forecast" / "output" / "multihorizon_runner_plane_166"
    best_df = pd.read_csv(base / "best_models_by_horizon.csv")
    best_df = best_df.copy()
    best_df["horizon"] = best_df["target"].str.extract(r"(\d+)").astype(int)
    metric_col = "backtest_mean_level_mae" if "backtest_mean_level_mae" in best_df.columns else "level_mae"
    best_df = best_df.sort_values("horizon")

    target_name = "latent_flight_5"
    row = best_df.loc[best_df["target"].eq(target_name)].iloc[0]
    pred_df = pd.read_csv(base / target_name / f"{target_name}_predictions.csv", parse_dates=["event_datetime"])
    actual_col = "next_latent_soh_causal_flight_5_pct"
    model_col = row["best_model"]

    example = pred_df.loc[pred_df["battery_id"].eq(1)].copy()
    for split_name in ["final_test", "holdout", "train_dev", "test", "valid", "train"]:
        candidate = example.loc[example["split"].eq(split_name)].sort_values("cumulative_flight_count")
        if len(candidate) >= 10:
            example = candidate
            example_split = split_name
            break
    else:
        example = example.sort_values("cumulative_flight_count")
        example_split = "all"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    axes[0].bar(best_df["horizon"].astype(str), best_df[metric_col], color="#2563eb", alpha=0.9)
    for _, item in best_df.iterrows():
        axes[0].text(
            x=str(item["horizon"]),
            y=item[metric_col] + 0.01,
            s=item["best_model"].replace("_with_latent", "").replace("_no_latent", ""),
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0].set_title("Best Backtest MAE by Forecast Horizon")
    axes[0].set_xlabel("Forecast horizon (flights)")
    axes[0].set_ylabel("Level MAE (SOH %)")

    axes[1].plot(
        example["cumulative_flight_count"],
        example[actual_col],
        color="#111827",
        linewidth=2.0,
        label="Actual 5-flight SOH",
    )
    axes[1].plot(
        example["cumulative_flight_count"],
        example[model_col],
        color="#0f766e",
        linestyle="--",
        linewidth=2.0,
        label=f"Predicted ({model_col})",
    )
    axes[1].set_title(f"Example 5-Flight Forecast Trajectory ({example_split} split)")
    axes[1].set_xlabel("Cumulative flight count")
    axes[1].set_ylabel("Future SOH after 5 flights (%)")
    axes[1].legend(loc="best")

    plt.tight_layout()
    out = FIG_DIR / "poster_short_term_forecasting.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def save_long_term_backbone() -> Path:
    base = REPO_ROOT / "ml_workspace" / "soh_forecast" / "output" / "backbone_curve_plane_166"
    external_curve = pd.read_csv(base / "external_backbone_curve.csv")
    combined_curve = pd.read_csv(base / "combined_backbone_curve.csv")
    combined_points = pd.read_csv(base / "combined_backbone_points.csv")
    plane_df = pd.read_csv(base / "plane166_backbone_dataset.csv", parse_dates=["event_datetime"])
    trajectory_df = pd.read_csv(base / "plane166_backbone_trajectory.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    sample_points = combined_points.sample(min(len(combined_points), 1500), random_state=7)
    for source_name, color in [("evtol", "#94a3b8"), ("plane", "#2563eb")]:
        subset = sample_points.loc[sample_points["source"].eq(source_name)]
        axes[0].scatter(subset["progress"], subset["health_pct"], s=10, alpha=0.3, color=color, label=source_name)
    axes[0].plot(external_curve["progress"], external_curve["health_pct"], color="#dc2626", linewidth=2.0, label="External backbone")
    axes[0].plot(combined_curve["progress"], combined_curve["health_pct"], color="#0f766e", linewidth=2.2, label="Combined backbone")
    axes[0].set_title("Normalized Long-Run Backbone Shape")
    axes[0].set_xlabel("Normalized life progress")
    axes[0].set_ylabel("Adjusted SOH (%)")
    axes[0].set_ylim(-2.0, 100.0)
    axes[0].legend(loc="best")

    colors = {1: "#1d4ed8", 2: "#7c3aed"}
    for battery_id in sorted(trajectory_df["battery_id"].unique()):
        observed = plane_df.loc[plane_df["battery_id"].eq(battery_id)].sort_values("cumulative_flight_count")
        traj = trajectory_df.loc[trajectory_df["battery_id"].eq(battery_id)].sort_values("cumulative_flight_count")
        color = colors.get(battery_id, "#111827")
        axes[1].plot(
            observed["cumulative_flight_count"],
            observed["current_soh_pct"],
            color=color,
            linewidth=2.0,
            label=f"Causal latent batt {battery_id}",
        )
        axes[1].plot(
            traj["cumulative_flight_count"],
            traj["backbone_soh_pct"],
            color=color,
            linestyle="--",
            linewidth=2.0,
            alpha=0.9,
            label=f"Backbone batt {battery_id}",
        )
    axes[1].axhline(0.0, color="#b91c1c", linestyle="--", linewidth=1.0)
    axes[1].set_title("Plane-Calibrated Backbone vs Latent SOH")
    axes[1].set_xlabel("Cumulative flight count")
    axes[1].set_ylabel("Adjusted SOH (%)")
    axes[1].set_ylim(-2.0, 100.0)
    axes[1].legend(loc="best", fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "poster_long_term_backbone.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    outputs = [
        save_latent_smoothing_and_estimation(),
        save_short_term_forecast(),
        save_long_term_backbone(),
    ]
    for path in outputs:
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
