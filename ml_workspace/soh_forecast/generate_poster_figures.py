from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
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

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.4))

    x = flight_df["event_datetime"]

    ax.plot(
        x,
        flight_df["observed_soh_pct"],
        color="#64748b",
        linewidth=1.2,
        alpha=0.6,
        zorder=1,
    )
    ax.scatter(
        x,
        flight_df["observed_soh_pct"],
        s=28,
        color="#475569",
        edgecolors="#f8fafc",
        linewidths=0.6,
        alpha=0.95,
        zorder=2,
        label="Observed raw SOH (flight events)",
    )
    ax.plot(
        x,
        flight_df["latent_soh_smooth_pct"],
        color="#0f766e",
        linewidth=2.8,
        zorder=3,
        label="Kalman-smoothed latent SOH",
    )
    ax.set_title("Plane 166 Battery 1: Raw Flight SOH vs Kalman-Smoothed Latent SOH")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_xlabel("Event date")
    ax.set_ylabel("SOH (%)")
    ax.legend(loc="best")

    plt.tight_layout()
    out = FIG_DIR / "poster_latent_smoothing_estimation.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def save_raw_soh_spike_trust_figure() -> Path:
    latent_path = REPO_ROOT / "ml_workspace" / "latent_soh" / "output" / "plane_166" / "latent_soh_event_table.csv"
    spike_path = REPO_ROOT / "ml_workspace" / "latent_soh" / "output" / "plane_166" / "diagnostics" / "top_raw_spike_events.csv"

    df = pd.read_csv(latent_path, parse_dates=["event_datetime"])
    df = df.loc[df["battery_id"].eq(1)].sort_values("event_datetime").copy()
    spikes = pd.read_csv(spike_path, parse_dates=["event_datetime"])
    spikes = (
        spikes.loc[spikes["battery_id"].eq(1)]
        .sort_values("delta_observed_soh_pct", key=lambda s: s.abs(), ascending=False)
        .head(3)
        .copy()
    )

    start = spikes["event_datetime"].min() - pd.Timedelta(days=10)
    stop = spikes["event_datetime"].max() + pd.Timedelta(days=10)
    window = df.loc[df["event_datetime"].between(start, stop)].copy()

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    ax.plot(
        window["event_datetime"],
        window["observed_soh_pct"],
        color="#334155",
        linewidth=1.1,
        alpha=0.6,
        zorder=1,
    )
    ax.scatter(
        window["event_datetime"],
        window["observed_soh_pct"],
        s=20,
        color="#475569",
        edgecolors="#f8fafc",
        linewidths=0.4,
        alpha=0.9,
        zorder=3,
        label="Raw observed SOH",
    )
    for _, row in spikes.iterrows():
        previous = df.loc[
            df["battery_id"].eq(row["battery_id"])
            & df["event_datetime"].lt(row["event_datetime"]),
            ["event_datetime", "observed_soh_pct"],
        ].tail(1)
        if previous.empty:
            continue
        ax.plot(
            [previous["event_datetime"].iloc[0], row["event_datetime"]],
            [previous["observed_soh_pct"].iloc[0], row["observed_soh_pct"]],
            color="#dc2626",
            linewidth=1.6,
            alpha=0.8,
            zorder=2,
        )
    ax.scatter(
        spikes["event_datetime"],
        spikes["observed_soh_pct"],
        s=60,
        color="#dc2626",
        edgecolors="#7f1d1d",
        linewidths=0.6,
        zorder=4,
        label="Largest raw SOH jumps",
    )

    for _, row in spikes.iterrows():
        ax.annotate(
            f"{row['delta_observed_soh_pct']:+.0f} pts",
            xy=(row["event_datetime"], row["observed_soh_pct"]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8,
            color="#7f1d1d",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fee2e2", "edgecolor": "#fecaca"},
        )

    ax.set_xlabel("Event date", fontsize=11)
    ax.set_ylabel("SOH (%)", fontsize=11)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    out = FIG_DIR / "poster_raw_soh_spikes_untrusted.png"
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


def save_best_model_horizon_mae() -> Path:
    base = REPO_ROOT / "ml_workspace" / "soh_forecast" / "output" / "multihorizon_runner_plane_166"
    best_df = pd.read_csv(base / "best_models_by_horizon.csv")
    best_df = best_df.copy()
    best_df["horizon"] = best_df["target"].str.extract(r"(\d+)").astype(int)
    metric_col = "backtest_mean_level_mae" if "backtest_mean_level_mae" in best_df.columns else "level_mae"
    best_df = best_df.sort_values("horizon")

    target_mae = 0.35
    horizons = best_df["horizon"].astype(int).tolist()
    horizon_labels = [f"{horizon} Flight" if horizon == 1 else f"{horizon} Flights" for horizon in horizons]
    mae_values = best_df[metric_col].to_numpy()

    fig, ax = plt.subplots(1, 1, figsize=(11.2, 4.08), constrained_layout=True)

    ax.plot(
        horizon_labels,
        mae_values,
        color="#2563eb",
        linewidth=2.6,
        marker="o",
        markersize=7.2,
        markerfacecolor="#2563eb",
        markeredgecolor="#1d4ed8",
        markeredgewidth=1.0,
        label="MAE",
        zorder=3,
    )
    ax.axhline(
        target_mae,
        color="#22c55e",
        linestyle="--",
        linewidth=2.2,
        label=f"Target ({target_mae:.2f})",
        zorder=3,
    )

    for idx, value in enumerate(mae_values):
        ax.text(
            idx,
            value + 0.045,
            f"{value:.3f}".rstrip("0").rstrip("."),
            ha="center",
            va="bottom",
            fontsize=13,
            color="#111827",
        )

    ax.text(
        -0.055,
        1.075,
        "MAE (SOH pts)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        color="#111827",
    )
    ax.set_xlabel("Forecast Horizon", fontsize=15, labelpad=12)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=13)
    ax.set_ylim(0, max(max(mae_values) * 1.22, target_mae * 1.35))
    ax.set_yticks([0.0, 0.3, 0.6, 0.9, 1.2])
    ax.grid(axis="y", color="#d1d5db", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        frameon=False,
        fontsize=14,
        handlelength=2.4,
        columnspacing=1.8,
    )

    out = FIG_DIR / "poster_best_model_horizon_mae.png"
    fig.savefig(out, dpi=220)
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


def add_estimated_rul_hour_axis(ax: plt.Axes, plane_df: pd.DataFrame, calibration_df: pd.DataFrame) -> None:
    total_life_hours = []
    for battery_id, g in plane_df.groupby("battery_id"):
        calib = calibration_df.loc[calibration_df["battery_id"].astype(str).eq(str(battery_id))]
        if calib.empty:
            continue

        duration_h = pd.to_numeric(g["duration_h"], errors="coerce").dropna()
        mean_duration_h = duration_h.loc[duration_h.gt(0)].mean()
        if pd.isna(mean_duration_h):
            continue
        total_life_hours.append(float(calib["total_life_flights_from_start"].iloc[0]) * float(mean_duration_h))

    if not total_life_hours:
        return

    median_total_life_hours = float(np.median(total_life_hours))
    progress_ticks = np.linspace(0.0, 1.0, 6)
    rul_hour_ticks = (1.0 - progress_ticks) * median_total_life_hours

    rul_ax = ax.secondary_xaxis("bottom", functions=(lambda x: x, lambda x: x))
    rul_ax.spines["bottom"].set_position(("outward", 48))
    rul_ax.set_xlabel("Estimated RUL (flight hours)", fontsize=12, labelpad=8)
    rul_ax.set_xticks(progress_ticks)
    rul_ax.set_xticklabels([f"{tick:.0f} h" for tick in rul_hour_ticks])
    rul_ax.tick_params(axis="x", labelsize=10, pad=2)


def save_normalized_long_run_backbone_shape() -> Path:
    base = REPO_ROOT / "ml_workspace" / "soh_forecast" / "output" / "backbone_curve_plane_166"
    external_curve = pd.read_csv(base / "external_backbone_curve.csv")
    combined_curve = pd.read_csv(base / "combined_backbone_curve.csv")
    combined_points = pd.read_csv(base / "combined_backbone_points.csv")
    plane_df = pd.read_csv(base / "plane166_backbone_dataset.csv", parse_dates=["event_datetime"])
    calibration_df = pd.read_csv(base / "plane166_backbone_calibration.csv")

    fig, ax = plt.subplots(1, 1, figsize=(11.44, 5.56))

    sample_points = combined_points.sample(min(len(combined_points), 1500), random_state=7)
    for source_name, color, marker, size, alpha in [
        ("evtol", "#8f9aa6", "o", 13, 0.38),
        ("plane", "#1d4ed8", "^", 26, 0.72),
    ]:
        subset = sample_points.loc[sample_points["source"].eq(source_name)]
        ax.scatter(
            subset["progress"],
            subset["health_pct"],
            s=size,
            alpha=alpha,
            color=color,
            marker=marker,
            label=source_name,
        )
    ax.plot(
        external_curve["progress"],
        external_curve["health_pct"],
        color="#dc2626",
        linewidth=2.2,
        label="External backbone",
    )
    ax.plot(
        combined_curve["progress"],
        combined_curve["health_pct"],
        color="#0f766e",
        linewidth=2.4,
        label="Combined backbone",
    )
    ax.set_title("Normalized Long-Run Backbone Shape", fontsize=17)
    ax.set_xlabel("Normalized life progress", fontsize=13)
    ax.set_ylabel("Adjusted SOH (%)", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-2.0, 100.0)
    ax.legend(loc="best", fontsize=12)
    add_estimated_rul_hour_axis(ax, plane_df, calibration_df)

    plt.tight_layout()
    out = FIG_DIR / "poster_normalized_long_run_backbone_shape.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    outputs = [
        save_latent_smoothing_and_estimation(),
        save_raw_soh_spike_trust_figure(),
        save_short_term_forecast(),
        save_best_model_horizon_mae(),
        save_long_term_backbone(),
        save_normalized_long_run_backbone_shape(),
    ]
    for path in outputs:
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
