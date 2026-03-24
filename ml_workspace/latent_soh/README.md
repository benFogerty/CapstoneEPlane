# Latent SOH

This workspace builds a condition-aware latent SOH label from parquet telemetry using an event-level state-space model.

It defaults to the corrected downstream parquet file:
- `data/event_timeseries.parquet`

Key points:
- Input observations are the per-event BMS SOH estimates from aux telemetry.
- The pipeline does **not** treat those observations as ground truth.
- Measurement noise is made time-varying with condition scores driven by current, `dI/dt`, `dT/dt`, SOC edge behavior, estimator disagreement, and estimator reset flags.
- `FilterPy` is the canonical backend (filter + RTS smoother).
- Canonical SOH series: `latent_soh_smooth_pct`; uncertainty: `latent_soh_smooth_std_pct`.
- `R_t` aggressiveness is controlled by named profiles:
  - `balanced` is the default
  - `current` reproduces the original heavier profile
  - `light` and `instability_focused` are available for tuning

Main entrypoint:

```bash
.venv/bin/python -m ml_workspace.latent_soh.build_latent_soh --plane-id 166 --rt-profile balanced
```

Outputs:
- `output/plane_<plane_id>/event_observation_table.csv`
- `output/plane_<plane_id>/latent_soh_event_table.csv`
- `output/plane_<plane_id>/diagnostics/smoother_summary.json`
- `output/plane_<plane_id>/diagnostics/condition_score_summary.csv`
- `output/plane_<plane_id>/diagnostics/top_raw_spike_events.csv`
- `output/plane_<plane_id>/diagnostics/spike_feature_summary.csv`
- `output/plane_<plane_id>/diagnostics/plots/`

The notebook [latent_soh_walkthrough.ipynb](/Users/benfogerty/Desktop/EPlaneCapstone/CapstoneEPlane/ml_workspace/latent_soh/latent_soh_walkthrough.ipynb) explains the method and graphs the main diagnostics.

## Canonical Method Selection

We select the canonical filter settings using a likelihood-based comparison rather than a weighted score.
The model is a 1D state-space system:
- State: latent SOH
- Process model: latent_soh_{t+1} = latent_soh_t + w_t, with w_t ~ N(0, Q_t)
- Measurement model: observed_soh_t = latent_soh_t + v_t, with v_t ~ N(0, R_t)

Where Q_t is controlled by `q_day_sigma_pct` and R_t is derived from condition-aware measurement noise
(`rt_profile`, plus current, dI/dt, dT/dt, SOC edge, instability, gap, and reset signals). This allows
the filter to trust clean telemetry more and down-weight events flagged as unstable or inconsistent.

For each candidate (`rt_profile`, `q_day_sigma_pct`), we run FilterPy (filter + RTS smoother) and compute
log-likelihood under the measurement model:
`observed_soh_pct ~ N(latent_soh_smooth_pct, measurement_sigma_pct^2)`.
We rank candidates by `avg_loglik` (higher is better), which is the standard Kalman optimality criterion.

We still track diagnostic metrics like spike removal and smoothness to confirm the chosen model is
behaviorally reasonable, but they do not determine the ranking. The best candidate is re-run to rebuild
the canonical `latent_soh_event_table.csv`. The canonical SOH series is `latent_soh_smooth_pct` with
uncertainty `latent_soh_smooth_std_pct`.

Method-search sweeps are intentionally not committed. The current forecasting pipeline uses:
- `rt_profile = current`
- `q_day_sigma_pct = 0.10`

## Forecasting and Data Leakage

RTS smoothing uses future observations, so `latent_soh_smooth_pct` is **not** causal. That is fine for
retrospective analysis but it will leak future information if used as a training target for forecasts
at time *t*. For forecasting, use the **causal** (filter-only) series:
- `latent_soh_filter_pct` (alias: `latent_soh_causal_pct`)
- `latent_soh_filter_std_pct` (alias: `latent_soh_causal_std_pct`)

These are computed using only past and current observations, so they are safe for model training,
validation, and evaluation on future time slices.

How we prevent leakage in practice:
- **Targets:** Forecasting targets are built from the causal series (`latent_soh_filter_pct`). The pipeline
  creates `next_latent_soh_causal_pct` and uses it as the target. If you see `next_latent_soh_smooth_pct`,
  it is explicitly overridden to the causal value in the forecasting feature pipeline.
- **Features:** Only features available at time *t* are used (no future events). Rolling/lag features are
  computed with `shift(1)` or past windows.
- **Splits:** Train/validation/test are assigned in chronological order per plane, and the holdout plane
  (e.g., 192) is never used for model selection.

If you want to visualize a clean degradation curve, you can still plot `latent_soh_smooth_pct`, but do
not use it as a label for predictive modeling.

To rebuild the latent SOH tables for a plane:
```bash
.venv/bin/python -m ml_workspace.latent_soh.build_latent_soh --plane-id 166 --rt-profile current --q-day-sigma-pct 0.10
```
