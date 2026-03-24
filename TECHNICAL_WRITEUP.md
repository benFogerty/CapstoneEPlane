# AeroCell Technical Write-Up

Last updated: 2026-03-24

This document is the repo's technical-depth reference. It consolidates the main modeling choices, the findings that originally lived across one-off notebooks, and the current runtime integration path used by the frontend and planning flows.

The short version:

- The canonical health label is a condition-aware latent SOH trajectory built from event telemetry with a FilterPy Kalman filter and RTS smoother.
- Charge-event physics and ICA-style analyses were useful for pack inference and for understanding when the raw BMS SOH stream is trustworthy, but they are not the primary production label.
- Forecasting is event-based and multi-horizon. The current best committed models are mostly LSTM/GRU sequence models on latent-flight horizons.
- The repo contains two RUL ideas:
  - an external-backbone calibration flow for mapping aircraft batteries onto eVTOL degradation shapes
  - the current frontend/runtime forecaster, which rolls the committed degradation model forward until SOH crosses the replacement threshold
- Operational optimization is a degradation-aware scheduling heuristic, not a full dispatch optimizer.

## 1. System Framing

The project solves three related problems:

1. Estimate a usable battery health signal from noisy aircraft telemetry.
2. Forecast how that health signal will move under future usage.
3. Turn those forecasts into decisions: replacement timing, flight-window recommendations, and battery-life-aware schedule scoring.

The challenge is that the raw aircraft/BMS SOH channel is visibly noisy, sometimes step-like, and often jumps during charge events or SOC-edge conditions. The pipeline therefore separates:

- raw observed SOH from the aircraft telemetry
- a latent health estimate used as the canonical training label
- auxiliary physics and charge-event estimators used for interpretation, pack inference, and validation

## 2. Data Model

The core inputs are:

- `data/event_manifest.parquet`
- `data/event_timeseries.parquet`
- `ml_workspace/latent_soh/output/plane_166/latent_soh_event_table.csv`
- `ml_workspace/latent_soh/output/plane_192/latent_soh_event_table.csv`

The modeling unit is the event. Each row is one battery-event instance for one aircraft battery, with:

- `event_type`
- `event_datetime`
- SOC summary fields such as `soc_min_pct`, `soc_mean_pct`, `soc_max_pct`
- current, thermal, and duration summaries
- observed SOH from the telemetry stream
- latent filter/smoother outputs

Current dataset summary from `ml_workspace/soh_forecast/output/dataset_summary.json`:

- 1,204 total event rows
- 1,200 rows with a next event available
- 10 rows with event gaps above the default training threshold
- plane `166`: 1,106 rows from `2023-05-16 11:00:00` to `2025-12-02 17:35:00`
- plane `192`: 98 rows from `2025-06-03 11:30:00` to `2025-07-05 00:47:00`

Plane `166` is the primary modeling aircraft. Plane `192` is useful as a sparse holdout/OOD check, but it is too small to support strong conclusions by itself.

## 3. Consolidated Notebook Findings

Many of the early capstone findings were developed in exploratory notebooks and later reduced to a smaller committed repo. The important findings still drive the production code, even where the one-off notebooks themselves were removed.

### 3.1 Raw SOH behavior

Across the exploratory work, the raw observed SOH stream showed three recurring issues:

- implausibly large positive jumps
- event-type dependence, especially around charge events
- SOC-edge sensitivity near top-of-charge and near estimator resets

Those findings are still visible in the committed diagnostics for plane `166`.

From `ml_workspace/latent_soh/output/plane_166/diagnostics/top_raw_spike_events.csv`:

- the largest raw upward jump is `+29.0` SOH points on a `charge` event at `2024-05-01 17:00:00`
- other large positive jumps also cluster around charge events with `soc_max_pct` near `99-100`

From `ml_workspace/latent_soh/output/plane_166/diagnostics/spike_feature_summary.csv`:

- spike events are about `71-73%` charge events and only about `8%` flight events
- non-spike events are about `53-54%` charge events and about `36%` flight events

Interpretation:

- the raw BMS SOH channel is not just measuring gradual electrochemical fade
- it is also reflecting charge-state context, reset behavior, and estimator logic
- this is the main reason the project moved to a latent state estimate instead of using observed SOH directly as ground truth

### 3.2 Charge-event notebooks

The physics-based charge-event notebook and earlier ICA/IV notebooks converged on a practical conclusion:

- with the available aircraft telemetry, charge-event capacity estimation is more defensible than a full electrochemical state-space reconstruction

The committed notebook `ml_workspace/soh_estimation/physics_based_soh_from_charge_events.ipynb` explicitly frames the chosen estimator as charge-event coulomb counting:

- `Q_delivered = ∫ |I| dt / 3600`
- `C_hat = Q_delivered / (ΔSOC / 100)`

That notebook also shows that the capacity-derived series is directionally useful, but less stable than the latent label and not strong enough to replace it as the canonical target.

### 3.3 ICA-specific findings

The original ICA work was exploratory rather than productionized. Its main value was diagnostic:

- incremental-capacity and IV-style features were most informative on clean, monotonic charge segments
- charge windows with poor SOC monotonicity or weak voltage span were much less reliable
- the charge-event SOH signal was best treated as a filtered subset problem, not as a universal estimator on every charging trace

That work influenced:

- the charge-event filtering thresholds
- the battery-inference workflow
- the decision to treat raw BMS SOH as noisy observation rather than truth

The cleaned repo no longer keeps the earlier one-off ICA notebooks, but the conclusions are preserved here and in the surviving charge-event and latent pipelines.

## 4. Latent SOH Method

The latent SOH workspace is in `ml_workspace/latent_soh/`. It builds a 1D event-level state-space model:

- state model: latent SOH follows a random walk
- measurement model: observed BMS SOH is a noisy observation of that latent state

### 4.1 Why a latent label was needed

The aircraft-level BMS SOH stream is useful, but not stable enough to serve as a direct supervised target. The latent approach was introduced to:

- remove implausible jumps
- preserve gradual degradation trends
- produce uncertainty estimates
- separate retrospective smoothing from causal forecasting labels

### 4.2 Measurement noise design

The important design choice is that measurement noise is time-varying rather than fixed.

`R_t` is increased when conditions suggest low trust in the observation, using features tied to:

- high current
- high `dI/dt`
- high `dT/dt`
- SOC-edge behavior
- disagreement between estimator channels
- reset flags and instability flags
- event gaps

The canonical implementation uses FilterPy. Multiple named aggressiveness profiles exist, but the current forecasting pipeline uses:

- `rt_profile = current`
- `q_day_sigma_pct = 0.10`

### 4.3 Filtered vs smoothed outputs

The workspace writes both:

- `latent_soh_filter_pct`
- `latent_soh_smooth_pct`

This distinction matters:

- `latent_soh_smooth_pct` uses future information and is appropriate for retrospective analysis and plots
- `latent_soh_filter_pct` is causal and is the forecasting-safe label used downstream

The feature pipeline explicitly overrides any smoothed next-target field with the causal filter target to prevent leakage.

### 4.4 What the diagnostics show

From `ml_workspace/latent_soh/output/plane_166/diagnostics/smoother_summary.json`:

- 1,106 total events
- raw total variation per battery: about `484` and `469`
- smoothed total variation per battery: about `46.9` and `45.5`
- raw max upward jump per battery: `29` and `25`
- smoothed max upward jump per battery: about `0.70`

From `ml_workspace/latent_soh/output/plane_192/diagnostics/smoother_summary.json`:

- 98 total events
- smoothed total variation per battery: about `0.856` and `0.798`

Interpretation:

- the smoother removes the pathological step changes seen in the raw stream
- the resulting latent curve is much more plausible as a degradation target
- plane `166` clearly benefits from aggressive observation down-weighting
- plane `192` is much quieter, but also much smaller

## 5. Charge-Event Physics and Battery Inference

There are two related workflows here:

- charge-event capacity/SOH estimation in `ml_workspace/soh_estimation/`
- battery-type inference in `ml_workspace/battery_inference/`

### 5.1 Physics-based SOH from charge events

The production script is `ml_workspace/soh_estimation/physics_based_soh.py`.

It consumes charge-event summaries and computes:

- `physics_soh_absolute_pct = 100 * capacity_est_ah / rated_capacity_ah`
- `physics_soh_relative_pct = 100 * capacity_est_ah / reference_capacity_ah`

The notebook comparison for plane `166` reports:

- absolute capacity SOH, overall: `n = 261`, `MAE = 7.904`, `RMSE = 9.919`, `R² = 0.106`
- relative/reference-normalized SOH, overall: `MAE = 10.422`, `RMSE = 12.183`, `R² = -0.348`

The notebook conclusion is directionally clear:

- absolute capacity-based SOH tracks observed charge-event SOH better than the reference-normalized version
- even so, the observed SOH stream is still more step-like than the charge-derived estimate
- this supports the idea that the BMS channel includes reporting/recalibration logic, not just physical fade

### 5.2 Plane-level battery inference

The battery-inference workflow estimates effective capacity from charging-event telemetry and uses top-of-charge voltage as a cross-check.

From `ml_workspace/battery_inference/output/plane_166/plane_battery_inference.json`:

- inferred battery type: `PB345V119E-L`
- confidence: `0.80`
- median effective capacity estimate: `26.13 Ah`
- capacity IQR: `2.16 Ah`
- median top-of-charge voltage: `404.2 V`
- valid event-battery segments: `261`

Interpretation:

- the pack inference is strong enough to support the `29 Ah` family rather than the `33 Ah` alternative
- the spread is still too wide to treat partial-charge capacity as a precise rated-capacity measurement on its own

## 6. Forecasting Pipeline

The forecasting workspace is `ml_workspace/soh_forecast/`.

### 6.1 Forecast target design

The event-based forecasting pipeline predicts future latent SOH at multiple horizons. For the tabular models, the main target form is:

- `target_delta = next_soh - current_soh`

and then the model output is converted back to a level prediction:

- `predicted_next_soh = current_soh + predicted_delta`

Sequence models predict the next level directly from a lookback window.

### 6.2 Why the target is event-based

The event framing makes it possible to:

- mix flights, charge events, and idle gaps in a single timeline
- model degradation per operational event, not only per calendar day
- forward-simulate future SOH under hypothetical future usage

### 6.3 Features and leakage control

Key feature families include:

- current, temperature, and SOC summaries
- throughput and equivalent-full-cycle proxies
- event duration and gap features
- lagged and rolling history features
- latent SOH-derived features
- event type and plane/battery identifiers

Leakage is controlled by:

- building forecast targets from `latent_soh_filter_pct`
- using only past-window and lagged features
- chronological train/validation/test assignment
- keeping the holdout plane outside model selection

### 6.4 Model families

The training pipeline supports:

- naive zero-delta baseline
- ridge and elastic-net tabular models
- spline/GAM-style tabular model
- histogram gradient boosting
- random forest
- LSTM sequence model
- GRU sequence model
- physics-hybrid and physics-informed neural variants

### 6.5 Best committed multi-horizon results

From `ml_workspace/soh_forecast/output/multihorizon_runner_plane_166/best_models_by_horizon.csv`:

- `latent_flight_1`: best model `lstm_sequence_no_latent_tune3`, backtest mean level MAE `0.192`
- `latent_flight_5`: best model `lstm_sequence_no_latent_tune0`, backtest mean level MAE `0.664`
- `latent_flight_10`: best model `gru_sequence_with_latent_tune0`, backtest mean level MAE `1.140`
- `latent_flight_15`: best model `gru_sequence_with_latent_tune1`, backtest mean level MAE `1.165`
- `latent_flight_20`: best model `gru_sequence_with_latent_tune2`, backtest mean level MAE `1.187`

Observed pattern:

- short horizons are reasonably accurate
- longer horizons degrade quickly
- holdout metrics become unstable once horizon coverage gets sparse

This is a useful modeling result in itself: the project has stronger evidence for short-term operational forecasting than for long-range deterministic life prediction from the current aircraft dataset alone.

## 7. Backbone RUL Method

The external-backbone workflow lives in:

- `ml_workspace/soh_forecast/external_backbone_common.py`
- `ml_workspace/soh_forecast/backbone_shape.py`
- `ml_workspace/soh_forecast/fit_backbone_curve.py`

This was designed to answer a different question than the direct forecast models:

- can we place an aircraft battery on a normalized degradation-progress curve learned from external eVTOL battery test files?

### 7.1 External dataset shaping

The workflow reads `VAH*.csv` eVTOL test files and constructs:

- capacity-test anchor points
- mission discharge indices
- interval-level features aligned to the aircraft feature schema

Each file gets an estimated end-of-life mission index. Progress is normalized as:

- `progress = mission_discharge_index / estimated_eol_index`

Health is expressed as an adjusted capacity scale clipped to `[0, 100]`.

### 7.2 Backbone fitting

The normalized external points are fit with a decreasing isotonic regression backbone. That choice enforces:

- monotonic degradation
- a shape that is data-driven but physically plausible
- bounded outputs between `0` and `100`

### 7.3 Plane calibration

For each aircraft battery, the calibration step grid-searches:

- `start_progress`
- `total_life_flights_from_start`

The objective is the MAE between observed plane SOH and the backbone-predicted health trajectory. The final output includes:

- fitted MAE
- estimated remaining flights to zero-health on the backbone scale

### 7.4 Interpretation

The backbone method is useful as a shape prior and explanatory tool. It is not the current frontend's primary RUL source, but it provides:

- a normalized degradation geometry
- a way to compare aircraft batteries against external life-test archetypes
- a plausible mechanism for extrapolation when aircraft data alone are sparse

## 8. Runtime RUL Prediction Used by the Frontend

The current live prediction path is in `frontend/scripts/live_model_outputs.py`.

This path is more operational than the external-backbone method.

### 8.1 Core rollout logic

The script:

1. builds a current plane profile
2. loads the committed model payload
3. simulates future charge and mission events
4. updates SOH event-by-event with `_simulate_event(...)`
5. stops when SOH falls below `REPLACEMENT_THRESHOLD_SOH = 40.0`

The forward rollout is generated by `_forecast_points(...)`.

### 8.2 Modeling assumptions in the rollout

The runtime forecaster:

- estimates future cadence from recent flight history
- alternates future flights and charge events as needed
- clamps unrealistic per-event degradation predictions
- imposes a baseline wear floor so the forecast cannot become unrealistically flat

The script then derives:

- `replacementDatePred`
- `rulDaysPred`
- `rulCyclesPred`
- `confidence`

This is the RUL path users actually see in the live dashboard.

### 8.3 Snapshot/exporter RUL path

`ml_workspace/integration/export_snapshots.py` contains a related but simpler horizon-aggregation method:

- it computes recent per-flight SOH drop from the best-horizon prediction CSVs
- weights horizons by inverse `delta_mae`
- falls back to empirical recent flight deltas if needed
- converts the resulting per-flight drop into `rulCyclesPred` and `rulDaysPred`

So the repo currently contains both:

- a simulation-style frontend forecaster
- a snapshot/exporter aggregation path

They are conceptually aligned, but they are not identical implementations.

## 9. Circuit Capacity and SOC Burn Models

The circuit-capacity workspace is `ml_workspace/circuit_capacity/`.

### 9.1 POH-based circuits model

The committed circuit model in `ml_workspace/circuit_capacity/output/circuit_model.json` uses:

- SOH grid: `[0, 20, 40, 60, 80, 100]`
- POH SOC-per-circuit grid: `[20, 16, 13, 12, 10, 9]`
- reserve SOC: `30%`

Per-plane calibration factors:

- plane `166`: `k_plane = 0.80`
- plane `192`: `k_plane = 0.74`

The core idea is simple:

- lower SOH implies more SOC cost per circuit
- the aircraft-specific scale factor maps the POH curve onto observed plane behavior

### 9.2 SOC-rate regression

The workspace also trains a flight SOC burn-rate model.

From `ml_workspace/circuit_capacity/output/soc_rate_model_metrics.json`:

- test MAE: `0.192` SOC-%/min
- test RMSE: `0.263`
- test `R² = 0.555`
- OOD `R² = 0.541`

This model is one of the stronger pieces of the stack because the target is more directly observable than SOH.

## 10. Operational Optimization and Scheduling

The scheduling code is in `ml_workspace/scheduling/`.

There are two levels of operational logic:

- offline scheduling heuristics in the ML workspace
- the live planner/recommendation scoring in the frontend

### 10.1 Offline window optimizer

`ml_workspace/scheduling/optimize_with_windows.py` is a greedy lookahead scheduler.

At a high level it:

- enumerates feasible candidate flight windows
- applies available charge windows first
- computes mission SOC demand using the circuit-capacity model
- predicts degradation for each candidate event
- scores candidates primarily by `pred_delta`
- selects the least-damaging option within the lookahead set

This is degradation-aware optimization, but it is still heuristic:

- no mixed-integer solver
- no global optimality guarantee
- limited lookahead depth by design

### 10.2 Frontend planner logic

The live planner path in `frontend/scripts/live_model_outputs.py` and `frontend/lib/planner-service.ts` adds operational constraints and user-facing scoring.

Important behaviors:

- weather can increase mission SOC demand
- reserve feasibility is checked explicitly
- charge-window feasibility is checked explicitly
- high-SOC waiting incurs a dwell penalty through `calendar_dwell_delta`
- ranking combines wear, charging burden, thermal/weather effects, and infeasibility penalties

This is best described as an advisory planner:

- it ranks better and worse days/windows
- it is battery-life-aware
- it is not a certified dispatch or safety optimizer

## 11. What the Current Stack Gets Right

- The latent label is much more credible than the raw SOH telemetry for supervised learning.
- Short-horizon health forecasting is meaningfully better than pretending SOH is static.
- Charge-event physics are useful for interpretation and pack inference.
- The planner is grounded in actual degradation and SOC models, not just calendar heuristics.
- The frontend is wired to committed artifacts instead of depending on ad hoc notebook outputs.

## 12. Main Limitations

- Plane `192` is too small to be a strong external validation set.
- Longer-horizon SOH forecasts remain unstable.
- ICA work was informative but not turned into a production estimator.
- The external-backbone method and the runtime frontend forecaster are not yet unified into one canonical RUL implementation.
- The optimizer is heuristic and intentionally lightweight.

## 13. Recommended Next Technical Steps

If this project were extended further, the highest-value improvements would be:

1. Increase aircraft-level holdout diversity before making stronger long-horizon claims.
2. Unify the backbone and runtime RUL stories into one documented canonical life-prediction path.
3. Promote the best ICA/charge-shape features into a production feature block only if they remain stable under stricter event-quality filtering.
4. Add a planner benchmark that measures not just predicted wear reduction, but realized constraint satisfaction and operational regret under simulated demand.

## 14. File Map

The most important files for this write-up are:

- `ml_workspace/latent_soh/README.md`
- `ml_workspace/latent_soh/output/plane_166/diagnostics/smoother_summary.json`
- `ml_workspace/latent_soh/output/plane_166/diagnostics/top_raw_spike_events.csv`
- `ml_workspace/latent_soh/output/plane_166/diagnostics/spike_feature_summary.csv`
- `ml_workspace/soh_estimation/physics_based_soh.py`
- `ml_workspace/soh_estimation/physics_based_soh_from_charge_events.ipynb`
- `ml_workspace/battery_inference/README.md`
- `ml_workspace/battery_inference/output/plane_166/plane_battery_inference.json`
- `ml_workspace/soh_forecast/models/README.md`
- `ml_workspace/soh_forecast/output/multihorizon_runner_plane_166/best_models_by_horizon.csv`
- `ml_workspace/soh_forecast/external_backbone_common.py`
- `ml_workspace/soh_forecast/backbone_shape.py`
- `ml_workspace/circuit_capacity/output/circuit_model.json`
- `ml_workspace/circuit_capacity/output/soc_rate_model_metrics.json`
- `ml_workspace/scheduling/optimize_with_windows.py`
- `ml_workspace/integration/export_snapshots.py`
- `frontend/scripts/live_model_outputs.py`
