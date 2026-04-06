# ML Workspace README

This directory contains the full modeling workflow that took the project from noisy aircraft telemetry to an operational battery-health forecasting stack for the Pipistrel Velis Electro fleet data in this repo.

This README is written as a process document first, not just a file index. The goal is to explain:

- what we did
- why we did it
- what each experiment taught us
- how those lessons changed the next stage of the pipeline
- why the final SOH modeling and forecasting approach looks the way it does

## Final Outcome

The final modeling stance in this workspace is:

1. Do not trust the raw BMS SOH channel as ground truth.
2. Clean event labels before doing any health analysis.
3. Treat charge events as diagnostically useful, but not as direct evidence of smooth degradation.
4. Build a condition-aware latent SOH label with a Kalman filter and RTS smoother.
5. Use the causal filter output, not the smoother, for forecasting targets to avoid future leakage.
6. Forecast event-based latent SOH at multiple flight horizons.
7. Convert predicted SOH into operational quantities such as circuits and estimated flight minutes through a calibrated circuit-capacity layer.

In short: the final solution is a cleaned, uncertainty-aware, event-level latent SOH pipeline followed by multi-horizon SOH forecasting and a downstream circuit-capacity model.

## Workspace Structure

- `EDA/`: exploratory notebooks used to audit label quality, inspect SOH behavior, and understand feature effects
- `battery_inference/`: charging-based battery-type identification for the aircraft
- `soh_estimation/`: charge-event physics-based SOH estimation and sanity-check comparisons
- `latent_soh/`: canonical latent SOH label generation with condition-aware Kalman filtering and smoothing
- `soh_forecast/`: feature engineering, train/validation/test split logic, and multi-horizon forecasting models
- `circuit_capacity/`: translation from SOH to usable circuits and SOC burn rate
- `integration/`: export of committed ML artifacts for the frontend/runtime path
- `scheduling/`: degradation-aware scheduling and planning experiments

## 1. Exploratory Data Analysis and Data Trust Audit

This was the first major stage because the early data did not behave like a clean degradation dataset.

### What we did

We used the notebooks in `EDA/` to inspect the raw event stream from multiple angles:

- `EDA/soh_visualization.ipynb`
- `EDA/observed_soh_event_timeseries.ipynb`
- `EDA/soh_vs_soc_throughput.ipynb`
- `EDA/soh_vs_soc_bands.ipynb`
- `EDA/soh_without_charging_and_charge_event_diagnostics.ipynb`
- `EDA/Mislabelling/event_direction_label_audit.ipynb`

The EDA work focused on two views of the same system:

- time-series views of SOH over calendar time
- event-by-event views of SOH over flights/charges

That second framing turned out to be critical. Calendar-time plots showed that the signal was noisy. Event-level plots showed why: jumps were often tied to specific event types rather than physically plausible gradual battery fade.

### 1.1 Mislabelled events

One of the first discoveries was that some events were mislabeled as flight or charge.

The correction workflow is implemented in:

- `EDA/Mislabelling/create_corrected_event_parquets.py`

That script infers event direction directly from the auxiliary SOC traces by:

- reading per-pack SOC from the aux parquet rows
- computing start-of-event and end-of-event SOC medians
- measuring `delta_soc` for each battery event
- inferring whether the event behaved like charge, discharge, weak charge, weak discharge, flat, or mixed
- overwriting the manifest/timeseries event labels when SOC behavior clearly contradicted the original label

The logic is intentionally physical instead of metadata-driven. If SOC rises materially during the event, the event is charge-like. If SOC falls materially, it is discharge-like. That is a much better basis for downstream modeling than trusting a bad label column.

### What we learned

- Label quality was not good enough to treat the raw manifest as authoritative.
- Before modeling health, we had to fix the meaning of the events themselves.
- Once event direction was audited from SOC behavior, many of the strange SOH patterns made more sense.

This is why later pipelines are designed to prefer corrected event files when available:

- `data/event_manifest_corrected.parquet`
- `data/event_timeseries_corrected.parquet`

## 2. Why Raw SOH Could Not Be Used as Ground Truth

The next question was whether the aircraft-reported SOH could be used directly as the training label.

EDA showed three recurring problems:

- implausibly large upward jumps in SOH
- strong event-type dependence, especially around charging
- sensitivity near SOC extremes and estimator reset conditions

These issues are still visible in the committed diagnostics for plane `166`.

From `latent_soh/output/plane_166/diagnostics/smoother_summary.json`:

- raw total variation is `484.0` and `469.0` SOH points for the two batteries
- raw maximum upward jumps are `29.0` and `25.0` SOH points

Those magnitudes are not credible as true electrochemical recovery. They indicate that the BMS SOH stream is partly a reporting/estimation channel, not a pure physics measurement.

### What we learned

- The raw SOH signal contains real information, but it is not a reliable supervised-learning target by itself.
- Any final SOH model had to explicitly account for observation uncertainty.
- The health pipeline needed to separate "what the aircraft reported" from "what we believe the underlying battery health probably was."

## 3. Why Charge Events Were Treated Differently

One of the biggest EDA findings was that charging events behaved differently from flight events.

The notebooks repeatedly suggested that charge events were associated with:

- abrupt positive SOH jumps
- higher trust issues near top-of-charge
- likely recalibration or internal estimator-reset behavior in the BMS

This is reflected directly in the latent diagnostics. The spike analysis in the committed writeup shows that spike events were dominated by charge events, and many of the biggest jumps occurred when `soc_max_pct` was near `99-100`.

### Important nuance

We did not simply throw away charging data everywhere.

Instead, we changed how we used it:

- we stopped treating charge-event SOH jumps as direct evidence of smooth battery degradation
- we used charging data for battery inference and charge-physics capacity estimation
- we still kept charge events in the event timeline for forecasting context
- we down-weighted charge-event observations in the latent SOH measurement model when they looked unstable

So the lesson was not "charging events are useless." The lesson was "charging events are useful, but the raw SOH reported during/around them is not trustworthy enough to be treated naively."

### What we learned

- Charge events are where the raw SOH estimator most obviously shows recalibration-like behavior.
- For health modeling, charge events needed special handling, not blind inclusion.
- For battery inference, charge events were still extremely valuable because they expose partial capacity information and top-of-charge voltage behavior.

## 4. Battery Inference: Determining Which Velis Electro Battery We Had

Before we could interpret nominal behavior correctly, we needed to know which battery configuration the aircraft actually had.

The issue was that the Velis Electro handbook gave two plausible battery options, and the plane would have one of them. That uncertainty matters because rated capacity and top-of-charge behavior affect every physics-based interpretation of the telemetry.

This workflow lives in:

- `battery_inference/infer_battery_type.py`
- `battery_inference/README.md`

### What we did

We used charging-event telemetry to estimate effective capacity from partial charge windows.

The script:

- loads charging events from the event parquet
- cleans each event to keep only physically valid charging rows
- requires strong SOC monotonicity
- selects a usable SOC window such as `40-90`, `30-80`, or `50-90`
- computes delivered amp-hours over that window
- scales that partial-charge amount back to an estimated full-event capacity
- uses top-of-charge voltage as a cross-check

This is a pragmatic inference method: it does not assume the entire charge event is ideal, but it does extract enough consistent information from many charge traces to distinguish between the candidate battery families.

### Result for plane 166

From `battery_inference/output/plane_166/plane_battery_inference.json`:

- inferred battery type: `PB345V119E-L`
- confidence: `0.80`
- median effective capacity: `26.13 Ah`
- capacity IQR: `2.16 Ah`
- median top-of-charge voltage: `404.2 V`
- valid event-battery segments: `261`

The capacity estimate was much closer to the `29 Ah` family than the `33 Ah` alternative, and the voltage cross-check supported the same conclusion.

### What we learned

- The plane is much more consistent with the `PB345V119E-L` battery family.
- Charge telemetry was good enough to identify pack family with useful confidence, even if it was not precise enough to recover a perfect rated-capacity estimate on every event.
- This gave us a defensible battery assumption for the later SOH estimation and operational modeling stages.

## 5. Physics-Based SOH from Charge Events

Once we knew how to interpret the battery more credibly, we built a direct charge-physics SOH estimator.

This work lives in:

- `soh_estimation/physics_based_soh.py`
- `soh_estimation/physics_based_soh_from_charge_events.ipynb`
- `soh_estimation/soh_capacity_vs_ica_anchor_comparison.ipynb`

### What we did

The charge-physics estimator uses coulomb counting during charging windows:

- `Q_delivered = integral(|I| dt) / 3600`
- `C_hat = Q_delivered / (delta_SOC / 100)`

Then SOH can be formed in two ways:

- absolute SOH: estimated capacity divided by rated capacity
- relative SOH: estimated capacity divided by a battery-specific early-life reference capacity

The goal here was not just to create another SOH number. It was to ask whether physically motivated capacity estimates told a similar long-term story as the black-box BMS SOH.

### What we found

The notebook and script results showed:

- absolute capacity-based SOH worked better than reference-normalized SOH
- the charge-derived series was directionally useful, but still noisy
- it was not stable enough to replace the final latent label

The committed writeup reports for plane `166`:

- absolute capacity SOH: `MAE = 7.904`, `RMSE = 9.919`, `R^2 = 0.106`
- relative/reference-normalized SOH: `MAE = 10.422`, `RMSE = 12.183`, `R^2 = -0.348`

### What we learned

- Charge-event physics can recover a meaningful degradation trend, but not a production-ready ground truth by itself.
- It is more useful as a sanity check and interpretation layer than as the canonical label.
- This supported the next decision: we needed a latent-state approach that explicitly modeled observation noise and uncertainty.

## 6. ICA and Cross-Method Sanity Checking

The repo also keeps the comparison notebook:

- `soh_estimation/soh_capacity_vs_ica_anchor_comparison.ipynb`

This was used as a convergence sanity check across different SOH estimation ideas.

### Why this mattered

We did not want to commit to one method only because it was mathematically convenient. We wanted to see whether multiple physically motivated or semi-physical methods pointed in the same general direction.

The methods compared were conceptually:

- raw/black-box observed SOH behavior
- charge-capacity-derived SOH
- ICA-anchored or charge-shape-informed SOH reasoning
- latent-state SOH

### What we learned

- Different methods did not match event-by-event, but they did converge on the same overall degradation direction.
- That convergence increased confidence that the latent trend was not an artifact of smoothing alone.
- ICA-style work was valuable diagnostically, but the available aircraft telemetry was not rich or clean enough to make ICA the production estimator.

This is why the final pipeline uses latent SOH as the canonical label, while keeping charge/ICA work as supporting evidence.

## 7. Canonical SOH Label: Condition-Aware Latent State Modeling

This is the core modeling contribution in the workspace.

The canonical implementation lives in:

- `latent_soh/build_latent_soh.py`
- `latent_soh/condition_noise.py`
- `latent_soh/state_space.py`
- `latent_soh/latent_soh_walkthrough.ipynb`
- `latent_soh/README.md`

### Why a latent approach was necessary

The aircraft telemetry gave us noisy, jumpy, event-dependent SOH observations. A latent state model lets us separate:

- the hidden battery-health trajectory we actually care about
- the noisy observation channel produced by the BMS

This is a standard aerospace-style idea: when the observable channel is imperfect, estimate the hidden system state and carry uncertainty forward instead of pretending the raw measurement is exact.

### State-space formulation

The model is an event-level 1D state-space system:

- state model: latent SOH follows a random walk with process variance controlled by `q_day_sigma_pct`
- measurement model: observed BMS SOH is a noisy measurement of latent SOH

The important choice is that measurement noise is not fixed. It is made condition-aware and event-specific.

### 7.1 How uncertainty is modeled

`condition_noise.py` computes several risk scores from each event, including:

- current severity
- `dI/dt`
- `dT/dt`
- SOC edge behavior
- observed SOH instability within the event
- disagreement in coulomb/Kalman gap features
- estimator switch/reset flags
- event type
- missingness

Those scores are combined into a `condition_multiplier`, which scales the measurement sigma:

- low-trust events get larger `R_t`
- stable events get smaller `R_t`

This is the critical mechanism that tells the filter when to trust the observation and when to mostly ignore it.

The current committed forecasting pipeline uses:

- `rt_profile = current`
- `q_day_sigma_pct = 0.10`

### 7.2 Filtering and smoothing

The latent pipeline writes both:

- `latent_soh_filter_pct`
- `latent_soh_smooth_pct`

These are not interchangeable:

- `latent_soh_filter_pct` is causal and only uses information available up to that event
- `latent_soh_smooth_pct` is retrospective and uses future observations through RTS smoothing

The smoother is ideal for analysis because it reconstructs a cleaner global degradation trajectory. The filter is the safe label for forecasting because it does not leak future information.

### 7.3 What the results show

For plane `166`, from `latent_soh/output/plane_166/diagnostics/smoother_summary.json`:

- total events modeled: `1106`
- raw total variation per battery: `484.0` and `469.0`
- smoothed total variation per battery: `46.89` and `45.48`
- raw max upward jumps: `29.0` and `25.0`
- smoothed max upward jumps: `0.701` and `0.725`

That is a huge reduction in implausible volatility while preserving the long-run degradation trend.

### What we learned

- The latent state approach solved the central problem of the project: the raw SOH signal was too noisy to use directly.
- Explicitly modeling uncertainty was not optional. It was the reason the final label became credible.
- The smoothed curve is the best retrospective explanation of battery health, but the filtered curve is the correct training label for prediction.

## 8. Leakage Prevention and Forecast-Safe Labeling

Once we had a good latent label, the next issue was preventing future leakage.

This is handled explicitly in:

- `soh_forecast/feature_pipeline.py`

### What we did

The feature pipeline computes both smoothed and causal next-step targets, but then intentionally overwrites any smoothed next-label field with the causal target.

In practice:

- forecasting targets are built from `latent_soh_filter_pct`
- lag and rolling features use `shift(1)` or past-only windows
- train/validation/test splits are chronological
- the holdout plane is excluded from model selection

### Why this matters

RTS smoothing uses future events. If we trained a forecast model against smoothed future labels without guarding against that, the benchmark would look better than reality.

The project explicitly avoids that trap.

### What we learned

- Good labels are not enough; they also need to be causally valid for prediction.
- Leakage prevention had to be designed into the pipeline, not added later as an afterthought.
- This made the forecasting results much more defensible.

## 9. SOH Forecasting

The forecasting workspace lives in:

- `soh_forecast/train_horizon_models.py`
- `soh_forecast/feature_pipeline.py`
- `soh_forecast/models/README.md`
- `soh_forecast/models/`

### Forecasting design choice

The forecast problem is event-based, not purely calendar-based.

That means each training example represents an operational event for a specific battery, with features describing:

- the current latent SOH state
- event type
- current, temperature, SOC, and duration summaries
- condition-noise diagnostics
- throughput, thermal, storage, and resistance stress proxies
- cumulative usage
- lags and rolling history

This framing was chosen because battery degradation is driven by operations, not just the passage of time.

### 9.1 Multi-horizon targets

The default committed horizons are:

- next flight
- next 5 flights
- next 10 flights
- next 15 flights
- next 20 flights

These are defined in `DEFAULT_MULTI_HORIZON_CONFIGS` in `feature_pipeline.py`.

### 9.2 Training approach for time-series data

This was not trained as a random IID regression problem.

The pipeline uses chronological splits and walk-forward evaluation:

- expanding walk-forward backtests for model selection
- a final untouched in-plane test block
- an optional sparse holdout plane for out-of-distribution checking

The default training script uses:

- `split_scheme = walk_forward`
- `backtest_folds = 3`
- `final_test_frac = 0.15`
- `fold_valid_frac = 0.10`

So the training protocol mirrors how the model would actually be used: fit on earlier life, validate on later life, and test on the future.

### 9.3 Model families compared

The project compared multiple model classes:

- naive zero-delta baseline
- ridge regression
- elastic net
- GAM/spline regression
- histogram gradient boosting
- random forest
- LSTM sequence models
- GRU sequence models
- physics-hybrid neural models
- physics-informed neural models

This breadth mattered because we did not know in advance whether the dataset would reward:

- simple linear structure
- nonlinear tabular models
- sequence memory
- or explicit physics-inspired latent parameterization

### 9.4 What we found

From `soh_forecast/output/multihorizon_runner_plane_166/best_models_by_horizon.csv`:

- `latent_flight_1`: best model `lstm_sequence_no_latent_tune3`, backtest mean level MAE `0.192`
- `latent_flight_5`: best model `lstm_sequence_no_latent_tune0`, backtest mean level MAE `0.664`
- `latent_flight_10`: best model `gru_sequence_with_latent_tune0`, backtest mean level MAE `1.140`
- `latent_flight_15`: best model `gru_sequence_with_latent_tune1`, backtest mean level MAE `1.165`
- `latent_flight_20`: best model `gru_sequence_with_latent_tune2`, backtest mean level MAE `1.187`

### Interpretation

- Very short-horizon forecasting is credible.
- Accuracy degrades as the horizon expands.
- Sequence models ended up outperforming most tabular baselines in the committed benchmark winners.
- Longer-range deterministic battery-life forecasting remains much less certain than short-term operational forecasting.

### What we learned

- The latent label unlocked predictive signal that was hidden by raw SOH noise.
- Event history matters, which is why LSTM/GRU models did well.
- The current dataset supports stronger claims for short-horizon SOH forecasting than for long-horizon life prediction.

## 10. Circuit Capacity Model: Turning SOH into Operational Usefulness

The final modeling stage was not just "predict SOH." It was "translate SOH into something operationally meaningful."

That work lives in:

- `circuit_capacity/fit_circuit_model.py`
- `circuit_capacity/predict_circuits.py`
- `circuit_capacity/circuit_capacity_model_demo.ipynb`

### Why this was needed

A predicted SOH percentage is useful to a data scientist, but it is not the most natural quantity for operators.

Operators want answers like:

- how many circuits can the aircraft still fly?
- how much usable time is left in a session?
- how will those capabilities decline as SOH degrades?

So we built a translation layer from health to operations.

### 10.1 How the circuit model works

The model combines:

- a POH-based relationship between SOH and SOC cost per circuit
- a per-plane calibration factor
- a reserve SOC threshold

The committed `circuit_model.json` uses:

- SOH grid: `[0, 20, 40, 60, 80, 100]`
- POH SOC-per-circuit grid: `[20, 16, 13, 12, 10, 9]`
- reserve SOC: `30%`
- plane factor `k_plane`: `0.80` for plane `166`, `0.74` for plane `192`

Operationally, that means lower SOH implies higher SOC burn per circuit, which reduces the number of feasible circuits before the reserve limit is hit.

### 10.2 SOC-rate regression

The workspace also trains a direct SOC burn-rate model for flight events.

From `circuit_capacity/output/soc_rate_model_metrics.json`:

- test MAE: `0.192` SOC-%/min
- test RMSE: `0.263`
- test `R^2 = 0.555`
- OOD `R^2 = 0.541`

This is one of the stronger predictive components in the stack because SOC burn is more directly observable than SOH.

### What we learned

- A good SOH pipeline becomes much more useful once it is connected to a capacity model.
- The latent SOH signal can be converted into practical future capability estimates, not just an abstract health score.
- This is what allows the frontend and planner to talk about future circuits and approximate flight endurance instead of only reporting SOH.

## 11. End-to-End Research Progression

The project did not start with the final latent forecast model. It evolved through the following sequence:

1. Audit the data and discover that some event labels are wrong.
2. Visualize raw SOH in time and by event and realize the signal is not trustworthy as-is.
3. Notice that charging events behave differently and likely trigger recalibration-like BMS behavior.
4. Use charging telemetry to infer which battery family the aircraft most likely has.
5. Build a physics-based charge SOH estimator and compare it to other methods.
6. Use cross-method agreement as a sanity check that the long-run trend is real.
7. Build a latent state model to convert noisy SOH observations into a credible canonical health label.
8. Enforce causal target construction to avoid future leakage.
9. Train and benchmark multiple time-series forecasting models on the latent label.
10. Convert forecasted SOH into future operational capacity through the circuit-capacity model.

That progression is important because every stage answered a different uncertainty:

- Is the data labeled correctly?
- Is the raw SOH trustworthy?
- What battery are we actually modeling?
- Can physics-based estimates recover the same degradation direction?
- How do we estimate health when observations are noisy?
- How do we predict the future without leakage?
- How do we make those predictions useful for real aircraft operations?

## 12. Final Modeling Approach We Stand Behind

The final approach this workspace supports is:

- corrected event labels
- event-level telemetry features
- condition-aware latent SOH estimation with FilterPy Kalman filtering and RTS smoothing
- causal latent filter output as the forecasting label
- walk-forward multi-horizon sequence forecasting
- SOH-to-circuit translation through a calibrated capacity model

Why this is the final approach:

- it is robust to noisy SOH observations
- it preserves uncertainty instead of ignoring it
- it reflects the operational event structure of the aircraft
- it avoids future leakage
- it produces outputs that can be consumed by the planner and frontend

## 13. Reproducing the Main Pipeline

### Rebuild latent SOH

```bash
python -m ml_workspace.latent_soh.build_latent_soh --plane-id 166 --rt-profile current --q-day-sigma-pct 0.10
python -m ml_workspace.latent_soh.build_latent_soh --plane-id 192 --rt-profile current --q-day-sigma-pct 0.10
```

### Run battery inference

```bash
python ml_workspace/battery_inference/infer_battery_type.py --plane-id 166
```

### Retrain forecasting models

```bash
python ml_workspace/soh_forecast/train_horizon_models.py
```

### Refit the circuit-capacity layer

```bash
python ml_workspace/circuit_capacity/fit_circuit_model.py --planes 166,192
```

## 14. Suggested Reading Order Inside This Folder

If someone new wants to understand the project in the right order, the best reading path is:

1. `EDA/`
2. `battery_inference/README.md`
3. `soh_estimation/`
4. `latent_soh/README.md`
5. `soh_forecast/models/README.md`
6. `circuit_capacity/`
7. `../TECHNICAL_WRITEUP.md`

That order mirrors the actual development logic of the capstone.

## 15. Main Limitations

The workspace is strong, but not perfect.

- Plane `192` is too small to be a strong standalone validation aircraft.
- Long-horizon forecasting is still much weaker than short-horizon forecasting.
- Charge-event and ICA methods were informative but not reliable enough to become the primary SOH label.
- The forecasting stack is event-based and operationally useful, but not a full electrochemical digital twin.

## 16. Short Version

The raw telemetry SOH was noisy, mislabeled data made it worse, and charging events exposed estimator recalibration effects that made naive modeling unsafe. We cleaned the labels, used charge data to infer the correct battery family, compared multiple SOH estimation ideas, and then adopted a condition-aware latent Kalman approach as the canonical health label. From there we trained leakage-safe multi-horizon event-based forecasters and translated predicted SOH into future circuits and estimated usable flight capability.
