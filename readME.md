# AeroCell Capstone

Battery-health intelligence and operations support for electric aircraft.

This repository combines telemetry processing, latent state-of-health estimation, multi-horizon SOH forecasting, circuit-capacity modeling, and a Next.js dashboard that turns the model outputs into operational views for students and operators.

## What This Repo Contains

- event-level telemetry built from aircraft scrape data in `data/`
- a condition-aware latent SOH pipeline in `ml_workspace/latent_soh/`
- battery-type and charge-event capacity inference in `ml_workspace/battery_inference/`
- SOH forecasting code and minimal runtime artifacts in `ml_workspace/soh_forecast/`
- circuit-capacity and SOC-rate models in `ml_workspace/circuit_capacity/`
- JSON snapshot export code in `ml_workspace/integration/`
- a frontend app in `frontend/`

## Repository Layout

- `data/`: checked-in parquet inputs used by the modeling and frontend snapshot pipelines
- `ml_workspace/latent_soh/`: canonical latent SOH label generation
- `ml_workspace/battery_inference/`: partial-charge capacity and battery-type inference
- `ml_workspace/soh_forecast/`: feature engineering and multi-horizon forecasting
- `ml_workspace/circuit_capacity/`: SOH-to-capacity translation and SOC-rate regression
- `ml_workspace/integration/`: ML-to-frontend snapshot export
- `frontend/`: Next.js application and API routes
- `tests/`: Python regression tests for forecasting split logic and planner behavior

## Quick Start

### 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Frontend dependencies

```bash
cd frontend
npm install
```

### 3. Build frontend snapshots

```bash
cd frontend
npm run snapshots
```

### 4. Run the app locally

```bash
cd frontend
npm run dev
```

The production container path is defined by [`Dockerfile`](Dockerfile) and [`render.yaml`](render.yaml).

## Common Workflows

Rebuild latent SOH tables:

```bash
python -m ml_workspace.latent_soh.build_latent_soh --plane-id 166 --rt-profile current --q-day-sigma-pct 0.10
python -m ml_workspace.latent_soh.build_latent_soh --plane-id 192 --rt-profile current --q-day-sigma-pct 0.10
```

Run battery inference:

```bash
python ml_workspace/battery_inference/infer_battery_type.py --plane-id 166
```

Retrain multi-horizon forecast artifacts:

```bash
python ml_workspace/soh_forecast/train_horizon_models.py
```

Refresh exported snapshots consumed by the frontend:

```bash
python ml_workspace/integration/export_snapshots.py --plane-ids 166,192
cd frontend
npm run snapshots
```

## Tests

Python regression tests:

```bash
python -m unittest tests.test_walk_forward tests.test_planner_scenarios
```

Frontend checks:

```bash
cd frontend
npm run lint
npm run build
```

## Checked-In Artifacts

The repo intentionally keeps only the generated artifacts needed to run the current demo and integration path:

- latent SOH event tables and diagnostics for planes `166` and `192`
- battery inference summaries for planes `166` and `192`
- circuit-capacity calibration and SOC-rate model artifacts
- tabular SOH forecast model bundles used by planner scripts
- multi-horizon prediction CSVs and horizon winner summaries used by frontend/export code

One-off notebooks, comparison sweeps, ad hoc benchmark dumps, and extra model checkpoints are intentionally excluded so the repo stays reviewable.

## Additional Docs

- [`TECHNICAL_WRITEUP.md`](TECHNICAL_WRITEUP.md): technical methodology, notebook findings, latent/ICA analysis, forecasting, RUL, and optimization details
- [`frontend/README.md`](frontend/README.md): frontend-specific commands and deployment notes
- [`ml_workspace/latent_soh/README.md`](ml_workspace/latent_soh/README.md): latent SOH method details
- [`ml_workspace/integration/README.md`](ml_workspace/integration/README.md): snapshot and API lineage
