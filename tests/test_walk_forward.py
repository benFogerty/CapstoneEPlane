from __future__ import annotations

import unittest

import pandas as pd

from ml_workspace.soh_forecast.benchmarking import select_backtest_winner, summarize_backtest_fold_metrics
from ml_workspace.soh_forecast.feature_pipeline import EVENT_GROUP_KEY_COLS, assign_walk_forward_splits


class WalkForwardSplitTests(unittest.TestCase):
    def _build_dataset(self) -> pd.DataFrame:
        rows = []
        for plane_id, n_groups in [("166", 20), ("192", 4)]:
            for group_idx in range(n_groups):
                event_dt = pd.Timestamp("2024-01-01 08:00:00") + pd.Timedelta(days=group_idx)
                for battery_id in (1, 2):
                    rows.append(
                        {
                            "plane_id": plane_id,
                            "battery_id": battery_id,
                            "battery_id_str": str(battery_id),
                            "flight_id": 1000 + group_idx,
                            "event_datetime": event_dt,
                            "event_id": f"{plane_id}_{battery_id}_{1000 + group_idx}_{event_dt.strftime('%Y%m%d%H%M%S')}",
                            "event_type": "flight",
                            "observed_soh_pct": 90.0 - group_idx - battery_id * 0.1,
                            "latent_soh_filter_pct": 89.5 - group_idx - battery_id * 0.1,
                            "next_latent_soh_causal_flight_1_pct": 89.0 - group_idx - battery_id * 0.1,
                        }
                    )
        return pd.DataFrame(rows)

    def test_walk_forward_assignments_are_group_consistent(self) -> None:
        df = self._build_dataset()
        assigned = assign_walk_forward_splits(
            df,
            primary_plane="166",
            holdout_plane="192",
            final_test_frac=0.15,
            backtest_folds=3,
            fold_valid_frac=0.10,
            required_target_cols=["next_latent_soh_causal_flight_1_pct"],
        )

        grouped = assigned.groupby(EVENT_GROUP_KEY_COLS, dropna=False)
        self.assertTrue((grouped["final_split"].nunique() == 1).all())
        self.assertTrue((grouped["refit_role"].nunique() == 1).all())
        for fold_id in range(1, 4):
            self.assertTrue((grouped[f"backtest_fold_{fold_id}_role"].nunique() == 1).all())

        primary = assigned.loc[assigned["plane_id"].eq("166")].copy()
        primary_groups = primary[EVENT_GROUP_KEY_COLS + ["final_split", "refit_role", "backtest_fold_1_role", "backtest_fold_2_role", "backtest_fold_3_role"]].drop_duplicates()
        primary_groups = primary_groups.sort_values(["event_datetime", "flight_id"]).reset_index(drop=True)

        final_test = primary_groups.loc[primary_groups["final_split"].eq("final_test")]
        self.assertGreater(len(final_test), 0)
        self.assertLess(primary_groups.loc[primary_groups["final_split"].eq("train_dev"), "event_datetime"].max(), final_test["event_datetime"].min())

        for fold_id in range(1, 4):
            role_col = f"backtest_fold_{fold_id}_role"
            fold_train = primary_groups.loc[primary_groups[role_col].eq("train")]
            fold_valid = primary_groups.loc[primary_groups[role_col].eq("valid")]
            self.assertGreater(len(fold_train), 0)
            self.assertGreater(len(fold_valid), 0)
            self.assertLess(fold_train["event_datetime"].max(), fold_valid["event_datetime"].min())
            self.assertLess(fold_valid["event_datetime"].max(), final_test["event_datetime"].min())

        holdout = assigned.loc[assigned["plane_id"].eq("192")]
        self.assertTrue(holdout["final_split"].eq("holdout").all())
        self.assertTrue(holdout["refit_role"].eq("holdout").all())


class BacktestWinnerTests(unittest.TestCase):
    def test_backtest_winner_prefers_lower_mean_then_std_then_delta(self) -> None:
        fold_metrics = pd.DataFrame(
            [
                {"model": "model_a", "eval_split": "valid", "fold_id": 1, "n": 10, "level_mae": 0.20, "level_rmse": 0.21, "level_r2": 0.9, "delta_mae": 0.10, "delta_rmse": 0.11, "delta_r2": 0.8},
                {"model": "model_a", "eval_split": "valid", "fold_id": 2, "n": 10, "level_mae": 0.24, "level_rmse": 0.25, "level_r2": 0.9, "delta_mae": 0.12, "delta_rmse": 0.13, "delta_r2": 0.8},
                {"model": "model_b", "eval_split": "valid", "fold_id": 1, "n": 10, "level_mae": 0.21, "level_rmse": 0.22, "level_r2": 0.9, "delta_mae": 0.09, "delta_rmse": 0.10, "delta_r2": 0.8},
                {"model": "model_b", "eval_split": "valid", "fold_id": 2, "n": 10, "level_mae": 0.23, "level_rmse": 0.24, "level_r2": 0.9, "delta_mae": 0.08, "delta_rmse": 0.09, "delta_r2": 0.8},
                {"model": "model_c", "eval_split": "valid", "fold_id": 1, "n": 10, "level_mae": 0.22, "level_rmse": 0.23, "level_r2": 0.9, "delta_mae": 0.05, "delta_rmse": 0.06, "delta_r2": 0.8},
                {"model": "model_c", "eval_split": "valid", "fold_id": 2, "n": 10, "level_mae": 0.22, "level_rmse": 0.23, "level_r2": 0.9, "delta_mae": 0.04, "delta_rmse": 0.05, "delta_r2": 0.8},
            ]
        )
        summary = summarize_backtest_fold_metrics(fold_metrics)
        winner = select_backtest_winner(summary)
        self.assertIsNotNone(winner)
        self.assertEqual(winner["model"], "model_c")

    def test_backtest_winner_uses_delta_mae_as_final_tie_break(self) -> None:
        fold_metrics = pd.DataFrame(
            [
                {"model": "model_x", "eval_split": "valid", "fold_id": 1, "n": 10, "level_mae": 0.20, "level_rmse": 0.21, "level_r2": 0.9, "delta_mae": 0.08, "delta_rmse": 0.09, "delta_r2": 0.8},
                {"model": "model_x", "eval_split": "valid", "fold_id": 2, "n": 10, "level_mae": 0.20, "level_rmse": 0.21, "level_r2": 0.9, "delta_mae": 0.08, "delta_rmse": 0.09, "delta_r2": 0.8},
                {"model": "model_y", "eval_split": "valid", "fold_id": 1, "n": 10, "level_mae": 0.20, "level_rmse": 0.21, "level_r2": 0.9, "delta_mae": 0.05, "delta_rmse": 0.06, "delta_r2": 0.8},
                {"model": "model_y", "eval_split": "valid", "fold_id": 2, "n": 10, "level_mae": 0.20, "level_rmse": 0.21, "level_r2": 0.9, "delta_mae": 0.05, "delta_rmse": 0.06, "delta_r2": 0.8},
            ]
        )
        summary = summarize_backtest_fold_metrics(fold_metrics)
        winner = select_backtest_winner(summary)
        self.assertIsNotNone(winner)
        self.assertEqual(winner["model"], "model_y")


if __name__ == "__main__":
    unittest.main()
