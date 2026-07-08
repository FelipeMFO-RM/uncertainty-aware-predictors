"""Evaluation utilities for one-step and multi-step rollout predictions.

Lives at: src/modeling/Evaluation.py

Designed for chained / recursive surrogates (cold drawing pass-by-pass)
but the one-step part is reusable for single-pass surrogates too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


class Evaluation:
    """Metrics for one-step and multi-step rollout evaluation.

    One-step metrics evaluate each row independently using real inputs.
    Multi-step rollout metrics evaluate the model recursively, where each
    pass feeds the previous prediction back as input.
    """

    # =========================================================
    # One-step metrics (per-row)
    # =========================================================
    @staticmethod
    def one_step_metrics(
        y_true: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray | None = None,
        alphas: tuple = (0.5, 0.8, 0.9, 0.95),
    ) -> dict:
        """Standard regression metrics + optional calibration coverage.

        Parameters
        ----------
        y_true : (N,) array of true target values.
        mu : (N,) array of point predictions.
        sigma : (N,) array of predictive standard deviations.
                If None, coverage is not computed.
        alphas : nominal coverage levels for empirical coverage table.

        Returns
        -------
        dict with keys: rmse, mae, mape, r2, and coverage (if sigma given).
        """
        y_true = np.asarray(y_true, dtype=float)
        mu = np.asarray(mu, dtype=float)
        resid = y_true - mu

        rmse = float(np.sqrt(np.mean(resid ** 2)))
        mae = float(np.mean(np.abs(resid)))
        # MAPE: skip rows where y_true is zero
        nonzero = y_true != 0
        mape = float(
            np.mean(np.abs(resid[nonzero] / y_true[nonzero])) * 100.0
        ) if nonzero.any() else float("nan")

        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        out = {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}

        if sigma is not None:
            sigma = np.asarray(sigma, dtype=float)
            coverage = {}
            for a in alphas:
                z = norm.ppf(0.5 + a / 2.0)
                lo = mu - z * sigma
                hi = mu + z * sigma
                coverage[a] = float(
                    ((y_true >= lo) & (y_true <= hi)).mean()
                )
            out["coverage"] = coverage

        return out

    # =========================================================
    # Multi-step rollout (recursive)
    # =========================================================
    @staticmethod
    def rollout_single_group(
        group_df: pd.DataFrame,
        predict_fn,
        features: list,
        target: str,
        recursive_feature: str,
        sort_column: str = "pass_number",
    ) -> pd.DataFrame:
        """Run a recursive rollout on one group of sequential passes.

        At each pass:
          - the first pass uses the real input row;
          - subsequent passes overwrite the ``recursive_feature`` column
            with the previous pass's predicted target;
          - all other features keep their real values (they are
            controlled or observable inputs, not predicted ones).

        Parameters
        ----------
        group_df : the rows of one wire / experiment group, ordered.
        predict_fn : callable(X_df) -> DataFrame with columns
                     'mu', 'sigma_total' (and optionally 'sigma2_*').
                     This is the surrogate's predict_with_uncertainty.
        features : list of feature column names expected by predict_fn.
        target : name of the target column.
        recursive_feature : name of the feature that is replaced by the
                            previous pass's predicted target
                            (e.g. 'initial_tensile_strength' for UTS).
        sort_column : column used to order the passes within the group.

        Returns
        -------
        DataFrame with one row per pass containing:
            pass_number, y_true, mu_pred, sigma_pred,
            abs_error, in_90_interval.
        """
        ordered = group_df.sort_values(sort_column).reset_index(drop=True)
        n_passes = len(ordered)

        rows = []
        prev_pred = None  # holds μ̂ of the previous pass

        for i in range(n_passes):
            row = ordered.iloc[[i]].copy()

            # On all passes except the first, overwrite the recursive
            # input feature with the previous predicted target.
            if i > 0 and prev_pred is not None:
                row[recursive_feature] = prev_pred

            pred = predict_fn(row[features])
            mu = float(pred["mu"].iloc[0])
            sigma = float(pred["sigma_total"].iloc[0])
            y_true = float(ordered.iloc[i][target])
            err = abs(y_true - mu)

            z90 = norm.ppf(0.95)
            lo = mu - z90 * sigma
            hi = mu + z90 * sigma

            rows.append({
                "pass_index": i,
                "pass_number": float(ordered.iloc[i][sort_column]),
                "y_true": y_true,
                "mu_pred": mu,
                "sigma_pred": sigma,
                "abs_error": err,
                "in_90_interval": bool(lo <= y_true <= hi),
            })

            prev_pred = mu

        return pd.DataFrame(rows)

    @staticmethod
    def rollout_all_groups(
        df: pd.DataFrame,
        group_col: str,
        predict_fn,
        features: list,
        target: str,
        recursive_feature: str,
        sort_column: str = "pass_number",
    ) -> pd.DataFrame:
        """Run the recursive rollout on every group in df.

        Returns one tidy DataFrame concatenating all per-group rollouts,
        with an extra column ``group_id`` to identify the source group.
        """
        out_frames = []
        for gid, gdf in df.groupby(group_col):
            roll = Evaluation.rollout_single_group(
                gdf, predict_fn, features, target,
                recursive_feature, sort_column,
            )
            roll["group_id"] = gid
            out_frames.append(roll)
        return pd.concat(out_frames, ignore_index=True)

    # =========================================================
    # Multi-step aggregate metrics
    # =========================================================
    @staticmethod
    def final_state_metrics(
        rollout_df: pd.DataFrame,
        group_col: str = "group_id",
        alphas: tuple = (0.5, 0.8, 0.9, 0.95),
    ) -> dict:
        """Metrics computed on the LAST pass of each group only.

        This is what the controller cares about: the predicted final
        state of a candidate route vs. its true final state.
        """
        last = (
            rollout_df.sort_values(["group_id", "pass_index"])
                      .groupby(group_col)
                      .tail(1)
                      .reset_index(drop=True)
        )
        return Evaluation.one_step_metrics(
            y_true=last["y_true"].to_numpy(),
            mu=last["mu_pred"].to_numpy(),
            sigma=last["sigma_pred"].to_numpy(),
            alphas=alphas,
        )

    @staticmethod
    def pass_by_pass_curve(
        rollout_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Average rollout error and coverage by pass index.

        Useful to visualise compounding: if abs_error grows quickly with
        pass_index, the model has serious error compounding.
        """
        curve = rollout_df.groupby("pass_index").agg(
            mean_abs_error=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            mean_sigma=("sigma_pred", "mean"),
            coverage_90=("in_90_interval", "mean"),
            n_groups=("group_id", "nunique"),
        ).reset_index()
        return curve

    @staticmethod
    def compounding_ratio(
        rollout_metrics: dict,
        one_step_metrics: dict,
        n_passes_typical: int,
    ) -> float:
        """Ratio of final-state RMSE to one-step RMSE times sqrt(n).

        A value near 1.0 indicates errors are roughly independent across
        passes (best realistic case). Higher values indicate correlated
        errors compounding faster than independent random walk.
        """
        rmse_final = rollout_metrics["rmse"]
        rmse_one = one_step_metrics["rmse"]
        if rmse_one == 0:
            return float("nan")
        return rmse_final / (np.sqrt(n_passes_typical) * rmse_one)

    @staticmethod
    @staticmethod
    def one_step_metrics_by(
        df: pd.DataFrame,
        y_col: str,
        mu_col: str,
        sigma_col: str,
        by: str = "pass_number",
        alphas: tuple = (0.9,),
    ) -> pd.DataFrame:
        """One-step metrics grouped by a column (typically pass_number).

        The one-step counterpart of ``pass_by_pass_curve``: whereas the
        rollout curve shows how errors COMPOUND when predictions feed
        predictions, this table shows how the model performs per pass when
        every input is TRUE data. Comparing the two isolates the
        compounding contribution from the intrinsic per-pass difficulty.
        """
        rows = []
        for g, gdf in df.groupby(by):
            m = Evaluation.one_step_metrics(
                y_true=gdf[y_col].to_numpy(),
                mu=gdf[mu_col].to_numpy(),
                sigma=gdf[sigma_col].to_numpy(),
                alphas=alphas,
            )
            row = {by: g, "n_rows": len(gdf),
                   "rmse": m["rmse"], "mae": m["mae"]}
            for a in alphas:
                row[f"cov_{a}"] = m["coverage"][a]
            rows.append(row)
        return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)

    def metrics_on_dataframe(
        df: pd.DataFrame,
        target: str,
        mu: np.ndarray,
        sigma: np.ndarray | None = None,
        alphas: tuple = (0.5, 0.8, 0.9, 0.95),
    ) -> dict:
        """one_step_metrics taking the validation DataFrame directly.

        Target-agnostic convenience wrapper: pulls ``y_true`` from
        ``df[target]`` and forwards to ``one_step_metrics``. Works for
        IACS, UTS or any other surrogate as long as ``target`` names the
        ground-truth column in ``df``.

        Parameters
        ----------
        df : pd.DataFrame
            Validation set containing the ground-truth ``target`` column.
        target : str
            Name of the ground-truth column in ``df``.
        mu : (N,) array of point predictions (same order as ``df``).
        sigma : (N,) array of predictive std devs, or None to skip
            coverage.
        alphas : nominal coverage levels.

        Returns
        -------
        dict with keys: rmse, mae, mape, r2, and coverage (if sigma).
        """
        return Evaluation.one_step_metrics(
            y_true=df[target].to_numpy(dtype=float),
            mu=np.asarray(mu, dtype=float),
            sigma=sigma,
            alphas=alphas,
        )