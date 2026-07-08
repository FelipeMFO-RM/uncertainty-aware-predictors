"""Monte Carlo recursive rollout for cold-drawing surrogates (spec §4.3).

Lives at: src/modeling/CDRollout.py

Cold drawing applies surrogates pass after pass: the predicted state of
pass ``i`` is the input state of pass ``i+1``. Feeding only the mean forward
silently drops the per-pass variance, so this module runs ``K`` Monte Carlo
trajectories in parallel: at every pass, each surrogate's calibrated
(truncated) predictive distribution is SAMPLED per trajectory, and the
samples — not the means — feed the next pass. The empirical spread of the
K trajectories at pass ``n`` is therefore the honest, compounded predictive
distribution of the final state (copper_digital_twin_v4 §4.3).

Design notes
------------
* Same philosophy as ``MBCInference``: all statistics are delegated to
  ``Modeling`` (``predict_with_uncertainty``, ``sample_truncated``); this
  module only orchestrates. Bundles are the same ``SurrogateBundle`` objects
  the annealing controller loads.
* State chaining is detected STRUCTURALLY, mirroring
  ``detect_downstream_consumers`` in the annealing pipeline. For a surrogate
  with short name ``s`` (target ``{s}_final``):
    - a feature named ``s`` or ``initial_{s}`` in ANY bundle is a
      previous-pass state feature fed by ``s``'s prediction at pass i-1
      (e.g. ``initial_tensile_strength`` <- previous ``tensile_strength_final``);
    - a feature named ``{s}_final`` in ANOTHER bundle is a same-pass upstream
      feed (e.g. grain size predicted first, consumed by IACS within the
      same pass), which also fixes the within-pass execution order.
  Nothing is hardcoded; retrain any surrogate with or without these
  features and the rollout adapts. Explicit overrides are supported.
* One batched ``predict_with_uncertainty`` call per surrogate per pass
  (K rows), so the cost is ``n_passes x n_surrogates`` predictor calls per
  route/wire — negligible for tree ensembles.

Deviation from the spec worth stating: §4.3 resamples a base learner per
trajectory (epistemic) and injects aleatoric noise separately. Here each
trajectory samples from the full calibrated predictive law
N(mu, sigma_epist^2 + sigma_aleat^2) (truncated to the physical support),
which propagates both components jointly per pass and stays consistent with
the single-stage propagation already used in the annealing chain.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ColdDrawingRollout:
    """MC recursive rollout: evaluation on real wires + MBC route scoring.

    Parameters
    ----------
    modl : Modeling
        The uncertainty-aware ``Modeling`` instance (predict_with_uncertainty,
        sample_truncated).
    """

    def __init__(self, modl) -> None:
        self.modl = modl

    # ------------------------------------------------------------------
    # Structural detection of chained state features
    # ------------------------------------------------------------------
    @staticmethod
    def detect_state_features(
        bundles: dict,
        overrides: dict | None = None,
    ) -> tuple[dict, dict, list]:
        """Map chained features to producing surrogates; order the pass.

        Parameters
        ----------
        bundles : dict[str, SurrogateBundle]
            Surrogates keyed by alias (e.g. ``{"uts": ..., "iacs": ...}``).
        overrides : dict or None
            Optional explicit ``{feature_name: producing_bundle_key}`` for
            previous-pass state features, merged over the structural
            detection (use when a dataset departs from the naming
            conventions).

        Returns
        -------
        prev_pass_map : dict[str, str]
            ``{feature_name -> bundle_key}``: features overwritten at pass
            i>1 by the producing surrogate's pass i-1 SAMPLE.
        same_pass_map : dict[str, str]
            ``{feature_name -> bundle_key}``: features filled within the
            CURRENT pass by an upstream surrogate's sample (feature name is
            the upstream target, e.g. ``grain_size_final``).
        order : list[str]
            Bundle keys in within-pass execution order (upstream first).
        """
        produced_prev: dict[str, str] = {}
        produced_same: dict[str, str] = {}
        for key, b in bundles.items():
            s = b.short_name
            produced_prev[s] = key
            produced_prev[f"initial_{s}"] = key
            produced_same[b.target] = key

        prev_pass_map: dict[str, str] = {}
        same_pass_map: dict[str, str] = {}
        edges: dict[str, set] = defaultdict(set)  # upstream -> {downstream}

        for key, b in bundles.items():
            for f in b.features:
                if f in produced_same and produced_same[f] != key:
                    same_pass_map[f] = produced_same[f]
                    edges[produced_same[f]].add(key)
                elif f in produced_prev:
                    prev_pass_map[f] = produced_prev[f]

        if overrides:
            prev_pass_map.update(overrides)

        # Kahn topological sort on same-pass edges (upstream first).
        indeg = {k: 0 for k in bundles}
        for up, downs in edges.items():
            for d in downs:
                indeg[d] += 1
        queue = deque(sorted(k for k, v in indeg.items() if v == 0))
        order: list[str] = []
        while queue:
            k = queue.popleft()
            order.append(k)
            for d in sorted(edges.get(k, ())):
                indeg[d] -= 1
                if indeg[d] == 0:
                    queue.append(d)
        if len(order) != len(bundles):
            raise ValueError(
                "Cyclic same-pass dependency between surrogates: "
                f"{ {k: sorted(v) for k, v in edges.items()} }"
            )

        for f, k in prev_pass_map.items():
            logger.info("State feature '%s' <- previous-pass prediction of '%s'.", f, k)
        for f, k in same_pass_map.items():
            logger.info("Same-pass feature '%s' <- upstream surrogate '%s'.", f, k)
        logger.info("Within-pass execution order: %s", order)
        return prev_pass_map, same_pass_map, order

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _predict_bundle(self, bundle, X: pd.DataFrame) -> pd.DataFrame:
        """Raw calibrated predictive distribution for one bundle (thin plumb).

        Same kwargs plumbing as ``MBCInference.predict_surrogate``, kept raw
        (un-prefixed ``mu`` / ``sigma_total`` columns) because the rollout
        samples from it immediately.
        """
        art = bundle.artifacts
        y_min = art.get("y_min")
        y_max = art.get("y_max")
        return self.modl.predict_with_uncertainty(
            X=X,
            predictor_a=bundle.predictor_a,
            predictor_b=bundle.predictor_b,
            weights=art["weights"],
            model_names=art["base_model_names"],
            features=bundle.features,
            variance_floor=art["variance_floor"],
            recalibration_c=art["recalibration_c"],
            use_weights=art.get("use_weighted_variance", True),
            y_min=-np.inf if y_min is None else y_min,
            y_max=np.inf if y_max is None else y_max,
            ood_ref=art.get("ood_ref"),
        )

    def predict_bundle(self, bundle, X: pd.DataFrame) -> pd.DataFrame:
        """Public alias of the raw per-bundle predictive call.

        Handy as the ``predict_fn`` adapter for the mean-chained
        ``Evaluation.rollout_*`` family (which expects a callable returning
        ``mu`` / ``sigma_total`` columns).
        """
        return self._predict_bundle(bundle, X)

    def _sample_bundle(self, bundle, mu, sigma, rng) -> np.ndarray:
        """One truncated sample per row (per trajectory) from N(mu, sigma)."""
        art = bundle.artifacts
        y_min = art.get("y_min")
        y_max = art.get("y_max")
        samples = self.modl.sample_truncated(
            mu=np.asarray(mu), sigma=np.asarray(sigma), n_samples=1,
            y_min=-np.inf if y_min is None else y_min,
            y_max=np.inf if y_max is None else y_max,
            rng=rng,
        )
        return np.asarray(samples).reshape(-1)

    @staticmethod
    def resolve_initial_state(
        prev_pass_map: dict,
        first_row: pd.Series | None = None,
        fixed_state: dict | None = None,
        init_state: dict | None = None,
    ) -> dict:
        """Value of every previous-pass state feature at pass 1.

        Resolution order per feature ``f`` (short name ``s`` when
        ``f == initial_{s}``): explicit ``init_state[f]``; the true value in
        ``first_row`` (evaluation mode); ``fixed_state[f]``;
        ``fixed_state["original_{s}"]`` (the CDHelper convention: at pass 1
        the initial state IS the original one). Raises listing what is
        missing otherwise.
        """
        fixed_state = fixed_state or {}
        init_state = init_state or {}
        out: dict = {}
        missing = []
        for f in prev_pass_map:
            s = f[len("initial_"):] if f.startswith("initial_") else f
            if f in init_state:
                out[f] = init_state[f]
            elif first_row is not None and f in first_row.index:
                out[f] = first_row[f]
            elif f in fixed_state:
                out[f] = fixed_state[f]
            elif f"original_{s}" in fixed_state:
                out[f] = fixed_state[f"original_{s}"]
            else:
                missing.append(f)
        if missing:
            raise KeyError(
                f"Cannot resolve pass-1 value for state features {missing}. "
                f"Provide them in init_state, or as original_<name> in "
                f"material_properties/fixed_state."
            )
        return out

    def _roll_passes(
        self,
        bundles: dict,
        pass_frames: list[pd.DataFrame],
        state0: dict,
        prev_pass_map: dict,
        same_pass_map: dict,
        order: list,
        k_samples: int,
        rng,
    ) -> tuple[list[dict], dict]:
        """Core loop shared by evaluation and route modes.

        ``pass_frames`` is one single-row DataFrame per pass carrying every
        NON-chained feature (geometry + fixed descriptors). Returns per-pass
        summaries and the final-state samples per bundle key.
        """
        state = {f: np.full(k_samples, float(v)) for f, v in state0.items()}
        per_pass: list[dict] = []
        samples_now: dict[str, np.ndarray] = {}

        for i, frame in enumerate(pass_frames):
            X = pd.concat([frame] * k_samples, ignore_index=True)
            for f, vals in state.items():
                X[f] = vals
            samples_now = {}
            summary: dict = {"pass_index": i + 1}
            if "pass_number" in frame.columns:
                summary["pass_number"] = float(frame["pass_number"].iloc[0])

            for key in order:
                b = bundles[key]
                # same-pass upstream feeds first (already sampled this pass)
                for f, up_key in same_pass_map.items():
                    if f in b.features:
                        X[f] = samples_now[up_key]
                missing = [f for f in b.features if f not in X.columns]
                if missing:
                    raise KeyError(
                        f"Surrogate '{key}' needs features {missing} absent "
                        f"from the pass frame/state. Provide them in the "
                        f"route geometry or fixed_state."
                    )
                preds = self._predict_bundle(b, X)
                mu = preds["mu"].to_numpy()
                sigma = preds["sigma_total"].to_numpy()
                y_k = self._sample_bundle(b, mu, sigma, rng)
                samples_now[key] = y_k
                s = b.short_name
                summary[f"{s}_mu"] = float(np.mean(y_k))
                summary[f"{s}_sigma"] = float(np.std(y_k, ddof=1))
                summary[f"{s}_sigma_model_mean"] = float(np.mean(sigma))

            # previous-pass state update for the NEXT pass
            for f, prod_key in prev_pass_map.items():
                state[f] = samples_now[prod_key]
            per_pass.append(summary)

        finals = {k: v.copy() for k, v in samples_now.items()}
        return per_pass, finals

    # ------------------------------------------------------------------
    # Mode 1 — evaluation on real wires (df with true geometry + truth)
    # ------------------------------------------------------------------
    def rollout_group_mc(
        self,
        bundles: dict,
        group_df: pd.DataFrame,
        k_samples: int = 200,
        seed: int = 0,
        sort_column: str = "pass_number",
        overrides: dict | None = None,
    ) -> pd.DataFrame:
        """MC rollout of one wire: TRUE geometry per pass, predicted state fed.

        Pass 1 initialises the state features from the wire's TRUE first
        row; from pass 2 on, state features carry the sampled predictions.
        Truth columns (``{target}``) are attached when present, yielding a
        tidy frame directly comparable to ``Evaluation.rollout_all_groups``
        (same ``mu_pred/sigma_pred/abs_error/in_90_interval`` semantics),
        in long format with a ``surrogate`` column.
        """
        prev_map, same_map, order = self.detect_state_features(bundles, overrides)
        ordered = group_df.sort_values(sort_column).reset_index(drop=True)
        chained = set(prev_map) | set(same_map)
        pass_frames = [
            ordered.iloc[[i]].drop(columns=[c for c in chained if c in ordered.columns])
            for i in range(len(ordered))
        ]
        state0 = self.resolve_initial_state(prev_map, first_row=ordered.iloc[0])
        rng = np.random.default_rng(seed)
        per_pass, _ = self._roll_passes(
            bundles, pass_frames, state0, prev_map, same_map, order,
            k_samples, rng,
        )

        z90 = 1.6448536269514722
        rows = []
        for i, summ in enumerate(per_pass):
            for key in order:
                s = bundles[key].short_name
                target = bundles[key].target
                mu = summ[f"{s}_mu"]; sg = summ[f"{s}_sigma"]
                rec = {
                    "pass_index": summ["pass_index"],
                    "pass_number": summ.get("pass_number", summ["pass_index"]),
                    "surrogate": s,
                    "mu_pred": mu,
                    "sigma_pred": sg,
                }
                if target in ordered.columns:
                    y = float(ordered.iloc[i][target])
                    rec["y_true"] = y
                    rec["abs_error"] = abs(y - mu)
                    rec["in_90_interval"] = float(mu - z90 * sg <= y <= mu + z90 * sg)
                rows.append(rec)
        return pd.DataFrame(rows)

    def rollout_all_groups_mc(
        self,
        bundles: dict,
        df: pd.DataFrame,
        group_col: str = "group_experiment_id",
        k_samples: int = 200,
        seed: int = 0,
        sort_column: str = "pass_number",
        overrides: dict | None = None,
    ) -> pd.DataFrame:
        """MC rollout over every wire in ``df`` (long format + group_id)."""
        frames = []
        for j, (gid, gdf) in enumerate(df.groupby(group_col, sort=False)):
            roll = self.rollout_group_mc(
                bundles, gdf, k_samples=k_samples, seed=seed + j,
                sort_column=sort_column, overrides=overrides,
            )
            roll["group_id"] = gid
            frames.append(roll)
        out = pd.concat(frames, ignore_index=True)
        logger.info(
            "MC rollout: %d wires, K=%d, surrogates=%s.",
            df[group_col].nunique(), k_samples, sorted({b.short_name for b in bundles.values()}),
        )
        return out

    # ------------------------------------------------------------------
    # Mode 2 — MBC: score candidate routes from the CDGraph enumeration
    # ------------------------------------------------------------------
    def rollout_route_mc(
        self,
        bundles: dict,
        route_df: pd.DataFrame,
        fixed_state: dict,
        init_state: dict | None = None,
        k_samples: int = 200,
        seed: int = 0,
        overrides: dict | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """MC rollout of ONE candidate route (rows = passes of one sequence).

        ``route_df`` comes from ``CDGraph.generate_all_sequences_df`` (one
        sequence's rows, reset_index'ed: pass_number, original/initial/final
        diameter, step, total_strain, reduction_ratio). ``fixed_state``
        carries every non-geometry, non-chained feature (purity,
        original_tensile_strength, ...). Returns (per-pass summary frame,
        final-state samples per bundle key) — the shared trajectories in the
        finals enable a JOINT feasibility probability across targets.
        """
        prev_map, same_map, order = self.detect_state_features(bundles, overrides)
        route = route_df.sort_values("pass_number").reset_index(drop=True)
        pass_frames = []
        for i in range(len(route)):
            frame = route.iloc[[i]].copy()
            for k, v in fixed_state.items():
                frame[k] = v
            pass_frames.append(frame.reset_index(drop=True))
        state0 = self.resolve_initial_state(
            prev_map, fixed_state=fixed_state, init_state=init_state,
        )
        rng = np.random.default_rng(seed)
        per_pass, finals = self._roll_passes(
            bundles, pass_frames, state0, prev_map, same_map, order,
            k_samples, rng,
        )
        return pd.DataFrame(per_pass), finals

    def predict_sequences(
        self,
        bundles: dict,
        sequences_df: pd.DataFrame,
        fixed_state: dict,
        init_state: dict | None = None,
        k_samples: int = 200,
        seed: int = 0,
        overrides: dict | None = None,
        log_every: int = 25,
    ) -> tuple[pd.DataFrame, dict]:
        """Score EVERY enumerated route probabilistically.

        Parameters
        ----------
        sequences_df : pd.DataFrame
            MultiIndex (sequence_id, pass_number) frame from
            ``CDGraph.generate_all_sequences_df``.

        Returns
        -------
        df_routes : pd.DataFrame
            One row per route: sequence_id, route (D0->...->Df string),
            n_passes, cum_reduction (sum of steps, mm), plus per-surrogate
            final ``{s}_mu`` / ``{s}_sigma``.
        finals : dict[int, dict[str, np.ndarray]]
            ``{sequence_id -> {bundle_key -> K final samples}}`` — shared
            trajectories, consumed by
            ``ProbabilisticEvaluation.add_route_success_probabilities`` for
            the joint chance constraint.
        """
        seq_ids = sequences_df.index.get_level_values("sequence_id").unique()
        rows, finals_all = [], {}
        for j, sid in enumerate(seq_ids, start=1):
            route = sequences_df.loc[sid].reset_index()
            per_pass, finals = self.rollout_route_mc(
                bundles, route, fixed_state, init_state,
                k_samples=k_samples, seed=seed + int(sid), overrides=overrides,
            )
            diams = [route["initial_diameter"].iloc[0], *route["final_diameter"].tolist()]
            rec = {
                "sequence_id": sid,
                "route": " -> ".join(f"{d:g}" for d in diams),
                "n_passes": len(route),
                "cum_reduction": float(route["step"].sum()),
            }
            last = per_pass.iloc[-1]
            for key, b in bundles.items():
                s = b.short_name
                rec[f"{s}_final_mu"] = last[f"{s}_mu"]
                rec[f"{s}_final_sigma"] = last[f"{s}_sigma"]
            rows.append(rec)
            finals_all[int(sid)] = finals
            if j % log_every == 0 or j == len(seq_ids):
                logger.info("Scored %d / %d routes (K=%d).", j, len(seq_ids), k_samples)
        return pd.DataFrame(rows), finals_all
