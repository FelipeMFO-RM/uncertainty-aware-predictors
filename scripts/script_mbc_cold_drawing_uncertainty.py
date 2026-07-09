"""
Author : Felipe Matheus
Created: 07/2026

Uncertainty-Aware MBC Cold-Drawing Pipeline — probabilistic route selection
===========================================================================
Probabilistic counterpart of ``script_mbc_cold-drawing.py`` (deterministic).
Same combinatorial skeleton — enumerate every admissible drawing route
D0 -> Df on the ported DAG (steps in S, reduction ratio <= rr_max,
<= max_passes) — with the predictive core replaced:

- Each route is scored by a K-trajectory MONTE CARLO ROLLOUT
  (``ColdDrawingRollout.predict_sequences``): every pass samples the
  calibrated truncated predictive distribution and the SAMPLES feed the
  next pass (copper_digital_twin_v4 §4.3), so the final-state uncertainty
  compounds honestly.
- Feasibility is the chance constraint Pr(final state meets ALL setpoints)
  >= 1 - delta, computed JOINTLY from the shared trajectories (no
  independence assumption between targets).
- Ranking: highest joint Pr(success), then fewest passes, then smallest
  cumulative die reduction — the deterministic criteria, risk-aware.
- State chaining (initial_tensile_strength <- previous pass; grain size ->
  IACS/UTS within a pass) is detected STRUCTURALLY from the bundles'
  features; add bundles to the config and the rollout adapts by itself.

Shared-method policy: every step calls the same ``ColdDrawingRollout`` /
``ProbabilisticEvaluation`` methods as
``notebooks/experiments/mbc_cold_drawing_uncertainty.ipynb`` — fix a method
once, both entry points are fixed.

Run:
    python script_mbc_cold_drawing_uncertainty.py

Config: config/config_script_cold_drawing_uncertainty.yaml (see
EXAMPLE_CONFIG at the bottom of this file for the expected schema).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.DataLoader import LoaderHelper
from src.modeling.Modeling import Modeling
from src.modeling.MBCInference import MBCInference
from src.modeling.CDRollout import ColdDrawingRollout
from src.modeling.CDGraph import ColdDrawingMBCHelper
from src.metrics.ProbabilisticEvaluation import ProbabilisticEvaluation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROCESS = "cold_drawing"
CONFIG_PATH = "config/config_script_cold_drawing_uncertainty.yaml"

LOGGER = logging.getLogger("log_mbc_cold_drawing_uncertainty")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_logging() -> None:
    """Configure root logger with console and file handlers."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("mbc_cold_drawing_uncertainty.log"),
        ],
    )


def log_config_snapshot(config: dict) -> None:
    """Log a concise snapshot of the most relevant config values."""
    seq = config["sequence_generation"]
    LOGGER.info(
        "material=%s | D0=%s -> Df=%s | steps=%s | rr_max=%s | "
        "min_setpoints=%s | delta=%s | k_samples=%s",
        config["material_properties"],
        seq["original_diameter"], seq["final_diameter"], seq["steps"],
        seq["rr_max"], config["min_setpoints"],
        config.get("delta", 0.1), config.get("k_samples", 200),
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(config_path: str = CONFIG_PATH) -> None:
    """Execute the uncertainty-aware cold-drawing route-selection pipeline."""
    setup_logging()

    loader = LoaderHelper()
    modl = Modeling()
    mbc = MBCInference(modl, loader)
    roll = ColdDrawingRollout(modl)
    pev = ProbabilisticEvaluation()

    LOGGER.info("Starting uncertainty-aware cold-drawing MBC pipeline")
    LOGGER.info("Loading config from %s", config_path)
    config = loader.load_config(config_path)
    log_config_snapshot(config)

    # --- Load surrogate bundles + fail-fast setpoint validation --------------
    LOGGER.info("Loading surrogate bundles")
    metric = config.get("best_metric", "rmse")
    bundles = mbc.load_surrogates_best(config["bundle_dirs"], metric=metric)
    max_setpoints = config.get("max_setpoints") or None
    targets = pev.validate_setpoints(
        bundles, config["min_setpoints"], max_setpoints=max_setpoints,
    )

    # --- Structural state-chaining detection (audit trail) -------------------
    prev_map, same_map, order = roll.detect_state_features(bundles)
    LOGGER.info("State features (prev-pass): %s", prev_map)
    LOGGER.info("Same-pass upstream feeds : %s", same_map)
    LOGGER.info("Within-pass order        : %s", order)

    # --- Enumerate admissible routes (ported deterministic graph) ------------
    seq = config["sequence_generation"]
    LOGGER.info("Enumerating admissible routes")
    sequences_df = ColdDrawingMBCHelper.generate_all_sequences_df(
        original_diameter=seq["original_diameter"],
        final_diameter=seq["final_diameter"],
        steps=seq["steps"],
        rr_max=seq["rr_max"],
        max_passes=seq.get("max_passes"),
        max_sequences=seq.get("max_sequences"),
    )
    n_routes = sequences_df.index.get_level_values("sequence_id").nunique()
    LOGGER.info("%d admissible routes (D0=%s -> Df=%s)",
                n_routes, seq["original_diameter"], seq["final_diameter"])
    if n_routes == 0:
        LOGGER.warning("No admissible route under the given steps/rr_max. Stop.")
        return

    # --- Score every route: K-trajectory Monte Carlo rollout -----------------
    LOGGER.info("Scoring routes with K=%d Monte Carlo trajectories",
                config.get("k_samples", 200))
    df_routes, finals = roll.predict_sequences(
        bundles, sequences_df,
        fixed_state=config["material_properties"],
        init_state=config.get("init_state") or None,
        k_samples=config.get("k_samples", 200),
        seed=config.get("seed", 0),
    )

    # --- Joint chance-constrained feasibility on the final state -------------
    LOGGER.info("Computing joint success probabilities (chance constraints)")
    df_prob = pev.add_route_success_probabilities(
        df_routes, finals, config["min_setpoints"], bundles,
        max_setpoints=max_setpoints,
    )
    delta = config.get("delta", 0.1)
    df_feasible = pev.apply_probabilistic_setpoints(df_prob, delta=delta)
    LOGGER.info("Feasible routes (Pr >= %.2f): %d / %d",
                1 - delta, len(df_feasible), len(df_prob))

    if df_feasible.empty:
        max_pr = float(df_prob["pr_success_all"].max()) if len(df_prob) else float("nan")
        LOGGER.warning(
            "No route satisfies the chance constraint at delta=%.2f "
            "(max Pr over routes was %.3f). Relax delta, widen steps/rr_max, "
            "or revisit setpoints/models. Exporting the ranked full set for "
            "inspection instead.", delta, max_pr,
        )
        df_feasible = df_prob  # export everything, ranked, for the audit

    # --- Rank: safest first, then simplest ------------------------------------
    ranked = pev.rank_routes(df_feasible)
    top = ranked.iloc[0]
    LOGGER.info("Top route: %s | passes=%d | Pr(all)=%.3f",
                top["route"], int(top["n_passes"]), float(top["pr_success_all"]))

    # --- Export (shared method with the notebook) ------------------------------
    output_dir = Path(config["output_path"])
    out_path = pev.export_routes(
        ranked, output_dir, config["identifier"], targets=targets,
        top_n=config.get("top_n_export"),
    )
    if out_path is not None:
        LOGGER.info("Routes saved to %s", out_path)

    # full audit dump (every scored route, ranked, with probabilities)
    full_path = output_dir / Path(config["identifier"]) / "mbc_cd_unc_all_routes.csv"
    pev.rank_routes(df_prob).to_csv(full_path, index=False)
    LOGGER.info("Full audit grid saved to %s", full_path)

    LOGGER.info("Pipeline finished successfully.")


# ---------------------------------------------------------------------------
# Example config (write to config/config_script_cold_drawing_uncertainty.yaml)
# ---------------------------------------------------------------------------

EXAMPLE_CONFIG = """
# config/config_script_cold_drawing_uncertainty.yaml
identifier: cd_routes_pilot
output_path: data/enriched/mbc_cold_drawing_uncertainty_script_results

# Downstream bundles: run dir OR TAG folder (best run auto-picked by metric).
# ALL THREE surrogates active; comment out any bundle not trained yet — the
# rollout detects the state chaining structurally and degrades gracefully.
bundle_dirs:
  grain_size: models/cold_drawing_grain_size/experiments/cd-gs-v1-best_quality
  iacs: models/cold_drawing_iacs/experiments/cd-iacs-v1-best_quality
  uts: models/cold_drawing_uts/experiments/cd-uts-v1-best_quality

# Fixed material state: every non-geometry, non-chained feature the
# surrogates need. original_<name> doubles as the pass-1 fallback for the
# chained state features (initial_tensile_strength <- original_tensile_strength,
# initial_iacs <- original_iacs, grain_size <- original_grain_size).
material_properties:
  purity: 99.9
  original_tensile_strength: 280.0
  original_iacs: 100.5
  original_grain_size: 40.0   # canonical um scalar (synthesized from *_equivalent_circle_diameter)

# Optional explicit pass-1 state overrides ({feature: value}); usually empty
# because the original_* fallback covers it.
init_state: {}

# Route generation (mirrors the deterministic config_script_cold-drawing.yaml)
sequence_generation:
  original_diameter: 2.0    # mm
  final_diameter: 1.2       # mm
  steps: [0.1, 0.2, 0.3, 0.4]
  rr_max: 80.0              # % per pass
  max_passes: 20
  max_sequences: 2000

# Chance constraints on the FINAL state (short_name -> bound)
min_setpoints:
  iacs: 99.0
  tensile_strength: 445.0
max_setpoints: {}           # e.g. {grain_size: 30.0} for a refinement spec

delta: 0.10        # keep routes with Pr(all setpoints) >= 1 - delta
k_samples: 200     # Monte Carlo trajectories per route
seed: 0
best_metric: rmse  # metric used to auto-pick the best run from a TAG folder
top_n_export: 20   # rows in the stakeholder CSV (full audit dump is separate)
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
