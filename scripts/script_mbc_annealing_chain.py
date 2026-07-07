"""
Author : Felipe Matheus
Created: 06/2026

Uncertainty-Aware MBC Annealing Pipeline — WITH UPSTREAM PROPAGATION
===================================================================
Chained version of ``script_mbc_annealing_uncertainty.py``: a grain-size
surrogate predicts grain_size_final as a distribution, which is propagated
(Monte Carlo, copper_digital_twin_v4 §4.2) into the IACS surrogate so the
IACS prediction's uncertainty includes the upstream grain-size uncertainty.

For the proof-of-concept, the grain-size surrogate can be a DUMMY trained on
synthetic data (set ``train_dummy_grain_size: true`` in config). In production,
point ``grain_size_bundle_dir`` at the real grain-size bundle and set the flag
to false.

Run:
    python -m src.mbc.script_mbc_annealing_chain
Config: config/config_script_annealing_chain.yaml (schema in EXAMPLE_CONFIG).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.DataLoader import LoaderHelper
from src.modeling.Modeling import Modeling
from src.modeling.Evaluation import Evaluation
from src.mbc.MBCInference import MBCInference
from src.mbc.ProbabilisticEvaluation import ProbabilisticEvaluation
from src.mbc.DummySurrogate import DummySurrogate

CONFIG_PATH = "config/config_script_annealing_chain.yaml"
LOGGER = logging.getLogger("log_mbc_annealing_chain")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("mbc_annealing_chain.log"),
        ],
    )


def export_selections(selected, output_dir: Path, identifier: str) -> Path:
    rows = []
    for label, sub in selected.items():
        if len(sub):
            r = sub.iloc[0].copy()
            r["selection"] = label
            rows.append(r)
    export_df = pd.DataFrame(rows)
    keep = (["selection", "temperature", "time"]
            + [c for c in export_df.columns
               if c.endswith(("_mu", "_sigma")) or c.startswith("pr_success")
               or c in ("cost", "cost_phys", "iacs_lcb")])
    export_df = export_df[[c for c in keep if c in export_df.columns]]
    if "temperature" in export_df.columns:
        export_df["temperature_C"] = export_df["temperature"] - 273
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"mbc_ann_chain_{identifier}.csv"
    export_df.to_csv(out_path, index=False)
    return out_path


def run_pipeline(config_path: str = CONFIG_PATH) -> None:
    """Execute the chained (grain-size -> IACS) uncertainty-aware MBC."""
    setup_logging()

    loader = LoaderHelper()
    modl = Modeling()
    evla = Evaluation()
    mbc = MBCInference(modl, loader)
    pev = ProbabilisticEvaluation()

    LOGGER.info("Starting chained uncertainty-aware MBC annealing pipeline")
    config = loader.load_config(config_path)

    # --- Grain-size surrogate: dummy (synthetic) or real bundle -------------
    gs_dir = config["grain_size_bundle_dir"]
    if config.get("train_dummy_grain_size", False):
        LOGGER.info("Training DUMMY grain-size surrogate on synthetic data")
        dummy = DummySurrogate(modl, evla)
        df_gs = dummy.generate_data(
            n=config.get("dummy_n", 100), seed=config.get("dummy_seed", 7)
        )
        dummy.train_and_persist(
            df_gs, bundle_dir=gs_dir,
            presets=config.get("dummy_presets", "medium_quality"),
            time_limit_a=config.get("dummy_time_limit_a", 60),
            time_limit_b=config.get("dummy_time_limit_b", 30),
            y_min=0.0, y_max=None,
        )
    gs_bundle = mbc.load_surrogate(gs_dir)

    # --- Downstream surrogates (best run auto-picked from tag folders) ------
    LOGGER.info("Loading IACS/UTS surrogates (best run by RMSE)")
    down_bundles = mbc.load_surrogates_best(
        config["downstream_tag_dirs"], metric=config.get("best_metric", "rmse")
    )
    iacs_bundle = down_bundles["iacs"]
    uts_bundle = down_bundles.get("uts")

    # --- Grid ----------------------------------------------------------------
    param_grid = mbc.build_param_grid(
        config["operational_limits"], config["step"]
    )
    df_grid = mbc.build_state_grid_df(config["material_properties"], param_grid)
    LOGGER.info("Grid: %d cells", len(df_grid))

    # --- Propagate grain-size -> IACS (or predict directly) -----------------
    gs_feat = config["downstream_gs_feature"]
    if gs_feat in iacs_bundle.features:
        LOGGER.info("Propagating grain-size -> IACS (K=%d)",
                    config.get("k_samples", 200))
        df_prop = mbc.propagate_upstream_to_downstream(
            upstream=gs_bundle, downstream=iacs_bundle, df_grid=df_grid,
            downstream_feature=gs_feat,
            k_samples=config.get("k_samples", 200),
            ood_gamma=config.get("ood_gamma", 1.0), seed=config.get("seed", 0),
        )
    else:
        LOGGER.warning("IACS does not use '%s'; predicting IACS directly.", gs_feat)
        df_prop = mbc.predict_surrogate(iacs_bundle, df_grid)

    # --- UTS (predicted directly) -------------------------------------------
    if uts_bundle is not None:
        uts_preds = mbc.predict_surrogate(uts_bundle, df_grid)
        for c in [c for c in uts_preds.columns if c.startswith("uts_")]:
            df_prop[c] = uts_preds[c].to_numpy()

    # --- Probabilistic feasibility ------------------------------------------
    surrogate_bounds = {
        key: (b.artifacts.get("y_min"), b.artifacts.get("y_max"))
        for key, b in {"iacs": iacs_bundle, **(
            {"uts": uts_bundle} if uts_bundle is not None else {})}.items()
    }
    df_prob = pev.add_success_probabilities(
        df_prop, config["min_setpoints"],
        surrogate_bounds=surrogate_bounds, modl=modl,
    )
    delta = config.get("delta", 0.1)
    df_feasible = pev.apply_probabilistic_setpoints(df_prob, delta=delta)
    LOGGER.info("Feasible cells (Pr >= %.2f): %d", 1 - delta, len(df_feasible))
    if df_feasible.empty:
        LOGGER.warning("No feasible cell at delta=%.2f; relax delta/setpoints.", delta)
        return

    # --- Cost + selection ----------------------------------------------------
    df_cost = pev.add_cost_function(
        pev.add_cost_function_physics(df_feasible),
        operational_limits=config["operational_limits"],
    )
    for short in config["min_setpoints"]:
        df_cost = pev.add_risk_adjusted_value(
            df_cost, short, kappa=config.get("kappa", 1.645)
        )
    selected = {}
    if "selection_spec_low" in config:
        selected.update(
            pev.select_lowest_by_features(df_cost, config["selection_spec_low"])
        )
    if "selection_spec_high" in config:
        selected.update(
            pev.select_highest_by_features(df_cost, config["selection_spec_high"])
        )

    # --- Export --------------------------------------------------------------
    output_dir = Path(config["output_path"])
    out_path = export_selections(selected, output_dir, config["identifier"])
    LOGGER.info("Selections saved to %s", out_path)
    full = output_dir / f"mbc_ann_chain_{config['identifier']}_full_grid.csv"
    df_cost.to_csv(full, index=False)
    LOGGER.info("Full feasible grid saved to %s", full)
    LOGGER.info("Pipeline finished successfully.")


EXAMPLE_CONFIG = """
# config/config_script_annealing_chain.yaml
identifier: grain_size_chain_pilot
output_path: data/enriched/mbc_annealing_chain

# --- grain-size surrogate (upstream) ---
train_dummy_grain_size: true                       # false to use a real bundle
grain_size_bundle_dir: models/annealing_grain_size/dummy_uncertainty_aware
dummy_n: 100
dummy_seed: 7

# --- downstream surrogates: TAG folders, best run auto-picked ---
downstream_tag_dirs:
  iacs: models/annealing_iacs/experiments/annealing-v1-best-quality
  uts:  models/annealing_uts/experiments/annealing-uts-v1-best_quality
best_metric: rmse

# Name of the IACS feature fed by the grain-size target.
downstream_gs_feature: grain_size

material_properties:
  purity: 99.95
  iacs: 98.47
  initial_diameter: 1.2
  initial_grain_size: 45.0
  tensile_strength: 250.0

operational_limits:
  temperature: [523, 723]
  time: [30, 120]
step:
  temperature: 10
  time: 5

min_setpoints:
  iacs: 99.5
  uts: 210.0

delta: 0.10
kappa: 1.645
k_samples: 200
ood_gamma: 1.0
seed: 0

selection_spec_low:
  lowest_time: time
  lowest_temperature: temperature
  lowest_cost: cost
  lowest_cost_phys: cost_phys
selection_spec_high:
  safest_iacs: pr_success_iacs
  safest_combined: pr_success_all
"""


if __name__ == "__main__":
    run_pipeline()
