# Copper Alloy Digital Twin

> Uncertainty-aware machine learning surrogates for sequential copper alloy treatment processes — rotary swaging, cold drawing, and annealing.

[![Status](https://img.shields.io/badge/status-active--development-yellow)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue)]()

---

## Overview

This project builds a probabilistic **surrogate model** (predictor, ML supervised model) to support the **model-based controller** (digital twin) that chooses process parameters to hit target material properties during a chain of three industrial treatment steps. In production, intermediate measurements are not always available — only a final quality check after annealing — so the controller must reason about the entire process chain in advance.

For each process, three surrogate machine learning models predict three material properties (grain size, electrical conductivity in %IACS, and ultimate tensile strength in MPa). Unlike [deterministic surrogate models](https://github.com/FelipeMFO-RM/predictors/tree/cold_drawing_test_mbc), **every surrogate here returns a full predictive distribution** with explicit decomposition between epistemic uncertainty (model ignorance, reducible with more data) and aleatoric uncertainty (irreducible process noise).

## Why uncertainty-aware?

TLDR; low data -> high uncertainty, we want to at least know and be honest about how far from real life we are (especially during the whole digital twin process). Plus data augmentation techniques and assessment of how to improve models.

A deterministic model trained on ~100 rows can hide significant variation behind a single MAE number. By making uncertainty explicit and calibrated, the controller can:

- Rank candidate process routes by **probability of success** instead of binary feasibility, following the setpoints on the MBC phase.
- Identify regions of the input space where more data would actually help (high epistemic uncertainty) vs. regions limited by process noise (high aleatoric uncertainty). During the surrogate model training phase.
- Propagate uncertainty across chained surrogates instead of pretending the output of one model is a precise input to the next. That can be useful especially on cold-drawing phase.

## Architecture

TLDR; A model will predict the value of the setpoint variable using many other models, their discordance will be statistically interpreted as an epistemic error. A second model will learn how to infer the aleatoric error (if not that smaller than the epistemic); and in the end we obtain a distribution of the prediction instead of a single value.

```
                          ┌────────────────────────┐
                          │      Model A           │
   features  ──────────►  │  AutoGluon ensemble    │  ──►  μ̂(x) + σ²_epist(x)
                          │  K-fold bagging        │
                          └────────────────────────┘
                                      │
                          ┌────────────────────────┐
                          │   OOF residuals        │
                          │   r̃² = max(r² − σ²_e,0)│
                          └────────────────────────┘
                                      │
                          ┌────────────────────────┐
   features  ──────────►  │      Model B           │  ──►  σ²_aleat(x)
                          │  AutoGluon (log-space) │
                          └────────────────────────┘

                  Final predictive distribution per surrogate:
                  y | x  ~  N( μ̂(x),  σ²_epist(x) + σ²_aleat(x) )
```

The mean and epistemic variance come from the AutoGluon `WeightedEnsemble`, whose weights are recovered via **non-negative least squares** against the out-of-fold predictions. The predictive distribution is **calibrated** with a scalar `σ → c·σ`, **bounded** to the physical support of each property via a truncated Gaussian (e.g. the %IACS ceiling), and **inflated out-of-domain** via a Mahalanobis distance so inputs outside the training cloud get honestly wider intervals.

For chained surrogates (grain size → IACS within annealing, cold drawing → annealing across steps, or pass-by-pass within cold drawing), the upstream uncertainty is propagated with **Monte Carlo sampling**: draw K=200 realisations from the upstream predictive distribution, run the downstream model on each, aggregate the empirical distribution of outputs. This is what keeps the downstream model from being overconfident about an input it actually only knows approximately.

## The model-based controller (MBC)

Once surrogates emit calibrated distributions, the controller sweeps a dense (temperature, time) grid for a fixed material state and, per cell:

1. predicts every target's distribution (`MBCInference`),
2. optionally propagates an upstream surrogate's uncertainty into a downstream one,
3. keeps only cells satisfying the **chance constraint** `Pr(Y ≥ setpoint) ≥ 1 − δ` (`ProbabilisticEvaluation`),
4. ranks the survivors by operational cost (and, optionally, a risk-aware lower confidence bound `μ − κσ`),
5. exports the chosen operating points.

This replaces the deterministic controller's binary `pred ≥ min` test with a probabilistic one, and is the `P(y ≥ y_target) ≥ 1 − δ` format the digital twin is graded on.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── config/
│   └── Variables.py                          # YAML-backed config accessor
├── data/
│   └── raw/                                   # one CSV per surrogate
├── models/                                    # persisted AutoGluon bundles + experiment logs
├── notebooks/
│   ├── modeling/                              # reference per-surrogate pipelines
│   │   ├── annealing_iacs_fixed.ipynb         #   <- reference implementation
│   │   ├── annealing_uts_fixed.ipynb
│   │   └── cold_drawing_uts_*.ipynb
│   ├── experiments/                           # grid sweeps + controller
│   │   ├── run_experiments_iacs.ipynb
│   │   ├── run_experiments_uts.ipynb
│   │   └── mbc_annealing_uncertainty.ipynb
│   ├── exploration/                           # EDA
│   └── deprecated/                            # superseded notebooks
├── scripts/
│   └── script_mbc_annealing_uncertainty.py    # stakeholder-runnable controller
└── src/
    ├── modeling/
    │   ├── Modeling.py                        # core: all uncertainty-aware statistics
    │   ├── Evaluation.py                      # regression + calibration metrics, rollouts
    │   ├── Experiments.py                     # config-driven training runner + logging
    │   └── MBCInference.py                    # bundle loading, grid, Monte Carlo propagation
    ├── metrics/
    │   └── ProbabilisticEvaluation.py         # chance constraints, cost, selection
    ├── processing/Processing.py
    ├── feature_engineering/
    │   ├── FeatureEngineering.py              # labelling, stratified folds, augmentation
    │   └── Scaler.py                          # deterministic path only
    ├── visualization/Plots.py
    ├── DataLoader.py                          # LoaderHelper: pickle/model loading
    └── DataDumper.py
```

A trained surrogate is persisted as a self-contained **bundle**:

```
<bundle_dir>/
  artifacts.pkl   # weights, base_model_names, recalibration_c, variance_floor,
                  # ood_ref, y_min, y_max, calibration tables
  model_a/        # AutoGluon predictor (mean + epistemic)
  model_b/        # AutoGluon predictor (aleatoric variance, log-space)
```

A full developer-facing walkthrough of every module and method is in
[`docs/code_architecture.pdf`](docs/code_architecture.tex).

## Quick start

### Installation

```bash
git clone <repo-url>
cd uncertainty-aware-predictors
python -m venv .venv
source .venv/bin/activate                       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Train and calibrate a surrogate

Open the reference notebook:

```bash
jupyter lab notebooks/modeling/annealing_iacs_fixed.ipynb
```

The notebook walks through the full pipeline: setup, data loading, Model A training, OOF extraction, ensemble weight recovery via NNLS, epistemic variance, aleatoric targets, Model B training, calibration diagnostics, a validation-set evaluation, inference with physical constraints, persistence, and a deployment-simulation that reloads the bundle from disk.

All hyperparameters are in a single configuration cell at the top — adapting to a new surrogate is typically a one-cell edit:

```python
TARGET = "iacs_final"
FEATURES = ["purity", "iacs", "temperature", "time"]
PRESETS_A = "medium_quality"
NUM_BAG_FOLDS_A = 5
USE_WEIGHTED_VARIANCE = True
VARIANCE_FLOOR_FRAC = 0.01
RECALIBRATION_TARGET_ALPHA = 0.9
Y_MAX = 106.0          # physical ceiling for %IACS (None for unbounded targets)
USE_SHARED_FOLDS = True
```

### Sweep many configurations

For grid sweeps over time limits, features, fold counts, etc., use the experiment runner instead of editing cells by hand:

```bash
jupyter lab notebooks/experiments/run_experiments_iacs.ipynb
```

```python
from src.modeling.Experiments import ExperimentConfig, ExperimentRunner

runner = ExperimentRunner(modl, evla, models_root="../../models")
base = ExperimentConfig(process="annealing_iacs", tag="annealing-v1")

grid = {"time_limit_a": [120, 300], "num_bag_folds_a": [5, 10]}
log = runner.run_grid(df, base, grid, df_val=df_val)   # results in experiments_log.csv
```

Each run lands in its own `models/<process>/experiments/<tag>/<run_id>/` folder, and `list_of_the_best.csv` ranks every run per metric with a direct path to its model directory.

### Predict with uncertainty

```python
from src.modeling.Modeling import Modeling
from autogluon.tabular import TabularPredictor
import pickle, pandas as pd

modl = Modeling()
predictor_a = TabularPredictor.load("models/annealing_iacs/<run>/model_a")
predictor_b = TabularPredictor.load("models/annealing_iacs/<run>/model_b")
art = pickle.load(open("models/annealing_iacs/<run>/artifacts.pkl", "rb"))

new_data = pd.DataFrame({
    "purity": [99.95],
    "iacs": [98.0],
    "temperature": [573],
    "time": [60],
})

preds = modl.predict_with_uncertainty(
    X=new_data,
    predictor_a=predictor_a,
    predictor_b=predictor_b,
    weights=art["weights"],
    model_names=art["base_model_names"],
    features=art["features"],
    variance_floor=art["variance_floor"],
    recalibration_c=art["recalibration_c"],
    use_weights=art["use_weighted_variance"],
    y_min=art.get("y_min"),
    y_max=art.get("y_max"),
    ood_ref=art.get("ood_ref"),
)
print(preds)
#         mu  sigma2_epist  sigma2_aleat  sigma2_total  sigma_total  ood_multiplier  ...
# 0   95.32          0.84          1.43          2.27         1.51             1.0
```

### Run the controller

For a fixed material state, sweep the grid and export the best operating points:

```bash
jupyter lab notebooks/experiments/mbc_annealing_uncertainty.ipynb
# or, headless, for stakeholders:
python -m scripts.script_mbc_annealing_uncertainty
```

## Key concepts

**Upstream / downstream.** Within a process step, the grain-size surrogate is upstream of IACS and UTS surrogates (its prediction is their feature). Across steps, post-cold-drawing state is upstream of annealing. Errors propagate, and the Monte Carlo step propagates them honestly.

**Out-of-fold (OOF) predictions.** Honest estimates of generalization error: a prediction made by a model that did not see that row during training. Required for residuals that drive Model B.

**Weighted vs. uniform statistics.** AutoGluon's `WeightedEnsemble_L2` assigns non-uniform weights to base learners via the Caruana et al. (2004) ensemble selection algorithm. The pipeline recovers these weights via non-negative least squares and uses them for both the mean and the epistemic variance. The flag `use_weighted_variance` toggles to uniform statistics as a robustness check.

**Calibration.** A predictive interval at nominal level α is calibrated when the true value falls inside it on a fraction ≈ α of test points. The pipeline includes diagnostics and a scalar recalibration step (σ → c·σ) to correct mild over/underconfidence.

**Physical constraints.** Each property has a physical support (e.g. %IACS ≤ ~106). The predictive Gaussian is truncated to that support at inference via closed-form moments — training is untouched, only the output distribution is reshaped. The `p_above_max` diagnostic flags cells whose prediction leans heavily on the ceiling.

**Out-of-domain inflation.** Tree ensembles extrapolate flat and overconfidently. A Mahalanobis reference fitted on the training inputs inflates `σ_epist` for inputs outside the training cloud (`σ` grows smoothly, capped), so the controller sees wider intervals — and lower `Pr(success)` — exactly where the model knows least.

**Chance constraints (δ) and risk ranking (κ).** The controller keeps cells where `Pr(Y ≥ setpoint) ≥ 1 − δ` (δ is the risk tolerance), then ranks survivors — optionally by a lower confidence bound `μ − κσ` (κ prefers robust guarantees) or directly by `Pr(success)`.

**MAE vs. σ.** MAE and RMSE are tracked alongside calibration. They are not replaced by the predictive distribution — they answer different questions. A typical σ should match the RMSE in order of magnitude; if not, the calibration step adjusts it.

## Coding standards

- Python 3.11+, PEP 8, line length 100.
- Type hints on public methods.
- Docstrings in NumPy style.
- Logging via the `logging` module, not `print` (notebook examples excluded).
- `pathlib.Path` over `os.path`.
- AutoGluon imports inside functions (heavy import cost).
- Reusable statistical helpers go into the `Modeling` class in `src/modeling/Modeling.py`; controller helpers into `MBCInference` / `ProbabilisticEvaluation`.

## Status

| Process       | IACS          | UTS           | Grain size    |
|---------------|---------------|---------------|---------------|
| Annealing     | ✅ Calibrated | ✅ Calibrated | 🧪 Dummy (proof-of-concept) |
| Cold drawing  | ⏳ Pending    | 🚧 In progress | ⏳ Pending   |
| Rotary swaging| —             | —             | —             |

Annealing IACS is the reference implementation; the remaining surrogates follow the same template via the configuration cell. The grain-size surrogate currently exists only as a synthetic dummy used to validate the upstream→downstream Monte Carlo propagation.

## Roadmap

- [x] Annealing IACS surrogate (Model A + Model B + calibration)
- [x] Annealing UTS surrogate
- [x] Config-driven experiment runner with logging and best-run tracking
- [x] Physical truncation (IACS ceiling) and out-of-domain σ inflation
- [x] Probabilistic annealing controller (grid search over T, t, chance constraints)
- [x] Monte Carlo propagation across chained surrogates (single-stage)
- [ ] Real grain-size surrogate (replace the synthetic dummy)
- [ ] `GroupKFold` for cold drawing (rows within a wire are correlated)
- [ ] Cold drawing surrogates (one-step and multi-step rollout)
- [ ] Probabilistic cold drawing controller (DAG path ranking)
- [ ] Physical jittering for input-noise robustness
- [ ] Errors-in-variables augmentation for downstream surrogates
- [ ] Rotary swaging surrogates
- [ ] End-to-end multi-process controller

Full step-by-step in `docs/copper_digital_twin_v4` §5; module-level reference in `docs/code_architecture.pdf`.

## References

- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS.
- Wilson, A. G., & Izmailov, P. (2020). *Bayesian Deep Learning and a Probabilistic Perspective of Generalization.* NeurIPS.
- Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. (2004). *Ensemble Selection from Libraries of Models.* ICML.
- Erickson, N., et al. (2020). *AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data.*
- Kennedy, M. C., & O'Hagan, A. (2001). *Bayesian Calibration of Computer Models.* JRSS-B.