import os
import re
import json
import h2o
import pickle
import pandas as pd

from .utils import pick_best_non_ensemble


class DataDumper:
    @staticmethod
    def _infer_algo_from_model_id(model_id: str) -> str:
        """Infer algorithm family name from a model_id string.

        Args:
            model_id (str): H2O model identifier
            (e.g., 'GBM_grid_1_AutoML_...').

        Returns:
            str: Algorithm family name (e.g., 'gbm', 'xgboost', 'glm').
        """
        patterns = {
            r"^GBM_": "gbm",
            r"^XGBoost_": "xgboost",
            r"^DRF_": "drf",
            r"^GLM_": "glm",
            r"^DeepLearning_": "deeplearning",
            r"^StackedEnsemble_": "stackedensemble",
            r"^RuleFit_": "rulefit",
            r"^IsolationForest_": "isolationforest",
        }
        for pat, fam in patterns.items():
            if re.match(pat, model_id):
                return fam
        return "other"

    @staticmethod
    def _get_leaderboard_with_algo(aml) -> pd.DataFrame:
        """Get leaderboard as DataFrame including algorithm family.

        Args:
            aml (h2o.automl.H2OAutoML): Trained AutoML object.

        Returns:
            pd.DataFrame: Leaderboard with an 'algo' column for
            algorithm family.
        """
        try:
            lb = aml.get_leaderboard(extra_columns="ALL").as_data_frame(
                use_multi_thread=True
            )
            if "algo" not in lb.columns:
                raise RuntimeError
        except Exception:
            lb = aml.leaderboard.as_data_frame(use_multi_thread=True)
            lb["algo"] = lb["model_id"].map(
                DataDumper._infer_algo_from_model_id
            )
        return lb

    @staticmethod
    def best_by_family(aml, metric: str = "rmse"):
        """Select the best model of each algorithm family.

        Args:
            aml (h2o.automl.H2OAutoML): Trained AutoML object.
            metric (str, optional): Metric to minimize for selection
            (e.g., 'rmse', 'mae'). Defaults to 'rmse'.

        Returns:
            tuple:
                - pd.DataFrame: Leaderboard rows of best models by family.
                - list[str]: List of model_ids for the best models.
        """
        lb = DataDumper._get_leaderboard_with_algo(aml)
        if metric not in lb.columns:
            raise ValueError(
                f"Metric '{metric}' not found on leaderboard."
            )
        lb = lb[pd.notna(lb[metric])].copy()
        idx = lb.groupby("algo")[metric].idxmin()
        best_df = lb.loc[idx].sort_values(metric).reset_index(drop=True)
        best_ids = best_df["model_id"].tolist()
        return best_df, best_ids

    @staticmethod
    def top_n_models(aml, n: int = 10, metric: str = "rmse"):
        """Select the top-N models from the leaderboard.

        Args:
            aml (h2o.automl.H2OAutoML): Trained AutoML object.
            n (int, optional): Number of models to select.
            Defaults to 10.
            metric (str, optional): Metric to minimize for selection.
            Defaults to 'rmse'.

        Returns:
            tuple:
                - pd.DataFrame: Leaderboard rows of the top-N models.
                - list[str]: List of model_ids for the top-N models.
        """
        lb = DataDumper._get_leaderboard_with_algo(aml)
        if metric not in lb.columns:
            raise ValueError(
                f"Metric '{metric}' not found on leaderboard."
            )
        lb = lb[pd.notna(lb[metric])].copy()
        lb_sorted = lb.sort_values(
            metric, ascending=True
        ).reset_index(drop=True)
        top_df = lb_sorted.head(n).copy()
        top_ids = top_df["model_id"].tolist()
        return top_df, top_ids

    @staticmethod
    def save_selection(
        aml,
        out_dir: str,
        best_fam_ids: list,
        topn_ids: list,
        save_mojo: bool = True,
        save_binary: bool = False,
        save_scalers: dict | None = None,
        save_metrics: dict | None = None,
        changelog: str | None = None,
        snapshot_df: pd.DataFrame | None = None,
        main_model=None,
    ):
        """Save selected models + leaderboards + metadata + extras."""

        aml_dir = os.path.join(out_dir, "aml_object")
        models_dir = os.path.join(aml_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        paths = {"models": [], "aml_dir": aml_dir}

        # --- Leaderboards ------------------------------------------------
        try:
            lb_all = aml.get_leaderboard(
                extra_columns="ALL"
            ).as_data_frame(use_multi_thread=True)
            lb_all.to_csv(
                os.path.join(aml_dir, "leaderboard_ALL.csv"),
                index=False,
            )
            paths["leaderboard_all_csv"] = os.path.join(
                aml_dir, "leaderboard_ALL.csv"
            )
        except Exception:
            lb = aml.leaderboard.as_data_frame(use_multi_thread=True)
            lb.to_csv(
                os.path.join(aml_dir, "leaderboard.csv"), index=False
            )
            paths["leaderboard_csv"] = os.path.join(
                aml_dir, "leaderboard.csv"
            )

        # --- Select model IDs --------------------------------------------
        selected_ids: list[str] = []
        for mid in best_fam_ids + topn_ids:
            if mid not in selected_ids:
                selected_ids.append(mid)

        # --- Save candidate models ---------------------------------------
        for mid in selected_ids:
            m = h2o.get_model(mid)
            entry = {"model_id": mid}
            if save_binary:
                entry["binary_path"] = h2o.save_model(
                    m, path=models_dir, force=True
                )
            if save_mojo:
                try:
                    entry["mojo_path"] = m.download_mojo(
                        path=models_dir, get_genmodel_jar=True
                    )
                except Exception as e:
                    entry["mojo_error"] = str(e)
            paths["models"].append(entry)

        # --- Main model: force deterministic ----------------------------
        if main_model is None:
            main_model = pick_best_non_ensemble(aml)

        main_model_path = h2o.save_model(
            model=main_model, path=out_dir, force=True
        )
        paths["main_model_path"] = main_model_path
        paths["main_model_id"] = main_model.model_id

        # --- Metadata ----------------------------------------------------
        meta = {
            "leader_id": aml.leader.model_id,
            "saved_model_ids": selected_ids,
            "best_family_ids": best_fam_ids,
            "topn_ids": topn_ids,
            "n_models_saved": len(selected_ids),
            "h2o_version": h2o.__version__,
            "main_model_id": main_model.model_id,
            "main_model_path": main_model_path,
        }

        meta_path = os.path.join(aml_dir, "run_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        paths["metadata_json"] = meta_path

        # --- Extras ------------------------------------------------------
        if save_scalers is not None:
            p = os.path.join(out_dir, "scalers.pkl")
            with open(p, "wb") as f:
                pickle.dump(save_scalers, f)
            paths["scalers_pickle"] = p

        if save_metrics is not None:
            p = os.path.join(out_dir, "metrics.pkl")
            with open(p, "wb") as f:
                pickle.dump(save_metrics, f)
            paths["metrics_pickle"] = p

        if changelog is not None:
            p = os.path.join(out_dir, "changelog.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(changelog.strip() + "\n")
            paths["changelog_txt"] = p

        if snapshot_df is not None and isinstance(
            snapshot_df, pd.DataFrame
        ):
            p = os.path.join(out_dir, "snapshot_df.csv")
            snapshot_df.to_csv(p, index=False)
            paths["snapshot_df_csv"] = p

        return paths
