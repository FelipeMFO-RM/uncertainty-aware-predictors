import os
import json
import pickle
import pandas as pd
import h2o


class LoaderHelper:
    """Helper class for loading models and pickled objects."""

    def load_pickle(self, folder_path: str, file: str) -> object:
        """
        Load a pickle file from folder_path.

        Args:
            folder_path (str): Base folder.
            file (str): File name (with or without .pkl).

        Returns:
            object: Loaded pickle object.
        """
        if not file.endswith(".pkl"):
            file += ".pkl"
        with open(os.path.join(folder_path, file), "rb") as f:
            return pickle.load(f)

    def get_best_model(self, folder_path: str):
        """Load deterministic main model saved in run_metadata.json."""
        meta_path = os.path.join(
            folder_path, "aml_object", "run_metadata.json"
        )
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Missing metadata: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model_path = meta.get("main_model_path")
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Missing main model file: {model_path}"
            )

        return h2o.load_model(model_path)

    def load_scalers(self, folder_path: str):
        """
        Load scalers.pkl from a model folder.

        Args:
            folder_path (str): Model folder.

        Returns:
            dict: Scalers dictionary.
        """
        return self.load_pickle(folder_path, "scalers.pkl")

    def load_bundle(
        self, bundle_dir: str, load_models: bool = True
    ) -> dict:
        """
        Load an AutoML bundle previously saved with DataDumper.save_selection.

        Args:
            bundle_dir (str): Path to the 'aml_object' directory
            containing models and metadata.
            load_models (bool, optional): Whether to reload model objects.
            Defaults to True.

        Returns:
            dict: {
                "metadata": dict with run metadata,
                "leaderboard": pd.DataFrame (if exists),
                "leaderboard_all": pd.DataFrame (if exists),
                "models": dict of {model_id: h2o_model}
                (if load_models=True)
            }
        """
        out = {}
        # 1) Metadata
        meta_path = os.path.join(bundle_dir, "run_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                out["metadata"] = json.load(f)

        # 2) Leaderboards
        lb_path = os.path.join(bundle_dir, "leaderboard.csv")
        if os.path.exists(lb_path):
            out["leaderboard"] = pd.read_csv(lb_path)

        lb_all_path = os.path.join(bundle_dir, "leaderboard_ALL.csv")
        if os.path.exists(lb_all_path):
            out["leaderboard_all"] = pd.read_csv(lb_all_path)

        # 3) Load models (if requested)
        if load_models:
            models_dir = os.path.join(bundle_dir, "models")
            models = {}
            if os.path.exists(models_dir):
                for fname in os.listdir(models_dir):
                    fpath = os.path.join(models_dir, fname)
                    if fname.endswith(".zip"):  # MOJO
                        try:
                            m = h2o.import_mojo(fpath)
                            models[m.model_id] = m
                        except Exception as e:
                            print(
                                f"[WARN] Could not import MOJO "
                                f"{fname}: {e}"
                            )
                    else:  # Assume binary model
                        try:
                            m = h2o.load_model(fpath)
                            models[m.model_id] = m
                        except Exception as e:
                            print(
                                f"[WARN] Could not load model "
                                f"{fname}: {e}"
                            )
            out["models"] = models

        return out
