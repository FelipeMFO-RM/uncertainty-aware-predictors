import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Any, Dict, Optional
from collections import deque
import itertools



logger = logging.getLogger(__name__)

class FeatureEngineering:
    """Class of feature engineering."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # - Normalization
    # ------------------------------------------------------------------ #

    def inverse_normalize(
        self,
        df_norm: pd.DataFrame,
        scalers: dict,
        alias_map: dict | None = None,
    ) -> pd.DataFrame:
        """Inverse Min-Max and log1p transformations to recover original scale.

        Args:
            df_norm (pd.DataFrame): Normalized DataFrame to invert.
            scalers (dict): Dictionary with fitted log and Min-Max scalers.
            alias_map (dict | None): Map original columns to prediction/alias
                columns (e.g. {"grain_size_final": "grain_size_final_pred"}).

        Returns:
            pd.DataFrame: DF with features restored to their original scale.
        """

        df_restored = df_norm.copy()
        alias_map = alias_map or {}

        # --- Inverse MinMax in one vectorized op ---
        mm_info = scalers.get("minmax")
        if mm_info:
            mm = mm_info["scaler"]
            mm_cols = mm_info["columns"]

            # Build full mapping: original + aliases
            col_map = {c: c for c in mm_cols}
            col_map.update(
                {
                    dst: src
                    for src, dsts in alias_map.items()
                    for dst in ([dsts] if isinstance(dsts, str) else dsts)
                    if src in mm_cols and dst in df_restored.columns
                }
            )

            # Align params with columns to restore
            idx_map = (
                {n: i for i, n in enumerate(mm.feature_names_in_)}
                if hasattr(mm, "feature_names_in_")
                else {n: i for i, n in enumerate(mm_cols)}
            )

            for col, src in col_map.items():
                i = idx_map[src]
                df_restored[col] = (df_restored[col] - mm.min_[i]) / mm.scale_[
                    i
                ]

        # --- Inverse log1p (vectorized) ---
        log_cols = scalers.get("log", [])
        log_targets = [
            c
            for src in log_cols
            for c in [src]
            + (
                []
                if src not in alias_map
                else (
                    [alias_map[src]]
                    if isinstance(alias_map[src], str)
                    else alias_map[src]
                )
            )
            if c in df_restored.columns
        ]

        if log_targets:
            df_restored[log_targets] = np.expm1(df_restored[log_targets])

        return df_restored

    def normalize(
        self, df: pd.DataFrame, log_cols: list, target_col: str
    ) -> tuple:
        """Normalize features with log1p and Min-Max scaling."""
        df_norm = df.copy()

        feature_cols = df_norm.columns.difference([target_col])
        log_columns = feature_cols.intersection(log_cols)

        if len(log_columns) > 0:
            if (df_norm[log_columns] <= -1).any().any():
                raise ValueError("log1p requires values > -1.")
            df_norm[log_columns] = np.log1p(df_norm[log_columns])

        minmax = MinMaxScaler()
        df_norm[feature_cols] = minmax.fit_transform(df_norm[feature_cols])

        scalers = {
            "log": list(log_columns),
            "minmax": {"scaler": minmax, "columns": list(feature_cols)},
        }
        return scalers, df_norm

    def normalize_with_scalers(
        self,
        df: pd.DataFrame,
        scalers: Dict[str, Any],
        target_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Apply training scalers (log1p and Min-Max) to a new DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame to normalize.
            scalers (Dict[str, Any]): Dictionary with fitted log and Min-Max
            scalers.
            target_col (Optional[str]): Target column to exclude from
            normalization.

        Raises:
            ValueError: If log1p is applied to values <= -1 or Min-Max
            columns are missing.

        Returns:
            pd.DataFrame: Normalized DataFrame with the same scaling as
            training.
        """
        df_norm = df.copy()

        if target_col and target_col in df_norm.columns:
            feature_cols = df_norm.columns.difference([target_col])
        else:
            feature_cols = df_norm.columns

        log_cols = pd.Index(scalers.get("log", []))
        log_cols_present = log_cols.intersection(feature_cols)
        if len(log_cols_present):
            df_norm[log_cols_present] = df_norm[log_cols_present].apply(
                pd.to_numeric, errors="coerce"
            )
            if (df_norm[log_cols_present] <= -1).any(axis=None):
                bad = df_norm[log_cols_present].le(-1).stack()
                bad_cols = list(bad[bad].index.get_level_values(1).unique())
                raise ValueError(
                    f"log1p requires values > -1. Bad columns: {bad_cols}"
                )
            df_norm[log_cols_present] = np.log1p(
                df_norm[log_cols_present].to_numpy()
            )

        mm_info = scalers.get("minmax")
        if mm_info is None:
            return df_norm

        mm = mm_info["scaler"]
        trained_cols = pd.Index(
            getattr(mm, "feature_names_in_", mm_info.get("columns", []))
        )
        mm_cols_present = trained_cols.intersection(feature_cols)

        if len(mm_cols_present):
            df_norm[mm_cols_present] = df_norm[mm_cols_present].apply(
                pd.to_numeric, errors="coerce"
            )
            idx = trained_cols.get_indexer(mm_cols_present)
            if (idx < 0).any():
                raise ValueError(
                    "Some MinMax-trained columns not found in scaler."
                )

            scale_vec = mm.scale_[idx]
            min_vec = mm.min_[idx]
            X = df_norm[mm_cols_present].to_numpy(dtype=float, copy=False)
            df_norm[mm_cols_present] = X * scale_vec + min_vec

        return df_norm

    def invert_single_column(
        self,
        col_scaled: pd.Series,
        scalers: Dict[str, Any],
        trained_feature: str,
        was_logged: Optional[bool] = None,
    ) -> pd.Series:
        """Invert scaling for a single column to its original values.

        Args:
            col_scaled (pd.Series): Scaled column to invert.
            scalers (Dict[str, Any]): Dictionary with fitted log and scalers.
            trained_feature (str): Name of the feature as used during training.
            was_logged (Optional[bool]): Whether the feature was log-scaled.
                If None, inferred from scalers.

        Returns:
            pd.Series: Restored column in original scale.
        """
        s = pd.to_numeric(col_scaled, errors="coerce")

        mm_info = scalers.get("minmax")
        mm = mm_info["scaler"]
        trained_cols = pd.Index(
            getattr(mm, "feature_names_in_", mm_info.get("columns", []))
        )

        i = trained_cols.get_loc(trained_feature)
        scale_i = mm.scale_[i]
        min_i = mm.min_[i]
        s = (s - min_i) / scale_i

        if was_logged is None:
            was_logged = trained_feature in set(scalers.get("log", []))
        if was_logged:
            s = np.expm1(s)

        s.name = col_scaled.name
        return s

    # ------------------------------------------------------------------ #
    # Expansion of cold-drawing simulations
    # ------------------------------------------------------------------ #
    @staticmethod
    def _expand_single_simulation(group: pd.DataFrame) -> pd.DataFrame:
        """Expand one simulation_number into all diameter pairs.

        For a given simulation_number:
        - constructs all combinations (initial_diameter, final_diameter)
        with initial_diameter > final_diameter;
        - assigns tensile_strength and tensile_strength_final from the
        corresponding diameters;
        - recalculates total_strain = 2 * ln(initial_diameter/final_diameter);
        - calculates number_of_passes as the minimum number of actual passes
        (lines of the df) needed to go from initial_diameter to
        final_diameter, always reducing the diameter.
        """
        sim = group["simulation_number"].iloc[0]
        purity = group["purity"].iloc[0]

        # ---- diameter map -> tensile strength ----------------------
        init_map = group[["initial_diameter", "tensile_strength"]].rename(
            columns={
                "initial_diameter": "diameter",
                "tensile_strength": "t_strength",
            }
        )
        final_map = group[["final_diameter", "tensile_strength_final"]].rename(
            columns={
                "final_diameter": "diameter",
                "tensile_strength_final": "t_strength",
            }
        )

        diameters = (
            pd.concat([init_map, final_map], ignore_index=True)
            .drop_duplicates(subset="diameter")
            .reset_index(drop=True)
        )

        # ---- real pass graph (init -> final) ---------------------
        diams_all = pd.unique(
            pd.concat(
                [group["initial_diameter"], group["final_diameter"]],
                ignore_index=True,
            )
        )

        edges = (
            group[["initial_diameter", "final_diameter"]]
            .drop_duplicates()
            .to_numpy()
        )

        graph: Dict[float, set[float]] = {float(d): set() for d in diams_all}
        for init_d, final_d in edges:
            init_d = float(init_d)
            final_d = float(final_d)
            # consider only decreasing passes
            if init_d > final_d:
                graph[init_d].add(final_d)

        # ---- BFS for the fewest number of passes among all pairs ------
        # dist_map[(d_init, d_final)] = number_of_passes
        dist_map: Dict[tuple[float, float], int] = {}
        for src in diams_all:
            src = float(src)
            q = deque([(src, 0)])
            seen = {src}
            while q:
                cur, d = q.popleft()
                for nxt in graph.get(cur, ()):
                    if nxt not in seen:
                        seen.add(nxt)
                        dist_map[(src, nxt)] = d + 1
                        q.append((nxt, d + 1))

        # DataFrame cast to join
        passes_df = pd.DataFrame(
            [(i, f, n) for (i, f), n in dist_map.items()],
            columns=[
                "initial_diameter",
                "final_diameter",
                "number_of_passes",
            ],
        )

        left = diameters.rename(
            columns={
                "diameter": "initial_diameter",
                "t_strength": "tensile_strength",
            }
        )
        right = diameters.rename(
            columns={
                "diameter": "final_diameter",
                "t_strength": "tensile_strength_final",
            }
        )

        left["key"] = 1
        right["key"] = 1
        pairs = left.merge(right, on="key").drop(columns="key")

        # only vertices initial > final
        pairs = pairs[pairs["initial_diameter"] > pairs["final_diameter"]]

        # ---- join number_of_passes --
        pairs = pairs.merge(
            passes_df,
            on=["initial_diameter", "final_diameter"],
            how="inner",
        )

        # ---- static columns --------------------------
        pairs["simulation_number"] = sim
        pairs["purity"] = purity
        pairs["total_strain"] = 2.0 * np.log(
            pairs["initial_diameter"] / pairs["final_diameter"]
        )

        cols = [
            "simulation_number",
            "purity",
            "tensile_strength",
            "initial_diameter",
            "final_diameter",
            "total_strain",
            "tensile_strength_final",
            "number_of_passes",
        ]
        return pairs[cols]

    def expand_simulations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand df_sim into df_sim_expanded."""
        expanded = (
            df.groupby("simulation_number", group_keys=False)
            .apply(self._expand_single_simulation)
            .reset_index(drop=True)
        )

        expanded = expanded.drop_duplicates(
            subset=["simulation_number", "initial_diameter", "final_diameter"]
        ).reset_index(drop=True)

        return expanded

    def remove_original_rows_expand_simulations(
        self, df_sim: pd.DataFrame, df_sim_expanded: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Remove from df_sim_expanded all rows that already exist in df_sim,
        based on (simulation_number, initial_diameter, final_diameter).

        Args:
            df_sim (pd.DataFrame): Original simulation dataset.
            df_sim_expanded (pd.DataFrame): Expanded dataset.

        Returns:
            pd.DataFrame: df_sim_expanded with original rows removed.
        """
        key_cols = ["simulation_number", "initial_diameter", "final_diameter"]

        # Mark original rows
        df_sim["_orig_row"] = 1

        # Merge to identify which expanded rows are original
        merged = df_sim_expanded.merge(
            df_sim[key_cols + ["_orig_row"]], on=key_cols, how="left"
        )

        # Keep only rows that were NOT in original df_sim
        df_new = merged[merged["_orig_row"].isna()].drop(columns="_orig_row")

        # Cleanup
        df_sim.drop(columns="_orig_row", inplace=True)

        return df_new.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # - Creation of tags
    # ------------------------------------------------------------------ #

    def make_tag(
        self,
        base_key: str,
        tag_dict: dict[str, str],
        feat_cols: dict[str, list[str]],
    ) -> str:
        """
        Build a tag like 'cual_wdim_npur_wtts'.
        Ordering follows feat_cols keys.
        """
        parts = [base_key]
        for group in feat_cols.keys():
            parts.append(f"{tag_dict[group]}{group}")
        return "_".join(parts)

    def add_tag(self, dfs_dict: dict, tag: str):
        """_summary_

        Args:
            dfs_dict (dict): dfs dictionary to duplicate with and wihout tag.
            tag (str): tag to insert. E.g. 'log' or 'dpl'
        """
        ans = {
            **{f"{t}_n{tag}": value for t, value in dfs_dict.items()},
            **{f"{t}_w{tag}": value for t, value in dfs_dict.items()},
        }
        return ans

    # ------------------------------------------------------------------ #
    # - Outliers treatment
    # ------------------------------------------------------------------ #

    def cut_outliers_single_col(
        self,
        df: pd.DataFrame,
        column: str,
        lower_p: float = 0.05,
        upper_p: float = 0.9,
    ) -> pd.DataFrame:
        """
        Remove outliers by keeping only rows between two percentiles.

        Args:
            df: Input DataFrame.
            column: Column on which to apply percentile filtering.
            lower_p: Lower percentile bound (0–1).
            upper_p: Upper percentile bound (0–1).

        Returns:
            Filtered DataFrame.
        """
        lower = df[column].quantile(lower_p)
        upper = df[column].quantile(upper_p)
        return df[(df[column] >= lower) & (df[column] <= upper)].copy()

    def cut_outliers_multi_col(
        self,
        df: pd.DataFrame,
        bounds: dict[str, tuple[float, float]],
        inclusive: str = "both",
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Keep only rows that satisfy all (lower, upper) percentile bounds.

        Args:
            df: Input DataFrame.
            bounds: Dict like {"col": (lower_p, upper_p)} with percentiles
                in [0, 1]. Example: {"grain_size_final": (0.05, 0.90)}.
            inclusive: "both" | "left" | "right" | "neither" for endpoints.
            verbose: If True, prints a small summary.

        Returns:
            (df_filtered, summary_df)
            - df_filtered: rows passing all bounds.
            - summary_df: per-column thresholds and dropped counts.
        """
        mask_all = np.ones(len(df), dtype=bool)
        summary_rows = []

        for col, (lp, up) in bounds.items():
            s = pd.to_numeric(df[col], errors="coerce")
            lo = s.quantile(lp)
            hi = s.quantile(up)

            if inclusive == "both":
                m = (s >= lo) & (s <= hi)
            elif inclusive == "left":
                m = (s >= lo) & (s < hi)
            elif inclusive == "right":
                m = (s > lo) & (s <= hi)
            else:  # "neither"
                m = (s > lo) & (s < hi)

            kept = int(m.sum())
            dropped = int((~m).sum())
            summary_rows.append(
                {
                    "column": col,
                    "lower": lo,
                    "upper": hi,
                    "kept": kept,
                    "dropped": dropped,
                }
            )
            mask_all &= m.fillna(False).to_numpy()

        df_out = df[mask_all].copy()
        summary = pd.DataFrame(summary_rows)

        if verbose:
            total_drop = len(df) - len(df_out)
            print(summary)
            print(
                f"\nTotal kept: {len(df_out)}  |  Total dropped: {total_drop}"
            )

        return df_out, summary

    # ------------------------------------------------------------------ #
    # - Further methods
    # ------------------------------------------------------------------ #
    def cols_to_drop(
        self,
        tag_dict: dict[str, str],
        feat_cols: dict[str, list[str]],
    ) -> list[str]:
        """
        Columns to drop: all groups tagged 'n'.
        """
        drop = []
        for group, val in tag_dict.items():
            if val == "n":
                drop.extend(feat_cols.get(group, []))
        return drop

    def build_dfs_to_train(
        self,
        dfs_source: dict[str, pd.DataFrame],
        features_to_vary: dict[str, list[str]],
        skip_all_n: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        From source DFs and their feature-groups (features_to_vary), generate
        all datasets with columns dropped according to every 'w'/'n'
        combination.

        Example features_to_vary:
            {
                "dim": ["initial_diameter", "final_diameter"],
                "pur": ["purity"],
                "tts": ["total_strain"],
            }

        Returns:
            dict: {"cual_wdim_npur_wtts": df_modified, ...}
        """
        dfs_to_train: dict[str, pd.DataFrame] = {}
        groups = list(features_to_vary.keys())

        for base_key, df in dfs_source.items():
            for combo in itertools.product(("w", "n"), repeat=len(groups)):
                tag_dict = dict(zip(groups, combo))

                # optional: skip case where all groups are dropped
                if skip_all_n and all(v == "n" for v in tag_dict.values()):
                    continue

                tag_name = self.make_tag(base_key, tag_dict, features_to_vary)
                drop_cols = self.cols_to_drop(tag_dict, features_to_vary)

                dfs_to_train[tag_name] = (
                    df.drop(columns=drop_cols, errors="ignore")
                    .drop_duplicates()
                    .copy()
                )

        return dfs_to_train

    def label_element(
        self, df, element: str = "Cu", material_col_name: str = "material"
    ) -> pd.DataFrame:
        """Label rows based on the presence of a specific element in the
        'material' column.

        Args:
            df (pd.DataFrame): Input DataFrame with an 'alloy' column.
            element (str): Element to check for in the 'alloy' strings.

        Returns:
            pd.DataFrame: DataFrame with an added boolean column
            'has_{element}'.
        """
        if (f"is_{element}" or f"has_{element}") in df.columns:
            if f"is_{element}" in df.columns:
                return df.rename(columns={f"is_{element}": f"has_{element}"})
            return df
        df_labeled = df.copy()
        label_col = f"has_{element}"
        df_labeled[label_col] = df_labeled[material_col_name].str.contains(
            element, na=False
        )
        return df_labeled

    def concat_with_weight_column(
        self,
        df: pd.DataFrame,
        df_raw_schema: pd.DataFrame,
        weight_value: float,
        weight_col: str = "weight_col",
        target: str = "df_raw_schema",
        ignore_index: bool = True,
    ) -> pd.DataFrame:
        """Concatenate two DataFrames after tagging one with a weight column.

        Args:
            df (pd.DataFrame): First DataFrame to concatenate.
            df_raw_schema (pd.DataFrame): Second DataFrame to concatenate.
            weight_value (float): Constant value assigned in `weight_col`.
            weight_col (str): Name of the weight column to add.
            target (str): Which DataFrame receives `weight_col`,
                either "df" or "df_raw_schema".
            ignore_index (bool): Passed to `pd.concat` to reindex the result.

        Returns:
            pd.DataFrame: Concatenation of `df` and `df_raw_schema` with
            `weight_col` set on the chosen DataFrame.
        """
        if target not in ("df", "df_raw_schema"):
            raise ValueError("target must be 'df' or 'df_raw_schema'.")

        df, df_raw_schema = df.copy(), df_raw_schema.copy()

        if target == "df":
            df[weight_col] = weight_value
        else:
            df_raw_schema[weight_col] = weight_value

        return pd.concat([df, df_raw_schema], ignore_index=ignore_index)

    def add_ratio_mask_column(self, df, target_variable: str):
        """
        Add a ratio column comparing the final to the initial value
        of a target variable.

        Args:
            df (pd.DataFrame): Input dataframe containing both
                `<target_variable>` and `<target_variable>_final` columns.
            target_variable (str): Base column name to compute the ratio.

        Returns:
            pd.DataFrame: Dataframe with an added column
                `<target_variable>_ratio`.
        """
        df[f"{target_variable}_ratio"] = (
            df[f"{target_variable}_final"] / df[target_variable]
        )

        return df

    @staticmethod
    def add_group_fold_column(
        df: pd.DataFrame,
        group_col: str,
        n_folds: int = 5,
        seed: int = 42,
        col: str = "fold_id",
    ) -> pd.DataFrame:
        """Assign WHOLE groups (wires) to folds, balanced by group size.

        Cold-drawing rows of the same wire are statistically dependent
        (pass i+1 inherits the state of pass i), so plain row-level K-fold
        leaks information between train and validation. This builder keeps
        every wire intact in exactly one fold and greedily balances fold
        sizes by row count — since each wire spans passes 1..n, every fold
        automatically receives a MIX of pass numbers (no fold ends up
        holding all the 4th/5th passes), which is the stratification-by-pass
        behaviour the evaluation needs.

        The returned column is meant to be passed to AutoGluon as the
        ``groups`` bagging column (via ``fit_model_a(groups=col)``), making
        the internal bag folds group-aware and the OOF predictions honest.

        Parameters
        ----------
        df : pd.DataFrame
            Input frame containing ``group_col``.
        group_col : str
            Wire/sample identifier (e.g. ``group_experiment_id``).
        n_folds : int
            Number of folds (must match ``num_bag_folds`` of Model A).
        seed : int
            Shuffle seed (tie-breaking among equal-size groups).
        col : str
            Name of the fold column to create.

        Returns
        -------
        pd.DataFrame
            Copy with the integer fold column in ``[0, n_folds)``.
        """
        if group_col not in df.columns:
            raise KeyError(f"Missing group column: {group_col}")
        out = df.copy()
        sizes = out.groupby(group_col).size()
        if len(sizes) < n_folds:
            raise ValueError(
                f"Only {len(sizes)} groups for {n_folds} folds; reduce "
                f"n_folds (or use leave-one-group-out)."
            )
        rng = np.random.default_rng(seed)
        order = sizes.sample(frac=1.0, random_state=seed).sort_values(
            ascending=False, kind="stable"
        )
        fold_rows = np.zeros(n_folds, dtype=int)
        assign: dict = {}
        for gid, n in order.items():
            f = int(np.argmin(fold_rows))
            assign[gid] = f
            fold_rows[f] += int(n)
        out[col] = out[group_col].map(assign).astype(int)
        logger.info(
            "Group folds on '%s': %d groups -> %d folds, rows per fold %s.",
            group_col, len(sizes), n_folds, fold_rows.tolist(),
        )
        return out

    def add_stratified_fold_column(
        self,
        df: pd.DataFrame,
        n_folds: int,
        seed: int,
        col: str = "fold_id",
        stratify_col: str | None = "is_essay",
    ) -> pd.DataFrame:
        """Assign a deterministic K-fold id, stratified on a label column.

        Used to share the exact same CV partition between AutoGluon
        (``groups=col``) and H2O (``fold_column=col``), so OOF metrics are
        coherently comparable. Stratification matters whenever a small
        subgroup exists (e.g. essay rows among literature rows): plain
        KFold can by chance concentrate that subgroup into 1-2 folds,
        leaving others with none, which silently breaks per-fold
        representativeness. ``df.groupby([col, stratify_col]).size()``
        is the quick way to confirm every fold got at least one row of
        the minority class after calling this.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe. Not mutated; a copy is returned.
        n_folds : int
            Number of folds (must be <= the smallest class count in
            ``stratify_col`` when stratifying).
        seed : int
            Random seed for reproducibility.
        col : str, default "fold_id"
            Name of the fold-id column to create. Keep it OUT of the
            model's feature list.
        stratify_col : str or None, default "is_essay"
            Column to stratify folds on. Falls back to plain KFold when
            None or absent from ``df`` (e.g. UTS, which has no essay
            marker yet).

        Returns
        -------
        pd.DataFrame
            Copy of ``df`` with an added integer ``col`` column.
        """
        out = df.copy()
        out[col] = -1

        if stratify_col is not None and stratify_col in out.columns:
            from sklearn.model_selection import StratifiedKFold
            splitter = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=seed
            )
            split_iter = splitter.split(out, out[stratify_col])
        else:
            from sklearn.model_selection import KFold
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(out)

        for k, (_, test_idx) in enumerate(split_iter):
            out.iloc[test_idx, out.columns.get_loc(col)] = k
        return out