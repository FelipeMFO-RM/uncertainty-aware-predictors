from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class _GroupPath:
    root_id: str
    leaf_id: str
    path_ids: tuple[str, ...]


class CDHelper:

    # Grouping methods
    @staticmethod
    def set_group_experiment(
        df: pd.DataFrame,
        experiment_id_col: str = "experiment_id",
        previous_experiment_id_col: str = "previous_experiment_id",
        pass_number_col: str = "pass_number",
        group_col: str = "group_experiment_id",
        root_suffix_sep: str = "_",
        group_prefix: str = "G",
        root_pass_number: int = 1,
    ) -> pd.DataFrame:
        """
        Expand a cold-drawing simulation dataframe by assigning a group id
        to each root-to-leaf path in the experiment dependency forest.

        Each row is an experiment node. The parent pointer is given by
        `previous_experiment_id_col` (each node has at most one parent),
        while a node may have multiple children (branching). A "primordial"
        root is a node whose `pass_number_col == root_pass_number`.

        A group corresponds to a unique path from a primordial root to a
        leaf. When branching happens, shared ancestors are duplicated
        across groups.

        Parameters
        ----------
        df:
            Input dataframe containing experiments and their parent linkage.
        experiment_id_col:
            Column with unique experiment identifiers.
        previous_experiment_id_col:
            Column with the parent experiment id (or missing if none).
        pass_number_col:
            Column that marks the pass number.
        group_col:
            Name of the new column to store the group identifier.
        root_suffix_sep:
            Separator used before appending the group suffix.
        group_prefix:
            Prefix used to form the group suffix (e.g., "G" -> "_G1").
        root_pass_number:
            Pass number that defines a primordial root.

        Returns
        -------
        pd.DataFrame
            Expanded dataframe with an additional `group_col`.
        """
        if df.empty:
            out = df.copy()
            out[group_col] = pd.Series(dtype="object")
            return out

        if experiment_id_col not in df.columns:
            raise KeyError(f"Missing column: {experiment_id_col}")
        if previous_experiment_id_col not in df.columns:
            raise KeyError(
                f"Missing column: {previous_experiment_id_col}"
            )
        if pass_number_col not in df.columns:
            raise KeyError(f"Missing column: {pass_number_col}")

        work = df.copy()

        ids = work[experiment_id_col].astype(str)
        if ids.duplicated().any():
            dup = ids[ids.duplicated()].iloc[0]
            raise ValueError(
                f"Duplicate {experiment_id_col} detected "
                f"(example: {dup})."
            )

        parent_raw = work[previous_experiment_id_col]
        parent = (
            parent_raw.where(parent_raw.notna(), None)
            .astype("object")
            .map(lambda x: None if x in ("-", "", "None") else x)
        )
        parent = parent.map(
            lambda x: None if x is None else str(x)
        )

        pass_num = work[pass_number_col]

        id_list = ids.tolist()
        id_pos = {eid: i for i, eid in enumerate(id_list)}
        parent_map = {
            eid: parent.iloc[i] for i, eid in enumerate(id_list)
        }

        children: dict[str, list[str]] = defaultdict(list)
        for child_id, parent_id in parent_map.items():
            if parent_id is None:
                continue
            if parent_id in id_pos:
                children[parent_id].append(child_id)

        roots = set(
            ids.loc[pass_num == root_pass_number].astype(str).tolist()
        )

        def _trace_to_root(leaf_id: str) -> list[str]:
            seen: set[str] = set()
            chain: list[str] = []
            cur = leaf_id
            while cur is not None and cur not in seen:
                seen.add(cur)
                chain.append(cur)
                cur = parent_map.get(cur)
                if cur is not None and cur not in id_pos:
                    cur = None
            chain.reverse()
            return chain

        leaves = [
            eid
            for eid in id_list
            if len(children.get(eid, [])) == 0
        ]

        group_paths: list[_GroupPath] = []
        for leaf_id in leaves:
            chain = _trace_to_root(leaf_id)
            if not chain:
                continue
            root_id = chain[0]
            if root_id not in roots:
                continue
            group_paths.append(
                _GroupPath(
                    root_id=root_id,
                    leaf_id=leaf_id,
                    path_ids=tuple(chain),
                )
            )

        if not group_paths:
            out = work.copy()
            out[group_col] = pd.Series(dtype="object")
            return out

        paths_by_root: dict[str, list[_GroupPath]] = defaultdict(list)
        for gp in group_paths:
            paths_by_root[gp.root_id].append(gp)

        for r in paths_by_root:
            paths_by_root[r].sort(
                key=lambda gp: (
                    id_pos.get(gp.leaf_id, 10**12),
                    gp.leaf_id,
                )
            )

        roots_sorted = sorted(
            paths_by_root.keys(),
            key=lambda r: id_pos.get(r, 10**12),
        )

        chunks: list[pd.DataFrame] = []
        for root_id in roots_sorted:
            for k, gp in enumerate(paths_by_root[root_id], start=1):
                group_id = (
                    f"{root_id}{root_suffix_sep}{group_prefix}{k}"
                )
                idx = [id_pos[eid] for eid in gp.path_ids]
                chunk = work.iloc[idx].copy()
                chunk[group_col] = group_id
                chunks.append(chunk)

        out = pd.concat(chunks, axis=0, ignore_index=True)
        return out

    # Sanity checks
    @staticmethod
    def check_n_pass(
        df: pd.DataFrame,
        group_col: str = "group_experiment_id",
        pass_number_col: str = "pass_number",
        correct: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Check and optionally correct pass number progression inside each
        group experiment.

        A valid group must have pass numbers starting at 1 and increasing
        strictly by +1 with no duplicates.

        If `correct=True`, the function returns a tuple:
            (df_corrected, report)

        Otherwise it returns only the report.
        """
        if group_col not in df.columns:
            raise KeyError(f"Missing column: {group_col}")
        if pass_number_col not in df.columns:
            raise KeyError(f"Missing column: {pass_number_col}")

        out = df.copy() if correct else df
        rows = []

        for gid, g in out.groupby(group_col, sort=False):
            seq = pd.to_numeric(
                g[pass_number_col], errors="coerce"
            ).tolist()
            n = len(seq)
            expected = list(range(1, n + 1))

            seq_int = [None if pd.isna(x) else int(x) for x in seq]
            has_nan = any(x is None for x in seq_int)

            seq_clean = [x for x in seq_int if x is not None]
            has_duplicates = len(seq_clean) != len(set(seq_clean))
            is_sequential = (not has_nan) and (seq_int == expected)

            needs_fix = has_nan or has_duplicates or (not is_sequential)

            if correct and needs_fix:
                out.loc[g.index, pass_number_col] = expected

            rows.append(
                {
                    group_col: gid,
                    "n_passes": n,
                    "has_nan": has_nan,
                    "has_duplicates": has_duplicates,
                    "is_sequential": is_sequential,
                    "is_valid": not needs_fix,
                    "corrected": bool(correct and needs_fix),
                }
            )

        report = pd.DataFrame(rows)

        if correct:
            return out, report
        return report

    @staticmethod
    def check_tensile_strength_augmentation(
        df: pd.DataFrame,
        group_col: str = "group_experiment_id",
        tensile_col: str = "tensile_strength",
        pass_number_col: str = "pass_number",
    ) -> pd.DataFrame:
        """
        Check that tensile strength increases monotonically along each
        group experiment.

        Parameters
        ----------
        df:
            Input dataframe containing grouped experiments.
        group_col:
            Column identifying the group experiment.
        tensile_col:
            Column with tensile strength values.
        pass_number_col:
            Column with pass numbers used for ordering.

        Returns
        -------
        pd.DataFrame
            A dataframe with one row per group and sanity check results.
        """
        results = []

        for gid, g in df.groupby(group_col, sort=False):
            g_sorted = g.sort_values(pass_number_col)
            ts = g_sorted[tensile_col].astype(float).tolist()

            is_strictly_increasing = all(
                ts[i] > ts[i - 1] for i in range(1, len(ts))
            )

            results.append(
                {
                    group_col: gid,
                    "n_passes": len(ts),
                    "is_strictly_increasing": is_strictly_increasing,
                }
            )

        return pd.DataFrame(results)

    # Ungroup, close the full cycle
    @staticmethod
    def ungroup_experiments(
        df: pd.DataFrame,
        group_col: str = "group_experiment_id",
        subset_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Collapse a grouped experiment dataframe back to its raw DAG form.

        Parameters
        ----------
        df:
            Input dataframe containing grouped experiments.
        group_col:
            Column identifying the group experiment.
        subset_cols:
            Columns used to identify unique raw experiments. If None,
            all columns except `group_col` are used.

        Returns
        -------
        pd.DataFrame
            Ungrouped dataframe equivalent to the original raw dataset.
        """
        if group_col not in df.columns:
            raise KeyError(f"Missing column: {group_col}")

        work = df.drop(columns=[group_col])

        if subset_cols is None:
            subset_cols = work.columns.tolist()

        out = work.drop_duplicates(
            subset=subset_cols, keep="first"
        ).reset_index(drop=True)

        return out

    # Add initial tensile strength
    @staticmethod
    def add_initial_tensile_strength(
        df: pd.DataFrame,
        group_col: str = "group_experiment_id",
        experiment_id_col: str = "experiment_id",
        previous_experiment_id_col: str = "previous_experiment_id",
        pass_number_col: str = "pass_number",
        tensile_final_col: str = "tensile_strength_final",
        original_tensile_col: str = "original_tensile_strength",
        output_col: str = "initial_tensile_strength",
    ) -> pd.DataFrame:
        """
        Add an initial tensile strength column to a grouped experiment
        dataframe.

        For pass_number == 1, the initial tensile strength is taken from
        the original tensile strength. For pass_number > 1, it is taken
        from the final tensile strength of the previous experiment within
        the same group.

        Parameters
        ----------
        df:
            Input grouped dataframe.
        output_col:
            Name of the column to be created.

        Returns
        -------
        pd.DataFrame
            Dataframe with the added initial tensile strength column.
        """
        out = df.copy()
        out[output_col] = pd.NA

        for _, g in out.groupby(group_col, sort=False):
            idx = g.index

            final_map = g.set_index(
                experiment_id_col
            )[tensile_final_col].to_dict()

            is_root = g[pass_number_col] == 1
            out.loc[idx[is_root], output_col] = (
                g.loc[is_root, original_tensile_col].values
            )

            non_root = g.loc[~is_root]
            prev_ids = non_root[
                previous_experiment_id_col
            ].astype(str)

            out.loc[non_root.index, output_col] = [
                final_map[pid] for pid in prev_ids
            ]

        return out

    @staticmethod
    def get_by_experiment_group(
        df: pd.DataFrame,
        group_col: str = "group_experiment_id",
        pass_number_col: str = "pass_number",
        material_col: str = "material",
        purity_col: str = "purity",
        tensile_original_col: str = "tensile_strength_original",
        tensile_final_col: str = "tensile_strength_final",
        original_diameter_col: str = "original_diameter",
        final_diameter_col: str = "final_diameter",
    ) -> pd.DataFrame:
        """
        Collapse grouped experiments into one row per group experiment.

        Each output row represents a full cold-drawing trajectory from
        the initial state (root pass) to the final pass.

        Returns
        -------
        pd.DataFrame
            One row per group_experiment_id with initial and final states.
        """
        rows = []

        for gid, g in df.groupby(group_col, sort=False):
            g_sorted = g.sort_values(pass_number_col)

            root = g_sorted.iloc[0]
            last = g_sorted.iloc[-1]

            rows.append(
                {
                    group_col: gid,
                    material_col: root[material_col],
                    purity_col: root[purity_col],
                    "n_passes": int(last[pass_number_col]),
                    "initial_tensile_strength": root[
                        tensile_original_col
                    ],
                    "initial_diameter": root[original_diameter_col],
                    tensile_final_col: last[tensile_final_col],
                    final_diameter_col: last[final_diameter_col],
                }
            )

        return pd.DataFrame(rows)
