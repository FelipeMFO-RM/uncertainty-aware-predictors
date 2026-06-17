import pandas as pd
import numpy as np


class Processing:
    """Processing data."""

    def __init__(self) -> None:
        pass

    def df_to_float(
        self,
        df: pd.DataFrame,
        drop_cols: list[str],
        precision_tag: str = "~",
        ignore_columns: list[str] = [],
    ) -> pd.DataFrame:
        """
        Clean and convert dataframe columns to float.

        Args:
            df (pd.DataFrame): Input dataframe.
            drop_cols (list[str]): Columns to drop.
            precision_tag (str, optional): Character to strip (default "~").

        Returns:
            pd.DataFrame: Processed dataframe with float values.
        """
        df_float = df.drop(
            columns=drop_cols + ignore_columns, errors="ignore"
        )
        df_float = df_float.applymap(
            lambda x: (
                str(x).replace(precision_tag, "") if pd.notnull(x) else x
            )
        )
        df_float = df_float.applymap(
            lambda x: (str(x).replace(" ", "") if pd.notnull(x) else x)
        )

        df_float = df_float.astype(float)
        ans = df_float
        if ignore_columns:
            for col in ignore_columns:
                ans = pd.concat([df[col], ans], axis=1)

        return ans

    def load_validation_data_chimie_paris(self, path: str) -> pd.DataFrame:
        """Load validation data from Chimie ParisTech, drop unnecessary columns,
        and filter out rows with NaN values."""
        df_val = (
            pd.read_csv(path)
            .drop(columns=["E_GPa", "temperature_C"], errors="ignore")
            .filter(regex="^(?!.*strength).*")
            .dropna(axis=0)
            .rename(columns={"Ref": "id"})
            .reset_index(drop=True)
            )
        return df_val

    def reorder_columns(self, df, columns_order):
        """Reorder df columns by columns_order;
        keep unseen columns at the end."""
        # normalize columns_order -> list of column names in desired order
        if isinstance(columns_order, dict):
            k = next(iter(columns_order))  # peek one item
            v = columns_order[k]
            if isinstance(k, int) and isinstance(v, str):  # {pos: name}
                desired = [
                    columns_order[i] for i in sorted(columns_order)
                ]
            elif isinstance(k, str) and isinstance(
                v, (int, float)
            ):  # {name: pos}
                desired = [
                    k
                    for k, _ in sorted(
                        columns_order.items(), key=lambda kv: kv[1]
                    )
                ]
            else:  # fallback: dict insertion order
                desired = list(columns_order.values())
        else:
            desired = list(columns_order)

        ordered = [c for c in desired if c in df.columns]
        extras = [c for c in df.columns if c not in ordered]
        return df[ordered + extras]

    def filter_cold_drawing_rules(
        self, df: pd.DataFrame, tol: float = 0.0
    ) -> pd.DataFrame:
        """
        Keep rows that satisfy:
        - grain_size_final < grain_size (when both exist)
        - tensile_strength_final > tensile_strength (when both exist)
        - iacs_final < iacs (when both exist)
        Rows with NaN in any of the compared pairs are preserved.
        """

        # --- Grain size rule ---
        gs = pd.to_numeric(df.get("grain_size"), errors="coerce")
        gsf = pd.to_numeric(df.get("grain_size_final"), errors="coerce")
        gs_pair = gs.notna() & gsf.notna()
        gs_ok = (~gs_pair) | (gsf < (gs - tol))

        # --- Tensile strength rule (optional) ---
        if {"tensile_strength", "tensile_strength_final"}.issubset(
            df.columns
        ):
            ts = pd.to_numeric(
                df["tensile_strength"], errors="coerce"
            )
            tsf = pd.to_numeric(
                df["tensile_strength_final"], errors="coerce"
            )
            ts_pair = ts.notna() & tsf.notna()
            ts_ok = (~ts_pair) | (tsf > (ts + tol))
        else:
            ts_ok = np.ones(len(df), dtype=bool)

        # --- IACS rule (optional) ---
        if {"iacs", "iacs_final"}.issubset(df.columns):
            i = pd.to_numeric(df["iacs"], errors="coerce")
            i_f = pd.to_numeric(df["iacs_final"], errors="coerce")
            i_pair = i.notna() & i_f.notna()
            i_ok = (~i_pair) | (i_f < (i - tol))
        else:
            i_ok = np.ones(len(df), dtype=bool)

        # Combine all rules
        mask = gs_ok & ts_ok & i_ok
        kept = df[mask].copy()

        print(
            f"Rows in: {len(df)} | Kept: {len(kept)} | "
            f"Dropped: {len(df) - len(kept)}"
        )
        return kept
