import numpy as np
import pandas as pd


class EditMinMaxScaler:
    """
    Min–Max scaler with fixed [data_min, data_max], optional log1p mapping.
    Uses precomputed `scale` and `offset`:
        y = z * scale + offset
        z = (y - offset) / scale
    where z = x or z = log1p(x) if `is_log=True`.
    """

    def __init__(
        self,
        values,
        data_min: float,
        data_max: float,
        is_log: bool = False,
        margin_factor: float = 1.1,
        feature_range: tuple = (0.0, 1.0),
    ):
        self.log = is_log
        self.values = values

        # 1) expand bounds in original space
        lo = float(data_min) / margin_factor
        hi = float(data_max) * margin_factor
        if self.log:
            lo, hi = np.log1p(lo), np.log1p(hi)

        a, b = map(float, feature_range)

        # 2) precompute affine mapping z -> y
        self._scale = (b - a) / (hi - lo)
        self._offset = a - lo * self._scale

        # keep a few attrs around for debugging/inspection
        self.eff_min = lo
        self.eff_max = hi
        self._a, self._b = a, b

        self._fitted = False

    def fit(self):
        """No learning; kept for API symmetry."""
        self._fitted = True
        return self

    def transform(self, X):
        """Original -> scaled:
        y = (log1p(x) if log else x) * scale + offset."""
        x = (
            X.to_numpy(dtype=float)
            if isinstance(X, pd.Series)
            else np.asarray(X, dtype=float)
        )
        z = np.log1p(x) if self.log else x
        y = z * self._scale + self._offset
        return (
            pd.Series(y, index=X.index, name=getattr(X, "name", None))
            if isinstance(X, pd.Series)
            else y
        )

    def inverse_transform(self, X_scaled):
        """Scaled -> original:
        x = expm1((y - offset)/scale) if log else (y - offset)/scale."""
        y = (
            X_scaled.to_numpy(dtype=float)
            if isinstance(X_scaled, pd.Series)
            else np.asarray(X_scaled, dtype=float)
        )
        z = (y - self._offset) / self._scale
        x = np.expm1(z) if self.log else z
        return (
            pd.Series(
                x,
                index=X_scaled.index,
                name=getattr(X_scaled, "name", None),
            )
            if isinstance(X_scaled, pd.Series)
            else x
        )


class ScalerHelpers:
    def __init__(self):
        pass

    def set_scaler(
        self,
        df: pd.DataFrame,
        features_range: dict,
        target_variable: str,
        log_features=[],
        log_target=False,
    ):

        log_features = list(log_features or [])

        columns_to_scale = [
            c for c in df.columns if (log_target or c != target_variable)
        ]

        if log_target:
            log_features.append(target_variable)

        log_aux = {
            col: True if col in log_features else False
            for col in columns_to_scale
        }
        ans = {
            col: EditMinMaxScaler(
                values=df[col],
                data_min=(
                    min(
                        features_range.get(
                            col, [df[col].min(), df[col].max()]
                        )[0],
                        df[col].min(),
                    )
                ),
                data_max=(
                    max(
                        features_range.get(
                            col, [df[col].min(), df[col].max()]
                        )[1],
                        df[col].max(),
                    )
                ),
                is_log=log_aux[col],
            )
            for col in columns_to_scale
        }
        return ans

    def apply_scalers(
        self, df: pd.DataFrame, scalers: dict
    ) -> pd.DataFrame:
        """Apply each scaler to its column (others are passed
        through unchanged)."""
        out = df.copy()
        for col, s in scalers.items():
            out[col] = s.transform(out[col])
        return out

    def invert_scalers(
        self, df_scaled: pd.DataFrame, scalers: dict
    ) -> pd.DataFrame:
        """Inverse-transform only the columns that have scalers."""
        out = df_scaled.copy()
        for col, s in scalers.items():
            out[col] = s.inverse_transform(out[col])
        return out
