import numpy as np
import pandas as pd
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from src.modeling.Modeling import Modeling


class Evaluations:
    """Class containing the methods to assess the models using specific
    evaluation metrics.
    """

    def __init__(self):
        self.modl = Modeling()
        pass

    def rmse(self, y_true, y_pred):
        return root_mean_squared_error(y_true=y_true, y_pred=y_pred)

    def mape(self, y_true, y_pred):
        """Mean Absolute Percentage Error"""
        return mean_absolute_percentage_error(
            y_true=y_true, y_pred=y_pred
        )

    def mae(self, y_true, y_pred):
        """Mean Absolute Error"""
        return mean_absolute_error(y_true=y_true, y_pred=y_pred)

    def gaussian_nll(
        self,
        y_true: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
    ) -> float:
        """Mean Gaussian negative log-likelihood.

        Measures the quality of the full probabilistic prediction
        N(mu, sigma^2). Lower is better: a well-calibrated sigma is
        neither too small (overconfident) nor too large (useless).

        Formula per sample i:
            NLL_i = log(sigma_i) + (y_true_i - mu_i)^2 / (2 * sigma_i^2)

        Args:
            y_true (np.ndarray): Ground-truth target values.
            mu (np.ndarray): Predicted mean values.
            sigma (np.ndarray): Predicted standard deviation (> 0).

        Returns:
            float: Mean NLL across all samples. Lower = better.
        """
        y_true = np.asarray(y_true, dtype=float)
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        nll = np.log(sigma) + (y_true - mu) ** 2 / (2.0 * sigma ** 2)
        return float(np.mean(nll))

    def calibration_coverage(
        self,
        y_true: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        z: float = 1.96,
    ) -> float:
        """Fraction of y_true inside the [mu ± z * sigma] interval.

        A well-calibrated model achieves coverage ≈ 0.95 on the
        held-out validation set (z=1.96 corresponds to 95 % CI).

        Interpretation:
            coverage < 0.95  ->  sigma underestimated (overconfident)
            coverage > 0.95  ->  sigma overestimated (too conservative)

        Args:
            y_true (np.ndarray): Ground-truth target values.
            mu (np.ndarray): Predicted mean values.
            sigma (np.ndarray): Predicted standard deviation (> 0).
            z (float, optional): Z-score for the interval.
            Defaults to 1.96 (95 % CI).

        Returns:
            float: Fraction of samples inside the interval. Target ≈ 0.95.
        """
        y_true = np.asarray(y_true, dtype=float)
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        in_interval = (y_true >= mu - z * sigma) & (
            y_true <= mu + z * sigma
        )
        return float(in_interval.mean())

    def select_models(
        self, df: pd.DataFrame, percentile: float, mode: str = "all"
    ):
        """
        Select rows from the dataframe where all metric values
        are below or equal to the given percentile threshold.

        Args:
            df (pd.DataFrame): DataFrame containing model metrics.
            percentile (float): Percentile between 0 and 1
            (e.g., 0.2 = 20%).

        Returns:
            pd.DataFrame: Subset of rows where all metrics are within
            the specified percentile.
        """
        thresholds = df.quantile(percentile)

        if mode == "any":
            mask = (df <= thresholds).any(axis=1)
        else:
            mask = (df <= thresholds).all(axis=1)
        return df[mask]

    def get_metrics_evaluation_set(
        self, df, df_results, metrics, target_variable, tag, model,
        scalers
    ):
        df_ = self.modl.scale_and_pred(
            df=df,
            target_variable=target_variable,
            scalers=scalers,
            model=model,
            identifier=tag,
        )
        df_results[f"{target_variable}_{tag}"] = df_[
            f"{target_variable}_{tag}"
        ]
        metrics[tag] = {
            "mape": self.mape(
                y_true=df[f"{target_variable}"],
                y_pred=df_[f"{target_variable}_{tag}"],
            ),
            "rmse": self.rmse(
                y_true=df[f"{target_variable}"],
                y_pred=df_[f"{target_variable}_{tag}"],
            ),
            "mae": self.mae(
                y_true=df[f"{target_variable}"],
                y_pred=df_[f"{target_variable}_{tag}"],
            ),
        }
