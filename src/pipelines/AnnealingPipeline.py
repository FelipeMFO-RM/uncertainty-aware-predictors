"""High-level dataset assembly for cold drawing experiments."""
#TODO STOPPED HERE NEED TO IMPELMENT THE CHFE MODULE
import pandas as pd

from ..feature_engineering.FeatureEngineering import FeatureEngineering
from ..processing.Processing import Processing


class DataPipeline:
    """Compose Processing and FeatureEngineering into one call.

    This is the notebook-facing API. It orchestrates lower-level
    helpers and must not implement transformation logic itself.
    """

    def __init__(self) -> None:
        self.proc = Processing()
        self.feat = FeatureEngineering()

    def build_cold_drawing_dataset(
        self,
        df: pd.DataFrame,
        drop_cols: list[str],
        columns_order: list[str],
        target_variable: str,
    ) -> pd.DataFrame:
        """Return the modeling-ready cold drawing dataframe.

        Parameters
        ----------
        df
            Raw simulation dataframe.
        ...

        Returns
        -------
        pd.DataFrame
            Filtered, engineered and reordered dataframe.
        """
        out = self.proc.df_to_float(df, drop_cols=drop_cols)
        out = self.proc.filter_cold_drawing_rules(out)
        out = self.feat.label_element(out, element="Cu")
        out = self.feat.add_ratio_mask_column(out, target_variable)
        return self.proc.reorder_columns(out, columns_order)