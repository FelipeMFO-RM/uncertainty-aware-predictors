import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Plots:
    """Plot class used on Notebooks."""

    def __init__(self) -> None:
        pass

    def plot_distributions(
        self,
        df,
        subplot_dim=(3, 3),
        bins=20,
        id="",
        save_pdf: str | None = None,
    ) -> None:
        """Plot distributions of numeric columns in subplots."""
        numeric_cols = df.select_dtypes(include=["number"]).columns
        n_rows, n_cols = subplot_dim

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3)
        )
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            axes[i].hist(df[col].dropna(), bins=bins, edgecolor="black")
            axes[i].set_title(col)
            axes[i].grid(False)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(
            f"{id} Distribution of Numeric Variables", fontsize=16
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_pdf:
            plt.savefig(save_pdf, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def plot_corr_matrix(
        self,
        df: pd.DataFrame,
        id: str | int = "",
        save_pdf: str | None = None,
    ) -> None:
        """Plot a numeric-only correlation matrix as a heatmap and close
        the figure.

        Args:
            df (pd.DataFrame): Input dataframe to compute correlations from.
            id (str | int, optional): Identifier to display in the figure
            title. Defaults to "".
        """
        corr = df.corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)

        ttl = "Correlation Matrix"
        if id != "":
            ttl = f"{id} {ttl} "
        ax.set_title(ttl)

        plt.tight_layout()
        if save_pdf:
            plt.savefig(save_pdf, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def compare_correlations(
        self,
        df: pd.DataFrame,
        target: str = "yield_strength_final",
        sort_by: str = "Pearson",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Compute Pearson and Spearman correlations of all numeric
        features vs. a target.

        Args:
            df (pd.DataFrame): Input dataframe.
            target (str, optional): Target column to correlate against.
            Defaults to "yield_strength_final".
            sort_by (str, optional): Column to sort by ("Pearson" or
            "Spearman"). Defaults to "Pearson".
            ascending (bool, optional): Sort order. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with columns ["Pearson", "Spearman"],
            indexed by feature names.
        """
        pearson = df.corr(method="pearson", numeric_only=True).get(
            target
        )
        spearman = df.corr(method="spearman", numeric_only=True).get(
            target
        )

        if pearson is None or spearman is None:
            raise ValueError(
                f"Target '{target}' not found among numeric columns. "
                "Ensure the target is numeric or present in the dataframe."
            )

        corr_df = pd.DataFrame(
            {"Pearson": pearson, "Spearman": spearman}
        ).drop(index=target, errors="ignore")

        corr_df = corr_df.dropna(how="all")

        if sort_by in corr_df.columns:
            corr_df = corr_df.sort_values(sort_by, ascending=ascending)

        return corr_df

    def plot_target_correlations(
        self,
        df: pd.DataFrame,
        target: str,
        id: str | int = "",
        top_n: int | None = None,
        save_pdf: str | None = None,
    ) -> None:
        """Plot Pearson & Spearman correlations (bars) of numeric features
        vs. a target and close the figure.

        Args:
            df (pd.DataFrame): Input dataframe.
            target (str, optional): Target column to correlate against.
            id (str | int, optional): Identifier to display in the figure
            title. Defaults to "".
            top_n (int | None, optional): If provided, plot only top-N
            features by |Pearson| magnitude. Defaults to None.
        """
        corr_df = self.compare_correlations(
            df, target=target, sort_by="Pearson", ascending=False
        )

        if top_n is not None and top_n > 0:
            corr_df = corr_df.reindex(
                corr_df["Pearson"]
                .abs()
                .sort_values(ascending=False)
                .head(top_n)
                .index
            )

        long_df = (
            corr_df.reset_index(names="feature")
            .melt(
                id_vars="feature",
                value_vars=["Pearson", "Spearman"],
                var_name="method",
                value_name="correlation",
            )
            .dropna(subset=["correlation"])
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(
            data=long_df,
            x="correlation",
            y="feature",
            hue="method",
            orient="h",
            palette="viridis",
            ax=ax,
        )

        ttl = f"Correlation with '{target}' (Pearson & Spearman)"
        if id != "":
            ttl = f"{id} {ttl} "
        ax.set_title(ttl)

        ax.set_xlabel("Correlation coefficient")
        ax.set_ylabel("Features")
        ax.axvline(0, linewidth=1, linestyle="--")
        ax.legend(title="Method", loc="best")

        plt.tight_layout()
        if save_pdf:
            plt.savefig(save_pdf, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def plot_predictions_vs_truth(
        self,
        df_results,
        target_col: str,
        ref_col: str,
        identifiers_to_select: list[str] | None = None,
        pred_tags: list[str] | None = None,
        path_to_save: str | None = None,
        file_name_to_save: str = "predictions.pdf",
        id_col: str = "id",
        variable_name: str | None = None,
        figsize: tuple[int, int] = (16, 8),
    ):
        """Plot model predictions against ground-truth and a reference
        series.

        Args:
            df_results: DataFrame with id column, ground-truth, reference
                feature, and prediction columns like
                f"{target_col}_{tag}".
            target_col: Ground-truth column name (y-axis reference).
            ref_col: Reference feature column (dashed line).
            pred_tags: Optional list of full tags to include.
            identifiers_to_select: Optional tokens to filter prediction
                columns (requires all tokens to appear).
            path_to_save: Directory to save the figure. If None, not saved.
            file_name_to_save: Output file name.
            id_col: X-axis column (default "id").
            variable_name: Label/title name. If None, uses `target_col`.
            figsize: Matplotlib figure size.

        Returns:
            (fig, ax, used_pred_cols)
        """
        if variable_name is None:
            variable_name = target_col

        all_pred_cols = [
            c
            for c in df_results.columns
            if c.startswith(f"{target_col}_")
        ]

        if pred_tags:
            allowed = {f"{target_col}_{tag}" for tag in pred_tags}
            pred_cols = [c for c in all_pred_cols if c in allowed]
        else:
            pred_cols = all_pred_cols

        if identifiers_to_select:
            tokens = list(identifiers_to_select)
            pred_cols = [
                c for c in pred_cols if all(tok in c for tok in tokens)
            ]

        if not pred_cols:
            pred_cols = all_pred_cols

        fig, ax = plt.subplots(figsize=figsize)

        for c in pred_cols:
            ax.plot(
                df_results[id_col],
                df_results[c],
                linewidth=1,
                alpha=0.6,
                label=c,
            )

        ax.plot(
            df_results[id_col],
            df_results[target_col],
            linewidth=3,
            color="black",
            label=target_col,
        )

        ax.plot(
            df_results[id_col],
            df_results[ref_col],
            linestyle="--",
            color="gray",
            linewidth=2,
            label=ref_col,
        )

        ax.set_xlabel(f"Sample ({id_col})")
        ax.set_ylabel(variable_name)
        ax.set_title(
            f"{variable_name} {identifiers_to_select} predictions vs "
            "ground truth per sample"
        )

        plt.xticks(rotation=45, ha="right")
        ax.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8
        )

        plt.tight_layout()

        if path_to_save:
            os.makedirs(path_to_save, exist_ok=True)
            outpath = os.path.join(path_to_save, file_name_to_save)
            plt.savefig(outpath, bbox_inches="tight")

        plt.show()
        return fig, ax, pred_cols

    # ---------- internal utilities (kept local to the class) ----------
    @staticmethod
    def _select_pred_cols(
        df: pd.DataFrame,
        target_col: str,
        pred_tags: list[str] | None = None,
        identifiers_to_select: list[str] | None = None,
    ) -> list[str]:
        """Return prediction columns matching `target_col` prefix and
        optional filters."""
        all_pred_cols = [
            c for c in df.columns if c.startswith(f"{target_col}_")
        ]
        if pred_tags:
            allowed = {f"{target_col}_{tag}" for tag in pred_tags}
            all_pred_cols = [
                c for c in all_pred_cols if c in allowed
            ]
        if identifiers_to_select:
            toks = list(identifiers_to_select)
            all_pred_cols = [
                c
                for c in all_pred_cols
                if all(t in c for t in toks)
            ]
        return all_pred_cols

    @staticmethod
    def _maybe_save(
        fig: plt.Figure, path_to_save: str | None, file_name: str
    ) -> None:
        """Save figure if a path is provided."""
        if path_to_save:
            os.makedirs(path_to_save, exist_ok=True)
            fig.savefig(
                os.path.join(path_to_save, file_name),
                bbox_inches="tight",
            )

    def plot_residual_boxplot(
        self,
        df: pd.DataFrame,
        target_col: str,
        pred_tags: list[str] | None = None,
        identifiers_to_select: list[str] | None = None,
        figsize: tuple[int, int] = (8, 4),
        path_to_save: str | None = None,
        file_name_to_save: str = "predictions_box_abs_error.pdf",
    ):
        """Boxplots of absolute error by model with short labels.

        Returns:
            (fig, ax, used_pred_cols)
        """
        pred_cols = self._select_pred_cols(
            df, target_col, pred_tags, identifiers_to_select
        )
        if not pred_cols:
            pred_cols = [
                c
                for c in df.columns
                if c.startswith(f"{target_col}_")
            ]

        short_labels = {c: f"M{i+1}" for i, c in enumerate(pred_cols)}
        long = pd.concat(
            [
                pd.DataFrame(
                    {
                        "model": short_labels[c],
                        "abs_error": (df[c] - df[target_col]).abs(),
                    }
                )
                for c in pred_cols
            ],
            ignore_index=True,
        )

        fig, ax = plt.subplots(figsize=figsize)
        long.boxplot(column="abs_error", by="model", ax=ax, grid=False)
        ax.set_title("Absolute error by model")
        ax.set_ylabel("|pred - true|")
        ax.set_xlabel("")
        plt.suptitle("")
        plt.tight_layout()

        self._maybe_save(fig, path_to_save, file_name_to_save)
        plt.show()
        return fig, ax, pred_cols

    def plot_error_heatmap(
        self,
        df: pd.DataFrame,
        target_col: str,
        pred_tags: list[str] | None = None,
        identifiers_to_select: list[str] | None = None,
        id_col: str = "id",
        order_samples_by_true: bool = True,
        use_abs: bool = True,
        vmax_quantile: float = 0.98,
        cmap: str = "viridis",
        figsize: tuple[int, int] = (14, 6),
        path_to_save: str | None = None,
        file_name_to_save: str = "predictions_error_heatmap.pdf",
    ):
        """Heatmap of residuals.

        Returns:
            (fig, ax, used_pred_cols)
        """
        pred_cols = self._select_pred_cols(
            df, target_col, pred_tags, identifiers_to_select
        )
        if not pred_cols:
            pred_cols = [
                c
                for c in df.columns
                if c.startswith(f"{target_col}_")
            ]

        resid = np.vstack(
            [(df[c] - df[target_col]).values for c in pred_cols]
        )
        if use_abs:
            data = np.abs(resid)
            vmin = 0.0
            vmax = (
                np.quantile(data, vmax_quantile) if data.size else 1.0
            )
            title = "Absolute residuals heatmap |pred - true|"
        else:
            data = resid
            amax = (
                np.quantile(np.abs(data), vmax_quantile)
                if data.size
                else 1.0
            )
            vmin, vmax = -amax, amax
            cmap = "coolwarm"
            title = "Residuals heatmap (pred - true)"

        order = (
            np.argsort(df[target_col].values)
            if order_samples_by_true
            else np.arange(len(df))
        )
        xticklabels = df[id_col].values[order]

        short_labels = [f"M{i+1}" for i in range(len(pred_cols))]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            data[:, order],
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax.set_yticks(range(len(pred_cols)))
        ax.set_yticklabels(short_labels)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("|pred - true|" if use_abs else "pred - true")

        ax.set_title(title)
        plt.tight_layout()

        self._maybe_save(fig, path_to_save, file_name_to_save)
        plt.show()
        return fig, ax, pred_cols

    @staticmethod
    def _dumbbell_on_ax(
        ax: plt.Axes,
        df: pd.DataFrame,
        target_col: str,
        pred_col: str,
        id_col: str = "id",
        sort_by: str = "true",
        ms: float = 3.0,
        lw: float = 1.0,
        ytick_fontsize: int = 6,
        title: str | None = None,
        show_legend: bool = False,
    ) -> plt.Axes:
        """Draw a dumbbell plot on an existing Axes."""
        d = df[[id_col, target_col, pred_col]].copy()
        d = d.rename(
            columns={target_col: "true", pred_col: "pred"}
        )
        if sort_by == "true":
            d = d.sort_values("true")
        elif sort_by == "pred":
            d = d.sort_values("pred")

        y = np.arange(len(d))
        ax.hlines(y, d["true"], d["pred"], color="lightgray", lw=lw)
        ax.plot(d["true"], y, "o", ms=ms, label="True")
        ax.plot(d["pred"], y, "o", ms=ms, label="Pred")
        ax.set_yticks(y)
        ax.set_yticklabels(d[id_col], fontsize=ytick_fontsize)
        ax.set_xlabel(target_col)
        if title:
            ax.set_title(title, fontsize=9)
        if show_legend:
            ax.legend(fontsize=7)
        return ax

    @staticmethod
    def _bland_altman_on_ax(
        ax: plt.Axes,
        df: pd.DataFrame,
        target_col: str,
        pred_col: str,
        s: float = 12.0,
        title: str | None = None,
    ) -> plt.Axes:
        """Draw a Bland-Altman scatter on an existing Axes."""
        t = df[target_col].values
        p = df[pred_col].values
        mean = (t + p) / 2.0
        diff = p - t

        ax.scatter(mean, diff, s=s, alpha=0.7)
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_xlabel("Mean(True, Pred)")
        ax.set_ylabel("Pred - True")
        if title:
            ax.set_title(title, fontsize=9)
        return ax

    def plot_dumbbell(
        self,
        df: pd.DataFrame,
        target_col: str,
        pred_col: str,
        id_col: str = "id",
        sort_by: str = "true",
        figsize_per_sample: float = 0.22,
        min_height: int = 4,
        path_to_save: str | None = None,
        file_name_to_save: str = "predictions_dumbbell.pdf",
    ):
        """Dumbbell plot: true vs. predicted per sample."""
        d = df[[id_col]].copy()
        h = max(min_height, int(len(d) * figsize_per_sample))
        fig, ax = plt.subplots(figsize=(8, h))

        self._dumbbell_on_ax(
            ax=ax,
            df=df,
            target_col=target_col,
            pred_col=pred_col,
            id_col=id_col,
            sort_by=sort_by,
            title=f"Dumbbell: {pred_col}",
            show_legend=True,
        )

        plt.tight_layout()
        self._maybe_save(fig, path_to_save, file_name_to_save)
        plt.show()
        return fig, ax, [pred_col]

    def plot_bland_altman(
        self,
        df: pd.DataFrame,
        target_col: str,
        pred_col: str,
        figsize: tuple[int, int] = (7, 5),
        path_to_save: str | None = None,
        file_name_to_save: str = "predictions_bland_altman.pdf",
    ):
        """Bland-Altman plot: agreement between true and predicted."""
        fig, ax = plt.subplots(figsize=figsize)
        self._bland_altman_on_ax(
            ax=ax,
            df=df,
            target_col=target_col,
            pred_col=pred_col,
            title=f"BlandAltman: {pred_col}",
        )
        plt.tight_layout()
        self._maybe_save(fig, path_to_save, file_name_to_save)
        plt.show()
        return fig, ax, [pred_col]

    def grid_dumbbell_bland(
        self,
        df: pd.DataFrame,
        target_col: str,
        id_col: str,
        sels: list[str],
        orientation: str = "rows",
        row_height: float | None = None,
        save_path: str | None = None,
        file_name: str = "grid_dumbbell_bland.pdf",
    ):
        """Grid of Dumbbell (col 0) and Bland-Altman (col 1) plots for
        each model in `sels`.

        Returns:
            (fig, axes)
        """
        n = len(sels)
        if row_height is None:
            row_height = np.clip(0.20 * len(df), 3.0, 8.0)

        if orientation == "rows":
            nrows, ncols = n, 2
            figsize = (12, row_height * n)
        else:
            nrows, ncols = 2, n
            figsize = (6 * n, max(4.5, row_height))

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            squeeze=False,
        )

        for i, tag in enumerate(sels):
            pred_col = f"{target_col}_{tag}"

            r = i if orientation == "rows" else 0
            c = 0 if orientation == "rows" else i
            ax_db = axes[r, c]
            self._dumbbell_on_ax(
                ax=ax_db,
                df=df,
                target_col=target_col,
                pred_col=pred_col,
                id_col=id_col,
                sort_by="true",
                title=f"Dumbbell: {tag}",
                show_legend=(i == 0),
            )

            r2 = r if orientation == "rows" else 1
            c2 = 1 if orientation == "rows" else c
            ax_ba = axes[r2, c2]
            self._bland_altman_on_ax(
                ax=ax_ba,
                df=df,
                target_col=target_col,
                pred_col=pred_col,
                title=f"BlandAltman: {tag}",
            )

        plt.tight_layout()
        self._maybe_save(fig, save_path, file_name)
        return fig, axes
