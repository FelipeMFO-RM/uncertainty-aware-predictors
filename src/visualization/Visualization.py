import pandas as pd
import copy
from IPython.display import display_html
from itertools import chain, cycle


def front(self, n=5):
    return self.iloc[:, :n]


def back(self, n=5):
    return self.iloc[:, -n:]


def display_columns(self, n=5):
    return pd.concat([self.front(n), self.back(n)], axis=1)


def display_rows(self, n=5):
    return pd.concat([self.head(n), self.tail(n)])


def display(self, i=5, j=5):
    return self.display_columns(i).display_rows(j)


pd.DataFrame.front = front
pd.DataFrame.back = back
pd.DataFrame.display_columns = display_columns
pd.DataFrame.display_rows = display_rows
pd.DataFrame.display = display


class Visualization:
    """Class with methods for visualization of DataFrames."""

    def __init__(self):
        pass

    def visualize_percentiles(
        self, df: pd.DataFrame, column: str, percentile_fraction: float
    ):
        s = pd.to_numeric(df[column], errors="coerce")
        s = s.dropna()

        total_q = int(1 / percentile_fraction)
        qs = [(i + 1) * percentile_fraction for i in range(total_q)]

        out = pd.DataFrame(
            {f"{int(q*100)}%": [s.quantile(q)] for q in qs}
        ).T
        out.columns = [f"Percentiles of '{column}'"]
        return out

    def display_side_by_side(self, *dfs, titles=()):
        html_str = ""
        for i, df in enumerate(dfs):
            title = titles[i] if i < len(titles) else ""
            html_str += (
                '<div style="display:inline-block; vertical-align:top; '
                'padding-right:20px; width:45%">'
            )
            html_str += f"<h3>{title}</h3>"
            html_str += df.head(10).to_html(
                classes="dataframe", border=0
            )
            html_str += "</div>"
        display_html(html_str, raw=True)

    def display_vertically(self, *args, titles=cycle([""])):
        """Prints the given DataFrames showing their number of rows.
        Receive arguments as a dictionary.

        Args:
            titles ([type], optional): titles of the DataFrames provided.
            Defaults to cycle(['']).
        """
        html_str = ""
        for df, title in zip(args, chain(titles, cycle(["</br>"]))):
            df_ = copy.copy(df.display())
            html_str += (
                '<th style="text-align:center">'
                '<td style="vertical-align:top">'
            )
            html_str += f"<h3>{title}</h3>"
            html_str += df_.to_html().replace(
                "table", 'table style="display:inline"'
            )
            html_str += "</td></th><br>"
            html_str += f"{len(df)} rows x {len(df.columns)} columns"
        display_html(html_str, raw=True)
