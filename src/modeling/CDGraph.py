"""Cold-drawing route graph: DAG construction + DFS route enumeration.

Ported verbatim (logic unchanged) from the deterministic MBC repository
(``mbc-cold-drawing-deterministic``), so the probabilistic controller reuses
the exact same combinatorial machinery formalised in
``mathematical_formulation_cd_DETERMINISTIC.pdf``:

* nodes are reachable diameters (scaled integers to avoid float drift);
* edges are admissible passes (step in S, reduction ratio <= rr_max);
* routes are DFS-enumerated paths D0 -> Df (<= max_passes, <= max_sequences);
* ``generate_all_sequences_df`` returns one row per (sequence_id, pass_number)
  with the per-pass geometry the surrogates consume:
  original/initial/final diameter, step, total_strain, reduction_ratio.

Lives at: src/modeling/CDGraph.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

from src.utils import get_reduction_ratio, get_total_drawing_strain


@dataclass(frozen=True)
class Edge:
    """Represents the possible pass between the nodes that are
    the diameters"""
    d_in_i: int
    d_out_i: int
    step_i: int
    total_strain: float
    reduction_ratio: float


class ColdDrawingMBCHelper:
    @staticmethod
    def _max_decimals(x: float) -> int:
        """
        Get the number of decimal digits used in a float representation.

        Args:
            x (float): Input value.

        Returns:
            int: Number of decimal digits.
        """
        s = f"{x:.12f}".rstrip("0").rstrip(".")
        if "." not in s:
            return 0
        return len(s.split(".")[1])

    @staticmethod
    def _build_scale(d0: float, df: float, steps: Sequence[float]) -> int:
        """
        Build an integer scaling factor to represent diameters as ints.

        Args:
            d0 (float): Original diameter (mm).
            df (float): Final diameter (mm).
            steps (Sequence[float]): Allowed step sizes (mm).

        Returns:
            int: Scale factor (10^decimals).
        """
        decs = [
            ColdDrawingMBCHelper._max_decimals(d0),
            ColdDrawingMBCHelper._max_decimals(df),
        ]
        decs.extend(ColdDrawingMBCHelper._max_decimals(s) for s in steps)
        return 10 ** max(decs)

    @staticmethod
    def _to_int(x: float, scale: int) -> int:
        """
        Convert float to scaled integer.

        Args:
            x (float): Value in mm.
            scale (int): Scale factor.

        Returns:
            int: Scaled integer.
        """
        return int(round(x * scale))

    @staticmethod
    def _to_float(xi: int, scale: int) -> float:
        """
        Convert scaled integer to float.

        Args:
            xi (int): Scaled integer.
            scale (int): Scale factor.

        Returns:
            float: Value in mm.
        """
        return xi / scale

    @staticmethod
    def build_transitions(
        original_diameter: float,
        final_diameter: float,
        steps: Sequence[float],
        rr_max: float,
    ) -> Tuple[int, int, int, Dict[int, List[Edge]]]:
        """
        Build a DAG adjacency list of valid cold-drawing passes.

        Nodes are diameters represented as scaled integers. Edges represent
        a single pass using one step size. An edge is kept only if the per-pass
        reduction ratio is <= rr_max.

        Args:
            original_diameter (float): Start diameter (mm).
            final_diameter (float): Target diameter (mm).
            steps (Sequence[float]): Allowed step sizes (mm).
            rr_max (float): Max allowed reduction ratio per pass (%).

        Returns:
            Tuple[int, int, int, Dict[int, List[Edge]]]:
                (d0_i, df_i, scale, adjacency)
        """
        if original_diameter <= 0 or final_diameter <= 0:
            raise ValueError("Diameters must be > 0.")
        if final_diameter > original_diameter:
            raise ValueError("final_diameter must be <= original_diameter.")
        if not steps:
            raise ValueError("steps cannot be empty.")

        scale = ColdDrawingMBCHelper._build_scale(
            original_diameter, final_diameter, steps
        )
        d0_i = ColdDrawingMBCHelper._to_int(original_diameter, scale)
        df_i = ColdDrawingMBCHelper._to_int(final_diameter, scale)
        steps_i = sorted(
            {ColdDrawingMBCHelper._to_int(s, scale) for s in steps if s > 0}
        )

        adjacency: Dict[int, List[Edge]] = {}
        frontier: List[int] = [d0_i]
        seen = {d0_i}

        while frontier:
            di = frontier.pop()
            if di == df_i:
                adjacency.setdefault(di, [])
                continue

            d_in = ColdDrawingMBCHelper._to_float(di, scale)
            out_edges: List[Edge] = []

            for si in steps_i:
                dj = di - si
                if dj < df_i:
                    continue

                d_out = ColdDrawingMBCHelper._to_float(dj, scale)
                rr = float(get_reduction_ratio(d_in, d_out))
                if rr > rr_max + 1e-12:
                    continue

                ts = float(get_total_drawing_strain(d_in, d_out))
                out_edges.append(
                    Edge(
                        d_in_i=di,
                        d_out_i=dj,
                        step_i=si,
                        total_strain=ts,
                        reduction_ratio=rr,
                    )
                )

                if dj not in seen:
                    seen.add(dj)
                    frontier.append(dj)

            out_edges.sort(key=lambda e: (e.d_out_i, e.step_i))
            adjacency[di] = out_edges

        adjacency.setdefault(df_i, [])
        return d0_i, df_i, scale, adjacency

    @staticmethod
    def iter_sequences(
        d0_i: int,
        df_i: int,
        adjacency: Dict[int, List[Edge]],
        max_passes: Optional[int] = None,
        max_sequences: Optional[int] = None,
    ) -> Iterator[List[Edge]]:
        """_summary_

        Args:
            d0_i (int): Start diameter node (scaled int).
            df_i (int): Target diameter node (scaled int).
            adjacency (Dict[int, List[Edge]]): DAG adjacency list.
            max_passes (Optional[int], optional): Cap on passes per sequence.
                Defaults to None.
            max_sequences (Optional[int], optional): Cap on sequences yielded.
                Defaults to None.

        Yields:
            Iterator[List[Edge]]: Each yielded item is a list of edges (passes)
        """
        stack: List[Tuple[int, int, List[Edge]]] = [(d0_i, 0, [])]
        produced = 0

        while stack:
            node, idx, path = stack.pop()

            if node == df_i:
                yield path
                produced += 1
                if max_sequences is not None and produced >= max_sequences:
                    return
                continue

            if max_passes is not None and len(path) >= max_passes:
                continue

            edges = adjacency.get(node, [])
            for j in range(len(edges) - 1, idx - 1, -1):
                e = edges[j]
                stack.append((e.d_out_i, 0, path + [e]))

    @staticmethod
    def sequences_to_dataframe(
        sequences: Iterable[List[Edge]],
        scale: int,
        original_diameter: float,
    ) -> pd.DataFrame:
        """
        Convert sequences into a single DataFrame with MultiIndex.

        The MultiIndex is (sequence_id, pass_number). Each row represents one
        pass within a sequence.

        Args:
            sequences (Iterable[List[Edge]]): Iterable of sequences.
            scale (int): Scale factor used for integer diameters.
            original_diameter (float): Original diameter (mm).

        Returns:
            pd.DataFrame: Pass-level data for all sequences.
        """
        records: List[dict] = []
        seq_id = 0

        for seq in sequences:
            seq_id += 1
            for pass_number, e in enumerate(seq, start=1):
                d_in = ColdDrawingMBCHelper._to_float(e.d_in_i, scale)
                d_out = ColdDrawingMBCHelper._to_float(e.d_out_i, scale)
                step = ColdDrawingMBCHelper._to_float(e.step_i, scale)

                records.append(
                    {
                        "sequence_id": seq_id,
                        "pass_number": pass_number,
                        "original_diameter": original_diameter,
                        "initial_diameter": d_in,
                        "final_diameter": d_out,
                        "step": step,
                        "total_strain": e.total_strain,
                        "reduction_ratio": e.reduction_ratio,
                    }
                )

        cols = [
            "original_diameter",
            "initial_diameter",
            "final_diameter",
            "step",
            "total_strain",
            "reduction_ratio",
        ]

        if not records:
            idx = pd.MultiIndex.from_arrays(
                [[], []], names=["sequence_id", "pass_number"]
            )
            return pd.DataFrame(columns=cols, index=idx)

        df = pd.DataFrame.from_records(records)
        df.set_index(["sequence_id", "pass_number"], inplace=True)
        return df[cols]

    @classmethod
    def generate_all_sequences_df(
        cls,
        original_diameter: float,
        final_diameter: float,
        steps: Sequence[float],
        rr_max: float,
        max_passes: Optional[int] = None,
        max_sequences: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate all valid sequences and return a single MultiIndex DataFrame.

        Args:
            original_diameter (float): Start diameter (mm).
            final_diameter (float): Target diameter (mm).
            steps (Sequence[float]): Allowed step sizes (mm).
            rr_max (float): Max allowed reduction ratio per pass (%).
            max_passes (Optional[int], optional): Cap on passes per sequence.
                Defaults to None.
            max_sequences (Optional[int], optional): Cap on sequences generated
                Defaults to None.

        Returns:
            pd.DataFrame: MultiIndex (sequence_id, pass_number) table.
        """
        d0_i, df_i, scale, adj = cls.build_transitions(
            original_diameter=original_diameter,
            final_diameter=final_diameter,
            steps=steps,
            rr_max=rr_max,
        )

        seq_iter = cls.iter_sequences(
            d0_i=d0_i,
            df_i=df_i,
            adjacency=adj,
            max_passes=max_passes,
            max_sequences=max_sequences,
        )

        return cls.sequences_to_dataframe(
            sequences=seq_iter,
            scale=scale,
            original_diameter=original_diameter,
        )
