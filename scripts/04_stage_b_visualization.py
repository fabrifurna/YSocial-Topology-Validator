"""Comparative figures for Stage B sensitivity analysis.

Produces per-condition modularity plots (the primary indicator of polarization
and Average Persona Bias), with a reference line for the baseline condition (c0).
"""

import logging
import os
import sys
from pathlib import Path
from typing import Union, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

logger = logging.getLogger(__name__)

CONDITION_LABELS = {
    "c0": "Baseline",
    "c1": "Neutral Persona",
    "c3": "Low Temperature",
    "c4": "High Temperature",
    "c8": "Aggressive RecSys",
}


class StageBVisualizer:
    """Generates comparative visualizations for Stage B (Sensitivity).

    Reads raw Stage B results (110 runs across 11 conditions) and produces
    comparative modularity plots — the key network polarization indicator
    under LLM and RecSys parameter variations.

    Args:
        csv_path: Path to the raw CSV file (data/01_processed/stage_b_raw.csv).

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV is empty.
    """

    def __init__(self, csv_path: Union[str, Path]) -> None:
        self.csv_path = Path(csv_path).resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: '{self.csv_path}'")

        self.df = pd.read_csv(self.csv_path)
        if self.df.empty:
            raise ValueError("CSV file is empty.")

        logger.info("stage B data loaded: %d rows", len(self.df))

    def plot_modularity_comparison(
        self,
        output_png: Union[str, Path],
        selected_conditions: Optional[List[str]] = None,
    ) -> None:
        """Generate a comparative boxplot of modularity across selected conditions.

        Boxplot with overlaid strip plot (10 points per condition). A dashed
        red reference line marks the baseline (c0) mean for visual comparison.

        Args:
            output_png: Output PNG path
                (e.g., data/01_processed/stage_b_modularity_comparison.png).
            selected_conditions: Condition IDs to plot (default: c0, c1, c3, c4, c8).
                All IDs must be present in the loaded DataFrame.

        Raises:
            ValueError: If no data remains after filtering.
        """
        if selected_conditions is None:
            selected_conditions = list(CONDITION_LABELS.keys())

        df_filtered = self.df[self.df["condition"].isin(selected_conditions)].copy()

        if df_filtered.empty:
            logger.error(
                "No data found for selected conditions: %s",
                selected_conditions,
            )
            raise ValueError("No data available after filtering by selected conditions.")

        df_filtered["condition_label"] = df_filtered["condition"].map(CONDITION_LABELS)

        missing_labels = df_filtered["condition_label"].isna().sum()
        if missing_labels > 0:
            logger.warning(
                "%d rows have a missing label (condition ID not in CONDITION_LABELS).",
                missing_labels,
            )
            df_filtered = df_filtered.dropna(subset=["condition_label"])

        condition_order = [CONDITION_LABELS[c] for c in selected_conditions if c in CONDITION_LABELS]
        df_filtered["condition_label"] = pd.Categorical(
            df_filtered["condition_label"],
            categories=condition_order,
            ordered=True,
        )

        baseline_mean = self.df[self.df["condition"] == "c0"]["modularity"].mean()

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(11, 7))

        sns.boxplot(
            data=df_filtered,
            x="condition_label",
            y="modularity",
            ax=ax,
            palette="Set2",
            width=0.6,
        )

        sns.stripplot(
            data=df_filtered,
            x="condition_label",
            y="modularity",
            ax=ax,
            alpha=0.6,
            color="black",
            size=6,
            jitter=True,
        )

        ax.axhline(
            y=baseline_mean,
            linestyle="--",
            color="red",
            linewidth=2.0,
            alpha=0.7,
            label=f"Baseline Mean: {baseline_mean:.4f}",
        )

        ax.set_xlabel("Condition", fontsize=12, fontweight="bold")
        ax.set_ylabel("Modularity (Q)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Modularity Sensitivity Analysis (Stage B)",
            fontsize=14, fontweight="bold",
        )
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)

        output_png = Path(output_png).resolve()
        output_png.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        logger.info("Figure saved — file: '%s'", output_png)
        plt.close()

    def get_summary_statistics(
        self,
        selected_conditions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Return a per-condition modularity summary (mean, std, sem).

        Args:
            selected_conditions: Condition IDs to summarize. Defaults to
                CONDITION_LABELS keys (c0, c1, c3, c4, c8).

        Returns:
            DataFrame with one row per condition and descriptive statistics.
        """
        if selected_conditions is None:
            selected_conditions = list(CONDITION_LABELS.keys())

        df_filtered = self.df[self.df["condition"].isin(selected_conditions)].copy()

        summary_rows = []
        for condition_id in selected_conditions:
            cond_data = df_filtered[df_filtered["condition"] == condition_id]["modularity"].dropna()
            if cond_data.empty:
                continue

            n = len(cond_data)
            m = float(cond_data.mean())
            s = float(cond_data.std(ddof=1)) if n > 1 else 0.0
            sem = s / (n ** 0.5) if n > 1 else 0.0

            summary_rows.append({
                "condition": condition_id,
                "label": CONDITION_LABELS.get(condition_id, condition_id),
                "n": n,
                "mean": round(m, 6),
                "std": round(s, 6),
                "sem": round(sem, 6),
            })

        return pd.DataFrame(summary_rows)
