"""Academic figures for Stage A topological stability.

Produces high-resolution (300 dpi) distribution plots of key metrics
across N simulation runs, supporting visual assessment of inter-run variability.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Union, Optional, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

logger = logging.getLogger(__name__)

KEY_METRICS = {
    "alpha_in_degree": "Power-law Exponent (α)",
    "modularity": "Modularity (Q)",
    "average_clustering": "Average Clustering",
    "density": "Network Density",
}


class StageAVisualizer:
    """Generates academic-style figures for Stage A topological stability.

    Reads raw Stage A results (N runs) and produces a boxplot grid with
    overlaid strip plots, one subplot per key topological metric.

    Args:
        csv_path: Path to the raw CSV file (data/01_processed/stage_a_raw.csv).

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV is empty or missing expected metric columns.
    """

    def __init__(self, csv_path: Union[str, Path]) -> None:
        self.csv_path = Path(csv_path).resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: '{self.csv_path}'")

        self.df = pd.read_csv(self.csv_path)
        if self.df.empty:
            raise ValueError("CSV file is empty.")

        missing = [m for m in KEY_METRICS.keys() if m not in self.df.columns]
        if missing:
            logger.warning(
                "Metrics missing from CSV: %s. Proceeding with available columns.",
                missing,
            )

        logger.info(
            "StageAVisualizer initialized — CSV loaded with %d rows.",
            len(self.df),
        )

    def plot_stability_distributions(self, output_png: Union[str, Path]) -> None:
        """Generate a 2×2 boxplot grid showing per-metric distributions.

        Each subplot contains a boxplot (quartiles, median, outliers) with
        an overlaid strip plot (alpha=0.6) showing individual run observations.

        Args:
            output_png: Output PNG path (e.g., data/01_processed/stage_a_stability.png).
        """
        sns.set_style("whitegrid")
        sns.set_palette("Set2")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, (metric, label) in enumerate(KEY_METRICS.items()):
            if metric not in self.df.columns:
                logger.warning(
                    "Metric '%s' not found in DataFrame. Subplot skipped.",
                    metric,
                )
                axes[idx].text(
                    0.5, 0.5,
                    f"Metric '{metric}' not available",
                    ha="center", va="center",
                    transform=axes[idx].transAxes,
                )
                axes[idx].set_title(label, fontsize=12, fontweight="bold")
                continue

            ax = axes[idx]

            sns.boxplot(
                data=self.df,
                y=metric,
                ax=ax,
                width=0.5,
                palette="Set2",
            )

            sns.stripplot(
                data=self.df,
                y=metric,
                ax=ax,
                alpha=0.6,
                color="black",
                size=6,
                jitter=True,
            )

            ax.set_title(label, fontsize=13, fontweight="bold")
            ax.set_ylabel(label, fontsize=11)
            ax.set_xlabel("", fontsize=0)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_png = Path(output_png).resolve()
        output_png.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        logger.info("Figure saved — file: '%s'", output_png)
        plt.close()

    def plot_summary_statistics(
        self,
        stability_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> str:
        """Return a formatted text summary of stability statistics.

        Args:
            stability_metrics: Pre-computed statistics dictionary (mean, std, cv
                per metric). If None, computed on-the-fly from the loaded DataFrame.

        Returns:
            Formatted string report with mean, std, and CV per metric.
        """
        if stability_metrics is None:
            stability_metrics = {}
            for metric in KEY_METRICS.keys():
                if metric not in self.df.columns:
                    continue
                series = self.df[metric].dropna()
                if series.empty:
                    continue
                m = float(series.mean())
                s = float(series.std(ddof=1))
                cv = s / m if m != 0 else None
                stability_metrics[metric] = {
                    "mean": m,
                    "std": s,
                    "cv": cv,
                }

        report_lines = ["=== Stability Report (Stage A) ===\n"]
        for metric, stats in stability_metrics.items():
            label = KEY_METRICS.get(metric, metric)
            mean = stats.get("mean", "N/A")
            std = stats.get("std", "N/A")
            cv = stats.get("cv", "N/A")
            report_lines.append(
                f"{label:.<30} mean={mean:.4f}, std={std:.4f}, CV={cv:.4f}"
                if isinstance(mean, float) else
                f"{label:.<30} {mean}, {std}, {cv}"
            )
        report_lines.append("")
        return "\n".join(report_lines)
