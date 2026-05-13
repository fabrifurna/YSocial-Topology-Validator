"""Stage B sensitivity analysis pipeline.

Handles the nested structure of N conditions (c0…c10) × M runs per condition.
Produces a raw dataset with all metrics and an aggregated report with
per-condition summary statistics.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ysocial_validator.ingestion import YSocialGraphBuilder
from ysocial_validator.topometrics import YSocialTopometrics

logger = logging.getLogger(__name__)

KEY_METRICS = [
    "alpha_in_degree",
    "modularity",
    "average_clustering",
    "density",
]


class StageBAnalyzer:
    """Orchestrates sensitivity analysis across N conditions × M runs.

    Manages the nested directory structure: each condition subdirectory
    (c0, c1, …, c10) contains M simulation runs (.sqlite files). Produces
    a raw dataset (N×M rows) and a per-condition aggregated report.

    Args:
        base_dir: Path to the directory containing condition subdirectories
            (e.g., data/raw/sensitivity_runs/).

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no condition subdirectories are found.
    """

    def __init__(self, base_dir: Union[str, Path]) -> None:
        self.base_dir = Path(base_dir).resolve()
        if not self.base_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: '{self.base_dir}'")

        self.conditions = self._discover_conditions()
        if not self.conditions:
            raise ValueError(
                f"No condition subdirectories found in '{self.base_dir}'. "
                "Expected names such as 'c0', 'c1', ..., 'c10'."
            )

        self.results: List[Dict[str, Any]] = []
        logger.info("found %d condition dirs in '%s'", len(self.conditions), self.base_dir.name)

    def _discover_conditions(self) -> List[str]:
        """Discover and sort condition directories alphabetically."""
        cond_dirs = sorted(
            [d for d in self.base_dir.iterdir() if d.is_dir()
             and (d.name.startswith("c") or d.name.startswith("condition_"))]
        )
        cond_ids = [d.name for d in cond_dirs]
        logger.debug("Conditions discovered: %s", cond_ids)
        return cond_ids

    def _discover_runs_in_condition(self, condition_dir: Path) -> List[Path]:
        return sorted(condition_dir.glob("*.sqlite"))

    def _extract_run_id(self, db_path: Path) -> str:
        return db_path.stem

    def _process_single_run(self, db_path: Path) -> dict:
        G = YSocialGraphBuilder(str(db_path)).load_follower_graph()
        report = YSocialTopometrics(G).generate_full_report()
        return report

    def process_all_runs(self) -> pd.DataFrame:
        """Execute the full pipeline across all conditions and runs.

        Nested iteration over 11 conditions × 10 runs (110 total), tracked
        by a single tqdm progress bar. Each record includes condition ID and
        run_id alongside all topological metrics.

        Returns:
            DataFrame with one row per run (110 rows expected). Failed runs
            include an 'error' column instead of metric values.
        """
        np.random.seed(42)
        self.results = []

        total_runs = sum(
            len(self._discover_runs_in_condition(self.base_dir / cond))
            for cond in self.conditions
        )

        with tqdm(total=total_runs, desc="Processing Stage B runs", unit="run") as pbar:
            for condition_id in self.conditions:
                condition_dir = self.base_dir / condition_id
                db_paths = self._discover_runs_in_condition(condition_dir)

                for db_path in db_paths:
                    run_id = self._extract_run_id(db_path)
                    try:
                        report = self._process_single_run(db_path)
                        report["condition"] = condition_id
                        report["run_id"] = run_id
                        report["db_path"] = str(db_path)
                        self.results.append(report)
                        logger.debug(
                            "Condition '%s', run '%s' completed — nodes: %d, edges: %d",
                            condition_id, run_id,
                            report.get("num_nodes", "?"),
                            report.get("num_edges", "?"),
                        )
                    except Exception as exc:
                        logger.error(
                            "Error in condition '%s', run '%s' (%s): %s",
                            condition_id, run_id, db_path.name, exc,
                        )
                        self.results.append({
                            "condition": condition_id,
                            "run_id": run_id,
                            "db_path": str(db_path),
                            "error": str(exc),
                        })
                    pbar.update(1)

        df = pd.DataFrame(self.results)
        n_ok = (~df.get("error", pd.Series([None] * len(df))).notna()).sum()
        logger.info(
            "Stage B completed — %d/%d runs processed successfully.",
            n_ok, len(df),
        )
        return df

    def save_raw_results(self, output_csv: Union[str, Path]) -> None:
        """Save the raw results DataFrame (all 110 runs) to CSV.

        Args:
            output_csv: Output CSV path (e.g., data/01_processed/stage_b_raw.csv).

        Raises:
            ValueError: If no runs have been processed yet.
        """
        if not self.results:
            raise ValueError("No results available. Run process_all_runs() first.")

        df = pd.DataFrame(self.results)
        output_csv = Path(output_csv).resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_csv, index=False)
        logger.info(
            "Raw results saved — file: '%s', rows: %d",
            output_csv, len(df),
        )

    def compute_aggregated_metrics(
        self,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Aggregate metrics per condition: mean, std, and 95% CI.

        Args:
            metrics: Metrics to aggregate. Defaults to KEY_METRICS.

        Returns:
            DataFrame with one row per condition and columns:
            ['{metric}_mean', '{metric}_std', '{metric}_ci95_low', '{metric}_ci95_high'].
        """
        if not self.results:
            raise ValueError("No results available. Run process_all_runs() first.")

        metrics = metrics or KEY_METRICS
        df = pd.DataFrame(self.results)

        aggregated_rows = []
        for condition_id in sorted(df["condition"].unique()):
            cond_data = df[df["condition"] == condition_id]
            agg_row = {"condition": condition_id}

            for metric in metrics:
                if metric not in cond_data.columns:
                    logger.warning(
                        "Metric '%s' not found for condition '%s'. Skipped.",
                        metric, condition_id,
                    )
                    continue

                series = cond_data[metric].dropna()
                if series.empty:
                    logger.warning(
                        "Metric '%s' contains only NaN values for condition '%s'. Skipped.",
                        metric, condition_id,
                    )
                    continue

                n = len(series)
                m = float(series.mean())
                s = float(series.std(ddof=1)) if n > 1 else 0.0
                sem = s / np.sqrt(n) if n > 1 else 0.0
                t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0

                agg_row[f"{metric}_mean"] = round(m, 6)
                agg_row[f"{metric}_std"] = round(s, 6)
                agg_row[f"{metric}_ci95_low"] = round(m - t_crit * sem, 6)
                agg_row[f"{metric}_ci95_high"] = round(m + t_crit * sem, 6)

            aggregated_rows.append(agg_row)

        result_df = pd.DataFrame(aggregated_rows)
        logger.info(
            "Aggregation completed — %d conditions, %d metrics.",
            len(result_df), len(metrics),
        )
        return result_df

    def save_aggregated_results(self, output_csv: Union[str, Path]) -> None:
        """Save the per-condition aggregated report to CSV.

        Args:
            output_csv: Output CSV path (e.g., data/01_processed/stage_b_aggregated.csv).

        Raises:
            ValueError: If no runs have been processed yet.
        """
        agg_df = self.compute_aggregated_metrics()
        output_csv = Path(output_csv).resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        agg_df.to_csv(output_csv, index=False)
        logger.info(
            "Aggregated results saved — file: '%s', rows: %d",
            output_csv, len(agg_df),
        )

    def full_sensitivity_report(
        self,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Assemble complete sensitivity analysis report (Stage B).

        Args:
            metrics: Metrics to include. Defaults to KEY_METRICS.

        Returns:
            Dictionary with keys:
            - ``'raw'``: raw DataFrame (110 rows, one per run)
            - ``'aggregated'``: aggregated DataFrame (11 rows, one per condition)
        """
        if not self.results:
            raise ValueError("No results available. Run process_all_runs() first.")

        raw_df = pd.DataFrame(self.results)
        agg_df = self.compute_aggregated_metrics(metrics=metrics)

        return {
            "raw": raw_df,
            "aggregated": agg_df,
        }
