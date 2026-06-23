"""Stage B sensitivity analysis pipeline.

Handles the nested structure of N conditions × M runs per condition.
Produces a raw dataset with all metrics and an aggregated report with
per-condition summary statistics.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from sdt_netval import load_network, GraphMetrics

logger = logging.getLogger(__name__)

KEY_METRICS = [
    "alpha_in_degree",
    "modularity",
    "average_clustering",
    "density",
]


class StageBAnalyzer:
    """Orchestrates sensitivity analysis across N conditions × M runs.

    Manages a nested directory structure where each condition subdirectory
    (e.g., c0, c1, …, c10) contains M simulation run files. Produces a raw
    dataset (N×M rows) and a per-condition aggregated report.

    Condition directories are identified by names starting with 'c' or
    'condition_'. All file formats supported by load_network are accepted.

    Args:
        base_dir: Path to the directory containing condition subdirectories.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no condition subdirectories are found.
    """

    _SUPPORTED_EXTENSIONS = (".sqlite", ".csv", ".zip")
    _CONDITION_PREFIXES = ("c", "condition_")

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
        logger.info(
            "StageBAnalyzer initialized — %d conditions found in '%s'",
            len(self.conditions), self.base_dir.name,
        )

    def _discover_conditions(self) -> List[str]:
        """Discover and sort condition directories alphabetically."""
        cond_dirs = sorted(
            d for d in self.base_dir.iterdir()
            if d.is_dir() and any(d.name.startswith(p) for p in self._CONDITION_PREFIXES)
        )
        cond_ids = [d.name for d in cond_dirs]
        logger.debug("Conditions discovered: %s", cond_ids)
        return cond_ids

    def _discover_runs_in_condition(self, condition_dir: Path) -> List[Path]:
        return sorted(
            p for p in condition_dir.iterdir()
            if p.suffix.lower() in self._SUPPORTED_EXTENSIONS
        )

    def _extract_run_id(self, db_path: Path) -> str:
        return db_path.stem

    def _process_single_run(self, db_path: Path) -> Dict[str, Any]:
        G = load_network(db_path)
        return GraphMetrics(G).generate_full_report()

    def process_all_runs(self) -> pd.DataFrame:
        """Execute the full pipeline across all conditions and runs.

        Nested iteration over all conditions × runs, tracked by a single
        tqdm progress bar. Each record includes condition ID and run_id
        alongside all topological metrics.

        Returns:
            DataFrame with one row per run. Failed runs include an 'error'
            column instead of metric values.
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
                            "Condition '%s', run '%s' — nodes: %d, edges: %d",
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
        n_ok = df.get("error", pd.Series(dtype=object)).isna().sum()
        logger.info("Stage B completed — %d/%d runs successful.", n_ok, len(df))
        return df

    def save_raw_results(self, output_csv: Union[str, Path]) -> None:
        """Save the raw results DataFrame to CSV.

        Args:
            output_csv: Output CSV path.

        Raises:
            ValueError: If no runs have been processed yet.
        """
        if not self.results:
            raise ValueError("No results available. Run process_all_runs() first.")

        output_csv = Path(output_csv).resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.results)
        df.to_csv(output_csv, index=False)
        logger.info("Raw results saved — file: '%s', rows: %d", output_csv, len(df))

    def compute_aggregated_metrics(
        self,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Aggregate metrics per condition: mean, std, and 95% CI.

        Args:
            metrics: Metrics to aggregate. Defaults to KEY_METRICS.

        Returns:
            DataFrame with one row per condition and columns
            ['{metric}_mean', '{metric}_std', '{metric}_ci95_low', '{metric}_ci95_high'].

        Raises:
            ValueError: If no results are available.
        """
        if not self.results:
            raise ValueError("No results available. Run process_all_runs() first.")

        metrics = metrics or KEY_METRICS
        df = pd.DataFrame(self.results)
        aggregated_rows = []

        for condition_id in sorted(df["condition"].unique()):
            cond_data = df[df["condition"] == condition_id]
            agg_row: Dict[str, Any] = {"condition": condition_id}

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
                        "Metric '%s' is all-NaN for condition '%s'. Skipped.",
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
            output_csv: Output CSV path.

        Raises:
            ValueError: If no runs have been processed yet.
        """
        agg_df = self.compute_aggregated_metrics()
        output_csv = Path(output_csv).resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        agg_df.to_csv(output_csv, index=False)
        logger.info("Aggregated results saved — file: '%s', rows: %d", output_csv, len(agg_df))

    def full_sensitivity_report(
        self,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Assemble complete sensitivity analysis report (Stage B).

        Args:
            metrics: Metrics to include. Defaults to KEY_METRICS.

        Returns:
            Dictionary with keys:
            - ``'raw'``: raw DataFrame (one row per run)
            - ``'aggregated'``: aggregated DataFrame (one row per condition)
        """
        if not self.results:
            raise ValueError("No results available. Run process_all_runs() first.")

        return {
            "raw": pd.DataFrame(self.results),
            "aggregated": self.compute_aggregated_metrics(metrics=metrics),
        }
