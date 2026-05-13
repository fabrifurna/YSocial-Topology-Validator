"""Stage A topological stability validation across N simulation runs.

Orchestrates loading, metric extraction, aggregation, and statistical
analysis of network topologies to assess stability of graph properties
under fixed LLM parameters.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Union, Optional, Dict, Any

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


class StageAValidator:
    """Orchestrates topological validation of N simulation runs (Stage A).

    Loads N SQLite databases from a directory, extracts topological metrics
    via YSocialGraphBuilder and YSocialTopometrics, aggregates into DataFrame,
    and computes stability statistics (mean, std, 95% confidence intervals).

    Args:
        data_dir: Path to directory containing .sqlite files.

    Raises:
        FileNotFoundError: If directory does not exist.
        ValueError: If no .sqlite files found in directory.
    """

    def __init__(self, data_dir: Union[str, Path]) -> None:
        self.data_dir = Path(data_dir).resolve()
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: '{self.data_dir}'")

        self.db_paths = self._discover_databases()
        if not self.db_paths:
            raise ValueError(f"No .sqlite files found in '{self.data_dir}'")

        self.results: list[Dict[str, Any]] = []
        logger.info(
            "StageAValidator initialized — %d databases found in '%s'",
            len(self.db_paths), self.data_dir.name,
        )

    def _discover_databases(self) -> list[Path]:
        """Discover and sort .sqlite files in directory alphabetically."""
        paths = sorted(self.data_dir.glob("*.sqlite"))
        logger.debug("Databases discovered: %s", [p.name for p in paths])
        return paths

    def _extract_run_id(self, db_path: Path) -> str:
        return db_path.stem

    def _process_single_run(self, db_path: Path) -> dict:
        G = YSocialGraphBuilder(str(db_path)).load_follower_graph()
        report = YSocialTopometrics(G).generate_full_report()
        return report

    def process_runs(self) -> pd.DataFrame:
        """Execute complete pipeline on all databases with progress bar.

        For each .sqlite file in directory:
        1. Extract run_id from filename
        2. Load graph via YSocialGraphBuilder
        3. Compute metrics via YSocialTopometrics
        4. Append run_id and db_path to report

        Failed runs are logged and skipped without interrupting iteration.

        Returns:
            DataFrame with one row per run and one column per metric.
            Failed runs include an 'error' column with exception message.
        """
        np.random.seed(42)  # Reproducibility
        self.results = []

        for db_path in tqdm(
            self.db_paths,
            desc="Processing Stage A runs",
            unit="run",
        ):
            run_id = self._extract_run_id(db_path)
            try:
                report = self._process_single_run(db_path)
                report["run_id"] = run_id
                report["db_path"] = str(db_path)
                self.results.append(report)
                logger.info(
                    "Run '%s' completed — nodes: %d, edges: %d",
                    run_id,
                    report.get("num_nodes", "?"),
                    report.get("num_edges", "?"),
                )
            except Exception as exc:
                logger.error(
                    "Error processing run '%s' (%s): %s",
                    run_id, db_path.name, exc,
                )
                self.results.append({
                    "run_id": run_id,
                    "db_path": str(db_path),
                    "error": str(exc),
                })

        df = pd.DataFrame(self.results)
        n_ok = (~df.get("error", pd.Series([None] * len(df))).notna()).sum()
        logger.info(
            "Stage A completed — %d/%d runs successful.",
            n_ok, len(df),
        )
        return df

    def save_raw_results(self, output_csv: Union[str, Path]) -> None:
        """Save raw results DataFrame to CSV.

        Args:
            output_csv: Output CSV file path (e.g., data/processed/stage_a_raw.csv).

        Raises:
            ValueError: If no runs have been processed yet.
        """
        if not self.results:
            raise ValueError("No results available. Run process_runs() first.")

        df = pd.DataFrame(self.results)
        output_csv = Path(output_csv).resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_csv, index=False)
        logger.info(
            "Raw results saved — file: '%s', rows: %d",
            output_csv, len(df),
        )

    def compute_stability_metrics(
        self,
        metrics: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Compute stability statistics for key topological metrics.

        For each requested metric (default: alpha_in_degree, modularity,
        average_clustering, density), computes:

        - **n**: number of non-missing observations
        - **mean**: arithmetic mean
        - **std**: sample standard deviation (ddof=1)
        - **sem**: standard error of the mean
        - **ci95_low, ci95_high**: 95% confidence interval via t-distribution
          (appropriate for N ≤ 30)
        - **cv**: coefficient of variation (σ/μ, relative stability indicator)

        Args:
            metrics: Metrics to aggregate. Defaults to KEY_METRICS
                (alpha_in_degree, modularity, average_clustering, density).

        Returns:
            DataFrame indexed by metric name with columns
            ['n', 'mean', 'std', 'sem', 'ci95_low', 'ci95_high', 'cv'].

        Raises:
            ValueError: If no results are available.
            KeyError: If a requested metric is absent from the dataset.
        """
        if not self.results:
            raise ValueError("No results available. Run process_runs() first.")

        metrics = metrics or KEY_METRICS
        df = pd.DataFrame(self.results)

        rows = []
        for metric in metrics:
            if metric not in df.columns:
                logger.warning("Metric '%s' not found in DataFrame. Skipped.", metric)
                continue

            series = df[metric].dropna()
            if series.empty:
                logger.warning("Metric '%s' contains only NaN values. Skipped.", metric)
                continue

            n = len(series)
            m = float(series.mean())
            s = float(series.std(ddof=1)) if n > 1 else 0.0
            sem = s / np.sqrt(n) if n > 1 else 0.0
            t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
            cv = s / m if m != 0 else None

            rows.append({
                "metric": metric,
                "n": n,
                "mean": round(m, 6),
                "std": round(s, 6),
                "sem": round(sem, 6),
                "ci95_low": round(m - t_crit * sem, 6),
                "ci95_high": round(m + t_crit * sem, 6),
                "cv": round(cv, 4) if cv is not None else None,
            })

        result_df = pd.DataFrame(rows).set_index("metric")
        logger.info(
            "Stability statistics computed — %d metrics.",
            len(result_df),
        )
        return result_df

    def full_stability_report(
        self,
        metrics: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """Assemble complete topological stability report (Stage A).

        Packages raw results and aggregated statistics into a single
        dictionary for downstream analysis and export.

        Args:
            metrics: Metrics to include in stability statistics.
                Defaults to KEY_METRICS.

        Returns:
            Dictionary with keys:
            - ``'raw'``: raw DataFrame (N rows, one per run)
            - ``'stability'``: aggregated statistics DataFrame, one row per metric
        """
        if not self.results:
            raise ValueError("No results available. Run process_runs() first.")

        raw_df = pd.DataFrame(self.results)
        stability_df = self.compute_stability_metrics(metrics=metrics)

        return {
            "raw": raw_df,
            "stability": stability_df,
        }
