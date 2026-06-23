"""Stage A topological stability pipeline.

Orchestrates loading, metric extraction, aggregation, and statistical
analysis of network topologies to assess stability of graph properties
under fixed simulation parameters.
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


class StageAValidator:
    """Orchestrates topological validation of N simulation runs (Stage A).

    Loads N simulation databases from a directory, extracts topological metrics
    via load_network and GraphMetrics, aggregates into DataFrame,
    and computes stability statistics (mean, std, 95% confidence intervals).

    Args:
        data_dir: Path to directory containing simulation files (.sqlite, .csv, .zip).

    Raises:
        FileNotFoundError: If directory does not exist.
        ValueError: If no supported simulation files found in directory.
    """

    _SUPPORTED_EXTENSIONS = (".sqlite", ".csv", ".zip")

    def __init__(self, data_dir: Union[str, Path]) -> None:
        self.data_dir = Path(data_dir).resolve()
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: '{self.data_dir}'")

        self.db_paths = self._discover_databases()
        if not self.db_paths:
            raise ValueError(
                f"No supported simulation files found in '{self.data_dir}'. "
                f"Accepted extensions: {', '.join(self._SUPPORTED_EXTENSIONS)}"
            )

        self.results: List[Dict[str, Any]] = []
        logger.info(
            "StageAValidator initialized — %d files found in '%s'",
            len(self.db_paths), self.data_dir.name,
        )

    def _discover_databases(self) -> List[Path]:
        """Discover and sort simulation files in directory alphabetically."""
        paths = sorted(
            p for p in self.data_dir.iterdir()
            if p.suffix.lower() in self._SUPPORTED_EXTENSIONS
        )
        logger.debug("Files discovered: %s", [p.name for p in paths])
        return paths

    def _extract_run_id(self, db_path: Path) -> str:
        return db_path.stem

    def _process_single_run(self, db_path: Path) -> Dict[str, Any]:
        G = load_network(db_path)
        return GraphMetrics(G).generate_full_report()

    def process_runs(self) -> pd.DataFrame:
        """Execute complete pipeline on all simulation files with progress bar.

        For each file in the directory:
        1. Extract run_id from filename stem
        2. Load graph via load_network (format auto-detected)
        3. Compute metrics via GraphMetrics
        4. Append run_id and db_path to report

        Failed runs are logged and skipped without interrupting the iteration.

        Returns:
            DataFrame with one row per run and one column per metric.
            Failed runs include an 'error' column with the exception message.
        """
        np.random.seed(42)
        self.results = []

        for db_path in tqdm(self.db_paths, desc="Processing Stage A runs", unit="run"):
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
                logger.error("Error processing run '%s' (%s): %s", run_id, db_path.name, exc)
                self.results.append({
                    "run_id": run_id,
                    "db_path": str(db_path),
                    "error": str(exc),
                })

        df = pd.DataFrame(self.results)
        n_ok = df.get("error", pd.Series(dtype=object)).isna().sum()
        logger.info("Stage A completed — %d/%d runs successful.", n_ok, len(df))
        return df

    def save_raw_results(self, output_csv: Union[str, Path]) -> None:
        """Save raw results DataFrame to CSV.

        Args:
            output_csv: Output CSV file path.

        Raises:
            ValueError: If no runs have been processed yet.
        """
        if not self.results:
            raise ValueError("No results available. Run process_runs() first.")

        output_csv = Path(output_csv).resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.results)
        df.to_csv(output_csv, index=False)
        logger.info("Raw results saved — file: '%s', rows: %d", output_csv, len(df))

    def compute_stability_metrics(
        self,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute stability statistics for key topological metrics.

        For each requested metric, computes:
        - **n**: number of non-missing observations
        - **mean**: arithmetic mean
        - **std**: sample standard deviation (ddof=1)
        - **sem**: standard error of the mean
        - **ci95_low, ci95_high**: 95% CI via t-distribution (appropriate for N ≤ 30)
        - **cv**: coefficient of variation (σ/μ)

        Args:
            metrics: Metrics to aggregate. Defaults to KEY_METRICS.

        Returns:
            DataFrame indexed by metric name with columns
            ['n', 'mean', 'std', 'sem', 'ci95_low', 'ci95_high', 'cv'].

        Raises:
            ValueError: If no results are available.
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
        logger.info("Stability statistics computed — %d metrics.", len(result_df))
        return result_df

    def full_stability_report(
        self,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Assemble complete topological stability report (Stage A).

        Args:
            metrics: Metrics to include in stability statistics. Defaults to KEY_METRICS.

        Returns:
            Dictionary with keys:
            - ``'raw'``: raw DataFrame (N rows, one per run)
            - ``'stability'``: aggregated statistics DataFrame, one row per metric
        """
        if not self.results:
            raise ValueError("No results available. Run process_runs() first.")

        return {
            "raw": pd.DataFrame(self.results),
            "stability": self.compute_stability_metrics(metrics=metrics),
        }
