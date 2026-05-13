"""Stage B hypothesis testing via Mann-Whitney U test.

Tests whether modularity differs significantly between the baseline condition (c0)
and each experimental condition, using a non-parametric two-sided test appropriate
for N=10 observations per group.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Union, List, Dict, Optional

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

logger = logging.getLogger(__name__)

TEST_CONDITIONS = ["c1", "c3", "c4", "c8"]
CONDITION_LABELS = {
    "c0": "Baseline",
    "c1": "Neutral Persona",
    "c3": "Low Temperature",
    "c4": "High Temperature",
    "c8": "Aggressive RecSys",
}


class StageBHypothesisTesting:
    """Executes Mann-Whitney U tests for Stage B modularity sensitivity.

    Compares the modularity distribution of the baseline condition (c0) against
    each experimental condition. Mann-Whitney U is used instead of a t-test
    because N=10 per group is insufficient to assume normality.

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

        self.results: List[Dict] = []
        logger.info("%d records loaded for hypothesis testing", len(self.df))

    @staticmethod
    def _significance_flag(p_value: float) -> str:
        """Convert a p-value to a standard academic significance marker.

        Args:
            p_value: p-value from a statistical test.

        Returns:
            '***' (p<0.001), '**' (p<0.01), '*' (p<0.05), or 'ns'.
        """
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return "ns"

    def run_tests(
        self,
        baseline_condition: str = "c0",
        test_conditions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Run two-sided Mann-Whitney U tests comparing baseline to each condition.

        For each test condition, extracts modularity values and computes the
        U statistic, p-value, and mean difference relative to the baseline.

        Args:
            baseline_condition: Baseline condition ID (default: "c0").
            test_conditions: Conditions to test against the baseline. Defaults
                to ["c1", "c3", "c4", "c8"].

        Returns:
            DataFrame with columns: Condition, Label, n_baseline, n_test,
            Mean_Baseline, Mean_Test, Mean_Difference, U_statistic, p_value,
            Significance.

        Raises:
            ValueError: If no data is found for the baseline condition.
        """
        if test_conditions is None:
            test_conditions = TEST_CONDITIONS

        baseline_data = (
            self.df[self.df["condition"] == baseline_condition]["modularity"]
            .dropna()
        )
        if baseline_data.empty:
            raise ValueError(
                f"No data found for baseline condition '{baseline_condition}'."
            )

        self.results = []
        baseline_mean = float(baseline_data.mean())

        for test_cond in test_conditions:
            test_data = (
                self.df[self.df["condition"] == test_cond]["modularity"].dropna()
            )
            if test_data.empty:
                logger.warning("No data found for condition '%s'. Skipped.", test_cond)
                continue

            U_stat, p_value = stats.mannwhitneyu(
                baseline_data, test_data, alternative="two-sided"
            )

            mean_test = float(test_data.mean())
            mean_diff = mean_test - baseline_mean

            self.results.append({
                "Condition": test_cond,
                "Label": CONDITION_LABELS.get(test_cond, test_cond),
                "n_baseline": len(baseline_data),
                "n_test": len(test_data),
                "Mean_Baseline": round(baseline_mean, 6),
                "Mean_Test": round(mean_test, 6),
                "Mean_Difference": round(mean_diff, 6),
                "U_statistic": round(float(U_stat), 2),
                "p_value": round(float(p_value), 6),
                "Significance": self._significance_flag(p_value),
            })

            logger.info(
                "%s vs %s: Δμ=%.4f, U=%.2f, p=%.6f %s",
                baseline_condition, test_cond, mean_diff, U_stat, p_value,
                self._significance_flag(p_value),
            )

        return pd.DataFrame(self.results)

    def print_results(self, verbose: bool = True) -> None:
        """Print a formatted results table to stdout.

        Args:
            verbose: If True, show all columns; if False, compact view
                (Condition, Label, Mean_Difference, p_value, Significance only).

        Raises:
            ValueError: If no results are available.
        """
        if not self.results:
            raise ValueError("No results available. Run run_tests() first.")

        df = pd.DataFrame(self.results)

        print("\n" + "=" * 110)
        print("MANN-WHITNEY U TEST RESULTS (Baseline: c0, Modularity)")
        print("=" * 110)

        if verbose:
            display_df = df[
                [
                    "Condition",
                    "Label",
                    "Mean_Baseline",
                    "Mean_Test",
                    "Mean_Difference",
                    "U_statistic",
                    "p_value",
                    "Significance",
                ]
            ]
        else:
            display_df = df[
                ["Condition", "Label", "Mean_Difference", "p_value", "Significance"]
            ]

        print(display_df.to_string(index=False))
        print("=" * 110)
        print(
            "Significance: *** p<0.001 (highly significant)"
            ", ** p<0.01 (very significant)"
            ", * p<0.05 (significant), ns (not significant)"
        )
        print("=" * 110 + "\n")

    def save_results(self, output_csv: Union[str, Path]) -> None:
        """Save hypothesis test results to CSV.

        Args:
            output_csv: Output CSV path (e.g., data/01_processed/stage_b_pvalues.csv).

        Raises:
            ValueError: If no results are available.
        """
        if not self.results:
            raise ValueError("No results available. Run run_tests() first.")

        df = pd.DataFrame(self.results)
        output_csv = Path(output_csv).resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_csv, index=False)
        logger.info(
            "Test results saved — file: '%s', rows: %d",
            output_csv, len(df),
        )

    def get_significant_conditions(self, alpha: float = 0.05) -> List[str]:
        """Return condition IDs with statistically significant modularity shift.

        Args:
            alpha: Significance threshold (default: 0.05).

        Returns:
            List of condition IDs where p-value < alpha.
        """
        if not self.results:
            raise ValueError("No results available. Run run_tests() first.")

        df = pd.DataFrame(self.results)
        return df[df["p_value"] < alpha]["Condition"].tolist()
