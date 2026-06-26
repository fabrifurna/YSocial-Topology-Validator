"""Stage B sensitivity analysis — entry point.

Runs the full Stage B pipeline across 11 OAT conditions (c0–c10),
each containing 10 simulation runs, and writes results to data/01_processed/.

Usage:
    python scripts/03_stage_b_analyzer.py
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sdt_netval.pipeline import StageBAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "00_raw" / "01_legacy_tomasevic" / "sensitivity_runs"
OUTPUT_RAW = Path(__file__).resolve().parents[1] / "data" / "01_processed" / "01_legacy_tomasevic" / "stage_b_raw.csv"
OUTPUT_AGGREGATED = Path(__file__).resolve().parents[1] / "data" / "01_processed" / "01_legacy_tomasevic" / "stage_b_aggregated.csv"


if __name__ == "__main__":
    logger.info("=== Stage B — Sensitivity Analysis ===")
    logger.info("Data directory: %s", DATA_DIR)

    analyzer = StageBAnalyzer(DATA_DIR)
    analyzer.process_all_runs()
    analyzer.save_raw_results(OUTPUT_RAW)
    analyzer.save_aggregated_results(OUTPUT_AGGREGATED)

    logger.info("=== Stage B complete ===")
