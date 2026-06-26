"""Stage A topological stability validation — entry point.

Runs the full Stage A pipeline on the 30 Y-Social benchmark databases and
writes results to data/01_processed/.

Usage:
    python scripts/01_stage_a_validator.py
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sdt_netval.pipeline import StageAValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "00_raw" / "01_legacy_tomasevic" / "benchmark_runs"
OUTPUT_RAW = Path(__file__).resolve().parents[1] / "data" / "01_processed" / "01_legacy_tomasevic" / "stage_a_raw.csv"
OUTPUT_STABILITY = Path(__file__).resolve().parents[1] / "data" / "01_processed" / "01_legacy_tomasevic" / "stage_a_stability.csv"


if __name__ == "__main__":
    logger.info("=== Stage A — Topological Stability ===")
    logger.info("Data directory: %s", DATA_DIR)

    validator = StageAValidator(DATA_DIR)
    validator.process_runs()
    validator.save_raw_results(OUTPUT_RAW)

    stability_df = validator.compute_stability_metrics()
    stability_df.to_csv(OUTPUT_STABILITY)
    logger.info("Stability report saved — file: '%s'", OUTPUT_STABILITY)

    print("\n" + stability_df.to_string())
    logger.info("=== Stage A complete ===")
