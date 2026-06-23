"""JSON serialization helpers for sdt-netval pipeline reports.

Converts pipeline output dictionaries (DataFrames + metadata) into
human-readable JSON files suitable for archiving and reproducibility.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_report_json(
    report: Dict[str, Any],
    path: Union[str, Path],
    *,
    indent: int = 2,
) -> None:
    """Serialize a pipeline report dictionary to a JSON file.

    Handles DataFrames (converted to list-of-records), numpy scalars,
    and NaN/Inf values (converted to None for JSON compliance).

    Args:
        report: Dictionary as returned by StageAValidator.full_stability_report()
            or StageBAnalyzer.full_sensitivity_report(). Values may be DataFrames
            or scalars.
        path: Destination JSON file path. Parent directories are created if needed.
        indent: JSON indentation level (default: 2).
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {k: _convert(v) for k, v in report.items()}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=indent, ensure_ascii=False)

    logger.info("Report saved as JSON — file: '%s'", path)


def _convert(obj: Any) -> Any:
    """Recursively convert an object to a JSON-serializable form."""
    if isinstance(obj, pd.DataFrame):
        # orient="records" → list of {col: val} dicts, one per row
        return json.loads(obj.to_json(orient="records", double_precision=6))
    if isinstance(obj, pd.Series):
        return json.loads(obj.to_json(double_precision=6))
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    return obj
