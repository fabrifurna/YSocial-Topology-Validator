"""sdt-netval: Universal topology validator for multi-agent simulation networks.

Quick start::

    from sdt_netval import load_network, GraphMetrics

    G = load_network("simulation.sqlite")   # or .zip or .csv
    report = GraphMetrics(G).generate_full_report()
"""

from sdt_netval.adapters import load_network
from sdt_netval.core.metrics import GraphMetrics
from sdt_netval.pipeline import StageAValidator, StageBAnalyzer
from sdt_netval.export import save_report_json

__all__ = ["load_network", "GraphMetrics", "StageAValidator", "StageBAnalyzer", "save_report_json"]
__version__ = "0.1.0"
