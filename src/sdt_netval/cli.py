"""Command-line interface for sdt-netval.

Usage examples:
    sdt-netval analyze simulation.sqlite
    sdt-netval analyze edges.csv --output results/
    sdt-netval stage-a data/benchmark_runs/
    sdt-netval stage-a data/benchmark_runs/ --output results/ --format json
    sdt-netval stage-b data/sensitivity_runs/ --output results/
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s — %(message)s",
)


def _setup_verbose(verbose: bool) -> None:
    if verbose:
        logging.getLogger().setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def _cmd_analyze(args: argparse.Namespace) -> int:
    from sdt_netval import load_network, GraphMetrics
    from sdt_netval.export import save_report_json

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: '{path}'", file=sys.stderr)
        return 1

    _setup_verbose(args.verbose)

    print(f"\nLoading '{path.name}' …")
    try:
        G = load_network(path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"  nodes: {G.number_of_nodes():,}   edges: {G.number_of_edges():,}\n")

    report = GraphMetrics(G).generate_full_report()

    _print_metrics_table(report)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = path.stem
        save_report_json(report, out_dir / f"{stem}_metrics.json")
        print(f"\nResults saved to '{out_dir}/'")

    return 0


def _print_metrics_table(report: dict) -> None:
    sections = {
        "Network structure": [
            ("num_nodes",     "Nodes"),
            ("num_edges",     "Edges"),
            ("density",       "Density"),
            ("avg_in_degree", "Avg in-degree"),
            ("std_in_degree", "Std in-degree"),
            ("reciprocity",   "Reciprocity"),
        ],
        "Degree distribution (power-law)": [
            ("alpha_in_degree", "Alpha (α)"),
            ("xmin",            "x_min"),
            ("ks_distance",     "KS distance"),
            ("is_scale_free",   "Scale-free (2 < α < 3)"),
            ("powerlaw_error",  "Fit error"),
        ],
        "Community structure": [
            ("average_clustering", "Avg clustering"),
            ("modularity",         "Modularity (Q)"),
            ("num_communities",    "Communities"),
        ],
    }

    for section, fields in sections.items():
        print(f"  {section}")
        print(f"  {'─' * 42}")
        for key, label in fields:
            val = report.get(key)
            if val is None:
                formatted = "n/a"
            elif isinstance(val, bool):
                formatted = "yes" if val else "no"
            elif isinstance(val, float):
                formatted = f"{val:.4f}"
            else:
                formatted = str(val)
            print(f"  {label:<28} {formatted}")
        print()


# ---------------------------------------------------------------------------
# stage-a
# ---------------------------------------------------------------------------

def _cmd_stage_a(args: argparse.Namespace) -> int:
    from sdt_netval.pipeline import StageAValidator
    from sdt_netval.export import save_report_json

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: directory not found: '{data_dir}'", file=sys.stderr)
        return 1

    _setup_verbose(args.verbose)

    try:
        validator = StageAValidator(data_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nStage A — {len(validator.db_paths)} run(s) found in '{data_dir.name}'\n")
    validator.process_runs()

    report = validator.full_stability_report()

    print("=== Stability Statistics ===\n")
    print(report["stability"].to_string())
    print()

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        report["raw"].to_csv(out_dir / "stage_a_raw.csv", index=False)
        report["stability"].to_csv(out_dir / "stage_a_stability.csv")

        if args.format == "json":
            save_report_json(report, out_dir / "stage_a_report.json")

        print(f"Results saved to '{out_dir}/'")

    return 0


# ---------------------------------------------------------------------------
# stage-b
# ---------------------------------------------------------------------------

def _cmd_stage_b(args: argparse.Namespace) -> int:
    from sdt_netval.pipeline import StageBAnalyzer
    from sdt_netval.export import save_report_json

    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        print(f"Error: directory not found: '{base_dir}'", file=sys.stderr)
        return 1

    _setup_verbose(args.verbose)

    try:
        analyzer = StageBAnalyzer(base_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nStage B — {len(analyzer.conditions)} condition(s): {analyzer.conditions}\n")
    analyzer.process_all_runs()

    report = analyzer.full_sensitivity_report()

    print("=== Per-condition Aggregated Metrics ===\n")
    print(report["aggregated"].to_string(index=False))
    print()

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        report["raw"].to_csv(out_dir / "stage_b_raw.csv", index=False)
        report["aggregated"].to_csv(out_dir / "stage_b_aggregated.csv", index=False)

        if args.format == "json":
            save_report_json(report, out_dir / "stage_b_report.json")

        print(f"Results saved to '{out_dir}/'")

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sdt-netval",
        description="Universal topology validator for multi-agent simulation networks.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed logs.")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # -- analyze --
    p_analyze = sub.add_parser(
        "analyze",
        help="Compute topology metrics for a single network file (.sqlite/.db, .csv, .zip).",
    )
    p_analyze.add_argument("file", help="Path to the network file.")
    p_analyze.add_argument(
        "--output", "-o", metavar="DIR",
        help="Directory where results are saved (JSON). Omit to only print to screen.",
    )
    p_analyze.set_defaults(func=_cmd_analyze)

    # -- stage-a --
    p_a = sub.add_parser(
        "stage-a",
        help="Run Stage A stability pipeline on a directory of simulation files.",
    )
    p_a.add_argument("data_dir", help="Directory containing simulation files.")
    p_a.add_argument("--output", "-o", metavar="DIR", help="Directory for output CSVs.")
    p_a.add_argument(
        "--format", choices=["csv", "json"], default="csv",
        help="Output format (default: csv).",
    )
    p_a.set_defaults(func=_cmd_stage_a)

    # -- stage-b --
    p_b = sub.add_parser(
        "stage-b",
        help="Run Stage B sensitivity pipeline on a nested conditions directory.",
    )
    p_b.add_argument("base_dir", help="Directory containing condition subdirectories (c0, c1, …).")
    p_b.add_argument("--output", "-o", metavar="DIR", help="Directory for output CSVs.")
    p_b.add_argument(
        "--format", choices=["csv", "json"], default="csv",
        help="Output format (default: csv).",
    )
    p_b.set_defaults(func=_cmd_stage_b)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
