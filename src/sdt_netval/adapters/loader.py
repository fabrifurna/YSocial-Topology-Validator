"""Universal network loader for multi-agent simulation exports.

Single entry-point: load_network(path) auto-detects the source format and
returns a standardized nx.DiGraph. Internal parsers are private — callers
never need to know which one ran.

Supported formats:
  .sqlite — Y-Social / Tomašević agent databases (follow/unfollow resolution)
  .zip    — Rossetti server exports (read in-memory, no disk extraction)
  .csv    — generic directed edge lists (source/target columns auto-detected)
"""

import io
import logging
import sqlite3
import zipfile
from pathlib import Path
from typing import Optional, Union

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

_SUPPORTED = frozenset({".sqlite", ".zip", ".csv"})
_FOLLOW_ACTIONS = frozenset({"follow", "create"})
_EDGE_KEYWORDS = frozenset({"edge", "follow", "link", "network", "relation"})

# Ordered alias lists: first match wins, most-specific names listed first
_SOURCE_ALIASES = ("source", "src", "from", "follower", "follower_id", "u")
_TARGET_ALIASES = ("target", "dst", "to", "followee", "user_id", "v")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_network(
    path: Union[str, Path],
    *,
    end_round: Optional[int] = None,
    source_col: str = "source",
    target_col: str = "target",
    edge_file: Optional[str] = None,
) -> nx.DiGraph:
    """Load a directed graph from any supported simulation export.

    Dispatches to the correct internal parser based on file extension.
    Format-specific kwargs are silently ignored when they do not apply.

    Args:
        path: Path to the input file (.sqlite, .zip, or .csv).
        end_round: [.sqlite] Include only follow events up to this round (inclusive).
        source_col: [.csv] Column name for edge sources. Falls back to auto-detection.
        target_col: [.csv] Column name for edge targets. Falls back to auto-detection.
        edge_file: [.zip] Filename of the edge-list inside the archive.
            Auto-detected from archive contents when omitted.

    Returns:
        Directed graph. Node and edge attributes are populated where the
        source format provides them.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the extension is unsupported, required columns are
            missing, or the archive contains no parseable edge list.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: '{p}'")

    ext = p.suffix.lower()
    if ext not in _SUPPORTED:
        raise ValueError(
            f"Unsupported format '{ext}'. "
            f"Accepted: {', '.join(sorted(_SUPPORTED))}"
        )

    logger.info("Loading '%s' (format: %s)", p.name, ext)

    if ext == ".sqlite":
        return _parse_sqlite(p, end_round=end_round)
    if ext == ".zip":
        return _parse_zip(p, edge_file=edge_file)
    return _parse_csv(p, source_col=source_col, target_col=target_col)


# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------


def _parse_sqlite(path: Path, end_round: Optional[int] = None) -> nx.DiGraph:
    # end_round applied at SQL level to avoid loading rows we'll discard
    where = f"WHERE CAST(round AS INTEGER) <= {int(end_round)}" if end_round else ""

    with sqlite3.connect(path) as conn:
        df_nodes = pd.read_sql_query("SELECT * FROM user_mgmt;", conn)
        df_edges = pd.read_sql_query(
            f"SELECT follower_id, user_id, action, "
            f"CAST(round AS INTEGER) AS round "
            f"FROM follow {where} ORDER BY round ASC;",
            conn,
        )

    if df_nodes.empty:
        raise ValueError(f"'user_mgmt' table is empty in '{path.name}'")

    G = nx.DiGraph()
    G.add_nodes_from(df_nodes["id"])
    nx.set_node_attributes(G, df_nodes.set_index("id").to_dict(orient="index"))

    if not df_edges.empty:
        active = _resolve_follow_sequence(df_edges)
        G.add_edges_from(
            (r.follower_id, r.user_id, {"round_created": r.round})
            for r in active.itertuples(index=False)
        )

    logger.info(
        "SQLite → nodes=%d, edges=%d, end_round=%s",
        G.number_of_nodes(), G.number_of_edges(), end_round,
    )
    return G


def _parse_zip(path: Path, edge_file: Optional[str] = None) -> nx.DiGraph:
    with zipfile.ZipFile(path, "r") as zf:
        target = edge_file or _find_edge_file(zf.namelist(), path.name)
        with zf.open(target) as raw:
            # TextIOWrapper converts the byte stream without writing to disk
            df = pd.read_csv(io.TextIOWrapper(raw, encoding="utf-8"))

    src, dst = _detect_edge_columns(df, path.name)
    return _build_digraph(df, src, dst)


def _parse_csv(path: Path, source_col: str, target_col: str) -> nx.DiGraph:
    df = pd.read_csv(path)

    # Explicit columns take priority; fall back to heuristic detection
    if source_col in df.columns and target_col in df.columns:
        return _build_digraph(df, source_col, target_col)

    try:
        src, dst = _detect_edge_columns(df, path.name)
    except ValueError:
        missing = [c for c in (source_col, target_col) if c not in df.columns]
        raise ValueError(
            f"Column(s) {missing} not found in '{path.name}' "
            f"and auto-detection also failed. "
            f"Available columns: {list(df.columns)}"
        )
    return _build_digraph(df, src, dst)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_follow_sequence(df: pd.DataFrame) -> pd.DataFrame:
    # groupby-last on a pre-sorted frame resolves the full follow/unfollow
    # history in a single pass — significantly faster than pair-by-pair iteration
    # on large follow tables (1000+ agent simulations)
    df = df.copy()
    df["action"] = df["action"].str.strip().str.lower()
    last = (
        df.sort_values("round")
        .groupby(["follower_id", "user_id"], as_index=False)
        .last()
    )
    return last[last["action"].isin(_FOLLOW_ACTIONS)].reset_index(drop=True)


def _find_edge_file(names: list[str], archive_name: str) -> str:
    # Prefer files whose name contains a network-related keyword over bare CSVs
    candidates = [
        n for n in names
        if n.endswith(".csv") and any(kw in n.lower() for kw in _EDGE_KEYWORDS)
    ]
    if not candidates:
        candidates = [n for n in names if n.endswith(".csv")]
    if not candidates:
        raise ValueError(
            f"No CSV found in '{archive_name}'. "
            f"Archive contents: {names}. "
            "Pass edge_file= to specify the correct file explicitly."
        )
    if len(candidates) > 1:
        logger.warning(
            "Multiple CSV candidates in '%s'; defaulting to '%s'. "
            "Pass edge_file= to override.",
            archive_name, candidates[0],
        )
    return candidates[0]


def _detect_edge_columns(df: pd.DataFrame, filename: str) -> tuple[str, str]:
    lower_map = {c.lower(): c for c in df.columns}
    src = next((lower_map[a] for a in _SOURCE_ALIASES if a in lower_map), None)
    dst = next((lower_map[a] for a in _TARGET_ALIASES if a in lower_map), None)

    if src is None or dst is None:
        missing_role = "source" if src is None else "target"
        raise ValueError(
            f"Cannot auto-detect {missing_role} column in '{filename}'. "
            f"Columns available: {list(df.columns)}. "
            "Pass source_col= / target_col= explicitly."
        )
    return src, dst


def _build_digraph(df: pd.DataFrame, source_col: str, target_col: str) -> nx.DiGraph:
    attr_cols = [c for c in df.columns if c not in (source_col, target_col)]
    # from_pandas_edgelist uses internal vectorized ops — much faster than iterrows
    # for the large edge tables produced by thousand-agent simulations
    G = nx.from_pandas_edgelist(
        df,
        source=source_col,
        target=target_col,
        edge_attr=attr_cols or None,
        create_using=nx.DiGraph(),
    )
    logger.info("Edge list → nodes=%d, edges=%d", G.number_of_nodes(), G.number_of_edges())
    return G
