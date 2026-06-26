"""Universal network loader for multi-agent simulation exports.

Single entry-point: load_network(path) auto-detects the source format and
returns a standardised nx.DiGraph. Internal parsers are private — callers
never need to know which one ran.

Supported inputs:
  .sqlite / .db / .sqlite3  — SQLite agent databases (follow/unfollow resolution)
  .zip                      — Archives containing a SQLite database or CSV edge list
  .csv                      — Generic directed edge lists (source/target auto-detected)
"""

import contextlib
import io
import logging
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Union

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

_SUPPORTED = frozenset({".sqlite", ".db", ".sqlite3", ".zip", ".csv"})
_DB_EXTENSIONS = frozenset({".db", ".sqlite", ".sqlite3"})
_FOLLOW_ACTIONS = frozenset({"follow", "create"})
_EDGE_KEYWORDS = frozenset({"edge", "follow", "link", "network", "relation"})

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
    For ZIP archives, the router inspects the contents and delegates
    automatically — callers never need to specify the internal format.

    Args:
        path: Path to the input file (.sqlite, .db, .sqlite3, .zip, or .csv).
        end_round: [.sqlite] Include only follow events up to this round (inclusive).
        source_col: [.csv] Column name for edge sources; falls back to auto-detection.
        target_col: [.csv] Column name for edge targets; falls back to auto-detection.
        edge_file: [.zip] Override auto-detection by naming a specific file inside the
            archive. Accepts .db/.sqlite (SQLite route) or .csv (edge-list route).

    Returns:
        Directed graph with node/edge attributes where the source format provides them.

    Raises:
        FileNotFoundError: Path does not exist.
        ValueError: Unsupported extension, missing columns, or unparseable archive.
        NotImplementedError: Archive contains only JSON network files (coming soon).
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

    if ext in _DB_EXTENSIONS:
        return _parse_sqlite(p, end_round=end_round)
    if ext == ".zip":
        return _parse_zip(p, edge_file=edge_file)
    return _parse_csv(p, source_col=source_col, target_col=target_col)


# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------


def _parse_sqlite(path: Path, end_round: Optional[int] = None) -> nx.DiGraph:
    where = f"WHERE CAST(round AS INTEGER) <= {int(end_round)}" if end_round else ""

    # contextlib.closing ensures conn.close() is called — on Windows, sqlite3's
    # context manager only commits/rolls back and leaves the file handle open,
    # which blocks tempfile cleanup when the DB was extracted from a ZIP
    with contextlib.closing(sqlite3.connect(path)) as conn:
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
        if edge_file:
            # Explicit override: trust the caller, route by entry extension
            return _dispatch_zip_entry(zf, edge_file, path)

        # Routing to specific parser based on archive contents to maintain
        # framework universality across simulation platforms
        return _route_zip_contents(zf, zf.namelist(), path)


def _parse_csv(path: Path, source_col: str, target_col: str) -> nx.DiGraph:
    df = pd.read_csv(path)

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
# ZIP routing
# ---------------------------------------------------------------------------


def _route_zip_contents(
    zf: zipfile.ZipFile,
    names: list[str],
    archive_path: Path,
) -> nx.DiGraph:
    # Examine only top-level entries — nested files belong to logs or sub-archives
    top_level = [n for n in names if "/" not in n and "\\" not in n]

    db_files = [n for n in top_level if Path(n).suffix.lower() in _DB_EXTENSIONS]
    if db_files:
        return _parse_zip_sqlite(zf, db_files[0], archive_path)

    csv_files = [
        n for n in names
        if n.endswith(".csv") and any(kw in n.lower() for kw in _EDGE_KEYWORDS)
    ]
    if not csv_files:
        csv_files = [n for n in names if n.endswith(".csv")]
    if csv_files:
        return _parse_zip_csv(zf, csv_files[0], archive_path)

    json_files = [n for n in top_level if n.endswith(".json")]
    if json_files:
        raise NotImplementedError(
            f"JSON network parsing is not yet implemented. "
            f"Found in archive: {json_files}. "
            "Convert to edge-list CSV, or pass edge_file= to target a .db directly."
        )

    raise ValueError(
        f"No parseable network data found in '{archive_path.name}'. "
        f"Archive top-level contents: {top_level}. "
        "Expected a .db/.sqlite (SQLite) or .csv (edge list)."
    )


def _dispatch_zip_entry(
    zf: zipfile.ZipFile,
    entry: str,
    archive_path: Path,
) -> nx.DiGraph:
    ext = Path(entry).suffix.lower()
    if ext in _DB_EXTENSIONS:
        return _parse_zip_sqlite(zf, entry, archive_path)
    if ext == ".csv":
        return _parse_zip_csv(zf, entry, archive_path)
    raise ValueError(
        f"Cannot parse '{entry}' from archive '{archive_path.name}' — "
        f"unsupported entry extension '{ext}'. "
        "Expected .db/.sqlite (SQLite) or .csv (edge list)."
    )


def _parse_zip_sqlite(
    zf: zipfile.ZipFile,
    db_name: str,
    archive_path: Path,
) -> nx.DiGraph:
    # Using tempfile to prevent I/O pollution when inspecting archives —
    # only the database is extracted; logs and config files stay inside the ZIP.
    # ignore_cleanup_errors=True guards against Windows handle-release races
    # even after the sqlite3 connection is explicitly closed via contextlib.closing
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        zf.extract(db_name, tmp)
        db_path = Path(tmp) / db_name
        logger.info("Extracted '%s' from '%s' → volatile temp dir", db_name, archive_path.name)
        return _parse_sqlite(db_path)


def _parse_zip_csv(
    zf: zipfile.ZipFile,
    csv_name: str,
    archive_path: Path,
) -> nx.DiGraph:
    csv_count = sum(1 for n in zf.namelist() if n.endswith(".csv"))
    if csv_count > 1:
        logger.warning(
            "Multiple CSV files in '%s'; selected '%s'. Pass edge_file= to override.",
            archive_path.name, csv_name,
        )
    with zf.open(csv_name) as raw:
        df = pd.read_csv(io.TextIOWrapper(raw, encoding="utf-8"))
    src, dst = _detect_edge_columns(df, archive_path.name)
    return _build_digraph(df, src, dst)


# ---------------------------------------------------------------------------
# Shared helpers
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
    # from_pandas_edgelist uses internal vectorised ops — much faster than iterrows
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
