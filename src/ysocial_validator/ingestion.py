"""Graph construction from Y-Social simulator SQLite databases.

Loads follower networks and resolves temporal follow/unfollow sequences
to produce directed graphs suitable for topological analysis.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


class YSocialGraphBuilder:
    """Constructs NetworkX directed graphs from Y-Social simulation databases.

    Handles loading user nodes and follower relationships from SQLite, resolving
    follow/unfollow action sequences to extract the final network state.
    For each (follower, followee) pair, only the last recorded action determines
    whether the edge exists in the final graph.

    Args:
        db_path: Path to Y-Social SQLite database file.

    Raises:
        FileNotFoundError: If database file does not exist.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path).resolve()
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: '{self.db_path}'")

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _load_raw_data(
        self, end_round: Optional[int]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Retrieve raw nodes and edges from database.

        Applies end_round filter at SQL level to minimize memory transfer.

        Args:
            end_round: Maximum simulation round to include, or None for all rounds.

        Returns:
            Tuple of (nodes DataFrame, edges DataFrame).
        """
        where = (
            f"WHERE CAST(round AS INTEGER) <= {end_round}"
            if end_round is not None
            else ""
        )
        query_nodes = """
            SELECT *
            FROM user_mgmt;
        """
        query_edges = f"""
            SELECT follower_id, user_id, action, CAST(round AS INTEGER) AS round
            FROM follow
            {where}
            ORDER BY round ASC;
        """
        with self._get_connection() as conn:
            df_nodes = pd.read_sql_query(query_nodes, conn)
            df_edges = pd.read_sql_query(query_edges, conn)

        return df_nodes, df_edges

    def _resolve_final_edges(self, df_edges: pd.DataFrame) -> pd.DataFrame:
        """Extract active edges after resolving follow/unfollow sequences.

        For each (follower, followee) pair, an edge exists if and only if
        the most recent action is a follow. Vectorized groupby approach
        is more efficient than row-by-row iteration for large datasets.

        Args:
            df_edges: Raw edges from database (may include both follow and unfollow actions).

        Returns:
            DataFrame containing only active edges (final state).
        """
        df = df_edges.copy()
        df["action"] = df["action"].str.strip().str.lower()

        # Actions indicating an active follow relationship
        follow_actions = {"follow", "create"}

        # Groupby identifies the last action per (follower, followee) pair
        df_last = (
            df.sort_values("round")
            .groupby(["follower_id", "user_id"], as_index=False)
            .last()
        )
        return df_last[df_last["action"].isin(follow_actions)].reset_index(drop=True)

    def load_follower_graph(self, end_round: Optional[int] = None) -> nx.DiGraph:
        """Construct directed graph of follower relationships.

        A directed edge u → v indicates that user u follows user v at the
        final state (or at end_round if specified).

        Args:
            end_round: Optional round cutoff; if provided, graph is constructed
                from events up to and including this round.

        Returns:
            DiGraph whose node attributes mirror all columns in user_mgmt
            (except id, which becomes the node key). Edge attribute: round_created.

        Raises:
            ValueError: If user_mgmt table is empty.
        """
        df_nodes, df_edges = self._load_raw_data(end_round)

        if df_nodes.empty:
            raise ValueError("No nodes found in 'user_mgmt'.")

        G = nx.DiGraph()
        G.add_nodes_from(df_nodes["id"])
        nx.set_node_attributes(G, df_nodes.set_index("id").to_dict(orient="index"))

        if not df_edges.empty:
            df_active = self._resolve_final_edges(df_edges)
            G.add_edges_from(
                (row.follower_id, row.user_id, {"round_created": row.round})
                for row in df_active.itertuples(index=False)
            )

        logger.info(
            "Graph constructed — nodes: %d, edges: %d, end_round=%s",
            G.number_of_nodes(), G.number_of_edges(), end_round,
        )
        return G
