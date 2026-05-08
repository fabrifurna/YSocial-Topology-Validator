"""
ingestion.py
============
Estrazione e costruzione di grafi topologici dai database SQLite
prodotti dal simulatore Y-Social.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


class YSocialGraphBuilder:
    """Costruisce grafi NetworkX a partire da un database SQLite di Y-Social.

    Ogni istanza è legata a un singolo file `.sqlite` prodotto da una run
    del simulatore. Il metodo principale, :meth:`load_follower_graph`,
    risolve correttamente le sequenze follow/unfollow: per ogni coppia
    (follower, followee) viene mantenuto solo lo stato dell'ultima azione.

    Args:
        db_path: Percorso al file SQLite di Y-Social.

    Raises:
        FileNotFoundError: Se il file non esiste al percorso indicato.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path).resolve()
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database non trovato: '{self.db_path}'")

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _load_raw_data(
        self, end_round: Optional[int]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Carica nodi e archi grezzi dal DB.

        Il filtro ``end_round`` viene applicato in SQL per evitare di
        trasferire dati inutili in memoria.
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
        """Restituisce gli archi attivi dopo aver risolto follow/unfollow.

        Per ogni coppia (follower, followee), l'arco esiste se e solo se
        l'ultima azione registrata è un follow. L'approccio vettorizzato
        tramite groupby è molto più efficiente del loop riga per riga su
        dataset di grandi dimensioni.
        """
        df = df_edges.copy()
        df["action"] = df["action"].str.strip().str.lower()

        follow_actions = {"follow", "create"}

        df_last = (
            df.sort_values("round")
            .groupby(["follower_id", "user_id"], as_index=False)
            .last()
        )
        return df_last[df_last["action"].isin(follow_actions)].reset_index(drop=True)

    def load_follower_graph(self, end_round: Optional[int] = None) -> nx.DiGraph:
        """Costruisce il grafo orientato della rete follower.

        Un arco u → v indica che l'utente u segue attualmente l'utente v.

        Args:
            end_round: Se specificato, la rete viene costruita considerando
                solo gli eventi fino a quel round (incluso).

        Returns:
            ``nx.DiGraph`` con nodi etichettati dall'ID utente e attributi
            ``username``, ``user_type``, ``archetype``, ``leaning``,
            ``is_page``. Gli archi hanno l'attributo ``round_created``.

        Raises:
            ValueError: Se la tabella ``user_mgmt`` è vuota.
        """
        df_nodes, df_edges = self._load_raw_data(end_round)

        if df_nodes.empty:
            raise ValueError("Nessun nodo trovato in 'user_mgmt'.")

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
            "Grafo costruito — nodi: %d | archi: %d | end_round=%s",
            G.number_of_nodes(), G.number_of_edges(), end_round,
        )
        return G