"""
topometrics.py
==============
Calcolo delle metriche di Social Network Analysis (SNA) sui grafi
estratti da Y-Social.

Le metriche implementate coprono tre aree diagnostiche principali:
struttura di base, distribuzione dei gradi (rilevamento Token Bias) e
struttura comunitaria (rilevamento Echo Chambers).
"""

import logging
from typing import Any, Dict

import networkx as nx
import numpy as np
import powerlaw

logger = logging.getLogger(__name__)


class YSocialTopometrics:
    """Calcola le metriche SNA su un grafo Y-Social.

    Il report prodotto da :meth:`generate_full_report` è un dizionario
    piatto serializzabile, progettato per essere inserito come riga in un
    DataFrame di aggregazione multi-run (Stage A).

    Args:
        G: Grafo orientato della rete sociale.

    Raises:
        TypeError: Se ``G`` non è un ``nx.DiGraph``.
        ValueError: Se il grafo è privo di nodi.
    """

    def __init__(self, G: nx.DiGraph) -> None:
        if not isinstance(G, nx.DiGraph):
            raise TypeError(f"Atteso nx.DiGraph, ricevuto {type(G).__name__}.")
        if G.number_of_nodes() == 0:
            raise ValueError("Il grafo non contiene nodi.")

        self.G = G
        self.G_undirected: nx.Graph = G.to_undirected()

    def compute_basic_stats(self) -> Dict[str, Any]:
        """Calcola le statistiche strutturali fondamentali della rete.

        Include la reciprocità (frazione di archi mutui) e la deviazione
        standard dei gradi. Una std bassa segnala potenziale Average Persona
        Bias, una reciprocità anomala può indicare comportamenti artificiali
        degli agenti LLM.
        """
        in_degrees = np.array([d for _, d in self.G.in_degree()])
        out_degrees = np.array([d for _, d in self.G.out_degree()])

        return {
            "num_nodes": self.G.number_of_nodes(),
            "num_edges": self.G.number_of_edges(),
            "density": float(nx.density(self.G)),
            "avg_in_degree": float(np.mean(in_degrees)),
            "std_in_degree": float(np.std(in_degrees)),
            "avg_out_degree": float(np.mean(out_degrees)),
            "std_out_degree": float(np.std(out_degrees)),
            "reciprocity": float(nx.reciprocity(self.G)),
        }

    def compute_powerlaw_alpha(self) -> Dict[str, Any]:
        """Stima l'esponente α della power-law sulla distribuzione in-degree.

        Una rete scale-free empirica ha tipicamente 2 < α < 3. Valori
        inferiori indicano un eccesso di hub, segnale di Token Bias.
        Il campo ``ks_distance`` quantifica la bontà del fit.
        """
        in_degrees = [d for _, d in self.G.in_degree() if d > 0]

        if len(in_degrees) < 10:
            logger.warning(
                "Power-law fit saltato: solo %d valori non nulli.", len(in_degrees)
            )
            return {
                "alpha_in_degree": None,
                "xmin": None,
                "ks_distance": None,
                "is_scale_free": None,
                "powerlaw_error": f"Dati insufficienti ({len(in_degrees)} in-degree > 0).",
            }

        try:
            fit = powerlaw.Fit(in_degrees, discrete=True, verbose=False)
            alpha = float(fit.alpha)
            return {
                "alpha_in_degree": round(alpha, 4),
                "xmin": float(fit.xmin),
                "ks_distance": round(float(fit.D), 4),
                "is_scale_free": 2.0 < alpha < 3.0,
                "powerlaw_error": None,
            }
        except Exception as exc:
            logger.error("Power-law fitting fallito: %s", exc)
            return {
                "alpha_in_degree": None,
                "xmin": None,
                "ks_distance": None,
                "is_scale_free": None,
                "powerlaw_error": str(exc),
            }

    def compute_clustering_and_modularity(self) -> Dict[str, Any]:
        """Calcola clustering medio e modularità della rete.

        La modularità Q ∈ [-1, 1] è la metrica principale per rilevare
        Echo Chambers (Q > 0.3 indica struttura comunitaria significativa)
        ed è l'indicatore più sensibile al Recommender System in Stage B.
        """
        avg_clustering = float(nx.average_clustering(self.G_undirected))
        communities = nx.community.greedy_modularity_communities(self.G_undirected)
        modularity = float(nx.community.modularity(self.G_undirected, communities))

        return {
            "average_clustering": round(avg_clustering, 4),
            "modularity": round(modularity, 4),
            "num_communities": len(communities),
            }   

    def generate_full_report(self) -> Dict[str, Any]:
        """Genera il report topologico completo, pronto per l'aggregazione multi-run."""
        report: Dict[str, Any] = {}
        report.update(self.compute_basic_stats())
        report.update(self.compute_powerlaw_alpha())
        report.update(self.compute_clustering_and_modularity())
        return report