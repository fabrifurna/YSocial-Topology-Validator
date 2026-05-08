"""
validator.py
============
Orchestrazione della pipeline per lo Stage A (Topological Stability Analysis).

La classe StageAValidator coordina il caricamento di N file .sqlite, l'estrazione
delle metriche topologiche, l'aggregazione in DataFrame e il calcolo delle
statistiche di stabilità (media, std, intervallo di confidenza al 95%).
"""

import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from ingestion import YSocialGraphBuilder
from topometrics import YSocialTopometrics

logger = logging.getLogger(__name__)

KEY_METRICS = [
    "alpha_in_degree",
    "modularity",
    "average_clustering",
    "density",
]


class StageAValidator:
    """Orchestrazione della validazione topologica su N run (Stage A).

    La classe carica N database SQLite da una cartella, estrae le metriche
    topologiche di ognuno tramite YSocialGraphBuilder + YSocialTopometrics,
    le aggrega in un DataFrame, e fornisce statistiche di stabilità
    (media, std, intervallo di confidenza al 95%).

    Args:
        data_dir: Percorso della cartella contenente i file .sqlite.

    Raises:
        FileNotFoundError: Se la cartella non esiste.
        ValueError: Se nessun file .sqlite viene trovato.
    """

    def __init__(self, data_dir: Union[str, Path]) -> None:
        self.data_dir = Path(data_dir).resolve()
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Cartella non trovata: '{self.data_dir}'")

        self.db_paths = self._discover_databases()
        if not self.db_paths:
            raise ValueError(f"Nessun file .sqlite trovato in '{self.data_dir}'")

        self.results: list[Dict[str, Any]] = []
        logger.info(
            "StageAValidator inizializzato — %d database trovati in '%s'",
            len(self.db_paths), self.data_dir.name,
        )

    def _discover_databases(self) -> list[Path]:
        """Scopre e ordina alfabeticamente i file .sqlite nella cartella."""
        paths = sorted(self.data_dir.glob("*.sqlite"))
        logger.debug("Database scoperti: %s", [p.name for p in paths])
        return paths

    def _extract_run_id(self, db_path: Path) -> str:
        """Estrae il run_id dal nome del file (es. run01.sqlite → run01)."""
        return db_path.stem

    def _process_single_run(self, db_path: Path) -> dict:
        """Processa un singolo database e restituisce il report."""
        G = YSocialGraphBuilder(str(db_path)).load_follower_graph()
        report = YSocialTopometrics(G).generate_full_report()
        return report

    def process_runs(self) -> pd.DataFrame:
        """Esegue la pipeline completa su tutti i database con progress bar.

        Per ogni file .sqlite nella cartella:
        1. Estrae il run_id dal nome file
        2. Carica il grafo tramite YSocialGraphBuilder
        3. Calcola le metriche tramite YSocialTopometrics
        4. Aggiunge le colonne run_id e db_path al report

        In caso di errore su una run, logga e continua alla successiva
        senza interrompere il ciclo.

        Returns:
            DataFrame con una riga per run e una colonna per metrica.
            Le run fallite hanno una colonna 'error' con il messaggio.
        """
        np.random.seed(42)
        self.results = []

        for db_path in tqdm(
            self.db_paths,
            desc="Processing Stage A runs",
            unit="run",
        ):
            run_id = self._extract_run_id(db_path)
            try:
                report = self._process_single_run(db_path)
                report["run_id"] = run_id
                report["db_path"] = str(db_path)
                self.results.append(report)
                logger.info(
                    "Run '%s' completata — nodi: %d, archi: %d",
                    run_id,
                    report.get("num_nodes", "?"),
                    report.get("num_edges", "?"),
                )
            except Exception as exc:
                logger.error(
                    "Errore durante il processing di run '%s' (%s): %s",
                    run_id, db_path.name, exc,
                )
                self.results.append({
                    "run_id": run_id,
                    "db_path": str(db_path),
                    "error": str(exc),
                })

        df = pd.DataFrame(self.results)
        n_ok = (~df.get("error", pd.Series([None] * len(df))).notna()).sum()
        logger.info(
            "Stage A completato — %d/%d run processate con successo.",
            n_ok, len(df),
        )
        return df

    def save_raw_results(self, output_csv: Union[str, Path]) -> None:
        """Salva il DataFrame grezzo dei risultati in CSV.

        Args:
            output_csv: Percorso del file CSV di output
                (es. data/processed/stage_a_raw.csv).

        Raises:
            ValueError: Se nessun run è stato ancora processato.
        """
        if not self.results:
            raise ValueError("Nessun risultato disponibile. Esegui process_runs() prima.")

        df = pd.DataFrame(self.results)
        output_csv = Path(output_csv).resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_csv, index=False)
        logger.info(
            "Risultati grezzi salvati — file: '%s', righe: %d",
            output_csv, len(df),
        )

    def compute_stability_metrics(
        self,
        metrics: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Calcola statistiche di stabilità per le metriche numeriche chiave.

        Per ogni metrica richiesta (default: alpha_in_degree, modularity,
        average_clustering, density), calcola:

        - **n**: numero di osservazioni non mancanti
        - **mean**: media aritmetica
        - **std**: deviazione standard (ddof=1)
        - **sem**: errore standard della media
        - **ci95_low, ci95_high**: intervallo di confidenza al 95%
          (basato su t-distribution, appropriato per N ≤ 30)
        - **cv**: coefficiente di variazione (σ/μ, indicatore di stabilità relativa)

        Args:
            metrics: Lista di metriche da aggregare. Se None, usa KEY_METRICS
                (alpha_in_degree, modularity, average_clustering, density).

        Returns:
            DataFrame con indice = nome metrica e colonne
            ['n', 'mean', 'std', 'sem', 'ci95_low', 'ci95_high', 'cv'].

        Raises:
            ValueError: Se nessun risultato è disponibile.
            KeyError: Se una metrica richiesta non esiste nel dataset.
        """
        if not self.results:
            raise ValueError("Nessun risultato disponibile. Esegui process_runs() prima.")

        metrics = metrics or KEY_METRICS
        df = pd.DataFrame(self.results)

        rows = []
        for metric in metrics:
            if metric not in df.columns:
                logger.warning("Metrica '%s' non trovata. Saltata.", metric)
                continue

            series = df[metric].dropna()
            if series.empty:
                logger.warning("Metrica '%s' contiene solo NaN. Saltata.", metric)
                continue

            n = len(series)
            m = float(series.mean())
            s = float(series.std(ddof=1)) if n > 1 else 0.0
            sem = s / np.sqrt(n) if n > 1 else 0.0
            t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
            cv = s / m if m != 0 else None

            rows.append({
                "metric": metric,
                "n": n,
                "mean": round(m, 6),
                "std": round(s, 6),
                "sem": round(sem, 6),
                "ci95_low": round(m - t_crit * sem, 6),
                "ci95_high": round(m + t_crit * sem, 6),
                "cv": round(cv, 4) if cv is not None else None,
            })

        result_df = pd.DataFrame(rows).set_index("metric")
        logger.info(
            "Statistiche di stabilità calcolate — %d metriche",
            len(result_df),
        )
        return result_df

    def full_stability_report(
        self,
        metrics: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """Genera un report completo di stabilità topologica (Stage A).

        Raccoglie i risultati grezzi e le statistiche aggregate in un
        unico dizionario per analisi e export.

        Args:
            metrics: Lista di metriche da includere nelle statistiche.
                Se None, usa KEY_METRICS.

        Returns:
            Dizionario con chiavi:
            - ``'raw'``: DataFrame grezzo (N righe, una per run)
            - ``'stability'``: DataFrame di statistiche aggregate per metrica
        """
        if not self.results:
            raise ValueError("Nessun risultato disponibile. Esegui process_runs() prima.")

        raw_df = pd.DataFrame(self.results)
        stability_df = self.compute_stability_metrics(metrics=metrics)

        return {
            "raw": raw_df,
            "stability": stability_df,
        }
