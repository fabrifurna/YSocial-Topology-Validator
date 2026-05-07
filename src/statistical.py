"""
statistical.py
==============
Analisi di stabilità topologica su N run di Y-Social (Stage A) e
sensitivity analysis sul Recommender System (Stage B).

Pipeline: .sqlite → grafo → metriche → DataFrame aggregato → statistiche.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from ingestion import YSocialGraphBuilder
from topometrics import YSocialTopometrics

logger = logging.getLogger(__name__)

_META_COLS = {"run_id", "db_path", "error"}


class YSocialStabilityAnalyzer:
    """Raccoglie e analizza le metriche topologiche su N run di Y-Social.

    Progettato per lo Stage A (stabilità su N run indipendenti) e lo Stage B
    (sensitivity analysis su varianti del Recommender System). Il metodo
    principale è :meth:`collect_runs`, che restituisce un DataFrame grezzo
    con una riga per run; i metodi :meth:`compute_descriptive_stats` e
    :meth:`compute_stability_tests` operano su quel DataFrame.

    Args:
        db_paths: Percorsi ai file .sqlite prodotti dal simulatore.

    Raises:
        ValueError: Se la lista è vuota.
    """

    def __init__(self, db_paths: list[Union[str, Path]]) -> None:
        if not db_paths:
            raise ValueError("db_paths non può essere vuota.")
        self.db_paths = [Path(p).resolve() for p in db_paths]

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        pattern: str = "*.sqlite",
    ) -> "YSocialStabilityAnalyzer":
        """Costruisce l'analizzatore da una cartella di file .sqlite.

        Args:
            directory: Cartella contenente i database delle run.
            pattern: Glob pattern per selezionare i file (default ``*.sqlite``).

        Raises:
            FileNotFoundError: Se la cartella non esiste.
            ValueError: Se nessun file corrisponde al pattern.
        """
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Cartella non trovata: '{directory}'")
        paths = sorted(directory.glob(pattern))
        if not paths:
            raise ValueError(f"Nessun file '{pattern}' in '{directory}'.")
        logger.info("Trovati %d file in '%s'.", len(paths), directory)
        return cls(paths)

    def _process_single(
        self,
        db_path: Path,
        run_id: int,
        end_round: Optional[int],
    ) -> dict:
        """Esegue la pipeline completa su un singolo .sqlite e restituisce un record."""
        G = YSocialGraphBuilder(str(db_path)).load_follower_graph(end_round=end_round)
        metrics = YSocialTopometrics(G).generate_full_report()
        metrics["run_id"] = run_id
        metrics["db_path"] = str(db_path)
        return metrics

    def collect_runs(
        self,
        end_round: Optional[int] = None,
        label_col: Optional[str] = None,
        labels: Optional[list] = None,
    ) -> pd.DataFrame:
        """Esegue la pipeline su tutti i file e restituisce un DataFrame per-run.

        Le run che falliscono vengono registrate con una colonna ``error`` e
        non interrompono l'elaborazione delle rimanenti.

        Args:
            end_round: Round massimo da considerare (``None`` = tutti i round).
            label_col: Nome della colonna di etichetta aggiuntiva (Stage B).
            labels: Lista di etichette parallela a ``db_paths`` (Stage B).

        Returns:
            DataFrame con una riga per run e una colonna per ogni metrica.

        Raises:
            ValueError: Se solo uno tra ``label_col`` e ``labels`` è fornito,
                o se le lunghezze non coincidono.
        """
        if (label_col is None) != (labels is None):
            raise ValueError(
                "'label_col' e 'labels' devono essere entrambi forniti o entrambi None."
            )
        if labels is not None and len(labels) != len(self.db_paths):
            raise ValueError(
                f"'labels' ha {len(labels)} elementi, attesi {len(self.db_paths)}."
            )

        records = []
        for run_id, db_path in enumerate(self.db_paths):
            try:
                row = self._process_single(db_path, run_id, end_round)
                if labels is not None:
                    row[label_col] = labels[run_id]
                logger.info(
                    "Run %d/%d completata — %s",
                    run_id + 1, len(self.db_paths), db_path.name,
                )
            except Exception as exc:
                logger.error("Run %d fallita (%s): %s", run_id, db_path.name, exc)
                row = {"run_id": run_id, "db_path": str(db_path), "error": str(exc)}
            records.append(row)

        df = pd.DataFrame(records)
        n_ok = df["error"].isna().sum() if "error" in df.columns else len(df)
        logger.info("Raccolta completata — %d/%d run riuscite.", n_ok, len(df))
        return df

    @staticmethod
    def _numeric_metric_cols(df: pd.DataFrame) -> list[str]:
        """Colonne numeriche escludendo i metadati di run."""
        return [
            c for c in df.select_dtypes(include=np.number).columns
            if c not in _META_COLS
        ]

    def compute_descriptive_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcola statistiche descrittive per ogni metrica numerica.

        Per ogni colonna numerica calcola: n, media, deviazione standard,
        errore standard della media (SEM) e intervallo di confidenza al 95%
        basato sulla distribuzione t (corretto per N piccoli). Il coefficiente
        di variazione (CV = σ/μ) quantifica la stabilità relativa.

        Args:
            df: DataFrame prodotto da :meth:`collect_runs`.

        Returns:
            DataFrame con indice = nome metrica e colonne
            ``['n', 'mean', 'std', 'sem', 'ci95_low', 'ci95_high', 'cv']``.
        """
        rows = []
        for col in self._numeric_metric_cols(df):
            series = df[col].dropna()
            n = len(series)
            if n == 0:
                continue
            m = float(series.mean())
            s = float(series.std(ddof=1)) if n > 1 else 0.0
            sem = s / np.sqrt(n) if n > 1 else 0.0
            t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
            rows.append({
                "metric": col,
                "n": n,
                "mean": round(m, 6),
                "std": round(s, 6),
                "sem": round(sem, 6),
                "ci95_low": round(m - t_crit * sem, 6),
                "ci95_high": round(m + t_crit * sem, 6),
                "cv": round(s / m, 4) if m != 0 else None,
            })
        return pd.DataFrame(rows).set_index("metric")

    def compute_stability_tests(self, df: pd.DataFrame) -> pd.DataFrame:
        """Testa la normalità e quantifica la variabilità inter-run per metrica.

        Per ogni metrica numerica calcola:

        - **Shapiro-Wilk** W e p-value (H₀: normale; p > 0.05 → non si
          rifiuta la normalità; ideale per N ≤ 50);
        - **CV** (σ/μ): stabile se < 0.10, molto stabile se < 0.05;
        - **range relativo** (max − min) / μ come controllo aggiuntivo.

        Richiede almeno 3 osservazioni; metriche con meno punti vengono saltate.

        Args:
            df: DataFrame prodotto da :meth:`collect_runs`.

        Returns:
            DataFrame con indice = nome metrica e colonne
            ``['n', 'shapiro_W', 'shapiro_p', 'is_normal', 'cv', 'rel_range', 'stable']``.
        """
        rows = []
        for col in self._numeric_metric_cols(df):
            series = df[col].dropna()
            n = len(series)
            if n < 3:
                logger.debug("Shapiro-Wilk saltato per '%s': n=%d < 3.", col, n)
                continue
            W, p = stats.shapiro(series)
            m = float(series.mean())
            s = float(series.std(ddof=1))
            cv = s / m if m != 0 else None
            rel_range = (float(series.max()) - float(series.min())) / m if m != 0 else None
            rows.append({
                "metric": col,
                "n": n,
                "shapiro_W": round(float(W), 4),
                "shapiro_p": round(float(p), 4),
                "is_normal": bool(p > 0.05),
                "cv": round(cv, 4) if cv is not None else None,
                "rel_range": round(rel_range, 4) if rel_range is not None else None,
                "stable": bool(cv is not None and cv < 0.10),
            })
        return pd.DataFrame(rows).set_index("metric")

    def full_stability_report(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Restituisce descrittive e test di stabilità in un'unica chiamata.

        Args:
            df: DataFrame prodotto da :meth:`collect_runs`.

        Returns:
            Coppia ``(desc_df, stab_df)`` dove ``desc_df`` contiene le
            statistiche descrittive e ``stab_df`` i test di stabilità.
        """
        return self.compute_descriptive_stats(df), self.compute_stability_tests(df)
