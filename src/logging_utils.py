"""
Este script fornece utilitários de logging do pipeline:
- salvar métricas por rodada (CSV)
- salvar previsões por rodada (CSV)
Cada linha recebe um run_id e timestamp para rastreabilidade.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Any, Iterable

import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def new_run_id(prefix: str = "run") -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    return f"{prefix}_{ts}"


def _append_csv(df: pd.DataFrame, filepath: str) -> None:
    file_exists = os.path.exists(filepath)
    df.to_csv(filepath, mode="a", header=not file_exists, index=False)


def log_metrics(
    metrics: Dict[str, Any],
    run_id: str,
    model_name: str,
    log_dir: str = "logs",
    filename: str = "metrics.csv",
) -> str:
    _ensure_dir(log_dir)
    payload = {**metrics, "run_id": run_id, "model_name": model_name,
               "timestamp": datetime.utcnow().isoformat()}
    df = pd.DataFrame([payload])
    path = os.path.join(log_dir, filename)
    _append_csv(df, path)
    return path


def log_predictions(
    predictions: pd.DataFrame,
    run_id: str,
    model_name: str,
    log_dir: str = "logs",
    filename: str = "predictions.csv",
) -> str:
    _ensure_dir(log_dir)
    df = predictions.copy()
    df["run_id"] = run_id
    df["model_name"] = model_name
    df["timestamp"] = datetime.utcnow().isoformat()
    path = os.path.join(log_dir, filename)
    _append_csv(df, path)
    return path


