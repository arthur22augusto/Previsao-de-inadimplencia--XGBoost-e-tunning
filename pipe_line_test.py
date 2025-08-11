"""
Este script simula a execução mensal da pipeline:
- agrupa `data_objetivo` por `SAFRA_REF` e `ID_CLIENTE`
- salva um CSV por mês em `data/teste_pipeline/`
- roda predição mensal para cada mês (sem `DATA_PAGAMENTO`)
- assume acurácia alvo entre 0.8 e 0.9 criando rótulos sintéticos
- registra métricas e atualiza logs (dashboard usa esses logs)
- opcional: chama atualização mensal para re-treino

Ao final, compara métricas do `testes.py` (uma rodada) com a
simulação mensal (médias por mês).
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

import src.data_ingestion as di
import src.data_processing as dp
import src.logging_utils as lu
import src.evaluation as ev
import src.modeling as md
from src.pipeline import predict_new_month


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _simulate_ground_truth(
    y_pred: np.ndarray,
    target_accuracy: float = 0.95,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    y_true = y_pred.copy()
    n = y_true.size
    flips = max(0, int(round((1.0 - target_accuracy) * n)))
    if flips > 0:
        idx = rng.choice(n, size=flips, replace=False)
        y_true[idx] = 1 - y_true[idx]
    return y_true


def _collect_month_files(base_dir: str) -> list[str]:
    files = []
    for name in sorted(os.listdir(base_dir)):
        if name.lower().endswith(".csv"):
            files.append(os.path.join(base_dir, name))
    return files


def main() -> None:
    run_id = lu.new_run_id("simulate")
    out_dir = os.path.join("data", "teste_pipeline")
    _ensure_dir(out_dir)

    # Carrega bases
    _, _, _, data_objetivo = di.carregar_dados()
    data_objetivo = dp.convert_data_types(data_objetivo)

    # Mantém apenas colunas necessárias para a inferência mensal
    cols_needed = [
        "ID_CLIENTE",
        "SAFRA_REF",
        "DATA_EMISSAO_DOCUMENTO",
        "DATA_VENCIMENTO",
        "VALOR_A_PAGAR",
        "TAXA",
    ]
    data_objetivo = data_objetivo[cols_needed]

    # Salva um arquivo por safra
    def _safe_name_from_safra(value) -> str:
        try:
            ts = pd.to_datetime(value)
            return ts.strftime("%Y-%m-%d")
        except Exception:
            name = str(value)
            for ch in "\\/:*?\"<>|":
                name = name.replace(ch, "-")
            name = name.replace(":", "-").replace(" ", "_")
            return name

    for safra, df_mes in data_objetivo.groupby("SAFRA_REF", dropna=False):
        nome = _safe_name_from_safra(safra)
        path = os.path.join(out_dir, f"pagamentos_{nome}.csv")
        df_mes.to_csv(path, sep=";", index=False)

    # Executa predição mês a mês e registra métricas simuladas
    metricas_sim = []
    for fpath in _collect_month_files(out_dir):
        df_mes = pd.read_csv(fpath, sep=";")
        df_mes = dp.convert_data_types(df_mes)

        preds = predict_new_month(df_mes)
        y_pred = preds["y_pred"].to_numpy()
        y_proba = preds["y_proba"].to_numpy()

        # Gera y_true sintético coerente com o ranking de probabilidade
        best_thr = md.load_threshold()
        y_true_base = (y_proba >= best_thr).astype(int)
        y_true = _simulate_ground_truth(y_true_base, target_accuracy=0.95)

        # Avalia e loga
        cm, cr, roc_auc, pr_auc = ev.evaluate_classification(
            y_true, y_pred, y_proba, print_results=False
        )
        lu.log_metrics(
            {
                "stage": "simulation_monthly",
                "roc_auc": float(roc_auc),
                "pr_auc": float(pr_auc),
                "file": os.path.basename(fpath),
            },
            run_id,
            model_name="xgb_best",
        )
        metricas_sim.append((roc_auc, pr_auc))

    # Resumo de comparação
    if metricas_sim:
        roc_mean = float(np.mean([m[0] for m in metricas_sim]))
        pr_mean = float(np.mean([m[1] for m in metricas_sim]))
        print(
            f"Métricas médias por mês (simulação): ROC_AUC={roc_mean:.4f}, "
            f"PR_AUC={pr_mean:.4f}"
        )

    print(
        "Simulação mensal concluída. Abra o dashboard: "
        "streamlit run src/dashboard.py"
    )


if __name__ == "__main__":
    main()


