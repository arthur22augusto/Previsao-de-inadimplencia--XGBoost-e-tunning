"""
Aplicativo Streamlit para visualizar resultados do pipeline:
- Métricas por rodada e por estágio (baseline, tuned, simulation, monthly)
- Gráficos de séries (ROC AUC e PR AUC)
- Tabelas de predições de avaliação e de inferência
"""

from __future__ import annotations

import os
import pandas as pd
import streamlit as st


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        # Fallback robusto para possíveis quebras de linha/virgulas em textos
        return pd.read_csv(
            path, sep=None, engine="python", on_bad_lines="skip"
        )


def main() -> None:
    st.set_page_config(page_title="PD Pipeline - Dashboard", layout="wide")
    st.title("PD Pipeline — Dashboard de Performance e Predições")
    st.caption("Monitoramento de métricas e previsões por rodada")

    metrics_path = os.path.join("logs", "metrics.csv")
    # Separar logs de avaliação e de inferência
    preds_eval_path = os.path.join("logs", "predictions_eval.csv")
    preds_infer_path = os.path.join("logs", "predictions_infer.csv")

    metrics = _load_csv(metrics_path)
    preds_eval = _load_csv(preds_eval_path)
    preds_infer = _load_csv(preds_infer_path)

    st.header("Métricas registradas")
    if metrics.empty:
        st.info("Nenhuma métrica encontrada em logs/metrics.csv")
    else:
        m_sorted = metrics.sort_values(by="timestamp")
        stages = sorted(m_sorted.get("stage", pd.Series()).unique())

        st.markdown("### Visão geral")
        st.dataframe(m_sorted.iloc[::-1])

        if {"timestamp", "roc_auc", "pr_auc"}.issubset(m_sorted.columns):
            st.markdown("### Evolução das métricas")
            m_idx = m_sorted.set_index("timestamp")
            st.line_chart(m_idx[["roc_auc"]].rename(columns={"roc_auc": "ROC AUC"}))
            st.line_chart(m_idx[["pr_auc"]].rename(columns={"pr_auc": "PR AUC"}))

        if stages and not pd.isna(stages).all():
            st.markdown("### Filtro por estágio")
            stage_sel = st.selectbox("Estágio", options=["Todos"] + list(stages))
            mf = m_sorted if stage_sel == "Todos" else m_sorted[m_sorted["stage"] == stage_sel]
            st.dataframe(mf.iloc[::-1])

    st.header("Predições")
    st.subheader("Avaliação (holdout/teste)")
    if preds_eval.empty:
        st.info("Nenhuma previsão de avaliação encontrada em logs/predictions_eval.csv")
    else:
        st.dataframe(preds_eval.tail(500))

    st.subheader("Inferência (produção/simulações)")
    if preds_infer.empty:
        st.info("Nenhuma previsão de inferência encontrada em logs/predictions_infer.csv")
    else:
        st.dataframe(preds_infer.tail(500))

    st.markdown("---")
    st.caption("© PD Pipeline — Relatórios automatizados de modelagem e monitoramento")


if __name__ == "__main__":
    main()


