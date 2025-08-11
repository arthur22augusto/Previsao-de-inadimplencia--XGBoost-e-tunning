"""
Script de orquestração principal do pipeline de modelagem:
- carregar dados
- pré-processar
- gerar features
- treinar modelo base
- avaliar modelo
- tunar hiperparâmetros
- salvar melhor modelo
- registrar métricas e previsões
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

import src.data_ingestion as di
import src.data_processing as dp
import src.featuring as ft
import src.modeling as md
import src.evaluation as ev
import src.logging_utils as lu


def _prepare_dataset() -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    data_cadastro, data_hist, data_pagamentos, data_objetivo = (
        di.carregar_dados()
    )

    # Cadastro
    data_cadastro = dp.convert_data_types(data_cadastro)
    data_cadastro = dp.adjust_flag_pf(data_cadastro)
    data_cadastro = dp.adjust_ddd(data_cadastro)
    data_cadastro = dp.adjust_email_domain(data_cadastro)

    # Histórico
    data_hist = dp.convert_data_types(data_hist)

    # Pagamentos
    data_pagamentos = dp.convert_data_types(data_pagamentos)
    data_pagamentos = dp.fill_null_values_taxa_valor(data_pagamentos)
    data_pagamentos = ft.create_new_columns_pgto(data_pagamentos)
    data_pagamentos = ft.create_hist_features(data_pagamentos)

    # Merge e pós-merge
    data_final = dp.merge_data(data_pagamentos, data_cadastro, data_hist)
    data_final = dp.fill_null_values_after_merge(data_final)
    data_final = dp.map_and_drop_regiao_cep(data_final)
    data_final = dp.apply_encoding(data_final)

    return data_final, data_cadastro, data_hist, data_pagamentos


def run_pipeline() -> None:
    run_id = lu.new_run_id("train")

    # Dados prontos
    data_final, _, _, _ = _prepare_dataset()

    # Modelagem inicial
    data_model = md.drop_columns_for_modeling(data_final)
    X_train, X_test, y_train, y_test = md.split_data(data_model)
    model = md.fit_first_model(
        X_train, y_train, md.get_scale_pos_weight(y_train)
    )

    # Previsões e avaliação inicial
    y_pred, y_proba = md.get_predictions(model, X_test)
    cm, cr, roc_auc, pr_auc = ev.evaluate_classification(
        y_test, y_pred, y_proba, print_results=True
    )

    # Log inicial
    metrics = {
        "stage": "baseline",
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    lu.log_metrics(metrics, run_id, model_name="xgb_baseline")

    preds_df = pd.DataFrame(
        {
            "y_true": y_test.reset_index(drop=True),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    )
    lu.log_predictions(
        preds_df, run_id, model_name="xgb_baseline",
        filename="predictions_eval.csv"
    )

    # Tuning
    model_2, best_threshold, best_params = md.tune_model(
        data_model, recall_minimo=0.80
    )

    # Reavaliação no holdout após tuning
    X_train2, X_test2, y_train2, y_test2 = md.split_data(data_model)
    y_pred2, y_proba2 = md.get_predictions(model_2, X_test2)
    cm2, cr2, roc_auc2, pr_auc2 = ev.evaluate_classification(
        y_test2, y_pred2, y_proba2, print_results=True
    )

    # Log final
    metrics2 = {
        "stage": "tuned",
        "roc_auc": float(roc_auc2),
        "pr_auc": float(pr_auc2),
        "best_threshold": float(best_threshold),
        "best_params": str(best_params),
        "n_train": int(len(X_train2)),
        "n_test": int(len(X_test2)),
    }
    lu.log_metrics(metrics2, run_id, model_name="xgb_tuned")

    preds_df2 = pd.DataFrame(
        {
            "y_true": y_test2.reset_index(drop=True),
            "y_pred": y_pred2,
            "y_proba": y_proba2,
        }
    )
    lu.log_predictions(
        preds_df2, run_id, model_name="xgb_tuned",
        filename="predictions_eval.csv"
    )

    # Salvando modelos e artefatos de inferência
    md.save_model(model, "model_base")
    md.save_model(model_2, "best_model")
    md.save_feature_schema(
        features=data_model.drop(columns=["INADIMPLENTE"]).columns
    )
    md.save_threshold(best_threshold)

    print(f"Pipeline finalizado. Run id: {run_id}")


if __name__ == "__main__":
    run_pipeline()


def _prepare_inference_dataset(
    df_pagamentos_mes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara dados de um novo mês sem DATA_PAGAMENTO para predição:
    - converter datas
    - criar colunas derivadas sem DATA_PAGAMENTO
    - juntar com cadastro e histórico
    - pós-processar e codificar
    """
    # Carrega bases auxiliares
    data_cadastro, data_hist, _, _ = di.carregar_dados()

    # Tipos
    df_new = dp.convert_data_types(df_pagamentos_mes)
    data_cadastro = dp.convert_data_types(data_cadastro)
    data_hist = dp.convert_data_types(data_hist)

    # Cadastro
    data_cadastro = dp.adjust_flag_pf(data_cadastro)
    data_cadastro = dp.adjust_ddd(data_cadastro)
    data_cadastro = dp.adjust_email_domain(data_cadastro)

    # Pagamentos (inferência)
    df_new = ft.create_new_columns_pgto_inference(df_new)

    # Merge e pós-merge
    data_final = dp.merge_data(df_new, data_cadastro, data_hist)
    data_final = dp.fill_null_values_after_merge(data_final)
    data_final = dp.map_and_drop_regiao_cep(data_final)
    data_final = dp.apply_encoding(data_final)
    return data_final


def predict_new_month(
    df_pagamentos_mes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Executa a predição para um novo mês de pagamentos (sem DATA_PAGAMENTO).
    Retorna DataFrame com ID_CLIENTE, SAFRA_REF, y_proba e y_pred.
    """
    run_id = lu.new_run_id("predict")
    data_final = _prepare_inference_dataset(df_pagamentos_mes)

    # Alinha features ao schema salvo
    feature_cols = md.load_feature_schema()
    X_new = data_final.copy()
    if "INADIMPLENTE" in X_new.columns:
        X_new = X_new.drop(columns=["INADIMPLENTE"])
    X_new = X_new.reindex(columns=feature_cols, fill_value=0)

    # Carrega melhor modelo e threshold
    model = md.load_latest_model(prefix="best_model")
    threshold = md.load_threshold()

    # Predição
    y_pred, y_proba = md.get_predictions(model, X_new)

    result = pd.DataFrame(
        {
            "ID_CLIENTE": df_pagamentos_mes["ID_CLIENTE"].values,
            "SAFRA_REF": df_pagamentos_mes["SAFRA_REF"].values,
            "y_proba": y_proba,
            "y_pred": y_pred,
        }
    )

    # Log das previsões
    lu.log_predictions(
        result, run_id, model_name="xgb_best",
        filename="predictions_infer.csv"
    )
    return result


