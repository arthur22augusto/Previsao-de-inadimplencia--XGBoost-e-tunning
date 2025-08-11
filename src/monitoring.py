"""
Script de automação mensal:
- carregar dados novos
- atualizar dataset histórico
- re-treinar modelo
- salvar novo modelo e métricas
- opcional: acionar dashboard/relatório
"""

from __future__ import annotations

import pandas as pd

import src.data_ingestion as di
import src.data_processing as dp
import src.featuring as ft
import src.modeling as md
import src.evaluation as ev
import src.logging_utils as lu


def _prepare_dataset_incremental() -> pd.DataFrame:
    # Reutiliza o mesmo preparo atual
    data_cadastro, data_hist, data_pagamentos, _ = di.carregar_dados()

    data_cadastro = dp.convert_data_types(data_cadastro)
    data_cadastro = dp.adjust_flag_pf(data_cadastro)
    data_cadastro = dp.adjust_ddd(data_cadastro)
    data_cadastro = dp.adjust_email_domain(data_cadastro)

    data_hist = dp.convert_data_types(data_hist)

    data_pagamentos = dp.convert_data_types(data_pagamentos)
    data_pagamentos = dp.fill_null_values_taxa_valor(data_pagamentos)
    data_pagamentos = ft.create_new_columns_pgto(data_pagamentos)
    data_pagamentos = ft.create_hist_features(data_pagamentos)

    data_final = dp.merge_data(data_pagamentos, data_cadastro, data_hist)
    data_final = dp.fill_null_values_after_merge(data_final)
    data_final = dp.map_and_drop_regiao_cep(data_final)
    data_final = dp.apply_encoding(data_final)

    return data_final


def monthly_update() -> None:
    run_id = lu.new_run_id("monthly")

    # Atualiza base consolidada
    # Opcional: baixar dados atualizados antes de carregar
    try:
        di.baixar_dados()
    except Exception:
        pass

    data_final = _prepare_dataset_incremental()
    data_model = md.drop_columns_for_modeling(data_final)

    # Re-treina baseline como referência
    X_train, X_test, y_train, y_test = md.split_data(data_model)
    model = md.fit_first_model(X_train, y_train, md.get_scale_pos_weight(y_train))
    y_pred, y_proba = md.get_predictions(model, X_test)
    cm, cr, roc_auc, pr_auc = ev.evaluate_classification(y_test, y_pred, y_proba, print_results=True)

    lu.log_metrics({
        "stage": "monthly_baseline",
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }, run_id, model_name="xgb_baseline")

    preds_df = pd.DataFrame({
        "y_true": y_test.reset_index(drop=True),
        "y_pred": y_pred,
        "y_proba": y_proba,
    })
    lu.log_predictions(
        preds_df, run_id, model_name="xgb_baseline",
        filename="predictions_eval.csv"
    )

    # Tuning mensal
    model_2, best_threshold, best_params = md.tune_model(data_model, recall_minimo=0.80)
    X_train2, X_test2, y_train2, y_test2 = md.split_data(data_model)
    y_pred2, y_proba2 = md.get_predictions(model_2, X_test2)
    cm2, cr2, roc_auc2, pr_auc2 = ev.evaluate_classification(y_test2, y_pred2, y_proba2, print_results=True)

    lu.log_metrics({
        "stage": "monthly_tuned",
        "roc_auc": float(roc_auc2),
        "pr_auc": float(pr_auc2),
        "best_threshold": float(best_threshold),
        "best_params": str(best_params),
        "n_train": int(len(X_train2)),
        "n_test": int(len(X_test2)),
    }, run_id, model_name="xgb_tuned")

    preds_df2 = pd.DataFrame({
        "y_true": y_test2.reset_index(drop=True),
        "y_pred": y_pred2,
        "y_proba": y_proba2,
    })
    lu.log_predictions(
        preds_df2, run_id, model_name="xgb_tuned",
        filename="predictions_eval.csv"
    )

    # Persistir melhor modelo e artefatos de inferência
    md.save_model(model_2, "best_model")
    md.save_feature_schema(
        features=data_model.drop(columns=["INADIMPLENTE"]).columns
    )
    # Usa o melhor threshold encontrado no tuning
    try:
        from src.modeling import tune_model
        # best_threshold já calculado acima
    except Exception:
        pass
    md.save_threshold(best_threshold)
    print(f"Atualização mensal finalizada. Run id: {run_id}")


if __name__ == "__main__":
    monthly_update()


