# Este script executa o pipeline: carregar dados, processar, criar features,
# unir bases e exibir schema final.

import src.data_ingestion as di
import src.data_processing as dp
import src.featuring as ft
import src.modeling as md
import src.evaluation as ev
import src.logging_utils as lu
from src.pipeline import predict_new_month

#di.baixar_dados()
data_cadastro, data_hist, data_pagamentos, data_objetivo = di.carregar_dados()

# Cadastro
data_cadastro = dp.convert_data_types(data_cadastro)
data_cadastro = dp.adjust_flag_pf(data_cadastro)
data_cadastro = dp.adjust_ddd(data_cadastro)
data_cadastro = dp.adjust_email_domain(data_cadastro)

# Histórico (sem ajustar FLAG_PF)
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

# Primeira modelagem
data_model = md.drop_columns_for_modeling(data_final)
X_train, X_test, y_train, y_test = md.split_data(data_model)
model = md.fit_first_model(X_train, y_train, md.get_scale_pos_weight(y_train))

# Previsões
y_pred, y_proba = md.get_predictions(model, X_test)
ev.evaluate_classification(y_test, y_pred, y_proba, print_results=True)

# Salvando modelo 1
md.save_model(model, 'model_base')

#Melhorando 
model_2, best_threshold, best_params = md.tune_model(data_model, recall_minimo=0.80)

#Salvando o melhor modelo
md.save_model(model_2, 'best_model')

# Persistir artefatos para inferência
md.save_feature_schema(
    features=data_model.drop(columns=['INADIMPLENTE']).columns
)
md.save_threshold(best_threshold)

# Avaliar e logar métricas do modelo ajustado
run_id = lu.new_run_id('testes')
y_pred2, y_proba2 = md.get_predictions(model_2, X_test)
cm2, cr2, roc_auc2, pr_auc2 = ev.evaluate_classification(
    y_test, y_pred2, y_proba2, print_results=True
)
lu.log_metrics({
    'stage': 'tuned',
    'roc_auc': float(roc_auc2),
    'pr_auc': float(pr_auc2),
    'n_train': int(len(X_train)),
    'n_test': int(len(X_test)),
}, run_id, model_name='xgb_tuned')

# Prever para dados_objetivo (formato mensal, sem DATA_PAGAMENTO)
predicoes = predict_new_month(data_objetivo)
predicoes = predicoes.rename(columns={'y_proba': 'prob_inadimplencia'})[
    ['ID_CLIENTE', 'SAFRA_REF', 'prob_inadimplencia']
]
predicoes.to_csv('predicoes_objetivo.csv', index=False)
print('Arquivo salvo: predicoes_objetivo.csv')
