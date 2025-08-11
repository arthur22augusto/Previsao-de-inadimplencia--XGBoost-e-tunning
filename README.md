## Projeto: Pipeline de Previsão de Inadimplência

Este repositório contém uma pipeline modular para previsão de
inadimplência (XGBoost), com orquestração, monitoramento, logging e
dashboard em Streamlit.

### Destaque: `main.py`

`main.py` é o script de referência para rodar o fluxo ponta‑a‑ponta:

- carrega dados de `data_ingestion`
- pré‑processa e cria features
- treina o modelo base e faz tuning (Optuna)
- salva modelos e artefatos de inferência
- gera predições para a base de teste (`base_pagamentos_teste.csv`)
- registra métricas em `logs/`

Execute:

```bash
python main.py
```

Saídas principais:

- modelos em `models/` (`model_base_*.json`, `best_model_*.json`)
- artefatos em `models/feature_schema.json` e `models/best_threshold.json`
- métricas em `logs/metrics.csv`
- predições de avaliação em `logs/predictions_eval.csv`
- predições de inferência em `logs/predictions_infer.csv`
- arquivo final `predicoes_objetivo.csv`

### Estrutura dos módulos (`src/`)

- `data_ingestion.py`: leitura das bases CSV.
- `data_processing.py`: limpeza, tipos, imputações e encoding.
- `featuring.py`: features de pagamentos e histórico. Versão
  específica para inferência sem `DATA_PAGAMENTO`.
- `modeling.py`: split, treino, tuning (50 trials), save/load modelo,
  schema de features e threshold.
- `evaluation.py`: métricas de classificação (ROC AUC, PR AUC, etc.).
- `logging_utils.py`: logging estruturado de métricas e predições.
- `pipeline.py`: orquestração e inferência mensal (funções abaixo).
- `monitoring.py`: rotina de atualização mensal (re‑treino e logs).
- `dashboard.py`: app Streamlit para visualização de resultados.

### Funções principais de pipeline

- `pipeline.run_pipeline()`
  - Treina, avalia, faz tuning, salva modelos/artefatos e loga tudo.

- `pipeline.predict_new_month(df_pagamentos_mes)`
  - Recebe DataFrame mensal no formato de pagamentos sem
    `DATA_PAGAMENTO`: colunas
    `ID_CLIENTE, SAFRA_REF, DATA_EMISSAO_DOCUMENTO, DATA_VENCIMENTO,
    VALOR_A_PAGAR, TAXA`.
  - Prepara dados, junta com cadastro e info, alinha schema e prediz.

- `monitoring.monthly_update()`
  - Carrega dados, atualiza dataset consolidado, re‑treina modelo,
    persiste artefatos e registra métricas/predições para o mês.

Comandos úteis:

```bash
# Orquestração de treino/tuning
python -c "from src.pipeline import run_pipeline; run_pipeline()"

# Predição mensal (exemplo de uso em Python)
python -c "import pandas as pd; from src.pipeline import predict_new_month;\
df=pd.read_csv('data/raw/base_pagamentos_teste.csv', sep=';');\
pred=predict_new_month(df); print(pred.head())"

# Atualização mensal (re‑treino e logs)
python -c "from src.monitoring import monthly_update; monthly_update()"

# Dashboard Streamlit
streamlit run src/dashboard.py
```

### Script de simulação mensal: `pipe_line_test.py`

Simula rodadas mensais a partir de `base_pagamentos_teste.csv`:

- agrupa por `SAFRA_REF` e salva um CSV por mês em `data/teste_pipeline/`
- executa `predict_new_month(...)` mês a mês
- gera rótulos sintéticos com acurácia alvo (ex.: 95%) a partir de
  `y_proba` e do threshold salvo, para avaliar ROC/PR por mês
- registra métricas no `logs/metrics.csv` (stage `simulation_monthly`)

Execute:

```bash
python pipe_line_test.py
```

Use o dashboard para comparar as métricas do `main.py` (baseline/tuned)
com as métricas mensais simuladas (`simulation_monthly`).

### Estrutura de diretórios

```
PD_Pipeline/
  data/
    raw/
    processed/
    teste_pipeline/
  logs/
    metrics.csv
    predictions_eval.csv
    predictions_infer.csv
  models/
    model_base_*.json
    best_model_*.json
    feature_schema.json
    best_threshold.json
  src/
    ... módulos descritos acima ...
```

### Instalação

```bash
python -m pip install -r requirements.txt
```

### Notas e boas práticas

- Mesmos passos de preparo entre treino e inferência (sem vazamento).
- `scale_pos_weight` para desbalanceamento.
- Datas convertidas com `errors='coerce'` e limpeza de tokens ruins.
- Logs robustos e separados por finalidade (avaliação vs. inferência).
- Schema de features e threshold persistidos para reprodutibilidade.


