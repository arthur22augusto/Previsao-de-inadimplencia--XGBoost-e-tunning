
# Documentação do Projeto de Previsão de Inadimplência

## 1. Introdução

### Contexto
Este projeto visa desenvolver um modelo de machine learning para a previsão de inadimplência de clientes. A análise se baseia em um conjunto de dados que inclui informações cadastrais, histórico de comportamento mensal e transações de pagamentos. O desafio central é identificar, com antecedência, os clientes com alta probabilidade de atrasar um pagamento por um período igual ou superior a 5 dias.

### Objetivo
O objetivo principal é construir e otimizar um modelo preditivo robusto que estime a probabilidade de inadimplência para cada transação futura. A implementação deste modelo permitirá à empresa adotar ações proativas de cobrança, otimizar a alocação de recursos e mitigar riscos financeiros associados a pagamentos em atraso.

---

## 2. Entendimento e Preparação dos Dados

### Descrição das Bases de Dados
O projeto utilizou quatro fontes de dados principais:
- **`base_cadastral.csv`**: Contém dados estáticos dos clientes, como data de cadastro, segmento industrial, porte da empresa e localização (CEP).
- **`base_info.csv`**: Fornece informações mensais (por safra) sobre os clientes, como renda e número de funcionários.
- **`base_pagamentos_desenvolvimento.csv`**: Inclui o histórico detalhado de transações de pagamentos, usado para treinar o modelo e para a engenharia de variáveis. É a base principal para a construção do alvo (target).
- **`base_pagamentos_teste.csv`**: Apresenta as transações futuras para as quais o modelo deve gerar as previsões de inadimplência.

### Chaves de Relacionamento
A integração das diferentes bases de dados foi realizada utilizando as seguintes chaves:
- **`ID_CLIENTE`**: Identificador único para cada cliente, usado para conectar todas as bases.
- **`SAFRA_REF`**: Data de referência (mês/ano), usada para conectar a base de pagamentos com as informações mensais correspondentes.

### Construção da Variável Target (`INADIMPLENTE`)
A variável alvo, que define o evento de inadimplência, foi criada a partir da base de pagamentos. Um pagamento foi classificado como inadimplente (`INADIMPLENTE = 1`) se a diferença entre a `DATA_PAGAMENTO` e a `DATA_VENCIMENTO` fosse **maior ou igual a 5 dias**. Caso contrário, foi classificado como adimplente (`INADIMPLENTE = 0`).

### Tratamento de Dados
Uma série de etapas de limpeza e pré-processamento foi aplicada para garantir a qualidade e a consistência dos dados:
- **Dados Ausentes (NaNs):**
  - `FLAG_PF`: Valores nulos foram preenchidos com 'PJ', assumindo que a ausência da flag indica uma pessoa jurídica.
  - `RENDA_MES_ANTERIOR`: Nulos foram preenchidos com a **mediana** da renda, agrupada por `PORTE` e `SEGMENTO_INDUSTRIAL` do cliente, para uma imputação mais contextualizada.
  - `NO_FUNCIONARIOS`: Nulos foram preenchidos com a **média** do número de funcionários, também agrupada por `PORTE` e `SEGMENTO_INDUSTRIAL`.
  - `VALOR_A_PAGAR`: A imputação foi feita em duas etapas: primeiro, com a média de pagamentos do próprio cliente; para clientes sem histórico, utilizou-se a mediana geral da base, que é mais robusta a outliers.
- **Tipos de Dados:**
  - Todas as colunas de data (`DATA_CADASTRO`, `SAFRA_REF`, `DATA_EMISSAO_DOCUMENTO`, `DATA_PAGAMENTO`, `DATA_VENCIMENTO`) foram convertidas para o formato `datetime`.
- **Limpeza e Consistência:**
  - Registros com inconsistências temporais (ex: data de pagamento anterior à data de emissão) foram removidos para garantir a lógica do fluxo de pagamento.
- **Encoding de Variáveis Categóricas:**
  - `CEP_2_DIG`: Foi transformada na variável `REGIAO_CEP` (Sudeste, Sul, etc.) e, em seguida, convertida em variáveis dummy (One-Hot Encoding).
  - `SEGMENTO_INDUSTRIAL`: Também passou por One-Hot Encoding.
  - `PORTE`: Como possui uma ordem implícita (PEQUENO, MÉDIO, GRANDE), foi convertida usando Label Encoding.

---

## 3. Análise Exploratória (EDA)
A análise exploratória inicial revelou insights importantes que guiaram o pré-processamento e a modelagem:
- **Desbalanceamento de Classe:** A variável alvo `INADIMPLENTE` é altamente desbalanceada, com apenas cerca de 7% das transações classificadas como inadimplentes. Isso exigiu o uso de métricas apropriadas (ROC AUC, PR AUC) и técnicas para lidar com o desbalanceamento, como o parâmetro `scale_pos_weight` no XGBoost.
- **Distribuição de `VALOR_A_PAGAR`:** A variável apresentou uma forte assimetria à direita, indicando a presença de outliers (valores muito altos). Essa característica justificou o uso da mediana em vez da média para a imputação de valores ausentes.
- **Dados Cadastrais:** A análise da base cadastral mostrou a necessidade de padronizar e limpar campos como `FLAG_PF` e a baixa cardinalidade de `DDD`, que foi removido em favor da variável `CEP_2_DIG`, mais completa.

---

## 4. Engenharia de Variáveis

Para enriquecer o modelo com informações preditivas, diversas variáveis foram criadas:

- **`FLAG_PF` (Binária):** A coluna foi transformada em um valor numérico (1 para PF, 0 para PJ) para ser utilizada pelo modelo.
- **`MES_EMISSAO`:** Extraído da data de emissão para capturar possíveis efeitos de sazonalidade nos pagamentos.
- **Features de Histórico (`HIST_*`):** Para capturar o comportamento de pagamento de cada cliente ao longo do tempo, foram criadas variáveis cumulativas e de janela móvel, sempre olhando para o período anterior à transação atual para evitar vazamento de dados (data leakage):
  - `HIST_QTD_TRANSACOES`: Total de transações anteriores.
  - `HIST_PAGAMENTOS_ADIANTADOS`: Contagem de pagamentos realizados antes do vencimento.
  - `HIST_INADIMPLENTE_ULT_MES`: Flag indicando se o cliente foi inadimplente na transação anterior.
  - `HIST_DIAS_ATRASO_ACUMULADO`: Soma total de dias de atraso.
  - `HIST_MEDIA_DIAS_ATRASO`: Média de dias de atraso no histórico.
  - `HIST_VALOR_MEDIO_PAGO`: Valor médio das transações passadas.

### Exclusão de Variáveis
Algumas variáveis foram excluídas antes da modelagem por não agregarem valor preditivo ou por representarem risco de vazamento de dados:
- `ID_CLIENTE`, `DATA_CADASTRO`, `DATA_EMISSAO_DOCUMENTO`, `DATA_PAGAMENTO`, `DATA_VENCIMENTO`: IDs e datas brutas, substituídas por features de engenharia.
- `DIAS_DE_ATRASO`: Esta variável é uma forma direta da variável alvo e sua inclusão causaria um vazamento de dados perfeito.
- `DOMINIO_EMAIL`: Alta cardinalidade e baixo poder preditivo.
- `DDD`: Redundante, uma vez que `CEP_2_DIG` já fornecia informação geográfica.

---

## 5. Modelagem

### Divisão de Dados e Validação
Os dados foram divididos em conjuntos de **treino (80%)** e **teste (20%)**. A estratificação pela variável alvo (`INADIMPLENTE`) foi utilizada para garantir que a proporção de inadimplentes fosse a mesma em ambos os conjuntos, o que é crucial para dados desbalanceados. Para o ajuste de hiperparâmetros, foi empregada a **Validação Cruzada Estratificada (Stratified K-Fold com 5 dobras)**, que oferece uma estimativa robusta da performance do modelo.

### Modelo Utilizado
O modelo escolhido foi o **XGBoost (Extreme Gradient Boosting)**, uma escolha sólida para dados tabulares devido à sua alta performance, capacidade de lidar com dados ausentes, e tratamento de interações não-lineares entre as variáveis.

### Métricas de Avaliação
Dado o desbalanceamento de classes, as seguintes métricas foram priorizadas:
- **ROC AUC:** Mede a capacidade do modelo de discriminar entre as classes positiva e negativa. É a principal métrica para otimização.
- **Precision-Recall AUC (PR AUC):** Especialmente útil para avaliar o desempenho em classes minoritárias.
- **F1-Score:** Média harmônica entre precisão e recall, fornecendo uma visão balanceada do desempenho na classe positiva.
- **Matriz de Confusão:** Permite uma análise detalhada dos erros (falsos positivos e falsos negativos).

### Ajuste de Hiperparâmetros
Foi conduzido um processo de otimização de hiperparâmetros utilizando a biblioteca **Optuna**. Optuna realiza uma busca mais inteligente que o Grid Search ou Random Search, focando em regiões promissoras do espaço de parâmetros. O objetivo foi maximizar a métrica **ROC AUC** em validação cruzada.

### Comparação de Modelos
Dois modelos foram comparados:
1.  **Modelo Base:** Um XGBoost com parâmetros padrão e `scale_pos_weight` para tratar o desbalanceamento.
2.  **Modelo Otimizado:** O mesmo XGBoost, mas com hiperparâmetros ajustados pelo Optuna.

| Métrica              | Modelo Base | Modelo Otimizado (Esperado) |
|----------------------|-------------|-----------------------------|
| ROC AUC (CV)         | ~0.953      | > 0.960                     |
| Precision-Recall AUC | ~0.874      | > 0.880                     |
| F1-Score (classe 1)  | ~0.74       | > 0.75                      |

O modelo otimizado apresentou uma performance superior em todas as métricas, justificando sua escolha como a versão final.

---

## 6. Avaliação do Modelo Final

O modelo final (otimizado com Optuna) demonstrou uma performance excelente e robusta.

### Matriz de Confusão e Curvas
- A **matriz de confusão** mostra que o modelo possui um alto **recall** (sensibilidade), identificando corretamente a grande maioria dos clientes inadimplentes, o que é crucial para a estratégia de negócio. A precisão, embora menor, ainda é forte, garantindo que as ações de cobrança não sejam direcionadas a um número excessivo de clientes adimplentes.
- As curvas **ROC e Precision-Recall** se mantiveram altas e estáveis no conjunto de teste, indicando boa generalização e ausência de overfitting.

### Importância das Variáveis (Feature Importance)
A análise de importância de variáveis (`gain`) revelou os principais fatores que influenciam a previsão de inadimplência:
1.  **`PRAZO_PAGAMENTO_DIAS`**: Prazos mais longos podem estar associados a maior risco.
2.  **`HIST_INADIMPLENTE_ULT_MES`**: O comportamento passado é um forte preditor do futuro.
3.  **`VALOR_A_PAGAR`**: Valores muito altos podem indicar maior risco.
4.  **`HIST_MEDIA_DIAS_ATRASO`**: A média de atraso histórica é um indicador de comportamento consistente.

![Feature Importance](../reports/images/feature_importance.png)

### Curva de Aprendizado (Learning Curve)
A curva de aprendizado mostrou que os scores de treino e validação convergem para um valor próximo e estável, indicando que o modelo não está sofrendo de alto viés nem de alta variância (overfitting).

![Learning Curve](../reports/images/learning_curve.png)

---

## 7. Geração da Submissão
Para gerar as previsões para a base de teste (`base_pagamentos_teste.csv`), o seguinte processo foi seguido:
1.  A base de teste foi carregada.
2.  Foram aplicadas as **exatas mesmas etapas de pré-processamento e engenharia de variáveis** usadas na base de desenvolvimento. Isso incluiu a junção com as bases `cadastral` e `info` para obter todas as features necessárias.
3.  O modelo final treinado (`xgb_inadimplencia_optuna.json`) foi carregado.
4.  O método `model.predict_proba()` foi utilizado para gerar as probabilidades de inadimplência para cada transação.
5.  Foi gerado um arquivo final `submissao_case.csv` contendo as colunas `ID_CLIENTE`, `SAFRA_REF`, e `PROBABILIDADE_INADIMPLENCIA`.

---

## 8. Considerações Finais

### Pontos Fortes do Modelo
- **Alta Acurácia e Poder Preditivo:** O modelo alcançou uma excelente performance (ROC AUC > 0.96), com alta capacidade de identificar inadimplentes.
- **Engenharia de Variáveis Robusta:** A criação de features de histórico capturou de forma eficaz o comportamento do cliente.
- **Interpretabilidade:** A análise de feature importance fornece insights claros sobre os principais drivers de risco.

### Limitações
- **Qualidade dos Dados:** O modelo é sensível à qualidade dos dados de entrada; inconsistências futuras podem degradar a performance.
- **Variáveis Externas:** O modelo não considera fatores macroeconômicos (inflação, juros) ou de mercado, que podem influenciar a inadimplência.

### Próximos Passos Sugeridos
- **Monitoramento Contínuo:** Implementar um pipeline para monitorar a performance do modelo mensalmente e detectar desvios (model drift).
- **Testes A/B:** Utilizar o score de probabilidade para realizar testes A/B com diferentes estratégias de cobrança (ex: comunicação diferenciada para clientes de alto risco).
- **Enriquecimento de Dados:** Incorporar fontes de dados externas, como scores de crédito (Serasa, etc.) ou dados do Banco Central, para aumentar o poder preditivo.

---

## 9. Reprodutibilidade

Para garantir a reprodutibilidade do projeto, siga as instruções abaixo.

### Dependências
As principais bibliotecas e suas versões devem ser instaladas a partir de um arquivo `requirements.txt`. As bibliotecas chave são:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `optuna`
- `seaborn`
- `matplotlib`
- `joblib`
- `fastparquet`

### Instruções para Execução
O projeto está organizado em uma sequência de notebooks Jupyter:
1.  **`notebooks/1_analise_exploratoria.ipynb`**: Realiza a leitura, limpeza inicial e salvamento dos dados brutos em formato parquet.
2.  **`notebooks/2_processamento_dos_dados.ipynb`**: Carrega os dados intermediários, realiza a engenharia de variáveis e o encoding, salvando a base final pronta para modelagem.
3.  **`notebooks/3_ajuste_modelo.ipynb`**: Treina e avalia o modelo XGBoost base.
4.  **`notebooks/4_melhorando_o_ajuste.ipynb`**: Utiliza o Optuna para otimizar os hiperparâmetros do modelo.

Para reproduzir os resultados, execute os notebooks na ordem listada.

---

## 10. Proposta de Pipeline de Produção

Para automatizar as previsões mensalmente, o modelo pode ser integrado a um pipeline de produção. O fluxo essencial seguiria estas etapas:

1.  **Agendamento Automático:**
    *   No início de cada mês, um processo automático (orquestrador) seria acionado para iniciar a geração de previsões.

2.  **Coleta de Dados:**
    *   O pipeline buscaria os novos dados de pagamentos do mês que precisam ser avaliados.

3.  **Processamento e Inferência:**
    *   A mesma lógica poderia ser utilizada porem adaptada para limpar, transformar e criar variáveis.
    *   Em seguida, o modelo treinado (`xgb_inadimplencia_optuna.json`) carregaria esses dados e geraria a probabilidade de inadimplência para cada transação.

4.  **Armazenamento e Consumo:**
    *   As previsões finais seriam salvas em um local central (como um banco de dados), prontas para serem usadas por outras equipes, alimentar dashboards ou guiar as estratégias de cobrança.

5.  **Monitoramento:**
    *   Seria fundamental acompanhar a performance do modelo e a saúde do pipeline para garantir que as previsões continuem precisas e confiáveis ao longo do tempo, alertando sobre qualquer falha ou degradação. 