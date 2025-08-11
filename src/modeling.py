from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
import optuna
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score, fbeta_score
)
from xgboost import XGBClassifier
import os
import glob
from functools import partial
from datetime import datetime
import json



# Função para dropar colunas erradas para treino
def drop_columns_for_modeling(df):
    columns_to_drop = [
    'ID_CLIENTE',              # ID único, não é uma feature preditiva
    'DATA_EMISSAO_DOCUMENTO',  # Datas originais, já temos features derivadas
    'DATA_PAGAMENTO',          # Vazamento de dados para prever inadimplência
    'DATA_VENCIMENTO',         # Datas originais, já temos features derivadas
    'DATA_CADASTRO',
    'DIAS_DE_ATRASO',
    'SAFRA_REF',
    'DIAS_ATE_PAGAMENTO'    
]
    return df.drop(columns=columns_to_drop)
    
# Função para fazer a divisão entre treino e teste
def split_data(df):
    X = df.drop(columns=['INADIMPLENTE'])
    y = df['INADIMPLENTE']

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test


# Função para pegar o scale_pos_weight e lidar com desbalanceamento
def get_scale_pos_weight(y_train):
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    return scale_pos_weight

# Função para ajustar o primeiro modelo
def fit_first_model(X_train, y_train, scale_pos_weight):
    model = XGBClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    return model

# Função para realizar predições
def get_predictions(model, X):
    y_proba = model.predict_proba(X)[:, 1]  # Probabilidade da classe positiva
    y_pred = model.predict(X)               # Classe prevista
    return y_pred, y_proba


# Função para salvar o modelo
def save_model(model, filename):
    folder = 'models'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Obter a data atual no formato AAAA-MM-DD
    today = datetime.today().strftime('%Y-%m-%d')

    # Criar o nome do arquivo com a data
    full_filename = f'{filename}_{today}.json'

    # Caminho completo
    filepath = os.path.join(folder, full_filename)

    model.save_model(filepath)


# Utilitários para persistir schema de features e threshold
def save_feature_schema(features, filename="feature_schema.json"):
    folder = 'models'
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"features": list(features)}, f, ensure_ascii=False, indent=2)
    return path


def load_feature_schema(filename="feature_schema.json"):
    path = os.path.join('models', filename)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["features"]


def save_threshold(value: float, filename="best_threshold.json"):
    folder = 'models'
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"best_threshold": float(value)}, f)
    return path


def load_threshold(filename="best_threshold.json") -> float:
    path = os.path.join('models', filename)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return float(data["best_threshold"])


def load_latest_model(prefix: str = 'best_model') -> XGBClassifier:
    pattern = os.path.join('models', f'{prefix}_*.json')
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum modelo encontrado com prefixo {prefix} em models/"
        )
    latest = max(candidates, key=os.path.getmtime)
    model = XGBClassifier()
    model.load_model(latest)
    return model


# Função para definir o objetivo da busca:

def objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        # peso balanceado: razão entre negativos e positivos
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", get_scale_pos_weight(y_train) - 5.0, get_scale_pos_weight(y_train) + 15),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "objective": "binary:logistic"
    }

    model = XGBClassifier(**params)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # PR AUC = average_precision_score
    scores = cross_val_score(
        model, X_train, y_train,
        cv=skf, scoring='average_precision', n_jobs=-1
    )

    return scores.mean()


# Função para realizar a busca de hiperparametros:
def tune_model(data, recall_minimo=0.80):
    X = data.drop(columns=['INADIMPLENTE'])
    y = data['INADIMPLENTE']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Otimização com Optuna
    objective_with_data = partial(objective, X_train=X_train, y_train=y_train)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_with_data, n_trials=50, show_progress_bar=True)
    best_params = study.best_params

    # Treina modelo final
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    # Busca pelo melhor threshold baseado no recall mínimo desejado
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # Encontra primeiro threshold que atenda o recall mínimo
    idx = np.where(recall >= recall_minimo)[0]
    if len(idx) > 0:
        best_threshold = thresholds[idx[-1]]  # mais alto threshold com recall >= meta
    else:
        best_threshold = 0.5  # fallback

    # Avalia também F2-score no threshold encontrado
    y_pred_thresh = (y_proba >= best_threshold).astype(int)
    f2 = fbeta_score(y_test, y_pred_thresh, beta=2)

    print(f"Best params: {best_params}")
    print(f"Threshold ótimo: {best_threshold:.4f}")
    print(f"Recall no threshold: {recall[idx[-1]] if len(idx) > 0 else recall.mean():.4f}")
    print(f"F2-score no threshold: {f2:.4f}")
    print(f"PR AUC no teste: {average_precision_score(y_test, y_proba):.4f}")

    return model, best_threshold, best_params






