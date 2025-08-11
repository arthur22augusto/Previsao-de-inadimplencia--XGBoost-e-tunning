import os
import joblib
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc
)


# Função para carregar o modelo selecionado
def load_model(model_name):
    folder = 'models'
    filepath = os.path.join(folder, f'{model_name}.json')
    return joblib.load(filepath)

# Função para avaliar modelo
def evaluate_classification(y_true, y_pred, y_proba, print_results=True):
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    if print_results:
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(cr)
        print(f"\nROC AUC: {roc_auc:.6f}")
        print(f"Precision-Recall AUC: {pr_auc:.6f}")

    return cm, cr, roc_auc, pr_auc
