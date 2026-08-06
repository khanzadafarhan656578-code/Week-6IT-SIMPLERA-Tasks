"""
Evaluation, Error Analysis, and Explainability Module.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def get_detailed_classification_report(y_true, y_pred, target_names: list) -> str:
    """
    Generates formatted classification report text.
    """
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    return report

def compute_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """
    Computes confusion matrix array.
    """
    return confusion_matrix(y_true, y_pred)

def compute_multiclass_roc_auc(y_true, y_probs, num_classes: int) -> float:
    """
    Computes One-vs-Rest macro ROC-AUC score.
    """
    if y_probs is None:
        return 0.0
    try:
        y_true_oh = np.eye(num_classes)[y_true]
        auc = roc_auc_score(y_true_oh, y_probs, multi_class='ovr', average='macro')
        return float(auc)
    except Exception:
        return 0.0

def analyze_model_errors(df_test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, label_encoder) -> pd.DataFrame:
    """
    Extracts and ranks misclassified articles with prediction confidence.
    """
    df_analysis = df_test.copy()
    df_analysis['true_label'] = label_encoder.inverse_transform(y_true)
    df_analysis['pred_label'] = label_encoder.inverse_transform(y_pred)
    df_analysis['is_misclassified'] = y_true != y_pred
    
    if y_probs is not None:
        df_analysis['confidence'] = np.max(y_probs, axis=1)
    else:
        df_analysis['confidence'] = 1.0

    df_errors = df_analysis[df_analysis['is_misclassified']].sort_values(by='confidence', ascending=False)
    return df_errors

def get_top_tfidf_features_per_class(model, vectorizer, class_names: list, top_n: int = 10) -> dict:
    """
    Extracts top positive TF-IDF features per class for Logistic Regression.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_features = {}

    if hasattr(model, 'coef_'):
        for i, class_name in enumerate(class_names):
            coefs = model.coef_[i]
            top_indices = np.argsort(coefs)[::-1][:top_n]
            top_words = feature_names[top_indices]
            weights = coefs[top_indices]
            top_features[class_name] = list(zip(top_words, weights))
            
    return top_features
