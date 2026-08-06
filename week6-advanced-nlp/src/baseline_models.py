"""
Baseline ML Models Module using TF-IDF and 5-Fold GridSearchCV Hyperparameter Tuning.
"""

import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

def train_tfidf_baselines(X_train: list, y_train: list, X_test: list, y_test: list) -> tuple:
    """
    Fits TF-IDF vectorizer and trains hyperparameter-tuned Logistic Regression, Linear SVM, and Naive Bayes models.
    """
    print("[BASELINE ML] Fitting TF-IDF Vectorizer...", flush=True)
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english', sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results = []
    models = {}

    # 1. Logistic Regression with GridSearchCV
    print("[BASELINE ML] Tuning Logistic Regression via 5-Fold GridSearchCV...", flush=True)
    start_train = time.time()
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid={'C': [0.1, 1.0, 10.0]},
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1
    )
    lr_grid.fit(X_train_vec, y_train)
    lr_train_time = time.time() - start_train
    best_lr = lr_grid.best_estimator_

    start_inf = time.time()
    lr_preds = best_lr.predict(X_test_vec)
    lr_probs = best_lr.predict_proba(X_test_vec) if hasattr(best_lr, "predict_proba") else None
    lr_inf_time = (time.time() - start_inf) / len(X_test) * 1000  # ms per sample

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, lr_preds, average='weighted')
    results.append({
        'Model': 'Logistic Regression (TF-IDF)',
        'Best_Params': str(lr_grid.best_params_),
        'Accuracy': accuracy_score(y_test, lr_preds),
        'Precision': prec,
        'Recall': rec,
        'F1_Score': f1,
        'Train_Time_s': lr_train_time,
        'Inference_Time_ms': lr_inf_time,
        'Predictions': lr_preds,
        'Probabilities': lr_probs
    })
    models['logistic_regression'] = best_lr

    # 2. Linear SVM with GridSearchCV
    print("[BASELINE ML] Tuning Linear SVM via 5-Fold GridSearchCV...", flush=True)
    start_train = time.time()
    svm_grid = GridSearchCV(
        LinearSVC(random_state=42, max_iter=2000),
        param_grid={'C': [0.1, 1.0, 10.0]},
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1
    )
    svm_grid.fit(X_train_vec, y_train)
    svm_train_time = time.time() - start_train
    best_svm = svm_grid.best_estimator_

    start_inf = time.time()
    svm_preds = best_svm.predict(X_test_vec)
    svm_inf_time = (time.time() - start_inf) / len(X_test) * 1000

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, svm_preds, average='weighted')
    results.append({
        'Model': 'Linear SVM (TF-IDF)',
        'Best_Params': str(svm_grid.best_params_),
        'Accuracy': accuracy_score(y_test, svm_preds),
        'Precision': prec,
        'Recall': rec,
        'F1_Score': f1,
        'Train_Time_s': svm_train_time,
        'Inference_Time_ms': svm_inf_time,
        'Predictions': svm_preds,
        'Probabilities': None
    })
    models['linear_svm'] = best_svm

    # 3. Multinomial Naive Bayes with GridSearchCV
    print("[BASELINE ML] Tuning Multinomial Naive Bayes via 5-Fold GridSearchCV...", flush=True)
    start_train = time.time()
    nb_grid = GridSearchCV(
        MultinomialNB(),
        param_grid={'alpha': [0.01, 0.1, 1.0]},
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1
    )
    nb_grid.fit(X_train_vec, y_train)
    nb_train_time = time.time() - start_train
    best_nb = nb_grid.best_estimator_

    start_inf = time.time()
    nb_preds = best_nb.predict(X_test_vec)
    nb_probs = best_nb.predict_proba(X_test_vec)
    nb_inf_time = (time.time() - start_inf) / len(X_test) * 1000

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, nb_preds, average='weighted')
    results.append({
        'Model': 'Multinomial Naive Bayes (TF-IDF)',
        'Best_Params': str(nb_grid.best_params_),
        'Accuracy': accuracy_score(y_test, nb_preds),
        'Precision': prec,
        'Recall': rec,
        'F1_Score': f1,
        'Train_Time_s': nb_train_time,
        'Inference_Time_ms': nb_inf_time,
        'Predictions': nb_preds,
        'Probabilities': nb_probs
    })
    models['naive_bayes'] = best_nb

    df_benchmark = pd.DataFrame(results)
    return df_benchmark, vectorizer, models, X_train_vec, X_test_vec
