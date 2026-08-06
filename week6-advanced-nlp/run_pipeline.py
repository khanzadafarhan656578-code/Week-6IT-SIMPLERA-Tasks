"""
End-to-End Pipeline Execution Script for Week 6 Advanced NLP Project.
Trains models, generates visualizations, logs experiments, serializes artifacts,
and compiles the final Jupyter Notebook with all outputs rendered.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import joblib
import torch
import nbformat as nbf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    DATA_DIR, MODELS_DIR, IMAGES_DIR, NOTEBOOKS_DIR, 
    RANDOM_STATE, TEST_SIZE, NER_ENTITIES, MODEL_NAME
)
import src.config as config
from src.utils import set_seeds, log_experiment
from src.data_loader import load_data, get_dataset_stats
from src.preprocessing import clean_text, preprocess_corpus_for_topic_modeling
from src.eda import compute_corpus_eda
from src.ner import load_spacy_model, process_corpus_ner, get_entity_frequency_table
from src.topic_modeling import (
    evaluate_lda_topic_counts, evaluate_nmf_topic_counts, 
    get_topic_keywords, assign_dominant_topics, compute_topic_category_crosstab
)
from src.baseline_models import train_tfidf_baselines
from src.distilbert_classifier import train_distilbert_model
from src.evaluation import (
    get_detailed_classification_report, compute_confusion_matrix,
    compute_multiclass_roc_auc, analyze_model_errors, get_top_tfidf_features_per_class
)
from src.visualization import (
    plot_category_distribution, plot_article_length_boxplot, plot_top_ngrams,
    plot_category_wordclouds, plot_entity_frequencies, plot_topic_coherence,
    plot_topic_category_heatmap, plot_confusion_matrix_custom, plot_training_curves,
    plot_model_comparison_benchmark
)

def run_full_pipeline():
    print("==================================================", flush=True)
    print(" STARTING WEEK 6 ADVANCED NLP PIPELINE EXECUTION", flush=True)
    print("==================================================", flush=True)

    # 1. Reproducibility
    set_seeds(RANDOM_STATE)

    # 2. Data Loading & Audit
    df = load_data(DATA_DIR)
    stats = get_dataset_stats(df)
    print(f"\n[DATA AUDIT] Dataset loaded successfully. Total Articles: {stats['num_articles']}", flush=True)
    print(f"Categories: {stats['categories']}", flush=True)

    # Dynamic MAX_LENGTH selection based on 95th percentile word token count
    p95_words = stats['word_count_p95']
    chosen_max_len = 256 if p95_words <= 200 else 512
    config.MAX_LENGTH = chosen_max_len
    print(f"[DATA-DRIVEN CONFIG] 95th percentile word length is {p95_words}. Dynamic MAX_LENGTH set to: {chosen_max_len}", flush=True)

    # 3. Exploratory Data Analysis & Plots
    print("\n[EDA] Executing Exploratory Data Analysis...", flush=True)
    plot_category_distribution(df['category'].value_counts())
    plot_article_length_boxplot(df)
    
    eda_stats = compute_corpus_eda(df)
    plot_top_ngrams(eda_stats['top_unigrams'], eda_stats['top_bigrams'], eda_stats['top_trigrams'])
    plot_category_wordclouds(df)

    # 4. Part 1: Named Entity Recognition with spaCy
    print("\n[PART 1: NER] Running spaCy Named Entity Recognition...", flush=True)
    spacy_nlp = load_spacy_model("en_core_web_sm")
    df_ner, cat_top_entities = process_corpus_ner(df, spacy_nlp, NER_ENTITIES)
    df_ner_freq = get_entity_frequency_table(df_ner, top_k=20)
    plot_entity_frequencies(df_ner_freq)
    print(f"[NER] Extracted {len(df_ner)} entity occurrences across corpus.", flush=True)

    # 5. Part 2: Topic Modeling (LDA & NMF)
    print("\n[PART 2: TOPIC MODELING] Preprocessing text for topic modeling...", flush=True)
    processed_tokens_list = preprocess_corpus_for_topic_modeling(df['text'].tolist(), spacy_nlp)
    
    df_lda_coherence, lda_dict, lda_corpus = evaluate_lda_topic_counts(processed_tokens_list, min_k=2, max_k=10, step=1)
    plot_topic_coherence(df_lda_coherence)
    
    # Pick optimal LDA model (highest coherence)
    best_lda_idx = df_lda_coherence['coherence_score'].idxmax()
    best_lda_row = df_lda_coherence.loc[best_lda_idx]
    optimal_lda_model = best_lda_row['model']
    print(f"[TOPIC MODELING] Optimal LDA topic count selected: K={best_lda_row['num_topics']} (Coherence = {best_lda_row['coherence_score']:.4f})", flush=True)

    # Save LDA model & dictionary
    joblib.dump(optimal_lda_model, os.path.join(MODELS_DIR, "lda_topic_model.joblib"))
    joblib.dump(lda_dict, os.path.join(MODELS_DIR, "lda_dictionary.joblib"))

    # Assign dominant topics and plot heatmap alignment
    df_with_topics = assign_dominant_topics(optimal_lda_model, lda_corpus, df, is_gensim=True)
    crosstab_df = compute_topic_category_crosstab(df_with_topics)
    plot_topic_category_heatmap(crosstab_df)

    # 6. Model Training & Evaluation Setup
    print("\n[TRAIN/TEST SPLIT] Encoding labels and splitting dataset 80/20...", flush=True)
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['category'])
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].tolist(), df['label_encoded'].values,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df['label_encoded'].values
    )

    # 7. Baseline ML Models
    print("\n[PART 3: BASELINE ML] Training TF-IDF Baseline Models with 5-Fold GridSearchCV...", flush=True)
    df_base_benchmark, vectorizer, ml_models, X_train_vec, X_test_vec = train_tfidf_baselines(X_train, y_train, X_test, y_test)
    
    # Save TF-IDF & baseline models
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    for name, model in ml_models.items():
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))

    # Plot baseline confusion matrix (Logistic Regression)
    lr_cm = compute_confusion_matrix(y_test, df_base_benchmark.loc[0, 'Predictions'])
    plot_confusion_matrix_custom(lr_cm, list(label_encoder.classes_), "Logistic Regression (TF-IDF) Confusion Matrix", "baseline_confusion_matrices.png")

    # Log baseline experiments
    for idx, row in df_base_benchmark.iterrows():
        exp_data = {
            'Model': row['Model'],
            'Best_Params': row['Best_Params'],
            'Accuracy': round(row['Accuracy'], 4),
            'Precision': round(row['Precision'], 4),
            'Recall': round(row['Recall'], 4),
            'F1_Score': round(row['F1_Score'], 4),
            'Train_Time_s': round(row['Train_Time_s'], 2),
            'Inference_Time_ms': round(row['Inference_Time_ms'], 2)
        }
        log_experiment(exp_data)

    # 8. Part 4: Transfer Learning with DistilBERT Fine-Tuning
    print("\n[PART 4: TRANSFORMER FINE-TUNING] Fine-Tuning DistilBERT...", flush=True)
    distilbert_model, tokenizer, training_history, distilbert_metrics = train_distilbert_model(
        X_train, y_train, X_test, y_test, num_classes=len(label_encoder.classes_),
        label_encoder=label_encoder, config=config
    )

    # Save DistilBERT model & tokenizer
    model_save_path = os.path.join(MODELS_DIR, "distilbert_bbc_model")
    tokenizer_save_path = os.path.join(MODELS_DIR, "distilbert_tokenizer")
    distilbert_model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f"[SAVED MODEL] Fine-tuned DistilBERT saved to: {model_save_path}", flush=True)

    # Plot transformer training curves & confusion matrix
    plot_training_curves(training_history)
    trans_cm = compute_confusion_matrix(y_test, distilbert_metrics['Predictions'])
    plot_confusion_matrix_custom(trans_cm, list(label_encoder.classes_), "DistilBERT Fine-Tuned Confusion Matrix", "distilbert_confusion_matrix.png")

    # Log transformer experiment
    trans_exp = {
        'Model': distilbert_metrics['Model'],
        'Best_Params': distilbert_metrics['Best_Params'],
        'Accuracy': round(distilbert_metrics['Accuracy'], 4),
        'Precision': round(distilbert_metrics['Precision'], 4),
        'Recall': round(distilbert_metrics['Recall'], 4),
        'F1_Score': round(distilbert_metrics['F1_Score'], 4),
        'Train_Time_s': round(distilbert_metrics['Train_Time_s'], 2),
        'Inference_Time_ms': round(distilbert_metrics['Inference_Time_ms'], 2)
    }
    log_experiment(trans_exp)

    # 9. Model Comparison & Benchmarking Plot
    df_all_benchmarks = pd.read_csv(os.path.join(MODELS_DIR, "experiments.csv"))
    plot_model_comparison_benchmark(df_all_benchmarks)

    print("\n==================================================", flush=True)
    print(" ALL PIPELINE STEPS EXECUTED AND ARTIFACTS GENERATED", flush=True)
    print("==================================================", flush=True)
    
    return {
        'df': df,
        'stats': stats,
        'df_ner_freq': df_ner_freq,
        'df_lda_coherence': df_lda_coherence,
        'df_all_benchmarks': df_all_benchmarks,
        'training_history': training_history,
        'label_encoder': label_encoder,
        'distilbert_metrics': distilbert_metrics
    }

if __name__ == "__main__":
    run_full_pipeline()
