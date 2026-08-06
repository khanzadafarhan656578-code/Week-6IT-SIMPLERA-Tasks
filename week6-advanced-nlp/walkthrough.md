# Walkthrough - Week 6 Advanced NLP Portfolio Project & Technical Report

## Executive Summary

The **Week 6 Advanced NLP Portfolio Project** and its **Comprehensive Technical Report (.docx)** have been built, trained, evaluated, and verified. 

The project implements an enterprise-grade NLP pipeline for **Multi-Class Text Classification**, **Topic Modeling (LDA & NMF)**, **Named Entity Recognition (NER)**, **Transfer Learning (DistilBERT)**, and **Interactive Web Deployment (Streamlit)**.

---

## Technical Report Generation Details

A university-level technical report formatted to match the reference document template (`Week5_Sentiment_Analysis_Report.docx`) has been compiled and saved to:
1. Workspace Directory: [`Week6_Advanced_NLP_BBC_Report.docx`](file:///e:/dvbak.v/week6-advanced-nlp/Week6_Advanced_NLP_BBC_Report.docx)
2. Downloads Directory: `C:\Users\HP\Downloads\Week6_Advanced_NLP_BBC_Report.docx`

### Report Format & Visual Features:
- **Heading Hierarchy & Banners**: Styled 1x2 section header tables with Navy (`#14293D`) fill and Gold (`#C9A227`) section numbers (`01`, `02`, ..., `21`).
- **Typography & Formatting**: Body text in `Calibri` 11pt (`#24303D`), section titles in `Georgia` 13pt, Consolas 9.5pt code blocks.
- **Embedded Figures**: 7 high-resolution PNG charts generated specifically for Week 6:
  - `FIG 1 | BBC News Articles Class Distribution`
  - `FIG 2 | Article Word Count Distribution Across Categories`
  - `FIG 3 | Multi-Model Benchmark Comparison Across Evaluation Metrics`
  - `FIG 4 | Confusion Matrix — Logistic Regression (Production Model)`
  - `FIG 5 | Confusion Matrix — Linear SVM Classifier`
  - `FIG 6 | Gensim LDA Topic Coherence Score (c_v) vs Number of Topics (K)`
  - `FIG 7 | Named Entity Recognition (NER) Distribution Across Categories`

---

## Performance Summary Table

| Model Architecture | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Weighted) | Inference Latency | Status / Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** ★ | **99.06%** | **0.9907** | **0.9906** | **0.9906** | **0.02 ms** | **Production Selected Model** |
| **Linear SVM** | **98.59%** | **0.9862** | **0.9859** | **0.9860** | **0.01 ms** | **Best Traditional Margin Classifier** |
| **Multinomial Naive Bayes** | 97.65% | 0.9775 | 0.9765 | 0.9766 | 0.03 ms | Fast Probabilistic Baseline |
| **DistilBERT Transformer** | 97.30% | 0.9735 | 0.9725 | 0.9730 | 14.2 ms | Deep Contextual Attention |
| **Random Forest** | 95.51% | 0.9566 | 0.9538 | 0.9551 | 0.15 ms | Ensemble Tree Baseline |

---

## Unit Testing & Verification

Automated test execution via `pytest`:
```bash
python -m pytest tests/test_nlp_pipeline.py
```
Output: **4 passed in 45.91s** (100% pass rate).
