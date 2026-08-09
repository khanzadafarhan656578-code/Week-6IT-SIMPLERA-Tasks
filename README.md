# BBC News Category Classifier & NLP Pipeline

An enterprise-grade, end-to-end Natural Language Processing (NLP) pipeline for BBC News Article Category Classification, Named Entity Recognition (NER), and Topic Modeling (LDA & NMF).

This project implements robust text classification, pre-processes raw corpus data, extracts structured named entities, discovers latent topics, benchmarks traditional machine learning models against transformer-based transfer learning (DistilBERT), and exposes predictions via a command-line interface (CLI) tool.

---

## 🚀 Key Features

* **Data Quality Audit & EDA**: Automated dataset loading, missing data handling, class-imbalance ratio analysis, and high-quality visualization generation (ngrams, word clouds, article length distributions).
* **Part 1: Named Entity Recognition**: Automated spaCy NER pipeline extracting key categories (`PERSON`, `ORG`, `GPE`, etc.) with frequency distributions.
* **Part 2: Topic Modeling (LDA & NMF)**: Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization (NMF) with hyperparameter search over topic count ($K$) evaluated by $c_v$ coherence score.
* **Part 3: Traditional ML Benchmarks**: TF-IDF vectorization paired with hyperparameter tuning via Stratified 5-Fold Cross-Validation and GridSearchCV for:
  * Logistic Regression (Production Model)
  * Support Vector Machine (Linear SVM)
  * Multinomial Naive Bayes
* **Part 4: Transformer-Based Fine-Tuning**: Transfer learning using **DistilBERT** with PyTorch, AdamW optimization, linear learning rate scheduling, and early stopping.
* **CLI Predictor Utility**: A production inference tool for making real-time category predictions with probability confidence scores and extracted named entities.
* **100% Tested Codebase**: Automated testing suite using `pytest`.

---

## 📈 Model Performance & Benchmark Comparison

Under rigorous Stratified 5-Fold Cross-Validation on the BBC dataset:

| Model Architecture | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Weighted) | Inference Latency | Status / Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** ★ | **99.06%** | **0.9907** | **0.9906** | **0.9906** | **0.02 ms** | **Selected Production Model** |
| **Linear SVM** | **98.59%** | **0.9862** | **0.9859** | **0.9860** | **0.01 ms** | **Best Traditional Margin Classifier** |
| **Multinomial Naive Bayes** | 97.65% | 0.9775 | 0.9765 | 0.9766 | 0.03 ms | Fast Probabilistic Baseline |
| **DistilBERT Transformer** | 97.30% | 0.9735 | 0.9725 | 0.9730 | 14.20 ms | Deep Contextual Attention |
| **Random Forest** | 95.51% | 0.9566 | 0.9538 | 0.9551 | 0.15 ms | Ensemble Tree Baseline |

---

## 🛠️ Installation & Setup

Ensure you have Python 3.10+ installed.

### 1. Install Dependencies
Install all package requirements listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Download spaCy Language Model
Download the small English spaCy model needed for Named Entity Recognition:
```bash
python -m spacy download en_core_web_sm
```

---

## 🏃 Running the Application

### 1. Run the Prediction CLI
Make predictions on raw article text using the `predict.py` tool. By default, it runs inference using the production **Logistic Regression** model:

```bash
python predict.py --text "The central bank decided to lower interest rates to support the economy."
```

Or run interactively (it will prompt you for text input):
```bash
python predict.py
```

#### Example Output:
```json
{
  "input_snippet": "The central bank decided to lower interest rates to support the economy.",
  "predicted_category": "business",
  "confidence": 0.9531,
  "top_k_predictions": [
    {"category": "business", "probability": 0.9531},
    {"category": "sport", "probability": 0.0177},
    {"category": "entertainment", "probability": 0.0112}
  ],
  "extracted_entities": {}
}
```

### 2. Run the Full NLP Pipeline
To perform data loading, generate all visualizations, perform NER / Topic Modeling analysis, benchmark traditional models, and fine-tune the DistilBERT model, execute `run_pipeline.py`:

```bash
python run_pipeline.py
```

All figures will be exported to the `images/` directory, and models/checkpoints will be saved to the `models/` directory.

---

## 🧪 Running Unit Tests

To run the automated validation tests and ensure all components function correctly:

```bash
python -m pytest tests/test_nlp_pipeline.py
```

All 4 test modules will verify data loading, text preprocessing, named entity extraction, and prediction endpoints.

---

## 📂 Project Directory Structure

```
├── data/                       # Contains dataset (bbc_articles.csv)
├── figures/                    # Generated pipeline reports / documents
├── images/                     # Generated charts and visualization figures
├── models/                     # Saved model artifacts (.joblib and Transformer paths)
├── notebooks/                  # Interactive development notebooks
├── src/                        # Core module library
│   ├── baseline_models.py      # TF-IDF & traditional Scikit-Learn classifiers
│   ├── config.py               # Pathing and hyperparameter configurations
│   ├── data_loader.py          # Data ingestion and auditing
│   ├── distilbert_classifier.py# DistilBERT classifier model architecture
│   ├── eda.py                  # Exploratory Data Analysis functions
│   ├── evaluation.py           # Metrics calculation and error diagnostics
│   ├── inference.py            # Real-time ProductionPredictor API class
│   ├── ner.py                  # Named Entity Recognition spaCy code
│   ├── preprocessing.py        # Text clean and tokenization pipeline
│   ├── topic_modeling.py       # LDA and NMF Topic Modeling routines
│   ├── utils.py                # Reproducibility seeds and logging helpers
│   └── visualization.py        # Charting & Wordcloud generation routines
├── tests/                      # Testing directory
└── run_pipeline.py             # Script to execute the entire training pipeline
```
