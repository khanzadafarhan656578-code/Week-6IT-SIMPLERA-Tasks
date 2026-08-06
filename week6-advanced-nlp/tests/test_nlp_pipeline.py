"""
Unit Tests for Week 6 Advanced NLP Portfolio Project.
"""

import pytest
import os
import pandas as pd
from src.data_loader import load_data, get_dataset_stats
from src.preprocessing import clean_text, preprocess_corpus_for_topic_modeling
from src.ner import load_spacy_model, extract_entities_from_text
from src.topic_modeling import evaluate_lda_topic_counts
from src.inference import ProductionPredictor

def test_data_loader(tmp_path):
    # Create test CSV
    csv_file = tmp_path / "bbc_articles.csv"
    df_dummy = pd.DataFrame({
        "text": ["Economy shows growth in Q3.", "Player scores winning goal."],
        "category": ["business", "sport"]
    })
    df_dummy.to_csv(csv_file, index=False)
    
    df = load_data(data_dir=str(tmp_path))
    assert df is not None
    assert len(df) == 2
    
    stats = get_dataset_stats(df)
    assert stats["num_articles"] == 2

def test_text_preprocessor():
    raw = "The QUICK brown fox jumps over 123 lazy dogs!"
    cleaned = clean_text(raw)
    assert "QUICK" in cleaned or "quick" in cleaned.lower()
    
    tokens = preprocess_corpus_for_topic_modeling([raw])
    assert len(tokens) > 0

def test_ner_extractor():
    nlp = load_spacy_model("en_core_web_sm")
    sample_text = "Apple CEO Tim Cook visited London in 2026."
    entities = extract_entities_from_text(sample_text, nlp)
    assert isinstance(entities, list)
    assert len(entities) > 0

def test_production_predictor():
    predictor = ProductionPredictor(model_type="logistic_regression")
    result = predictor.predict("Central bank cuts interest rates to boost economy")
    assert "predicted_category" in result
    assert "confidence" in result
