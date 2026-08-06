"""
Text Preprocessing Module for general cleaning and topic modeling tokenization/lemmatization.
"""

import re
import string
import pandas as pd

def clean_text(text: str) -> str:
    """
    General text cleaning:
    - Normalizes whitespace
    - Removes HTML tags & URLs if present
    - Fixes special encoding characters while preserving semantic entity markers.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    # Fix unusual whitespace / tabs / newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_corpus_for_topic_modeling(texts: list, nlp_model=None) -> list:
    """
    Batched preprocessing pipeline for Topic Modeling (LDA/NMF):
    - Tokenization
    - Lowercasing
    - Stopword removal
    - Lemmatization
    - Punctuation removal
    - Short word removal (length <= 2)
    """
    cleaned_texts = [clean_text(t).lower() for t in texts]
    processed_list = []

    if nlp_model is not None:
        print(f"[PREPROCESSING] Lemmatizing {len(texts)} documents via spaCy nlp.pipe...", flush=True)
        docs = nlp_model.pipe(cleaned_texts, batch_size=128, disable=["ner", "parser"])
        for doc in docs:
            tokens = [
                token.lemma_ for token in doc 
                if not token.is_stop 
                and not token.is_punct 
                and not token.is_space 
                and not token.like_num 
                and len(token.lemma_) > 2 
                and token.lemma_.isalpha()
            ]
            processed_list.append(tokens)
    else:
        for text in cleaned_texts:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
            processed_list.append([w for w in words if len(w) > 2])

    return processed_list

def preprocess_for_topic_modeling(text: str, nlp_model=None) -> list:
    """
    Single-text fallback wrapper.
    """
    return preprocess_corpus_for_topic_modeling([text], nlp_model)[0]
