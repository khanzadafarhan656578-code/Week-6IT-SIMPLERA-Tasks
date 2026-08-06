"""
Data Loader Module for flexible ingestion, schema auto-detection, and statistics computation.
"""

import os
import glob
import pandas as pd
import numpy as np

def load_data(data_dir: str = None) -> pd.DataFrame:
    """
    Scans the data directory, auto-detects CSV files, dynamically identifies text
    and label columns, cleans basic formatting, and returns a standardized DataFrame.
    """
    if data_dir is None:
        from src.config import DATA_DIR
        data_dir = DATA_DIR

    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {data_dir}")

    # Prioritize bbc_articles.csv if available
    target_csv = None
    for csv_file in csv_files:
        if "bbc_articles" in os.path.basename(csv_file):
            target_csv = csv_file
            break
    if target_csv is None:
        target_csv = csv_files[0]

    print(f"[DATA LOADER] Loading dataset from: {target_csv}")
    df = pd.read_csv(target_csv)

    # Schema Auto-Detection
    col_map_text = ["text", "content", "article", "news_text", "body"]
    col_map_label = ["labels", "category", "label", "topic", "class"]

    found_text_col = None
    found_label_col = None

    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower in col_map_text and found_text_col is None:
            found_text_col = col
        if col_lower in col_map_label and found_label_col is None:
            found_label_col = col

    if found_text_col is None or found_label_col is None:
        # Fallback heuristic: choose longest text column for text, lowest unique count column for category
        for col in df.columns:
            if df[col].dtype == object:
                avg_len = df[col].astype(str).str.len().mean()
                num_unique = df[col].nunique()
                if avg_len > 100 and found_text_col is None:
                    found_text_col = col
                elif num_unique < 20 and found_label_col is None:
                    found_label_col = col

    if found_text_col is None or found_label_col is None:
        raise ValueError(f"Could not automatically detect text and category columns in CSV columns: {df.columns.tolist()}")

    print(f"[DATA LOADER] Auto-detected columns -> Text Column: '{found_text_col}', Category Column: '{found_label_col}'")

    # Standardize column names
    df = df[[found_text_col, found_label_col]].copy()
    df.columns = ["text", "category"]

    # Initial Cleaning: missing values & duplicate rows
    initial_len = len(df)
    df.dropna(subset=["text", "category"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    cleaned_len = len(df)
    
    if initial_len != cleaned_len:
        print(f"[DATA CLEANING] Removed {initial_len - cleaned_len} missing/duplicate rows. Remaining rows: {cleaned_len}")

    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    
    return df.reset_index(drop=True)

def get_dataset_stats(df: pd.DataFrame) -> dict:
    """
    Computes summary statistics for dataset audit & EDA.
    """
    char_lengths = df["text"].apply(len)
    word_counts = df["text"].apply(lambda t: len(t.split()))
    sentence_counts = df["text"].apply(lambda t: t.count(".") + t.count("!") + t.count("?"))

    stats = {
        "num_articles": len(df),
        "num_categories": df["category"].nunique(),
        "categories": df["category"].value_counts().to_dict(),
        "char_len_mean": float(char_lengths.mean()),
        "char_len_min": int(char_lengths.min()),
        "char_len_max": int(char_lengths.max()),
        "word_count_mean": float(word_counts.mean()),
        "word_count_min": int(word_counts.min()),
        "word_count_max": int(word_counts.max()),
        "word_count_p95": int(np.percentile(word_counts, 95)),
        "sentence_count_mean": float(sentence_counts.mean())
    }
    return stats
