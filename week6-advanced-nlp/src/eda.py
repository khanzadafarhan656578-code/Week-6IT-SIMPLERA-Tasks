"""
Exploratory Data Analysis Module for computing n-grams, vocabulary metrics, and distributions.
"""

from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def get_top_ngrams(corpus: list, n: int = 1, top_k: int = 20, stop_words: str = 'english') -> pd.DataFrame:
    """
    Computes top unigrams, bigrams, or trigrams for a text corpus.
    """
    vec = CountVectorizer(ngram_range=(n, n), stop_words=stop_words, max_features=10000)
    bag_of_words = vec.fit_transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]
    
    df_ngrams = pd.DataFrame(words_freq, columns=['ngram', 'count'])
    return df_ngrams

def compute_corpus_eda(df: pd.DataFrame) -> dict:
    """
    Computes category breakdown, article word count statistics, and top terms.
    """
    category_counts = df['category'].value_counts()
    word_counts_by_cat = df.groupby('category')['text'].apply(lambda s: s.apply(lambda x: len(x.split())).mean())
    
    top_unigrams = get_top_ngrams(df['text'].tolist(), n=1, top_k=15)
    top_bigrams = get_top_ngrams(df['text'].tolist(), n=2, top_k=15)
    top_trigrams = get_top_ngrams(df['text'].tolist(), n=3, top_k=15)

    return {
        "category_counts": category_counts,
        "word_counts_by_cat": word_counts_by_cat,
        "top_unigrams": top_unigrams,
        "top_bigrams": top_bigrams,
        "top_trigrams": top_trigrams
    }
