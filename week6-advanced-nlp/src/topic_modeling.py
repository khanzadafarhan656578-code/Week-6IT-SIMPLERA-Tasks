"""
Topic Modeling Module using Gensim LDA and Scikit-Learn NMF with Coherence Score Optimization.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel

def evaluate_lda_topic_counts(tokens_list: list, min_k: int = 2, max_k: int = 15, step: int = 1) -> pd.DataFrame:
    """
    Evaluates LDA topic models across K=min_k..max_k using Gensim Coherence Score (c_v) and Perplexity.
    """
    dictionary = Dictionary(tokens_list)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]

    results = []
    print(f"[TOPIC MODELING] Evaluating LDA coherence scores for K={min_k} to K={max_k}...")

    for k in range(min_k, max_k + 1, step):
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=tokens_list,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        perplexity = lda_model.log_perplexity(corpus)

        results.append({
            'num_topics': k,
            'coherence_score': coherence_score,
            'perplexity': perplexity,
            'model': lda_model,
            'dictionary': dictionary,
            'corpus': corpus
        })
        print(f"  -> K={k}: Coherence Score (c_v) = {coherence_score:.4f}, Perplexity = {perplexity:.4f}")

    df_results = pd.DataFrame(results)
    return df_results, dictionary, corpus

def evaluate_nmf_topic_counts(texts: list, min_k: int = 2, max_k: int = 15) -> tuple:
    """
    Trains Scikit-Learn NMF across K=min_k..max_k and returns models & TF-IDF vectorizer.
    """
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=3, stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    nmf_results = []
    print(f"[TOPIC MODELING] Evaluating NMF models for K={min_k} to K={max_k}...")

    for k in range(min_k, max_k + 1):
        nmf = NMF(n_components=k, random_state=42, init='nndsvda', max_iter=200)
        nmf.fit(tfidf_matrix)
        
        # Calculate reconstruction error as metric
        reconstruction_err = nmf.reconstruction_err_
        nmf_results.append({
            'num_topics': k,
            'reconstruction_error': reconstruction_err,
            'model': nmf
        })

    return pd.DataFrame(nmf_results), tfidf_vectorizer, tfidf_matrix, feature_names

def get_topic_keywords(model, dictionary_or_features, num_words: int = 15, is_gensim: bool = True) -> dict:
    """
    Extracts top N keywords for each topic.
    """
    topic_keywords = {}

    if is_gensim:
        for topic_idx in range(model.num_topics):
            words = [word for word, prob in model.show_topic(topic_idx, topn=num_words)]
            topic_keywords[f"Topic_{topic_idx+1}"] = words
    else:
        # NMF sklearn model
        feature_names = dictionary_or_features
        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[:-num_words - 1:-1]
            words = [feature_names[i] for i in top_words_idx]
            topic_keywords[f"Topic_{topic_idx+1}"] = words

    return topic_keywords

def assign_dominant_topics(model, corpus_or_tfidf, df: pd.DataFrame, is_gensim: bool = True) -> pd.DataFrame:
    """
    Assigns dominant topic index and topic probability to every article.
    """
    df_out = df.copy()
    dominant_topics = []
    topic_probs = []

    if is_gensim:
        for doc in corpus_or_tfidf:
            topic_dist = model.get_document_topics(doc)
            sorted_topics = sorted(topic_dist, key=lambda x: x[1], reverse=True)
            if sorted_topics:
                dominant_topics.append(f"Topic_{sorted_topics[0][0]+1}")
                topic_probs.append(sorted_topics[0][1])
            else:
                dominant_topics.append("Topic_1")
                topic_probs.append(0.0)
    else:
        W = model.transform(corpus_or_tfidf)
        for row in W:
            top_idx = row.argmax()
            dominant_topics.append(f"Topic_{top_idx+1}")
            topic_probs.append(row[top_idx] / (row.sum() + 1e-10))

    df_out['dominant_topic'] = dominant_topics
    df_out['topic_probability'] = topic_probs

    return df_out

def compute_topic_category_crosstab(df_with_topics: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-tabulates ground-truth news categories against discovered dominant topics.
    """
    crosstab = pd.crosstab(df_with_topics['category'], df_with_topics['dominant_topic'], normalize='index')
    return crosstab
