"""
Publication-Quality Visualization Module (300 DPI, modern palettes, clear labels).
"""

import os
import matplotlib
matplotlib.use('Agg')  # Headless non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from wordcloud import WordCloud

# Global Plotting Aesthetic Setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

PALETTE = ['#2b5c8f', '#d95f02', '#7570b3', '#e7298a', '#66a61e']

def save_fig(fig, filename: str):
    """
    Saves figure at 300 DPI to images directory.
    """
    from src.config import IMAGES_DIR
    filepath = os.path.join(IMAGES_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[VISUALIZATION] Figure saved to: {filepath}", flush=True)

def plot_category_distribution(category_counts: pd.Series):
    """
    Bar & Pie chart of news categories.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=axes[0], palette=PALETTE)
    axes[0].set_title("BBC News Class Distribution", fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xlabel("News Category", fontsize=12)
    axes[0].set_ylabel("Number of Articles", fontsize=12)
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=11, xytext=(0, 3), textcoords='offset points')

    axes[1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', colors=PALETTE, startangle=140, explode=[0.03]*len(category_counts))
    axes[1].set_title("Category Share (%)", fontsize=14, fontweight='bold', pad=12)
    
    plt.tight_layout()
    save_fig(fig, "category_distribution.png")

def plot_article_length_boxplot(df: pd.DataFrame):
    """
    Boxplot of article word counts by category.
    """
    df_copy = df.copy()
    df_copy['word_count'] = df_copy['text'].apply(lambda x: len(x.split()))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='category', y='word_count', data=df_copy, palette=PALETTE, ax=ax, width=0.5, fliersize=3)
    ax.set_title("Article Word Count Distribution by Category", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("News Category", fontsize=12)
    ax.set_ylabel("Word Count per Article", fontsize=12)
    
    plt.tight_layout()
    save_fig(fig, "article_length_distribution.png")

def plot_top_ngrams(df_unigrams, df_bigrams, df_trigrams):
    """
    Combined barplot for top unigrams, bigrams, and trigrams.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.barplot(data=df_unigrams.head(10), x='count', y='ngram', ax=axes[0], color='#2b5c8f')
    axes[0].set_title("Top 10 Unigrams", fontsize=13, fontweight='bold')
    
    sns.barplot(data=df_bigrams.head(10), x='count', y='ngram', ax=axes[1], color='#d95f02')
    axes[1].set_title("Top 10 Bigrams", fontsize=13, fontweight='bold')
    
    sns.barplot(data=df_trigrams.head(10), x='count', y='ngram', ax=axes[2], color='#7570b3')
    axes[2].set_title("Top 10 Trigrams", fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    save_fig(fig, "top_ngrams.png")

def plot_category_wordclouds(df: pd.DataFrame):
    """
    Generates wordcloud grid for each category.
    """
    categories = df['category'].unique()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, cat in enumerate(categories):
        cat_text = " ".join(df[df['category'] == cat]['text'])
        wc = WordCloud(width=600, height=400, background_color='white', max_words=100, colormap='Dark2').generate(cat_text)
        axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].set_title(f"Category: {cat.upper()}", fontsize=14, fontweight='bold')
        axes[idx].axis('off')
        
    axes[-1].axis('off')
    plt.tight_layout()
    save_fig(fig, "category_wordclouds.png")

def plot_entity_frequencies(df_ner_freq: pd.DataFrame):
    """
    Barplot of top named entities across corpus.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_ner_freq.head(15), x='count', y='entity', hue='label', dodge=False, palette='Set2', ax=ax)
    ax.set_title("Top 15 Extracted Named Entities (spaCy)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Frequency across Corpus", fontsize=12)
    ax.set_ylabel("Named Entity", fontsize=12)
    
    plt.tight_layout()
    save_fig(fig, "entity_frequencies.png")

def plot_topic_coherence(df_coherence: pd.DataFrame):
    """
    Plot Coherence Score vs Number of Topics.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = '#2b5c8f'
    ax1.set_xlabel('Number of Topics (K)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Coherence Score (c_v)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(df_coherence['num_topics'], df_coherence['coherence_score'], marker='o', color=color, linewidth=2.5, label='Coherence Score')
    ax1.tick_params(axis='y', labelcolor=color)
    
    best_row = df_coherence.loc[df_coherence['coherence_score'].idxmax()]
    ax1.axvline(best_row['num_topics'], color='red', linestyle='--', alpha=0.7, label=f"Optimal K={int(best_row['num_topics'])}")
    ax1.set_title("LDA Topic Coherence Optimization (c_v Score)", fontsize=14, fontweight='bold', pad=12)
    ax1.legend(loc='lower right')
    
    plt.tight_layout()
    save_fig(fig, "topic_coherence_plot.png")

def plot_topic_category_heatmap(crosstab_df: pd.DataFrame):
    """
    Heatmap comparing ground truth categories against dominant topic clusters.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(crosstab_df, annot=True, fmt='.2f', cmap='Blues', cbar=True, ax=ax)
    ax.set_title("Dominant Topic vs Actual Category Alignment Matrix", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Discovered Topic", fontsize=12)
    ax.set_ylabel("Ground-Truth Category", fontsize=12)
    
    plt.tight_layout()
    save_fig(fig, "topic_category_heatmap.png")

def plot_confusion_matrix_custom(cm: np.ndarray, class_names: list, title: str, filename: str):
    """
    Plots styled confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    
    plt.tight_layout()
    save_fig(fig, filename)

def plot_training_curves(history: dict):
    """
    Plots training loss and validation metrics across epochs.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='#2b5c8f', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 's--', label='Val Loss', color='#d95f02', linewidth=2)
    axes[0].set_title("Loss Curves (DistilBERT Fine-Tuning)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend()
    
    axes[1].plot(epochs, history['val_accuracy'], 'o-', label='Val Accuracy', color='#7570b3', linewidth=2)
    axes[1].plot(epochs, history['val_f1'], 's--', label='Val Weighted F1', color='#66a61e', linewidth=2)
    axes[1].set_title("Validation Performance Curves", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Score", fontsize=11)
    axes[1].legend()
    
    plt.tight_layout()
    save_fig(fig, "training_curves.png")

def plot_model_comparison_benchmark(df_benchmarks: pd.DataFrame):
    """
    Bar chart comparing Accuracy, F1-Score, and Inference Time across models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.barplot(data=df_benchmarks, x='Accuracy', y='Model', palette='viridis', ax=axes[0])
    axes[0].set_title("Model Accuracy Comparison", fontsize=13, fontweight='bold')
    axes[0].set_xlim(0.8, 1.0)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_width():.4f}", (p.get_width(), p.get_y() + p.get_height() / 2.),
                         ha='left', va='center', fontsize=10, xytext=(5, 0), textcoords='offset points')

    sns.barplot(data=df_benchmarks, x='Inference_Time_ms', y='Model', palette='magma', ax=axes[1])
    axes[1].set_title("Inference Latency (ms per article)", fontsize=13, fontweight='bold')
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_width():.2f} ms", (p.get_width(), p.get_y() + p.get_height() / 2.),
                         ha='left', va='center', fontsize=10, xytext=(5, 0), textcoords='offset points')

    plt.tight_layout()
    save_fig(fig, "model_comparison_benchmark.png")
