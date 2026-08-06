"""
Named Entity Recognition Module using spaCy en_core_web_sm pipeline (High-Performance Batched).
"""

from collections import Counter, defaultdict
import pandas as pd
import spacy
import spacy.cli

def load_spacy_model(model_name: str = "en_core_web_sm"):
    """
    Loads spaCy pipeline gracefully with fallback handling.
    """
    try:
        return spacy.load(model_name)
    except Exception:
        spacy.cli.download(model_name)
        return spacy.load(model_name)

def extract_entities_from_text(text: str, nlp, target_labels: list = None) -> list:
    """
    Extracts entities from a single article text.
    Returns list of dicts: [{'text': ent.text, 'label': ent.label_}]
    """
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if target_labels is None or ent.label_ in target_labels:
            entities.append({'text': ent.text.strip(), 'label': ent.label_})
    return entities

def process_corpus_ner(df: pd.DataFrame, nlp, target_labels: list = None) -> tuple:
    """
    Extracts entities across the entire dataset using spacy.pipe for high throughput.
    """
    if target_labels is None:
        from src.config import NER_ENTITIES
        target_labels = NER_ENTITIES

    records = []
    category_entities = defaultdict(lambda: defaultdict(list))

    print(f"[NER] Extracting entities for {len(df)} articles across types: {target_labels}...", flush=True)
    
    text_slices = df['text'].astype(str).str.slice(0, 1000).tolist()
    categories = df['category'].tolist()

    # Disable parser to speed up NER
    disable_pipes = [p for p in nlp.pipe_names if p not in ["ner", "tok2vec"]]
    docs = list(nlp.pipe(text_slices, batch_size=512, disable=disable_pipes))

    for idx, (doc, cat) in enumerate(zip(docs, categories)):
        for ent in doc.ents:
            if ent.label_ in target_labels:
                ent_clean = ent.text.strip()
                if len(ent_clean) > 1:
                    records.append({
                        'article_id': idx,
                        'category': cat,
                        'entity': ent_clean,
                        'label': ent.label_
                    })
                    category_entities[cat][ent.label_].append(ent_clean)

    df_ner = pd.DataFrame(records)
    
    category_top_entities = {}
    for cat in category_entities:
        category_top_entities[cat] = {}
        for label in category_entities[cat]:
            counts = Counter(category_entities[cat][label]).most_common(10)
            category_top_entities[cat][label] = counts

    return df_ner, category_top_entities

def get_entity_frequency_table(df_ner: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """
    Generates an entity frequency table grouped by entity text and type.
    """
    freq = df_ner.groupby(['entity', 'label']).size().reset_index(name='count')
    freq = freq.sort_values(by='count', ascending=False).head(top_k)
    return freq
