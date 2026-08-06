"""
Centralized Configuration Settings for Week 6 Advanced NLP Project.
"""

import os
import torch

# Base Paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

# Set Hugging Face cache directory to project workspace on E drive
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
os.environ["HF_HOME"] = HF_CACHE_DIR
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# Ensure required directories exist
for folder in [DATA_DIR, MODELS_DIR, IMAGES_DIR, NOTEBOOKS_DIR, HF_CACHE_DIR]:
    os.makedirs(folder, exist_ok=True)

# Data Settings
DATA_FILE = os.path.join(DATA_DIR, "bbc_articles.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Hardware & Device Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0 if os.name == 'nt' else 2

# Model Hyperparameters
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256  # Set dynamically based on 95th percentile token length
BATCH_SIZE = 16 if torch.cuda.is_available() else 8
LEARNING_RATE = 2e-5
MAX_EPOCHS = 5
PATIENCE = 2  # Early stopping patience on validation loss
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# NER Settings
NER_ENTITIES = [
    "PERSON", "ORG", "GPE", "LOC", "EVENT", 
    "DATE", "PRODUCT", "MONEY", "NORP", "FAC"
]

# Topic Modeling Settings
MIN_TOPICS = 2
MAX_TOPICS = 15
TOPIC_KEYWORDS_COUNT = 15
