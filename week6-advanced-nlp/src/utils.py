"""
Utility functions for reproducibility, timing, logging, and system inspection.
"""

import os
import random
import time
import functools
import pandas as pd
import numpy as np
import torch

def set_seeds(seed: int = 42) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch for total reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[REPRODUCIBILITY] All random seeds globally set to: {seed}")

def time_it(func):
    """
    Decorator to measure execution time of functions.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"[TIMING] Function '{func.__name__}' executed in {elapsed_time:.4f} seconds.")
        return result
    return wrapper

def log_experiment(experiment_data: dict, csv_path: str = None) -> pd.DataFrame:
    """
    Logs experiment metrics to a CSV file and optionally logs to MLflow if installed.
    """
    if csv_path is None:
        from src.config import MODELS_DIR
        csv_path = os.path.join(MODELS_DIR, "experiments.csv")
    
    df_new = pd.DataFrame([experiment_data])
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_updated = df_new
        
    df_updated.to_csv(csv_path, index=False)
    print(f"[LOGGING] Experiment result logged to: {csv_path}")
    
    # Optional MLflow tracking
    try:
        import mlflow
        with mlflow.start_run(run_name=experiment_data.get("Model", "NLP_Model")):
            for k, v in experiment_data.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
                else:
                    mlflow.log_param(k, str(v))
        print("[MLflow] Successfully logged experiment to MLflow tracker.")
    except Exception:
        # MLflow optional fallback
        pass

    return df_updated
