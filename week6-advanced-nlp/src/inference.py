"""
Production Inference Pipeline Module returning structured JSON API response.
"""

import os
import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ProductionPredictor:
    """
    Production Predictor loading saved artifacts and executing inference with full metadata.
    """
    def __init__(self, model_type: str = "auto", models_dir: str = None):
        if models_dir is None:
            from src.config import MODELS_DIR
            models_dir = MODELS_DIR

        self.models_dir = models_dir
        self.model_type = model_type.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Label Encoder
        le_path = os.path.join(models_dir, "label_encoder.joblib")
        if os.path.exists(le_path):
            self.label_encoder = joblib.load(le_path)
            self.class_names = list(self.label_encoder.classes_)
        else:
            self.label_encoder = None
            self.class_names = ["business", "entertainment", "politics", "sport", "tech"]

        # Vectorizer path
        vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
        if os.path.exists(vec_path):
            self.vectorizer = joblib.load(vec_path)
        else:
            self.vectorizer = None

        self.use_transformer = False
        self.sklearn_model = None

        # Model Loading Logic based on model_type
        if self.model_type in ["distilbert", "auto"]:
            distilbert_path = os.path.join(models_dir, "distilbert_bbc_model")
            tokenizer_path = os.path.join(models_dir, "distilbert_tokenizer")
            if os.path.exists(distilbert_path) and os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                self.transformer_model = AutoModelForSequenceClassification.from_pretrained(distilbert_path)
                self.transformer_model.to(self.device)
                self.transformer_model.eval()
                self.use_transformer = True
                print("[PREDICTOR] Loaded DistilBERT transformer model.")

        if not self.use_transformer:
            # Select classical ML model
            model_filename = f"{self.model_type}.joblib" if self.model_type != "auto" else "logistic_regression.joblib"
            model_path = os.path.join(models_dir, model_filename)

            if not os.path.exists(model_path):
                # Fallback to logistic_regression if specific requested model isn't saved yet
                model_path = os.path.join(models_dir, "logistic_regression.joblib")

            if os.path.exists(model_path) and self.vectorizer is not None:
                self.sklearn_model = joblib.load(model_path)
                print(f"[PREDICTOR] Loaded Scikit-Learn model from: {os.path.basename(model_path)}")
            else:
                self.sklearn_model = None

        # Load spaCy for NER
        try:
            import spacy
            self.spacy_nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.spacy_nlp = None

    def predict(self, raw_text: str, top_k: int = 3) -> dict:
        """
        Executes prediction on raw text input and returns structured production JSON response.
        """
        if not raw_text or not isinstance(raw_text, str):
            return {"error": "Invalid input text."}

        # 1. Classification Prediction
        if self.use_transformer:
            inputs = self.tokenizer(
                raw_text,
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

            predicted_idx = int(np.argmax(probs))
            confidence = float(probs[predicted_idx])
            predicted_category = self.class_names[predicted_idx]

            # Top K predictions
            top_k_indices = np.argsort(probs)[::-1][:top_k]
            top_k_predictions = [
                {"category": self.class_names[idx], "probability": round(float(probs[idx]), 4)}
                for idx in top_k_indices
            ]
        elif hasattr(self, 'sklearn_model') and self.sklearn_model is not None:
            vec = self.vectorizer.transform([raw_text])
            probs = self.sklearn_model.predict_proba(vec)[0] if hasattr(self.sklearn_model, "predict_proba") else None
            if probs is not None:
                predicted_idx = int(np.argmax(probs))
                confidence = float(probs[predicted_idx])
                predicted_category = self.class_names[predicted_idx]
                top_k_indices = np.argsort(probs)[::-1][:top_k]
                top_k_predictions = [
                    {"category": self.class_names[idx], "probability": round(float(probs[idx]), 4)}
                    for idx in top_k_indices
                ]
            else:
                pred = self.sklearn_model.predict(vec)[0]
                predicted_category = str(pred)
                confidence = 1.0
                top_k_predictions = [{"category": predicted_category, "probability": 1.0}]
        else:
            return {"error": "No trained model available for inference."}

        # 2. Extract Named Entities
        extracted_entities = {}
        if self.spacy_nlp is not None:
            doc = self.spacy_nlp(raw_text[:10000])
            for ent in doc.ents:
                label = ent.label_
                if label not in extracted_entities:
                    extracted_entities[label] = []
                if ent.text.strip() not in extracted_entities[label]:
                    extracted_entities[label].append(ent.text.strip())

        # Construct production payload
        snippet = raw_text[:150] + "..." if len(raw_text) > 150 else raw_text

        payload = {
            "input_snippet": snippet,
            "predicted_category": predicted_category,
            "confidence": round(confidence, 4),
            "top_k_predictions": top_k_predictions,
            "extracted_entities": extracted_entities
        }

        return payload
