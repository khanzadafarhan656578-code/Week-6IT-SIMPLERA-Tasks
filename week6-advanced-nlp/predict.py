#!/usr/bin/env python
"""
CLI Predictor for Week 6 Advanced NLP Portfolio Project.
"""

import sys
import json
import argparse
from src.inference import ProductionPredictor

def main():
    parser = argparse.ArgumentParser(description="BBC News Category Classifier CLI Predictor")
    parser.add_argument("text_positional", nargs="?", type=str, help="Raw news article text (positional)")
    parser.add_argument("--text", type=str, help="Raw news article text or headline")
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "logistic_regression", "svm", "naive_bayes", "distilbert"], help="Model architecture")
    args = parser.parse_args()

    input_text = args.text or args.text_positional

    if not input_text:
        print("\n--- BBC News Classification CLI Predictor ---")
        input_text = input("Enter news article headline or text: ")

    if not input_text or not input_text.strip():
        print("Error: Empty input text provided.")
        sys.exit(1)

    predictor = ProductionPredictor(model_type=args.model)
    result = predictor.predict(input_text)
    print("\n--- Prediction Output (Structured JSON) ---")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
