"""
Transfer Learning Module fine-tuning DistilBERT with PyTorch, AdamW, Linear Scheduler, and Early Stopping.
"""

import time
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class BBCDataset(Dataset):
    """
    Custom PyTorch Dataset for Hugging Face Transformers.
    """
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten()
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def train_distilbert_model(
    X_train, y_train, X_val, y_val, num_classes, label_encoder, config
) -> tuple:
    """
    Fine-tunes DistilBERT for text classification with Early Stopping on validation loss.
    """
    device = torch.device(config.DEVICE)
    print(f"[TRANSFORMER] Loading model '{config.MODEL_NAME}' on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=num_classes
    )
    model.to(device)

    # Prepare DataLoaders
    train_dataset = BBCDataset(X_train, y_train, tokenizer, max_len=config.MAX_LENGTH)
    val_dataset = BBCDataset(X_val, y_val, tokenizer, max_len=config.MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    total_steps = len(train_loader) * config.MAX_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }

    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    print(f"[TRANSFORMER] Training started for up to {config.MAX_EPOCHS} epochs (Early Stopping Patience={config.PATIENCE})...")
    start_train_time = time.time()

    for epoch in range(1, config.MAX_EPOCHS + 1):
        # Training Phase
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        _, _, _, val_f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='weighted')

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f" Epoch {epoch}/{config.MAX_EPOCHS} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"[EARLY STOPPING] Triggered at epoch {epoch}. Restoring best model weights (Val Loss: {best_val_loss:.4f}).")
                break

    train_duration = time.time() - start_train_time
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # Final Inference Time & Probability Measurement
    model.eval()
    start_inf_time = time.time()
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.extend(probs)
            all_preds.extend(preds)

    inference_ms_per_sample = ((time.time() - start_inf_time) / len(X_val)) * 1000

    metrics = {
        'Model': 'DistilBERT Fine-Tuned',
        'Best_Params': f"lr={config.LEARNING_RATE}, batch_size={config.BATCH_SIZE}, max_len={config.MAX_LENGTH}",
        'Accuracy': accuracy_score(y_val, all_preds),
        'Precision': precision_recall_fscore_support(y_val, all_preds, average='weighted')[0],
        'Recall': precision_recall_fscore_support(y_val, all_preds, average='weighted')[1],
        'F1_Score': precision_recall_fscore_support(y_val, all_preds, average='weighted')[2],
        'Train_Time_s': train_duration,
        'Inference_Time_ms': inference_ms_per_sample,
        'Predictions': np.array(all_preds),
        'Probabilities': np.array(all_probs)
    }

    return model, tokenizer, history, metrics
