"""Fine-tuned DistilBERT for Jira priority classification."""
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.preprocessing import LabelEncoder

from config import (
    BERT_MODEL_NAME,
    BERT_MAX_LENGTH,
    BERT_BATCH_SIZE,
    BERT_EPOCHS,
    BERT_LEARNING_RATE,
    BERT_WEIGHT_DECAY,
    BERT_EARLY_STOPPING_PATIENCE,
    MODEL_DIR,
    RANDOM_STATE,
)


class JiraDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_label_encoder(labels: list) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(labels)
    return le


def build_bert_trainer(
    train_dataset: JiraDataset,
    eval_dataset: JiraDataset,
    num_labels: int,
    label_encoder: LabelEncoder,
):
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=num_labels,
    )
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "distilbert_checkpoints"),
        num_train_epochs=BERT_EPOCHS,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE * 2,
        learning_rate=BERT_LEARNING_RATE,
        weight_decay=BERT_WEIGHT_DECAY,
        warmup_ratio=0.1,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        seed=RANDOM_STATE,
    )
    def compute_metrics(eval_pred):
        from sklearn.metrics import f1_score
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"eval_f1": float(f1_score(labels, preds, average="macro", zero_division=0))}
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=BERT_EARLY_STOPPING_PATIENCE)],
    )
    return trainer, tokenizer, model


def train_bert(
    X_train: list[str],
    y_train: np.ndarray,
    X_val: list[str],
    y_val: np.ndarray,
    label_encoder: LabelEncoder,
) -> tuple[Trainer, object, object]:
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    train_ds = JiraDataset(X_train, y_train, tokenizer, BERT_MAX_LENGTH)
    eval_ds = JiraDataset(X_val, y_val, tokenizer, BERT_MAX_LENGTH)
    num_labels = len(label_encoder.classes_)
    trainer, tok, model = build_bert_trainer(train_ds, eval_ds, num_labels, label_encoder)
    trainer.train()
    return trainer, tok, model


def predict_bert(
    trainer,
    tokenizer,
    model,
    X: list[str],
    label_encoder: LabelEncoder,
    infer_batch_size: int = 32,
):
    """Batched inference to avoid OOM on large test sets."""
    device = next(model.parameters()).device
    model.eval()
    all_logits = []
    for start in range(0, len(X), infer_batch_size):
        batch = X[start : start + infer_batch_size]
        enc = tokenizer(
            batch,
            max_length=BERT_MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        all_logits.append(out.logits.cpu().numpy())
    logits = np.vstack(all_logits)
    preds = np.argmax(logits, axis=1)
    return label_encoder.inverse_transform(preds), logits
