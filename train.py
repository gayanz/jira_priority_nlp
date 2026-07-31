"""
Automated Jira Ticket Prioritization — DistilBERT fine-tuning (ASF data, n=5000).

- Data: Apache Software Foundation Jira (prepare with prepare_asf_data.py).
- Split: 80% train / 10% validation / 10% test.
- Model: Fine-tuned DistilBERT, 3–5 epochs, low learning rate.
- Metrics: Macro F1, Precision–Recall curves (plus confusion matrix).

Run:
  python prepare_asf_data.py   # once, after placing issues.bson.gz in data/
  python train.py
"""
import numpy as np

from config import (
    ASF_SAMPLE_CSV,
    OUTPUT_DIR,
    RANDOM_STATE,
    TRAIN_RATIO,
    VAL_RATIO,
    SAMPLE_SIZE,
)
from asf_data import load_prepared_csv
from preprocess import prepare_text, prepare_labels, normalize_column_names, _generate_sample_data
from bert_model import get_label_encoder, train_bert, predict_bert
from evaluate import full_evaluation, macro_f1


def load_dataframe():
    """Load stratified ASF sample CSV, or create a demo CSV if BSON was not prepared."""
    try:
        df = load_prepared_csv(ASF_SAMPLE_CSV)
    except FileNotFoundError:
        print(
            "[WARN] ASF sample CSV not found. Creating a DEMO dataset (not real ASF).\n"
            "       For research-grade results: download issues.bson.gz from Zenodo 5665896, "
            "       place it in data/, then run: python prepare_asf_data.py"
        )
        df = _generate_sample_data(n=SAMPLE_SIZE)
        df.to_csv(ASF_SAMPLE_CSV, index=False)
        print(f"       Wrote demo {ASF_SAMPLE_CSV}")
    df = normalize_column_names(df)
    return df


def main():
    print("Loading data...")
    df = load_dataframe()
    print(f"Loaded n={len(df)} rows")
    texts = prepare_text(df)
    raw_labels = prepare_labels(df)

    valid = (texts.str.len() > 0) & (raw_labels.astype(str).str.len() > 0)
    texts = texts[valid].tolist()
    raw_labels = raw_labels[valid]

    label_encoder = get_label_encoder(raw_labels.tolist())
    y = label_encoder.transform(raw_labels)
    label_names = list(label_encoder.classes_)
    class_indices = list(range(len(label_names)))
    n = len(texts)

    # 80 / 10 / 10 split (shuffle then slice)
    idx = np.arange(n)
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(idx)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    X_train = [texts[i] for i in train_idx]
    X_val = [texts[i] for i in val_idx]
    X_test = [texts[i] for i in test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    print(f"Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")
    print(f"Priority classes ({len(label_names)}): {label_names}")

    print("\n--- Fine-tuning DistilBERT ---")
    trainer, tokenizer, model = train_bert(X_train, y_train, X_val, y_val, label_encoder)
    y_pred_str, y_proba = predict_bert(trainer, tokenizer, model, X_test, label_encoder)
    y_pred = label_encoder.transform(y_pred_str)

    macro = macro_f1(y_test, y_pred, class_indices)
    print(f"\n** Macro-averaged F1 (test): {macro:.4f} **\n")

    eval_out = full_evaluation(
        y_test,
        y_pred,
        y_proba,
        class_indices,
        label_names,
        "DistilBERT",
    )
    print("Per-class report:")
    for k, v in eval_out["report"].items():
        if isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    model.save_pretrained(OUTPUT_DIR / "distilbert_final")
    tokenizer.save_pretrained(OUTPUT_DIR / "distilbert_final")
    print("\nModel saved to:", OUTPUT_DIR / "distilbert_final")
    print("Figures (confusion matrix, PR curves):", OUTPUT_DIR)


if __name__ == "__main__":
    main()
