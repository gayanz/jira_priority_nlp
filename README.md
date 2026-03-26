# Automated Jira Ticket Prioritization using NLP

Research-oriented pipeline: predict **Priority** from **Summary** and **Description** using a **fine-tuned DistilBERT**, aligned with an 80/10/10 split and evaluation via **Macro F1** and **Precision–Recall curves**.

## Data source

**Primary:** [Apache Software Foundation Jira dataset](https://zenodo.org/records/5665896) (BSON export).

1. Place `issues.bson.gz` in `data/` (optional: `priorities.bson.gz`).
2. Build a stratified sample of **5000** issues:

```bash
pip install -r requirements.txt
python prepare_asf_data.py
```

3. Train:

```bash
python train.py
```

## Preprocessing

- HTML stripped, long hex-like tokens removed, whitespace normalized  
- **Summary** and **Description** concatenated into one input sequence for the tokenizer  

## Model & training

- **DistilBERT** (`distilbert-base-uncased`), sequence classification  
- **Epochs:** up to 5, with early stopping on validation Macro F1  
- **Learning rate:** `2e-5` (low, to limit catastrophic forgetting)  
- **Split:** 80% train / 10% validation / 10% test  

## Evaluation

- **Macro-averaged F1** (primary)  
- **Per-class Precision–Recall curves**  
- **Confusion matrix** (saved under `output/`)  
- Accuracy is not emphasized (imbalanced priorities).  

## Configuration

Edit `config.py` for `SAMPLE_SIZE`, `ASF_MAX_SCAN_ISSUES`, BERT hyperparameters, and paths.

## Files

| File | Role |
|------|------|
| `asf_data.py` | Read BSON, stratified 5000-row CSV |
| `prepare_asf_data.py` | CLI to build `data/jira_asf_5000.csv` |
| `preprocess.py` | Text cleaning and label normalization |
| `bert_model.py` | Dataset, Trainer, DistilBERT fine-tuning |
| `evaluate.py` | Macro F1, PR curves, confusion matrix |
| `train.py` | End-to-end training and test evaluation |
