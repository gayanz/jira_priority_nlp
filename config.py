"""Configuration: ASF Jira sample + DistilBERT fine-tuning."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Apache Software Foundation Jira dump (Zenodo: https://zenodo.org/records/5665896)
# Place `issues.bson.gz` here after download, or use the prepared CSV below.
ASF_ISSUES_BSON_GZ = DATA_DIR / "issues.bson.gz"
ASF_PRIORITIES_BSON_GZ = DATA_DIR / "priorities.bson.gz"  # optional, for ObjectId -> name

# Prepared sample: 5000 records (Summary, Description, Priority)
ASF_SAMPLE_CSV = DATA_DIR / "jira_asf_5000.csv"

# How many issues to scan from BSON before stratified sampling (cap memory)
ASF_MAX_SCAN_ISSUES = 250_000
# Target sample size for this study
SAMPLE_SIZE = 500

RANDOM_STATE = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# DistilBERT — low LR to reduce catastrophic forgetting; 3–5 epochs
BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_MAX_LENGTH = 256
BERT_BATCH_SIZE = 16
BERT_EPOCHS = 1  # use early stopping; effective range 3–5
BERT_LEARNING_RATE = 2e-5
BERT_WEIGHT_DECAY = 0.01
BERT_EARLY_STOPPING_PATIENCE = 2

# Optional TF-IDF baseline (see baseline_model.py; not used by train.py)
DATA_CSV = ASF_SAMPLE_CSV
TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM_RANGE = (1, 2)
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 30
USE_SMOTE = True
SMOTE_K_NEIGHBORS = 5
