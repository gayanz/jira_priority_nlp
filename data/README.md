# ASF Jira data (primary source)

## 1. Apache Software Foundation Jira dataset (Zenodo)

- **Record:** [Apache Jira Issue Tracking Dataset](https://zenodo.org/records/5665896)  
- Download **`issues.bson.gz`** into this folder (`data/issues.bson.gz`).  
- Optional: **`priorities.bson.gz`** if issue documents store priority as an ObjectId (helps resolve names).

## 2. Build the 5000-record sample

```bash
python prepare_asf_data.py
```

This streams up to 250k valid issues (summary + priority), then writes a **stratified** sample of **5000** rows to **`jira_asf_5000.csv`** with columns:

- `summary`
- `description`
- `priority`

## 3. Train

```bash
python train.py
```

If `jira_asf_5000.csv` is missing, `train.py` will generate a **small synthetic demo** CSV (not ASF) so the pipeline runs; replace it with the real sample for your study.
