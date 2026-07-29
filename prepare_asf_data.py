#!/usr/bin/env python3
"""
Build jira_asf_5000.csv from ASF Jira BSON (Zenodo 5665896).

Usage:
  1. Download issues.bson.gz (and optionally priorities.bson.gz) into data/
  2. python prepare_asf_data.py
"""
from asf_data import build_asf_sample_csv

# Standardize priority labels to High / Medium / Low (drop unmapped/rare classes)
PRIORITY_LABEL_MAP = {
    "P0": "High",
    "P1": "High",
    "Blocker": "Blocker",
    "Urgent": "High",
    "Critical": "High",
    "Highest": "High",
    "High": "High",
    "Major": "Medium",
    "P2": "Medium",
    "P3": "Medium",
    "Normal": "Medium",
    "Medium": "Medium",
    "Minor": "Low",
    "P4": "Low",
    "Trivial": "Low",
    "Low": "Low",
    "Lowest": "Low",
}

if __name__ == "__main__":
    build_asf_sample_csv(priority_label_map=PRIORITY_LABEL_MAP)
    print("Done. Next: python train.py")
