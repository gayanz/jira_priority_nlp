#!/usr/bin/env python3
"""
Build jira_asf_5000.csv from ASF Jira BSON (Zenodo 5665896).

Usage:
  1. Download issues.bson.gz (and optionally priorities.bson.gz) into data/
  2. python prepare_asf_data.py
"""
from asf_data import build_asf_sample_csv

if __name__ == "__main__":
    build_asf_sample_csv()
    print("Done. Next: python train.py")
