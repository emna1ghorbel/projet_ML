import pandas as pd
import numpy as np
import sys
import os
from utils import clean_data, split_and_save

# ================================
# 1️⃣ Chargement des données
# ================================

# DATA_PATH   = "C:\\Users\\dell\\projet_ML\\data\\raw\\retail_customers_COMPLETE_CATEGORICAL.csv"
# OUTPUT_PATH = "C:\\Users\\dell\\projet_ML\\data\\processed\\retail_customers_cleaned.csv"
CLEAN_PATH = "data/processed/retail_customers_cleaned.csv"
RAW_PATH = "data/raw/retail_customers_COMPLETE_CATEGORICAL.csv"

if not os.path.exists(RAW_PATH):
    print(f"[ERREUR] Fichier introuvable : {RAW_PATH}")
    sys.exit(1)

if os.path.exists(CLEAN_PATH):
    print("[INFO] Loading already cleaned data...")
    df = pd.read_csv(CLEAN_PATH)
else:
    print("[INFO] Cleaning raw data...")
    df = pd.read_csv(RAW_PATH)
    df = clean_data(df)
    os.makedirs("data/processed/", exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

# 2. Split Data (using the logic in utils)
split_and_save(df)