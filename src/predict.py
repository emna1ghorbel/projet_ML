"""
predict.py — Batch prediction on new customers
Usage:
    python src/predict.py --input data/raw/new_customers.csv
    python src/predict.py --input data/raw/new_customers.csv --output reports/predictions.csv
"""

import os
import sys
import argparse
import joblib
import pandas as pd

# ================================
# Config
# ================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../models')

FEATURES = [
    'Recency',
    'Frequency',
    'MonetaryTotal',
    'TotalQuantity',
    'ReturnRatio',
    'CancelledTransactions',
    'AvgDaysBetweenPurchases',
    'Age',
    'SatisfactionScore',
    'SupportTicketsCount',
]

# ================================
# Load pipeline
# ================================
def load_pipeline():
    try:
        model  = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        pca    = joblib.load(os.path.join(MODEL_DIR, 'pca.pkl'))
        print("[OK] Pipeline chargé (model + scaler + pca)")
        return model, scaler, pca
    except FileNotFoundError as e:
        print(f"[ERREUR] Fichier modèle introuvable : {e}")
        print("         Lancez d'abord : python src/train_model.py")
        sys.exit(1)

# ================================
# Predict
# ================================
def predict(input_path, output_path=None):
    # 1. Load data
    if not os.path.exists(input_path):
        print(f"[ERREUR] Fichier introuvable : {input_path}")
        sys.exit(1)

    df_raw = pd.read_csv(input_path)
    print(f"[OK] Données chargées : {df_raw.shape[0]} clients, {df_raw.shape[1]} colonnes")

    # 2. Check required features
    missing = [f for f in FEATURES if f not in df_raw.columns]
    if missing:
        print(f"[ERREUR] Colonnes manquantes dans le fichier : {missing}")
        print(f"         Colonnes requises : {FEATURES}")
        sys.exit(1)

    # 3. Select & clean
    X = df_raw[FEATURES].copy()
    X = X.fillna(X.median())

    # 4. Load pipeline & predict
    model, scaler, pca = load_pipeline()

    X_scaled = scaler.transform(X)
    X_pca    = pca.transform(X_scaled)

    predictions = model.predict(X_pca)
    probas      = model.predict_proba(X_pca)[:, 1] * 100

    # 5. Build results dataframe
    df_result = df_raw.copy()
    df_result['Churn_Prediction'] = predictions
    df_result['Churn_Probability_%'] = probas.round(1)
    df_result['Churn_Label'] = df_result['Churn_Prediction'].map({1: 'Churn', 0: 'No Churn'})

    # 6. Print summary
    n_churn    = (predictions == 1).sum()
    n_no_churn = (predictions == 0).sum()
    print("\n" + "="*50)
    print("  RÉSULTATS DE PRÉDICTION")
    print("="*50)
    print(f"  Total clients analysés : {len(predictions)}")
    print(f"  🔴 Churn prédit        : {n_churn} ({n_churn/len(predictions)*100:.1f}%)")
    print(f"  🟢 Fidèles prédits     : {n_no_churn} ({n_no_churn/len(predictions)*100:.1f}%)")
    print("="*50)

    # Top 5 at-risk
    top_risk = df_result.nlargest(5, 'Churn_Probability_%')[
        FEATURES[:3] + ['Churn_Probability_%', 'Churn_Label']
    ]
    print("\n  Top 5 clients à risque :")
    print(top_risk.to_string(index=False))

    # 7. Save output
    if output_path is None:
        os.makedirs('reports', exist_ok=True)
        output_path = 'reports/predictions.csv'

    df_result.to_csv(output_path, index=False)
    print(f"\n[OK] Prédictions sauvegardées → {output_path}")

    return df_result


# ================================
# Entry point
# ================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch churn prediction')
    parser.add_argument('--input',  required=True,  help='Path to input CSV file')
    parser.add_argument('--output', required=False, help='Path to output CSV file (default: reports/predictions.csv)')
    args = parser.parse_args()

    predict(args.input, args.output)