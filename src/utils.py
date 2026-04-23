import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
import numpy as np


def clean_data(df):
    """Effectue un nettoyage complet du dataset de clients retail, incluant :"""

    # ================================
    # 2️⃣ Analyse initiale des données
    # ================================

    print(df.info())
    print(df.describe(include='all'))

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("Valeurs manquantes :")
    print(missing if not missing.empty else "Aucune valeur manquante détectée")

    nb_duplicates = df.duplicated().sum()
    print(f"Nombre de doublons : {nb_duplicates}")

    # ================================
    # 3️⃣ Nettoyage des données
    # ================================

    df = df.drop_duplicates()
    print(f"[OK] Doublons supprimés. Lignes restantes : {len(df)}")

    if 'MonetaryTotal' in df.columns:
        def parse_monetary(val):
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, pd.Timestamp):
                return np.nan
            try:
                return float(str(val).replace(",", "."))
            except ValueError:
                return np.nan

        nb_before = df['MonetaryTotal'].isnull().sum()
        df['MonetaryTotal'] = df['MonetaryTotal'].apply(parse_monetary)
        nb_after  = df['MonetaryTotal'].isnull().sum()
        df['MonetaryTotal'] = df['MonetaryTotal'].fillna(df['MonetaryTotal'].median())
        print(f"[OK] 'MonetaryTotal' converti en float "
            f"({nb_after - nb_before} valeur(s) aberrante(s) imputées par la médiane)")
    else:
        print("[AVERTISSEMENT] Colonne 'MonetaryTotal' introuvable, étape ignorée.")

    if 'Age' in df.columns:
        nb_missing_age = df['Age'].isnull().sum()
        df['Age'] = df['Age'].fillna(df['Age'].median())
        print(f"[OK] 'Age' : {nb_missing_age} valeurs manquantes imputées "
            f"par la médiane ({df['Age'].median():.1f})")
    else:
        print("[AVERTISSEMENT] Colonne 'Age' introuvable, étape ignorée.")

    if 'AvgDaysBetweenPurchases' in df.columns and df['AvgDaysBetweenPurchases'].isnull().any():
        nb_missing = df['AvgDaysBetweenPurchases'].isnull().sum()
        df['AvgDaysBetweenPurchases'] = df['AvgDaysBetweenPurchases'].fillna(df['AvgDaysBetweenPurchases'].median())
        print(f"[OK] 'AvgDaysBetweenPurchases' : {nb_missing} valeurs manquantes imputées par la médiane")

    # ================================
    # 4️⃣ Détection et correction des valeurs aberrantes
    # ================================

    COL_TICKETS = 'SupportTicketsCount'

    if COL_TICKETS in df.columns:
        nb_sentinel = df[COL_TICKETS].isin([-1, 999]).sum()
        df.loc[df[COL_TICKETS].isin([-1, 999]), COL_TICKETS] = np.nan
        df[COL_TICKETS] = df[COL_TICKETS].fillna(df[COL_TICKETS].median())
        print(f"[OK] '{COL_TICKETS}' : {nb_sentinel} valeurs sentinelles (-1, 999) imputées par la médiane")

        Q1 = df[COL_TICKETS].quantile(0.25)
        Q3 = df[COL_TICKETS].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[COL_TICKETS] < lower) | (df[COL_TICKETS] > upper)]
        print(f"Nombre de valeurs aberrantes détectées dans '{COL_TICKETS}' : {len(outliers)}")
        df[COL_TICKETS] = df[COL_TICKETS].clip(lower, upper)
        print(f"[OK] Capping appliqué sur '{COL_TICKETS}' → bornes [{lower:.2f}, {upper:.2f}]")
    else:
        print(f"[AVERTISSEMENT] Colonne '{COL_TICKETS}' introuvable, étape ignorée.")

    COL_SAT = 'SatisfactionScore'

    if COL_SAT in df.columns:
        nb_sentinel = (~df[COL_SAT].isin([1, 2, 3, 4, 5])).sum()
        df.loc[~df[COL_SAT].isin([1, 2, 3, 4, 5]), COL_SAT] = np.nan
        df[COL_SAT] = df[COL_SAT].fillna(df[COL_SAT].median())
        print(f"[OK] '{COL_SAT}' : {nb_sentinel} valeurs aberrantes (-1, 0, 99) imputées par la médiane")

        Q1 = df[COL_SAT].quantile(0.25)
        Q3 = df[COL_SAT].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[COL_SAT] < lower) | (df[COL_SAT] > upper)]
        print(f"Nombre de valeurs aberrantes détectées dans '{COL_SAT}' : {len(outliers)}")
        df[COL_SAT] = df[COL_SAT].clip(lower, upper)
        print(f"[OK] Capping appliqué sur '{COL_SAT}' → bornes [{lower:.2f}, {upper:.2f}]")
    else:
        print(f"[AVERTISSEMENT] Colonne '{COL_SAT}' introuvable, étape ignorée.")

    # ================================
    # 5️⃣ Formats inconsistants
    # ================================

    if 'RegistrationDate' in df.columns:
        nb_before = df['RegistrationDate'].isnull().sum()
        df["RegistrationDate"] = pd.to_datetime(
            df["RegistrationDate"], dayfirst=True, errors='coerce'
        )
        nb_after  = df['RegistrationDate'].isnull().sum()
        nb_failed = nb_after - nb_before

        df["RegYear"]    = df["RegistrationDate"].dt.year
        df["RegMonth"]   = df["RegistrationDate"].dt.month
        df["RegDay"]     = df["RegistrationDate"].dt.day
        df["RegWeekday"] = df["RegistrationDate"].dt.weekday

        ref_date = df["RegistrationDate"].max()
        df["DaysSinceRegistration"] = (ref_date - df["RegistrationDate"]).dt.days

        df["RegistrationDate"] = df["RegistrationDate"].dt.strftime("%Y-%m-%d")

        print(f"[OK] 'RegistrationDate' convertie au format YYYY-MM-DD "
            f"({nb_failed} date(s) non reconnue(s) → NaT)")
        print(f"     Features extraites : RegYear, RegMonth, RegDay, RegWeekday, DaysSinceRegistration")
    else:
        print("[AVERTISSEMENT] Colonne 'RegistrationDate' introuvable, étape ignorée.")

    # ================================
    # 6️⃣ Suppression des colonnes inutiles
    # ================================

    if 'NewsletterSubscribed' in df.columns:
        df = df.drop(columns=["NewsletterSubscribed"])
        print("[OK] Colonne 'NewsletterSubscribed' supprimée (variance nulle).")

    if 'CustomerID' in df.columns:
        df = df.drop(columns=["CustomerID"])
        print("[OK] Colonne 'CustomerID' supprimée (identifiant non prédictif).")

    # ================================
    # 7️⃣ Features depuis LastLoginIP
    # ================================

    if 'LastLoginIP' in df.columns:
        ip_split = df["LastLoginIP"].str.split(".", expand=True)
        df[["IP1", "IP2", "IP3", "IP4"]] = ip_split.iloc[:, :4].apply(
            pd.to_numeric, errors='coerce'
        ).fillna(0).astype(int)

        def ip_type(ip):
            if not isinstance(ip, str):
                return "Unknown"
            try:
                octets = list(map(int, ip.strip().split(".")))
                if octets[0] == 10:
                    return "Private"
                if octets[0] == 172 and 16 <= octets[1] <= 31:
                    return "Private"
                if octets[0] == 192 and octets[1] == 168:
                    return "Private"
                return "Public"
            except Exception:
                return "Unknown"

        df["IP_Type"] = df["LastLoginIP"].apply(ip_type)
        print(f"[OK] Features IP créées. Distribution IP_Type :\n{df['IP_Type'].value_counts()}")
        df = df.drop(columns=["LastLoginIP"])
    else:
        print("[AVERTISSEMENT] Colonne 'LastLoginIP' introuvable, étape ignorée.")

    # ================================
    # 8️⃣ Analyse distribution catégorielles
    # ================================

    for col in ["AccountStatus", "Churn"]:
        if col in df.columns:
            print(f"\nDistribution de '{col}' :")
            print(df[col].value_counts())

    # ================================
    # 9️⃣ Encodage et préparation ML  ← THIS IS WHAT WAS BROKEN BEFORE
    # ================================

    # Encode Churn: Yes/No → 1/0  (keeps it in the dataframe as a numeric column)
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        # If it was already 0/1 integers, map() returns NaN for them — fix that:
        df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce').fillna(0).astype(int)
        print(f"[OK] 'Churn' encodé en 0/1. Distribution :\n{df['Churn'].value_counts()}")

    # Encode IP_Type: Private=0, Public=1, Unknown=-1
    if 'IP_Type' in df.columns:
        df['IP_Type'] = df['IP_Type'].map({'Private': 0, 'Public': 1, 'Unknown': -1})
        print("[OK] 'IP_Type' encodé en numérique.")

    # Encode AccountStatus as numeric codes
    if 'AccountStatus' in df.columns:
        df['AccountStatus'] = pd.Categorical(df['AccountStatus']).codes
        print("[OK] 'AccountStatus' encodé en numérique.")

    # Drop the string date column (we already extracted RegYear, RegMonth, etc.)
    if 'RegistrationDate' in df.columns:
        df = df.drop(columns=['RegistrationDate'])

    # Keep only numeric columns
    df_numeric = df.select_dtypes(include=['number'])

    # ── Final safety net: kill any remaining NaNs ──
    # (e.g. DaysSinceRegistration NaT → NaN from failed date parsing)
    remaining_nans = df_numeric.isnull().sum().sum()
    if remaining_nans > 0:
        print(f"[INFO] {remaining_nans} NaN(s) résiduels détectés → imputés par la médiane")
        df_numeric = df_numeric.fillna(df_numeric.median())

    print(f"\n[OK] Nettoyage terminé. Shape finale : {df_numeric.shape}")
    print(f"     Colonnes : {df_numeric.columns.tolist()}")

    return df_numeric


def split_and_save(df, target_col='Churn'):
    """Splits data and saves it only if it doesn't already exist."""
    output_dir = "data/train_test/"
    files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]

    if all(os.path.exists(os.path.join(output_dir, f)) for f in files):
        print("[INFO] Split data already exists, skipping split.")
        return

    # Guard: make sure target column exists
    if target_col not in df.columns:
        raise ValueError(f"[ERREUR] Colonne cible '{target_col}' introuvable. "
                         f"Colonnes disponibles : {df.columns.tolist()}")

    print("[INFO] Splitting data...")
    os.makedirs(output_dir, exist_ok=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"[INFO] Distribution de la cible :\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train.to_csv(f"{output_dir}X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}y_test.csv", index=False)
    print("[OK] Data split and saved.")


def save_pipeline_objects(scaler, pca, model, folder='models/'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    joblib.dump(scaler, os.path.join(folder, 'scaler.pkl'))
    joblib.dump(pca, os.path.join(folder, 'pca.pkl'))
    joblib.dump(model, os.path.join(folder, 'model.pkl'))

    print(f"[OK] Pipeline objects saved successfully in '{folder}'")
    
    
def clean_for_prediction(df, features_list):
    """
    Clean ONLY the features needed for prediction.
    This should match the training preprocessing for those specific features.
    """
    # Select only the features we need
    X = df[features_list].copy()
    
    # Force numeric
    for col in features_list:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Special handling for SupportTicketsCount
    if 'SupportTicketsCount' in X.columns:
        X.loc[X['SupportTicketsCount'].isin([-1, 999]), 'SupportTicketsCount'] = np.nan
    
    # Special handling for SatisfactionScore (only 1-5 valid)
    if 'SatisfactionScore' in X.columns:
        X.loc[~X['SatisfactionScore'].isin([1,2,3,4,5]), 'SatisfactionScore'] = np.nan
    
    # Special handling for ReturnRatio (must be 0-1)
    if 'ReturnRatio' in X.columns:
        X.loc[(X['ReturnRatio'] < 0) | (X['ReturnRatio'] > 1), 'ReturnRatio'] = np.nan
    
    # Fill all remaining NaNs with median
    X = X.fillna(X.median(numeric_only=True))
    
    # Also clip outliers to match training capping (if needed)
    if 'SupportTicketsCount' in X.columns:
        # Match the capping from training (you should save these bounds from training)
        X['SupportTicketsCount'] = X['SupportTicketsCount'].clip(0, 15)  # Example bounds
    
    if 'SatisfactionScore' in X.columns:
        X['SatisfactionScore'] = X['SatisfactionScore'].clip(1, 5)
    
    if 'ReturnRatio' in X.columns:
        X['ReturnRatio'] = X['ReturnRatio'].clip(0, 1)
    
    return X