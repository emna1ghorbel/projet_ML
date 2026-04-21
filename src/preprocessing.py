import pandas as pd
import numpy as np
import sys
import os

# ================================
# 1️⃣ Chargement des données
# ================================

DATA_PATH   = "C:\\Users\\dell\\projet_ML\\data\\raw\\retail_customers_COMPLETE_CATEGORICAL.csv"
OUTPUT_PATH = "C:\\Users\\dell\\projet_ML\\data\\processed\\retail_customers_cleaned.csv"

# Vérifier que le fichier source existe avant de charger
if not os.path.exists(DATA_PATH):
    print(f"[ERREUR] Fichier introuvable : {DATA_PATH}")
    sys.exit(1)

# Charger le dataset depuis le dossier raw (.csv → read_csv)
df = pd.read_csv(DATA_PATH)
print(f"[OK] Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Afficher les 5 premières lignes pour inspection rapide
print(df.head())

# ================================
# 2️⃣ Analyse initiale des données
# ================================

# Afficher les informations générales (types, valeurs non nulles)
print(df.info())

# Afficher les statistiques descriptives (numériques + catégorielles)
print(df.describe(include='all'))

# Vérifier les valeurs manquantes par colonne
# (affiche uniquement les colonnes qui ont au moins une valeur manquante)
missing = df.isnull().sum()
missing = missing[missing > 0]
print("Valeurs manquantes :")
print(missing if not missing.empty else "Aucune valeur manquante détectée")

# Vérifier le nombre de doublons
nb_duplicates = df.duplicated().sum()
print(f"Nombre de doublons : {nb_duplicates}")

# ================================
# 3️⃣ Nettoyage des données
# ================================

# Supprimer les lignes dupliquées
df = df.drop_duplicates()
print(f"[OK] Doublons supprimés. Lignes restantes : {len(df)}")

# ── Correction de MonetaryTotal (type object : strings + timestamps parasites) ──
# Dans ce dataset, MonetaryTotal contient des chaînes numériques ET des objets
# datetime parasites. On force la conversion en float, les valeurs non
# convertibles deviennent NaN puis sont imputées par la médiane.
if 'MonetaryTotal' in df.columns:
    def parse_monetary(val):
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, pd.Timestamp):
            return np.nan          # timestamp parasite → NaN
        try:
            return float(str(val).replace(",", "."))
        except ValueError:
            return np.nan

    nb_before = df['MonetaryTotal'].isnull().sum()
    df['MonetaryTotal'] = df['MonetaryTotal'].apply(parse_monetary)
    nb_after  = df['MonetaryTotal'].isnull().sum()
    df['MonetaryTotal'].fillna(df['MonetaryTotal'].median(), inplace=True)
    print(f"[OK] 'MonetaryTotal' converti en float "
          f"({nb_after - nb_before} valeur(s) aberrante(s) imputées par la médiane)")
else:
    print("[AVERTISSEMENT] Colonne 'MonetaryTotal' introuvable, étape ignorée.")

# ── Remplacer les valeurs manquantes de 'Age' par la médiane ──
# (la médiane est préférable à la moyenne car elle est robuste aux valeurs extrêmes)
if 'Age' in df.columns:
    nb_missing_age = df['Age'].isnull().sum()
    df['Age'] = df['Age'].fillna(df['Age'].median())
    print(f"[OK] 'Age' : {nb_missing_age} valeurs manquantes imputées "
          f"par la médiane ({df['Age'].median():.1f})")
else:
    print("[AVERTISSEMENT] Colonne 'Age' introuvable, étape ignorée.")

# ── Imputation de AvgDaysBetweenPurchases ──
if 'AvgDaysBetweenPurchases' in df.columns and df['AvgDaysBetweenPurchases'].isnull().any():
    nb_missing = df['AvgDaysBetweenPurchases'].isnull().sum()
    df['AvgDaysBetweenPurchases'].fillna(df['AvgDaysBetweenPurchases'].median(), inplace=True)
    print(f"[OK] 'AvgDaysBetweenPurchases' : {nb_missing} valeurs manquantes imputées par la médiane")

# ================================
# 4️⃣ Détection et correction des valeurs aberrantes
# ================================

# ── SupportTicketsCount : valeurs sentinelles -1 et 999 → NaN puis capping IQR ──
COL_TICKETS = 'SupportTicketsCount'

if COL_TICKETS in df.columns:
    # Remplacer d'abord les valeurs sentinelles (-1 = non renseigné, 999 = erreur saisie)
    nb_sentinel = df[COL_TICKETS].isin([-1, 999]).sum()
    df.loc[df[COL_TICKETS].isin([-1, 999]), COL_TICKETS] = np.nan
    df[COL_TICKETS].fillna(df[COL_TICKETS].median(), inplace=True)
    print(f"[OK] '{COL_TICKETS}' : {nb_sentinel} valeurs sentinelles (-1, 999) imputées par la médiane")

    # Calcul des quartiles pour détecter les valeurs aberrantes résiduelles
    Q1 = df[COL_TICKETS].quantile(0.25)
    Q3 = df[COL_TICKETS].quantile(0.75)

    # Calcul de l'intervalle interquartile (IQR)
    # L'IQR mesure la dispersion centrale des données (Q3 - Q1)
    IQR = Q3 - Q1

    # Définition des bornes inférieure et supérieure
    # Toute valeur en dehors de [Q1 - 1.5*IQR, Q3 + 1.5*IQR] est considérée aberrante
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Identifier les valeurs aberrantes
    outliers = df[(df[COL_TICKETS] < lower) | (df[COL_TICKETS] > upper)]
    print(f"Nombre de valeurs aberrantes détectées dans '{COL_TICKETS}' : {len(outliers)}")

    # Corriger les valeurs aberrantes par capping :
    # les valeurs en dessous de 'lower' sont remplacées par 'lower'
    # les valeurs au dessus de 'upper' sont remplacées par 'upper'
    df[COL_TICKETS] = df[COL_TICKETS].clip(lower, upper)
    print(f"[OK] Capping appliqué sur '{COL_TICKETS}' → bornes [{lower:.2f}, {upper:.2f}]")
else:
    print(f"[AVERTISSEMENT] Colonne '{COL_TICKETS}' introuvable, étape ignorée.")

# ── SatisfactionScore : valeurs sentinelles -1, 0 et 99 → hors plage [1-5] ──
COL_SAT = 'SatisfactionScore'

if COL_SAT in df.columns:
    # Valeurs valides : 1 à 5 uniquement
    nb_sentinel = (~df[COL_SAT].isin([1, 2, 3, 4, 5])).sum()
    df.loc[~df[COL_SAT].isin([1, 2, 3, 4, 5]), COL_SAT] = np.nan
    df[COL_SAT].fillna(df[COL_SAT].median(), inplace=True)
    print(f"[OK] '{COL_SAT}' : {nb_sentinel} valeurs aberrantes (-1, 0, 99) imputées par la médiane")

    # Capping IQR sur les valeurs résiduelles
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

# Convertir la colonne RegistrationDate en format datetime uniforme (YYYY-MM-DD)
# errors='coerce' transforme les dates non reconnues en NaT au lieu de planter
if 'RegistrationDate' in df.columns:
    nb_before = df['RegistrationDate'].isnull().sum()
    df["RegistrationDate"] = pd.to_datetime(
        df["RegistrationDate"], dayfirst=True, errors='coerce'
    )
    nb_after  = df['RegistrationDate'].isnull().sum()
    nb_failed = nb_after - nb_before

    # Extraction de features utiles depuis la date
    df["RegYear"]    = df["RegistrationDate"].dt.year
    df["RegMonth"]   = df["RegistrationDate"].dt.month
    df["RegDay"]     = df["RegistrationDate"].dt.day
    df["RegWeekday"] = df["RegistrationDate"].dt.weekday  # 0 = Lundi

    # Ancienneté en jours depuis la date la plus récente du dataset
    ref_date = df["RegistrationDate"].max()
    df["DaysSinceRegistration"] = (ref_date - df["RegistrationDate"]).dt.days

    # Conserver la colonne formatée en string pour lisibilité
    df["RegistrationDate"] = df["RegistrationDate"].dt.strftime("%Y-%m-%d")

    print(f"[OK] 'RegistrationDate' convertie au format YYYY-MM-DD "
          f"({nb_failed} date(s) non reconnue(s) → NaT)")
    print(f"     Features extraites : RegYear, RegMonth, RegDay, RegWeekday, DaysSinceRegistration")
    print(df[["RegistrationDate", "RegYear", "RegMonth", "RegWeekday"]].head())
else:
    print("[AVERTISSEMENT] Colonne 'RegistrationDate' introuvable, étape ignorée.")

# ================================
# 6️⃣ Suppression des colonnes inutiles pour l'analyse
# ================================

# NewsletterSubscribed : variance nulle (toujours "Yes") → aucune information prédictive
if 'NewsletterSubscribed' in df.columns:
    df = df.drop(columns=["NewsletterSubscribed"])
    print("[OK] Colonne 'NewsletterSubscribed' supprimée (variance nulle).")
else:
    print("[AVERTISSEMENT] Colonne 'NewsletterSubscribed' introuvable, étape ignorée.")

# CustomerID : identifiant unique, non prédictif pour les modèles ML
if 'CustomerID' in df.columns:
    df = df.drop(columns=["CustomerID"])
    print("[OK] Colonne 'CustomerID' supprimée (identifiant non prédictif).")

# ================================
# 7️⃣ Création de nouvelles features à partir de "LastLoginIP"
# ================================

if 'LastLoginIP' in df.columns:
    # 1️⃣ Extraction des 4 octets de l'adresse IP en colonnes séparées
    # (chaque octet peut être utilisé comme feature numérique indépendante)
    ip_split = df["LastLoginIP"].str.split(".", expand=True)
    df[["IP1", "IP2", "IP3", "IP4"]] = ip_split.iloc[:, :4].apply(
        pd.to_numeric, errors='coerce'
    ).fillna(0).astype(int)

    # 2️⃣ Création de la feature IP_Type : distingue les IPs privées des publiques
    # Plages privées standards : 10.x.x.x, 172.16-31.x.x, 192.168.x.x
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

    # 3️⃣ Suppression de la colonne originale LastLoginIP
    # (remplacée par les nouvelles features IP1, IP2, IP3, IP4, IP_Type)
    df = df.drop(columns=["LastLoginIP"])

    print(df[["IP1", "IP2", "IP3", "IP4", "IP_Type"]].head())
else:
    print("[AVERTISSEMENT] Colonne 'LastLoginIP' introuvable, étape ignorée.")

# ================================
# 8️⃣ Analyse de la distribution des variables catégorielles
# ================================

# Compter les valeurs pour observer les déséquilibres de classes
for col in ["AccountStatus", "Churn"]:
    if col in df.columns:
        print(f"\nDistribution de '{col}' :")
        print(df[col].value_counts())

# ================================
# 9️⃣ Sauvegarde des données nettoyées
# ================================

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Enregistrer le dataset nettoyé dans le dossier processed
df.to_csv(OUTPUT_PATH, index=False)

print(f"\n[OK] Nettoyage terminé. Fichier sauvegardé → {OUTPUT_PATH}")
print(f"     Dimensions finales : {df.shape[0]} lignes, {df.shape[1]} colonnes")