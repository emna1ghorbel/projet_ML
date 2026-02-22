import pandas as pd
import sys
import os

# ================================
# 1️⃣ Chargement des données
# ================================

DATA_PATH = "data/raw/retail_customers_COMPLETE_CATEGORICAL.csv"
OUTPUT_PATH = "data/processed/retail_customers_cleaned.csv"

# Vérifier que le fichier source existe avant de charger
if not os.path.exists(DATA_PATH):
    print(f"[ERREUR] Fichier introuvable : {DATA_PATH}")
    sys.exit(1)

# Charger le dataset depuis le dossier raw
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

# Remplacer les valeurs manquantes de 'Age' par la médiane
# (la médiane est préférable à la moyenne car elle est robuste aux valeurs extrêmes)
if 'Age' in df.columns:
    nb_missing_age = df['Age'].isnull().sum()
    df['Age'] = df['Age'].fillna(df['Age'].median())
    print(f"[OK] 'Age' : {nb_missing_age} valeurs manquantes imputées par la médiane ({df['Age'].median():.1f})")
else:
    print("[AVERTISSEMENT] Colonne 'Age' introuvable, étape ignorée.")

# ================================
# 4️⃣ Détection et correction des valeurs aberrantes
# ================================

COL_TICKETS = 'SupportTicketsCount'

if COL_TICKETS in df.columns:
    # Calcul des quartiles pour détecter les valeurs aberrantes
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

# ================================
# 5️⃣ Formats inconsistants
# ================================

# Convertir la colonne RegistrationDate en format datetime uniforme (YYYY-MM-DD)
# errors='coerce' transforme les dates non reconnues en NaT au lieu de planter
if 'RegistrationDate' in df.columns:
    nb_before = df['RegistrationDate'].isnull().sum()
    df["RegistrationDate"] = pd.to_datetime(df["RegistrationDate"], dayfirst=True, errors='coerce')
    nb_after = df['RegistrationDate'].isnull().sum()
    df["RegistrationDate"] = df["RegistrationDate"].dt.strftime("%Y-%m-%d")

    nb_failed = nb_after - nb_before
    print(f"[OK] 'RegistrationDate' convertie au format YYYY-MM-DD ({nb_failed} date(s) non reconnue(s) → NaT)")
    print(df[["RegistrationDate"]].head())
else:
    print("[AVERTISSEMENT] Colonne 'RegistrationDate' introuvable, étape ignorée.")

# ================================
# 6️⃣ Suppression de la colonne "NewsletterSubscribed" inutile pour l'analyse
# ================================

# Cette colonne n'apporte pas d'information prédictive pour notre modèle
if 'NewsletterSubscribed' in df.columns:
    df = df.drop(columns=["NewsletterSubscribed"])
    print("[OK] Colonne 'NewsletterSubscribed' supprimée.")
else:
    print("[AVERTISSEMENT] Colonne 'NewsletterSubscribed' introuvable, étape ignorée.")

# ================================
# 7️⃣ Création de nouvelles features à partir de "LastLoginIP"
# ================================

if 'LastLoginIP' in df.columns:
    # 1️⃣ Extraction des 4 octets de l'adresse IP en colonnes séparées
    # (chaque octet peut être utilisé comme feature numérique indépendante)
    df[["IP1","IP2","IP3","IP4"]] = df["LastLoginIP"].str.split(".", expand=True)
    df[["IP1","IP2","IP3","IP4"]] = df[["IP1","IP2","IP3","IP4"]].astype(int)

    # 2️⃣ Création de la feature IP_Type : distingue les IPs privées des publiques
    # Plages privées standards : 10.x.x.x, 172.x.x.x, 192.x.x.x
    def ip_type(ip):
        first = int(ip.split('.')[0])
        if first in [10, 172, 192]:
            return "Private"
        else:
            return "Public"

    df["IP_Type"] = df["LastLoginIP"].apply(ip_type)
    print(f"[OK] Features IP créées. Distribution IP_Type :\n{df['IP_Type'].value_counts()}")

    # 3️⃣ Suppression de la colonne originale LastLoginIP
    # (remplacée par les nouvelles features IP1, IP2, IP3, IP4, IP_Type)
    df = df.drop(columns=["LastLoginIP"])

    print(df[["IP1","IP2","IP3","IP4","IP_Type"]].head())
else:
    print("[AVERTISSEMENT] Colonne 'LastLoginIP' introuvable, étape ignorée.")

# ================================
# 8️⃣ Analyse de la distribution des variables catégorielles
# ================================

# Exemple de dataset
df_example = pd.DataFrame({
    "AccountStatus": ["Active"]*50 + ["Suspended"]*30 + ["Pending"]*15 + ["Closed"]*5,
    "Churn": ["No"]*80 + ["Yes"]*20
})

# Compter les valeurs pour observer les déséquilibres de classes
print(df_example["AccountStatus"].value_counts())
print(df_example["Churn"].value_counts())

# ================================
# 9️⃣ Sauvegarde des données nettoyées
# ================================

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Enregistrer le dataset nettoyé dans le dossier processed
df.to_csv(OUTPUT_PATH, index=False)

print(f"[OK] Nettoyage terminé. Fichier sauvegardé → {OUTPUT_PATH}")
print(f"     Dimensions finales : {df.shape[0]} lignes, {df.shape[1]} colonnes")