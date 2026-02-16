import pandas as pd

# ================================
# 1️⃣ Chargement des données
# ================================

# Charger le dataset depuis le dossier raw
df = pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")

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
print("Valeurs manquantes :")
print(df.isnull().sum())

# Vérifier le nombre de doublons
print("Nombre de doublons :", df.duplicated().sum())

# ================================
# 3️⃣ Nettoyage des données
# ================================

# Supprimer les lignes dupliquées
df = df.drop_duplicates()

# Remplacer les valeurs manquantes de 'Age' par la médiane
# (médiane préférable si distribution asymétrique)
df['Age'] = df['Age'].fillna(df['Age'].median())

# ================================
# 4️⃣ Détection et correction des valeurs aberrantes
# ================================

# Calcul des quartiles pour la variable SupportTickets
Q1 = df['SupportTickets'].quantile(0.25)
Q3 = df['SupportTickets'].quantile(0.75)

# Calcul de l’intervalle interquartile (IQR)
IQR = Q3 - Q1

# Définition des bornes inférieure et supérieure
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Identifier les valeurs aberrantes
outliers = df[(df['SupportTickets'] < lower) | 
              (df['SupportTickets'] > upper)]

print("Nombre de valeurs aberrantes détectées :", len(outliers))

# Corriger les valeurs aberrantes par capping
# (remplacement des valeurs extrêmes par les limites calculées)
df['SupportTickets'] = df['SupportTickets'].clip(lower, upper)

# ================================
# 5️⃣ Sauvegarde des données nettoyées
# ================================

# Enregistrer le dataset nettoyé dans le dossier processed
df.to_csv("data/processed/retail_customers_cleaned.csv", index=False)

print("Nettoyage terminé et fichier sauvegardé.")
