# Analyse Comportementale Clientèle Retail
**Atelier Machine Learning — GI2**  
Prédiction du churn client pour un e-commerce de cadeaux.

---

## Structure du projet

```
projet_ml/
├── data/
│   ├── raw/                  # Données brutes originales
│   ├── processed/            # Données nettoyées
│   └── train_test/           # Données splittées (X_train, X_test, y_train, y_test)
├── notebooks/                # Notebooks Jupyter (prototypage)
├── src/
│   ├── preprocessing.py      # Chargement + nettoyage + split
│   ├── train_model.py        # Entraînement du modèle
│   ├── predict.py            # Prédiction batch sur nouveaux clients
│   └── utils.py              # Fonctions utilitaires (clean_data, split_and_save, ...)
├── models/                   # Modèles sauvegardés (.pkl)
│   ├── model.pkl
│   ├── scaler.pkl
│   └── pca.pkl
├── app/
│   ├── app.py                # Application Flask (prédiction client unique)
│   └── templates/
│       └── index.html        # Interface utilisateur
├── reports/                  # Prédictions et visualisations exportées
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/emna1ghorbel/projet_ML.git
cd projet_ml
```

### 2. Créer et activer l'environnement virtuel
```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## Utilisation

### Étape 1 — Préparation des données
Nettoie les données brutes, gère les valeurs manquantes/aberrantes, et sauvegarde les splits train/test.
```bash
python src/preprocessing.py
```

### Étape 2 — Entraînement du modèle
Sélectionne les 10 features clés, applique StandardScaler + PCA, entraîne un RandomForestClassifier.
```bash
python src/train_model.py
```

### Étape 3 — Prédiction batch (nouveaux clients)
```bash
# Prédiction sur fichier CSV de nouveaux clients
python src/predict.py --input data/raw/new_customers.csv

# Résultat sauvegardé dans reports/predictions.csv

# Chemin de sortie personnalisé
python src/predict.py --input data/raw/new_customers.csv --output reports/my_output.csv
```

### Étape 4 — Lancer l'application web (prédiction client unique)
```bash
python app/app.py
```
Ouvrir dans le navigateur : [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Pipeline ML

| Étape | Détail |
|---|---|
| **Features sélectionnées** | Recency, Frequency, MonetaryTotal, TotalQuantity, ReturnRatio, CancelledTransactions, AvgDaysBetweenPurchases, Age, SatisfactionScore, SupportTicketsCount |
| **Normalisation** | StandardScaler (fit sur X_train uniquement) |
| **Réduction de dimension** | PCA — 95% variance conservée |
| **Modèle** | RandomForestClassifier (100 arbres, class_weight=balanced) |
| **Performance** | Accuracy : 97% — F1 macro : 0.97 |

---

## Problèmes de qualité traités

| Problème | Feature | Traitement |
|---|---|---|
| Valeurs manquantes | Age | Imputation par médiane |
| Valeurs sentinelles | SupportTicketsCount (-1, 999) | Remplacement + capping IQR |
| Valeurs hors plage | SatisfactionScore (-1, 0, 99) | Remplacement + capping IQR |
| Format inconsistant | RegistrationDate | Parsing + extraction de features |
| Variance nulle | NewsletterSubscribed | Suppression |
| Identifiant non prédictif | CustomerID | Suppression |
| Données brutes | LastLoginIP | Feature engineering (octets + IP_Type) |

---

## Déploiement

L'application Flask expose une interface web permettant de :
- Saisir les informations d'un client manuellement via formulaire
- Obtenir une prédiction de churn en temps réel avec probabilité
- Visualiser le résultat avec un indicateur visuel (barre de progression, couleur)

**Note :** Les prédictions batch (plusieurs clients) sont disponibles via le script en ligne de commande `predict.py`.

---
