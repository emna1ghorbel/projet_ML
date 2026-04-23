import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils import save_pipeline_objects

# ================================
# 1. Load Data
# ================================
X_train = pd.read_csv('data/train_test/X_train.csv')
X_test  = pd.read_csv('data/train_test/X_test.csv')
y_train = pd.read_csv('data/train_test/y_train.csv').values.ravel()
y_test  = pd.read_csv('data/train_test/y_test.csv').values.ravel()

# ================================
# 2. Select only the features the form will collect
#    (doc says: drop features with zero importance in tree models)
# ================================
FEATURES = [
    'Recency',                   # days since last purchase
    'Frequency',                 # number of orders
    'MonetaryTotal',             # total spending £
    'TotalQuantity',             # total units bought
    'ReturnRatio',               # rate of returns
    'CancelledTransactions',     # number of cancellations
    'AvgDaysBetweenPurchases',   # purchase rhythm
    'Age',                       # customer age
    'SatisfactionScore',         # 1-5 rating
    'SupportTicketsCount',       # support tickets opened
]

# Verify all features exist
missing_features = [f for f in FEATURES if f not in X_train.columns]
if missing_features:
    raise ValueError(f"[ERREUR] Features manquantes dans X_train : {missing_features}\n"
                     f"Colonnes disponibles : {X_train.columns.tolist()}")

X_train = X_train[FEATURES]
X_test  = X_test[FEATURES]

print(f"[OK] Features sélectionnées : {FEATURES}")
print(f"[OK] X_train shape : {X_train.shape}")

# ================================
# 3. Scale
# ================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ================================
# 4. PCA — keep enough components to explain 95% variance
# ================================
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

print(f"[OK] PCA : {pca.n_components_} composantes → "
      f"{pca.explained_variance_ratio_.sum():.2%} variance expliquée")

# ================================
# 5. Train
# ================================
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_pca, y_train)

# ================================
# 6. Evaluate
# ================================
y_pred = model.predict(X_test_pca)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Feature importance — RF is trained on PCA components, so we show component importances
n_components = X_train_pca.shape[1]
importances = pd.Series(
    model.feature_importances_,
    index=[f"PC{i+1}" for i in range(n_components)]
).sort_values(ascending=False)
print("\n--- PCA Component Importances ---")
print(importances)

# ================================
# 7. Save
# ================================
save_pipeline_objects(scaler, pca, model)