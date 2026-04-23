import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from utils import clean_data

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../models')

model = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
pca = joblib.load(os.path.join(MODEL_DIR, 'pca.pkl'))

FEATURES = [
    'Recency', 'Frequency', 'MonetaryTotal', 'TotalQuantity',
    'ReturnRatio', 'CancelledTransactions', 'AvgDaysBetweenPurchases',
    'Age', 'SatisfactionScore', 'SupportTicketsCount',
]

def run_prediction(data: dict):
    df_raw = pd.DataFrame([data])
    df_clean = clean_data(df_raw)
    X = df_clean.reindex(columns=FEATURES)
    X = X.fillna(0)
    
    scaled = scaler.transform(X)
    pca_d = pca.transform(scaled)
    
    pred = model.predict(pca_d)[0]
    proba = round(model.predict_proba(pca_d)[0][1] * 100, 1)
    
    return ("Churn" if pred == 1 else "No Churn"), proba

@app.route('/', methods=['GET', 'POST'])
def index():
    result = proba = error = None
    form_data = {}

    if request.method == 'POST':
        try:
            form_data = {k: request.form[k] for k in request.form}
            data = {col: float(request.form.get(col, 0)) for col in FEATURES}
            result, proba = run_prediction(data)
        except Exception as e:
            error = str(e)

    return render_template('index.html', tab='single',
                           result=result, proba=proba,
                           error=error, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)