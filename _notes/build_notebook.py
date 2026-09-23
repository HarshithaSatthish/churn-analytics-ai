"""Builds notebooks/churn_project_colab.ipynb from cell definitions."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "churn_project_colab.ipynb"

PINS = "pandas==3.0.6 scikit-learn==1.9.1 matplotlib==3.11.1 seaborn==0.13.2 joblib"
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

cells = []


def md(source: str):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": source.splitlines(keepends=True)})


def code(source: str):
    cells.append({"cell_type": "code", "metadata": {},
                  "source": source.splitlines(keepends=True),
                  "outputs": [], "execution_count": None})


md("# Customer Churn Analytics with AI\n"
   "End-to-end pipeline: load -> clean -> EDA -> logistic-regression churn model.\n"
   "Runs top-to-bottom in Google Colab with no manual steps.")

code(f"%pip install -q {PINS}")

md("## 1. Load the data")
code(
    "import urllib.request\n"
    f"DATA_URL = '{DATA_URL}'\n"
    "urllib.request.urlretrieve(DATA_URL, 'telco_churn.csv')\n"
    "import pandas as pd\n"
    "df = pd.read_csv('telco_churn.csv')\n"
    "assert df.shape == (7043, 21), df.shape\n"
    "assert set(df['Churn'].unique()) == {'Yes', 'No'}\n"
    "print('loaded:', df.shape)"
)

md("## 2. Clean the data\n"
   "Fix: `TotalCharges` arrives as text with 11 blanks (all `tenure == 0`, never billed) "
   "-> coerce to numeric, fill with 0. No rows dropped.")
code(
    "SERVICE_COLS = ['PhoneService','MultipleLines','InternetService','OnlineSecurity',\n"
    "                'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']\n"
    "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')\n"
    "n_blank = int(df['TotalCharges'].isna().sum())\n"
    "assert (df.loc[df['TotalCharges'].isna(), 'tenure'] == 0).all()\n"
    "df['TotalCharges'] = df['TotalCharges'].fillna(0)\n"
    "df['churn_binary'] = (df['Churn'] == 'Yes').astype(int)\n"
    "df['tenure_band'] = pd.cut(df['tenure'], bins=[-1,6,12,24,200],\n"
    "                           labels=['0-6','7-12','13-24','25+']).astype(str)\n"
    "df['num_services'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype(int)\n"
    "df['is_fiber'] = (df['InternetService'] == 'Fiber optic').astype(int)\n"
    "df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)\n"
    "df['is_electronic_check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)\n"
    "assert df.shape[0] == 7043 and df.isna().sum().sum() == 0\n"
    "print(f'clean OK: {df.shape[0]} rows, {n_blank} TotalCharges fixed, churn={df[\"churn_binary\"].mean():.4f}')"
)

md("## 3. Exploratory analysis")
code(
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "t = 'churn_binary'\n"
    "s = df.groupby('Contract')[t].mean().sort_values()\n"
    "fig, ax = plt.subplots(figsize=(7,4))\n"
    "ax.bar(s.index, s.values*100); ax.set_ylabel('Churn rate (%)')\n"
    "ax.set_title('Churn rate by contract type'); plt.show()\n"
    "print('month-to-month:', round(s['Month-to-month']*100,1), '% | two-year:', round(s['Two year']*100,1), '%')\n"
    "order = ['0-6','7-12','13-24','25+']\n"
    "s2 = df.groupby('tenure_band')[t].mean().reindex(order)\n"
    "fig, ax = plt.subplots(figsize=(7,4))\n"
    "ax.bar(s2.index, s2.values*100); ax.set_ylabel('Churn rate (%)')\n"
    "ax.set_title('Churn rate by tenure band'); plt.show()\n"
    "num = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges','num_services',\n"
    "       'is_fiber','is_month_to_month','is_electronic_check', t]\n"
    "fig, ax = plt.subplots(figsize=(8,6))\n"
    "sns.heatmap(df[num].corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)\n"
    "ax.set_title('Feature correlations'); plt.show()"
)

md("## 4. Train the churn model\n"
   "80/20 stratified split -> ColumnTransformer (one-hot + scaling) -> "
   "logistic regression (class_weight='balanced').")
code(
    "from sklearn.compose import ColumnTransformer\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.metrics import (accuracy_score, precision_score, recall_score,\n"
    "                             f1_score, roc_auc_score, confusion_matrix,\n"
    "                             ConfusionMatrixDisplay)\n"
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.pipeline import Pipeline\n"
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
    "CAT = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService',\n"
    "       'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV',\n"
    "       'StreamingMovies','Contract','PaperlessBilling','PaymentMethod','tenure_band']\n"
    "NUM = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges','num_services',\n"
    "       'is_fiber','is_month_to_month','is_electronic_check']\n"
    "X, y = df[CAT + NUM], df['churn_binary']\n"
    "Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)\n"
    "pipe = Pipeline([\n"
    "  ('preprocess', ColumnTransformer([\n"
    "     ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT),\n"
    "     ('num', StandardScaler(), NUM)])),\n"
    "  ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])\n"
    "pipe.fit(Xtr, ytr)\n"
    "proba = pipe.predict_proba(Xte)[:,1]; pred = (proba >= 0.5).astype(int)\n"
    "metrics = {'accuracy': round(float(accuracy_score(yte, pred)),4),\n"
    "           'precision': round(float(precision_score(yte, pred)),4),\n"
    "           'recall': round(float(recall_score(yte, pred)),4),\n"
    "           'f1': round(float(f1_score(yte, pred)),4),\n"
    "           'roc_auc': round(float(roc_auc_score(yte, proba)),4)}\n"
    "print(metrics)\n"
    "fig, ax = plt.subplots(figsize=(5,4))\n"
    "ConfusionMatrixDisplay(confusion_matrix(yte, pred),\n"
    "                      display_labels=['Retained','Churned']).plot(ax=ax)\n"
    "ax.set_title('Confusion matrix (test set)'); plt.show()"
)

md("## 5. Save the artifacts")
code(
    "import joblib, json\n"
    "joblib.dump(pipe, 'churn_model.pkl')\n"
    "json.dump(metrics, open('metrics.json','w'), indent=2)\n"
    "print('saved churn_model.pkl + metrics.json')"
)

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                 "language": "python", "name": "python3"}},
      "cells": cells}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print("notebook written:", OUT, f"({len(cells)} cells)")
