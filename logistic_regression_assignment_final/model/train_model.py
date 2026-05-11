"""
Assignment 01: Logistic Regression Classification
Dataset: Iris Flower Dataset
Author: su92-bsaim-s23-015
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
import pickle
import os

# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

print("=" * 55)
print("   LOGISTIC REGRESSION - IRIS CLASSIFICATION")
print("=" * 55)
print(f"\n📦 Dataset Shape: {X.shape}")
print(f"🌸 Classes: {list(iris.target_names)}")
print(f"\n📊 First 5 rows:\n{X.head()}")

# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────
print("\n── Preprocessing ──")

# Check missing values
print(f"Missing values:\n{X.isnull().sum()}")
# Iris dataset has no missing values, but we handle it anyway
X = X.fillna(X.mean())

# Normalize / Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Data normalized using StandardScaler")

# ─────────────────────────────────────────────
# 3. Train/Test Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n🔀 Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 4. Train Logistic Regression Model
# ─────────────────────────────────────────────
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)
print("\n✅ Model trained successfully!")

# ─────────────────────────────────────────────
# 5. Evaluate Model
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall    = recall_score(y_test, y_pred, average='weighted')
f1        = f1_score(y_test, y_pred, average='weighted')

print("\n── Evaluation Metrics ──")
print(f"  Accuracy  : {accuracy  * 100:.2f}%")
print(f"  Precision : {precision * 100:.2f}%")
print(f"  Recall    : {recall    * 100:.2f}%")
print(f"  F1-Score  : {f1        * 100:.2f}%")

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ─────────────────────────────────────────────
# 6. Confusion Matrix Plot
# ─────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/confusion_matrix.png', dpi=150)
plt.close()
print("\n📈 Confusion matrix saved to plots/confusion_matrix.png")

# ─────────────────────────────────────────────
# 7. Feature Distribution Plot
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Feature Distributions by Species', fontsize=14)
colors = ['#E63946', '#457B9D', '#2A9D8F']

for idx, feature in enumerate(iris.feature_names):
    ax = axes[idx // 2][idx % 2]
    for cls_idx, cls_name in enumerate(iris.target_names):
        data = X[y == cls_idx][feature]
        ax.hist(data, alpha=0.6, label=cls_name, color=colors[cls_idx], bins=15)
    ax.set_title(feature)
    ax.legend(fontsize=8)
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig('plots/feature_distributions.png', dpi=150)
plt.close()
print("📊 Feature distributions saved to plots/feature_distributions.png")

# ─────────────────────────────────────────────
# 8. Save Model & Scaler
# ─────────────────────────────────────────────
os.makedirs('../saved_model', exist_ok=True)
with open('../saved_model/logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('../saved_model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n💾 Model and scaler saved to saved_model/")
print("\n✅ Training complete!")
