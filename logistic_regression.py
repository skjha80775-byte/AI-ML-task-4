import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, precision_score,
    recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve)

# 1. Dataset (Breast Cancer Wisconsin; target 1 = benign, 0 = malignant)
data = load_breast_cancer(as_frame=True)
df = data.frame
df.to_csv("breast_cancer.csv", index=False)
X, y = df.drop(columns="target"), df["target"]
print("Shape:", df.shape, "\nClass counts:\n", y.value_counts())

# 2. Split + standardize
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
sc = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

# 3. Fit
model = LogisticRegression(max_iter=1000).fit(X_tr_s, y_tr)
proba = model.predict_proba(X_te_s)[:, 1]

# 4. Evaluate at default threshold 0.5
def report(th):
    pred = (proba >= th).astype(int)
    return pred, dict(threshold=th, accuracy=accuracy_score(y_te, pred),
        precision=precision_score(y_te, pred), recall=recall_score(y_te, pred), f1=f1_score(y_te, pred))
pred, m = report(0.5)
print("\nThreshold 0.5:", {k: round(v, 4) for k, v in m.items()})
print("Confusion matrix:\n", confusion_matrix(y_te, pred))
auc = roc_auc_score(y_te, proba)
print("ROC-AUC:", round(auc, 4))

ConfusionMatrixDisplay(confusion_matrix(y_te, pred), display_labels=["malignant", "benign"]).plot(cmap="Blues")
plt.title("Confusion Matrix (threshold 0.5)"); plt.savefig("confusion_matrix.png", dpi=120); plt.close()

fpr, tpr, _ = roc_curve(y_te, proba)
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}"); plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.legend()
plt.savefig("roc_curve.png", dpi=120); plt.close()

# 5. Threshold tuning
print("\nThreshold tuning:")
rows = [report(t)[1] for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
print(pd.DataFrame(rows).round(4).to_string(index=False))
p, r, t = precision_recall_curve(y_te, proba)
plt.plot(t, p[:-1], label="Precision"); plt.plot(t, r[:-1], label="Recall")
plt.xlabel("Threshold"); plt.title("Precision/Recall vs Threshold"); plt.legend()
plt.savefig("threshold_tuning.png", dpi=120); plt.close()

# Sigmoid
z = np.linspace(-10, 10, 200)
plt.plot(z, 1 / (1 + np.exp(-z))); plt.axhline(0.5, color="gray", ls="--"); plt.axvline(0, color="gray", ls="--")
plt.xlabel("z = w·x + b"); plt.ylabel("σ(z)"); plt.title("Sigmoid Function"); plt.grid(alpha=.3)
plt.savefig("sigmoid.png", dpi=120); plt.close()
