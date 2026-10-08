# Task 4: Classification with Logistic Regression

**Dataset:** Breast Cancer Wisconsin (569 samples, 30 features; 1 = benign, 0 = malignant)
**Tools:** scikit-learn, pandas, matplotlib

## Steps
1. Loaded dataset, 80/20 stratified train/test split
2. Standardized features (scaler fit on train only)
3. Fit `LogisticRegression`
4. Evaluated: confusion matrix, precision, recall, ROC-AUC
5. Tuned threshold (0.3–0.9) and plotted the sigmoid

## Results (threshold 0.5)
| Metric | Value |
|---|---|
| Accuracy | 0.9825 |
| Precision | 0.9861 |
| Recall | 0.9861 |
| ROC-AUC | 0.9954 |

Confusion matrix: `[[41, 1], [1, 71]]`

## Threshold note
Lowering to 0.3 gives recall 1.00 (no missed benign) at slightly lower precision (0.973). In medical screening, choose the threshold based on which error is costlier. Raising it increases precision but drops recall.

## Sigmoid
σ(z) = 1 / (1 + e^(−z)) maps the linear score z = w·x + b to a probability in (0,1). Predict class 1 if σ(z) ≥ threshold.

## Files
`logistic_regression.py`, `breast_cancer.csv`, `output.txt`, `confusion_matrix.png`, `roc_curve.png`, `threshold_tuning.png`, `sigmoid.png`

## Interview Q&A
1. **Logistic vs linear regression:** linear predicts continuous values; logistic passes the linear score through a sigmoid to predict class probability, trained with log-loss.
2. **Sigmoid:** σ(z)=1/(1+e^−z), an S-curve squashing any real number into (0,1).
3. **Precision vs recall:** precision = TP/(TP+FP) (how many predicted positives are right); recall = TP/(TP+FN) (how many actual positives are found).
4. **ROC-AUC:** ROC plots TPR vs FPR across thresholds; AUC is the area under it (1 = perfect, 0.5 = random).
5. **Confusion matrix:** table of TP, FP, TN, FN counts comparing predictions to actual labels.
6. **Imbalanced classes:** accuracy becomes misleading and the model favors the majority class. Use class_weight='balanced', resampling, PR-AUC/F1, and threshold tuning.
7. **Choosing threshold:** use the precision–recall or ROC curve, based on the cost of FP vs FN (or maximize F1 / Youden's J).
8. **Multi-class:** yes, via one-vs-rest or multinomial (softmax) regression.
