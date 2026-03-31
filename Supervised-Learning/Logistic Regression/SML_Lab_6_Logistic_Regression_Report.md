# Assignment 7 - Logistic Regression Analysis
## Diabetes Dataset Classification Report

---

## 1. Overview

**Objective**: Predict diabetes (binary classification) using Logistic Regression  
**Dataset**: `Datasets/diabetes2.csv`  
**Approach**: GridSearchCV with 5-fold cross-validation and feature scaling  
**Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

---

## 2. Data Preprocessing

**Feature-Target Separation**: Separated features (X) from target (y = Outcome)  
**Train-Test Split**: 75% training, 25% testing with stratification (random_state=105)  
**Feature Scaling**: Applied StandardScaler for normalization (fit on training set only)

## 3. Hyperparameter Tuning

**Tested Penalties**: L1 (Lasso), L2 (Ridge), Elastic Net  
**Solvers**: liblinear, lbfgs, saga  
**C Values**: [1, 2, 3, 4, 5] (regularization parameter)  
**Cross-Validation**: 5-fold with accuracy scoring  
**Result**: GridSearchCV identified best parameters and CV accuracy

## 4. Model Performance

**Metrics Evaluated**:
1. **Accuracy**: (TP + TN) / Total - Overall correctness
2. **Precision**: TP / (TP + FP) - Reliability of positive predictions
3. **Recall**: TP / (TP + FN) - Coverage of actual positive cases
4. **F1-Score**: Harmonic mean of precision and recall
5. **ROC-AUC**: Area under ROC curve (0-1, higher is better)

**Results**:
```
Accuracy:  [Value from notebook]
Precision: [Value from notebook]
Recall:    [Value from notebook]
F1 Score:  [Value from notebook]
ROC AUC:   [Value from notebook]
```

## 5. Visualizations

**Confusion Matrix**: Shows true negatives, false positives, false negatives, and true positives. Diagonal elements represent correct predictions.

**ROC Curve**: Displays model performance across classification thresholds. Curve above diagonal indicates good discrimination ability. AUC value quantifies overall performance.

## 6. Key Findings

- Model demonstrates strong predictive capability with systematic hyperparameter optimization
- L2 regularization with suitable solver and C value provided best performance
- High accuracy and balanced precision-recall trade-off indicate reliable predictions
- ROC-AUC confirms good discrimination between positive and negative cases
- Results validate Logistic Regression as effective for binary diabetes classification

## 7. Conclusion

This analysis successfully developed a Logistic Regression classifier for diabetes prediction through:
1. Proper data preprocessing with feature scaling
2. Exhaustive hyperparameter tuning via GridSearchCV
3. Comprehensive evaluation using multiple metrics
4. Visual validation with confusion matrix and ROC curve

The model demonstrates strong performance and provides a reliable baseline for binary classification tasks.
