# SML Lab 7 - Support Vector Machine (SVM)
## Loan Status Classification Report

---

## 1. Introduction

Support Vector Machines (SVMs) are powerful supervised learning algorithms used for classification and regression tasks. This lab explores the application of SVM for binary classification on a loan status dataset, comparing model performance with and without feature engineering.

**Objective**: Classify loan approval status using SVM and evaluate the impact of feature engineering on model performance.

---

## 2. Dataset Description

### 2.1 Data Overview
- **Source**: Train dataset (`train_v9rqX0R.csv`)
- **Size**: Multiple records with financial and demographic features
- **Target Variable**: Loan approval status (binary classification)
- **Data Type**: Mix of numerical and categorical features

### 2.2 Data Preprocessing

#### Missing Values Handling:
- Numerical features: Filled with median values
- Categorical features: Dropped due to high cardinality

#### Feature Selection:
- **Retained**: All numerical features for SVM modeling
- **Removed**: Categorical columns for simplification
- **Final Feature Count**: Multiple numerical features after cleaning

#### Data Distribution:
Target variable classes are distributed across training and test sets using stratified sampling to maintain class balance.

### 2.3 Feature Correlation Analysis
A correlation heatmap was generated to identify relationships between features:
- Strong positive correlations indicate co-varying features
- Weak correlations suggest independent information
- Highly correlated features may introduce multicollinearity

---

## 3. Methodology

### 3.1 Model Architecture

**Algorithm**: Support Vector Machine (SVM)
- **Type**: Binary Classification using SVC (Support Vector Classifier)
- **Decision Function**: Non-linear kernel-based classification
- **Parameters Tuned**: C, gamma, and kernel type

### 3.2 Feature Scaling

Applied **RobustScaler** for normalization:
- Scales features to be robust against outliers
- Uses median and quartile range for scaling
- Formula: `(X - median) / IQR`
- **Advantages**: Less sensitive to extreme values compared to StandardScaler

### 3.3 Hyperparameter Tuning with GridSearchCV

**Search Strategy**: Exhaustive grid search with 5-fold cross-validation

**Parameter Space**:
| Parameter | Values | Total |
|-----------|--------|-------|
| C (Regularization) | [0.1, 1, 10, 100] | 4 |
| gamma (Kernel Coefficient) | ['scale', 0.1, 0.01, 0.001] | 4 |
| kernel (Decision Boundary) | ['rbf', 'poly', 'sigmoid'] | 3 |
| **Total Combinations** | - | **48** |

**Cross-Validation**: 5-fold with stratified splits to maintain class distribution

### 3.4 Target Variable Transformation

For multi-class capability, continuous target values were binned into discrete classes:
- **Binning Strategy**: Quantile-based binning into 5 equal-frequency bins
- **Purpose**: Convert to multi-class classification problem
- **Classes**: Integer labels 0-4 representing different approval probability ranges

---

## 4. Model 1: SVM without Feature Engineering

### 4.1 Model Configuration
- **Algorithm**: SVC with default parameters initially
- **Scaling**: RobustScaler applied to all features
- **Hyperparameter Optimization**: GridSearchCV (5-fold CV, 48 combinations)

### 4.2 Best Hyperparameters
Selected through exhaustive cross-validation search:
- **C**: Optimal regularization strength
- **kernel**: Selected kernel function (rbf, poly, or sigmoid)
- **gamma**: Kernel coefficient value

### 4.3 Performance Metrics

#### Training Phase:
- **Best Cross-Validation Score**: Highest accuracy achieved during 5-fold CV

#### Testing Phase:
- **Test Accuracy**: Model accuracy on held-out test set
- **Precision**: Of predicted positive cases, proportion actually positive
- **Recall**: Of actual positive cases, proportion correctly identified
- **F1-Score**: Harmonic mean balancing precision and recall

### 4.4 Prediction Analysis

**Confusion Matrix Analysis**:
```
                 Predicted Negative    Predicted Positive
Actual Negative   True Negatives       False Positives
Actual Positive   False Negatives      True Positives
```

**Classification Report**:
- Per-class metrics (precision, recall, F1-score)
- Support (number of instances per class)
- Weighted averages accounting for class imbalance

---

## 5. Model 2: SVM with Feature Engineering

### 5.1 Feature Engineering Strategy

Enhanced model with automatically generated features:

#### Polynomial Features:
- **Feature²**: Square of primary numerical feature
- **log(Feature)**: Log transformation to capture non-linear relationships
- **Advantage**: Captures non-linear patterns in data

#### Interaction Features:
- **Feature1 × Feature2**: Product of two features (interaction effect)
- **Feature1 / Feature2**: Ratio capturing proportional relationships
- **Advantage**: Enables SVM to capture feature dependencies

### 5.2 Expanded Feature Space
- **Original Features**: Number of numerical features after preprocessing
- **New Features**: Polynomial and interaction terms added
- **Total Features After Engineering**: Increased feature count
- **Dimensionality Trade-off**: More information vs. increased complexity

### 5.3 Model Configuration
- **Algorithm**: SVC with same parameter grid as Model 1
- **Scaling**: RobustScaler applied to engineered features
- **Hyperparameter Optimization**: GridSearchCV (5-fold CV)

### 5.4 Best Hyperparameters
Optimized through cross-validation on engineered feature space:
- **C**: Regularization strength (may differ from Model 1)
- **kernel**: Selected kernel type
- **gamma**: Kernel coefficient

### 5.5 Performance Metrics

#### Training Phase:
- **Best Cross-Validation Score**: Accuracy with engineered features

#### Testing Phase:
- **Test Accuracy**: Evaluation on test set with engineered features
- **Precision, Recall, F1-Score**: Detailed metric breakdown
- **Confusion Matrix**: Prediction error analysis

---

## 6. Results and Comparison

### 6.1 Model 1 Results (Baseline)

**Best Hyperparameters**:
- C = [Value from execution]
- kernel = [Value from execution]
- gamma = [Value from execution]

**Performance Metrics**:
- Best CV Score: [Value from execution]
- Test Set Accuracy: [Value from execution]
- Precision: [Value from execution]
- Recall: [Value from execution]
- F1-Score: [Value from execution]

### 6.2 Model 2 Results (with Feature Engineering)

**Best Hyperparameters**:
- C = [Value from execution]
- kernel = [Value from execution]
- gamma = [Value from execution]

**Performance Metrics**:
- Best CV Score: [Value from execution]
- Test Set Accuracy: [Value from execution]
- Precision: [Value from execution]
- Recall: [Value from execution]
- F1-Score: [Value from execution]

### 6.3 Comparative Analysis

| Metric | Model 1 | Model 2 | Difference |
|--------|---------|---------|------------|
| Test Accuracy | [M1_Acc] | [M2_Acc] | [Improvement] |
| Best C | [M1_C] | [M2_C] | - |
| Kernel | [M1_Kernel] | [M2_Kernel] | - |
| CV Score | [M1_CV] | [M2_CV] | [Diff] |

### 6.4 Key Observations

1. **Accuracy Improvement**: Feature engineering resulted in [+/-]X% accuracy change
2. **Optimal Parameters**: Both models selected similar/different hyperparameters
3. **Kernel Selection**: [rbf/poly/sigmoid] performed best for [Model 1/2]
4. **Regularization**: C value of [X] balanced overfitting and underfitting best
5. **Feature Importance**: Engineered features [did/did not] significantly improve performance

### 6.5 Error Analysis

**Model 1 Confusion Matrix**:
- True Negatives: [Count]
- True Positives: [Count]
- False Positives: [Count]
- False Negatives: [Count]

**Model 2 Confusion Matrix**:
- True Negatives: [Count]
- True Positives: [Count]
- False Positives: [Count]
- False Negatives: [Count]

---

## 7. Discussion

### 7.1 Why Feature Engineering Matters

Feature engineering helps SVM by:
1. **Capturing Non-linearity**: Polynomial features help capture quadratic relationships
2. **Creating Interactions**: Interaction terms enable SVM to learn feature combinations
3. **Enhancing Separation**: New features may provide better class separability
4. **Reducing Variance**: More informative features can reduce model variance

### 7.2 Hyperparameter Insights

**C Parameter** (Regularization Strength):
- Lower C → More regularization → Simpler decision boundary
- Higher C → Less regularization → Complex boundary, potential overfitting
- Optimal value: [M1/M2 value] indicating [tight/loose] constraint

**Kernel Function**:
- **RBF (Radial Basis Function)**: Maps to infinite-dimensional space, flexible
- **Polynomial**: Creates polynomial decision boundaries
- **Sigmoid**: Similar to neural networks, can underperform
- **Selected**: [Kernel] provided best generalization

**Gamma Parameter**:
- Low gamma → Far-reaching influence of support vectors
- High gamma → Close influence (local decision boundaries)
- Optimal value: [Value] indicates [global/local] decision patterns

### 7.3 Computational Considerations

- **GridSearchCV** evaluated 48 parameter combinations
- **5-fold cross-validation** = 240 model training iterations
- **n_jobs=-1**: Parallel processing across all CPU cores
- **Execution Time**: Manageable for dataset size

---

## 8. Conclusion

### 8.1 Key Findings

1. **Model Performance**: [Model 1/2] achieved [X]% test accuracy
2. **Feature Engineering Impact**: [Improved/No significant change] model performance
3. **Best Configuration**: SVM with [Kernel] kernel and C=[Value]
4. **Generalization**: [Model] shows better cross-validation to test accuracy ratio

### 8.2 Recommendations

1. **For Deployment**: Use [Model 1/2] based on [accuracy/computational efficiency/interpretability]
2. **Further Improvements**:
   - Experiment with additional feature engineering techniques
   - Test different scaling methods (StandardScaler, MinMaxScaler)
   - Implement ensemble methods (voting, stacking) combining both models
   - Perform class weight balancing for imbalanced datasets

3. **Domain Considerations**:
   - False Positives (approving bad loans) vs. False Negatives (rejecting good loans)
   - Adjust decision threshold based on business requirements
   - Validate with domain experts

---

## 9. References

1. **Support Vector Machines**: Vapnik, V. (1995). The Nature of Statistical Learning Theory
2. **RobustScaler**: Sklearn documentation on preprocessing scalers
3. **GridSearchCV**: Sklearn hyper-parameter optimization
4. **SVC Implementation**: Scikit-learn SVC classifier documentation

---

## 10. Appendix: Code Execution Details

### Key Libraries Used:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms
- **matplotlib/seaborn**: Data visualization

### Important Steps:
1. Load and clean data (missing value imputation)
2. Drop categorical features for numerical analysis
3. Feature scaling with RobustScaler
4. Exhaustive hyperparameter search (GridSearchCV)
5. Model comparison and analysis

### Random State:
- Model 1: random_state=3453 (reproducibility)
- Model 2: random_state=3422 (independent random seed)

---

**Report Generated**: March 29, 2026  
**Dataset**: Loan Status Classification  
**Models**: Support Vector Machine (SVM) with and without Feature Engineering  
**Best Performing Model**: [Model 1/2 with Accuracy: XX.XX%]