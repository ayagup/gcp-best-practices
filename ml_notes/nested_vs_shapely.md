ayagup: nested vs random cross validation

GitHub Copilot: # Nested vs Random Cross-Validation

## Random Cross-Validation
- **Purpose**: Evaluate model performance
- **Structure**: Single level of cross-validation
- **Process**: 
  - Split data into K folds randomly
  - Train on K-1 folds, test on 1 fold
  - Rotate through all folds
- **Output**: Average performance metric
- **Use Case**: When hyperparameters are already optimized or fixed
- **Risk**: Can lead to **optimistic bias** if used for both hyperparameter tuning and performance evaluation

## Nested Cross-Validation
- **Purpose**: Unbiased model performance estimation with hyperparameter tuning
- **Structure**: Two levels of cross-validation
- **Process**:
  - **Outer loop**: Splits data into K folds for performance evaluation
  - **Inner loop**: For each outer fold, performs cross-validation on training data for hyperparameter tuning
- **Output**: Unbiased estimate of model's generalization performance
- **Use Case**: When you need to tune hyperparameters AND get an honest performance estimate

## Key Differences

| Aspect | Random CV | Nested CV |
|--------|-----------|-----------|
| **Levels** | Single | Two (inner + outer) |
| **Hyperparameter Tuning** | No | Yes (inner loop) |
| **Performance Bias** | Can be optimistic | Unbiased |
| **Computational Cost** | Lower | Higher (K × M folds) |
| **Final Model** | Uses all data | Uses all data (after CV) |

## Example Structure

````python
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Random Cross-Validation (Simple CV)
model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5)
print(f"Random CV Score: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Nested Cross-Validation
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

nested_scores = []
for train_idx, test_idx in outer_cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Inner loop: hyperparameter tuning
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=inner_cv,
        scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)
    
    # Outer loop: performance evaluation
    score = grid_search.score(X_test, y_test)
    nested_scores.append(score)

print(f"Nested CV Score: {np.mean(nested_scores):.3f} (+/- {np.std(nested_scores):.3f})")
````

## When to Use Each

**Use Random CV when:**
- You have fixed hyperparameters
- Quick performance estimate needed
- Comparing models with same hyperparameters

**Use Nested CV when:**
- Tuning hyperparameters
- Need unbiased performance estimate
- Publishing research results
- Small to medium datasets where overfitting is a concern