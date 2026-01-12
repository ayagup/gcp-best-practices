# Comprehensive Guide to Model Evaluation Metrics

## Overview

This guide provides an exhaustive reference for model evaluation metrics across all major machine learning tasks. Understanding when and how to use each metric is crucial for properly assessing model performance on Google Cloud and in production environments.

## Table of Contents

1. [Classification Metrics](#1-classification-metrics)
2. [Regression Metrics](#2-regression-metrics)
3. [Ranking Metrics](#3-ranking-metrics)
4. [Clustering Metrics](#4-clustering-metrics)
5. [Object Detection Metrics](#5-object-detection-metrics)
6. [Segmentation Metrics](#6-segmentation-metrics)
7. [NLP Metrics](#7-nlp-metrics)
8. [Generative Model Metrics](#8-generative-model-metrics)
9. [Recommendation System Metrics](#9-recommendation-system-metrics)
10. [Time Series Metrics](#10-time-series-metrics)
11. [Fairness and Bias Metrics](#11-fairness-and-bias-metrics)

---

## 1. Classification Metrics

### 1.1 Binary Classification Metrics

```python
import numpy as np
from sklearn.metrics import *
from typing import Dict, Any, List
import tensorflow as tf

class BinaryClassificationMetrics:
    """Comprehensive binary classification metrics."""
    
    def __init__(self):
        """Initialize Binary Classification Metrics."""
        pass
    
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Compute all binary classification metrics.
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Confusion Matrix Based Metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # 1. Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # 2. Precision (Positive Predictive Value)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        
        # 3. Recall (Sensitivity, True Positive Rate)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['sensitivity'] = metrics['recall']  # Alias
        metrics['tpr'] = metrics['recall']  # Alias
        
        # 4. Specificity (True Negative Rate)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['tnr'] = metrics['specificity']  # Alias
        
        # 5. F1 Score (Harmonic mean of precision and recall)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # 6. F-beta Scores
        metrics['f2_score'] = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        metrics['f0.5_score'] = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        
        # 7. Matthews Correlation Coefficient
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # 8. Cohen's Kappa
        metrics['cohens_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # 9. Balanced Accuracy
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # 10. False Positive Rate
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # 11. False Negative Rate
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # 12. False Discovery Rate
        metrics['fdr'] = fp / (fp + tp) if (fp + tp) > 0 else 0
        
        # 13. Negative Predictive Value
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # 14. Positive Likelihood Ratio
        if metrics['fpr'] > 0:
            metrics['positive_likelihood_ratio'] = metrics['tpr'] / metrics['fpr']
        else:
            metrics['positive_likelihood_ratio'] = float('inf')
        
        # 15. Negative Likelihood Ratio
        if metrics['tnr'] > 0:
            metrics['negative_likelihood_ratio'] = metrics['fnr'] / metrics['tnr']
        else:
            metrics['negative_likelihood_ratio'] = 0
        
        # 16. Diagnostic Odds Ratio
        if metrics['negative_likelihood_ratio'] > 0:
            metrics['diagnostic_odds_ratio'] = (
                metrics['positive_likelihood_ratio'] / metrics['negative_likelihood_ratio']
            )
        else:
            metrics['diagnostic_odds_ratio'] = float('inf')
        
        # Probability-based metrics (if probabilities provided)
        if y_pred_proba is not None:
            # 17. ROC AUC
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
            # 18. PR AUC (Precision-Recall AUC)
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_auc'] = auc(recall_vals, precision_vals)
            
            # 19. Average Precision
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
            
            # 20. Log Loss (Binary Cross-Entropy)
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            
            # 21. Brier Score
            metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
        
        return metrics
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all metrics.
        
        Returns:
            Dictionary with metric descriptions
        """
        return {
            'accuracy': 'Proportion of correct predictions. Range: [0, 1]',
            'precision': 'TP / (TP + FP). Proportion of positive predictions that are correct',
            'recall': 'TP / (TP + FN). Proportion of actual positives correctly identified',
            'specificity': 'TN / (TN + FP). Proportion of actual negatives correctly identified',
            'f1_score': 'Harmonic mean of precision and recall. Range: [0, 1]',
            'f2_score': 'F-beta with beta=2 (weights recall higher than precision)',
            'f0.5_score': 'F-beta with beta=0.5 (weights precision higher than recall)',
            'mcc': 'Matthews Correlation Coefficient. Range: [-1, 1]. Balanced metric',
            'cohens_kappa': 'Agreement between predictions and truth. Range: [-1, 1]',
            'balanced_accuracy': 'Average of recall for each class. Good for imbalanced data',
            'roc_auc': 'Area under ROC curve. Measures probability ranking. Range: [0, 1]',
            'pr_auc': 'Area under Precision-Recall curve. Better for imbalanced data',
            'log_loss': 'Logarithmic loss. Lower is better. Range: [0, ∞)',
            'brier_score': 'Mean squared difference between predicted and actual. Range: [0, 1]'
        }


# Example usage
binary_metrics = BinaryClassificationMetrics()

# Sample data
y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
y_pred_proba = np.array([0.1, 0.9, 0.8, 0.3, 0.4, 0.95, 0.2, 0.6, 0.85, 0.15])

# Compute all metrics
metrics = binary_metrics.compute_all_metrics(y_true, y_pred, y_pred_proba)

print("Binary Classification Metrics:")
print(f"  Accuracy: {metrics['accuracy']:.4f}")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall: {metrics['recall']:.4f}")
print(f"  F1 Score: {metrics['f1_score']:.4f}")
print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
print(f"  PR AUC: {metrics['pr_auc']:.4f}")
print(f"  MCC: {metrics['mcc']:.4f}")
```

#### Metric Analysis: Binary Classification

| Metric | Definition | Pros | Cons | When to Use | When NOT to Use |
|--------|------------|------|------|-------------|-----------------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN)<br>Proportion of correct predictions | ✅ Simple to understand<br>✅ Single number summary<br>✅ Intuitive | ❌ Misleading with imbalanced data<br>❌ Doesn't distinguish error types<br>❌ Can be high with poor model | Balanced datasets<br>Equal cost for errors<br>Quick model comparison | Imbalanced datasets<br>Different error costs<br>Medical/fraud detection |
| **Precision** | TP / (TP + FP)<br>Proportion of positive predictions that are correct | ✅ Measures false positive rate<br>✅ Good for spam detection<br>✅ Focuses on positive predictions | ❌ Ignores false negatives<br>❌ Can be gamed (predict less)<br>❌ Unstable with few positives | Minimizing false positives critical<br>Spam/fraud detection<br>High confidence needed | False negatives costly<br>Need complete recall<br>Medical screening |
| **Recall** | TP / (TP + FN)<br>Proportion of actual positives correctly identified (Sensitivity, TPR) | ✅ Measures false negative rate<br>✅ Good for disease detection<br>✅ Captures all positives | ❌ Ignores false positives<br>❌ Can be 100% by predicting all<br>❌ May flood with false alarms | Catching all positives critical<br>Medical diagnosis<br>Security threats | False positives costly<br>Limited resources<br>Precision matters more |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall)<br>Harmonic mean of precision and recall | ✅ Balances precision/recall<br>✅ Single metric for both<br>✅ Robust to imbalance | ❌ Equal weight to P and R<br>❌ May not match business goal<br>❌ Less interpretable | Need balance of P and R<br>General classification<br>No clear priority | Different P/R importance<br>Need to tune threshold<br>Extreme imbalance |
| **ROC AUC** | Area under ROC curve (TPR vs FPR)<br>Probability that model ranks random positive higher than random negative | ✅ Threshold independent<br>✅ Robust to class imbalance<br>✅ Probability ranking quality | ❌ Optimistic for imbalanced data<br>❌ May not reflect real performance<br>❌ Ignores calibration | Comparing models<br>Threshold will vary<br>Balanced or moderate imbalance | Extreme imbalance<br>Care about calibration<br>Fixed threshold needed |
| **PR AUC** | Area under Precision-Recall curve<br>Average precision across all recall levels | ✅ Better for imbalanced data<br>✅ Focuses on positive class<br>✅ More realistic than ROC | ❌ Harder to interpret<br>❌ Baseline varies by dataset<br>❌ Less common | Imbalanced datasets<br>Positive class important<br>Fraud/anomaly detection | Balanced datasets<br>Both classes equally important<br>Need interpretability |
| **MCC** | (TP×TN - FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]<br>Correlation coefficient between predicted and actual | ✅ Accounts for all confusion matrix<br>✅ Balanced for imbalanced data<br>✅ Correlates well with true performance | ❌ Less intuitive<br>❌ Range [-1, 1] confusing<br>❌ Not widely known | Imbalanced datasets<br>Need single balanced metric<br>Comparing diverse models | Need interpretable metric<br>Stakeholder communication<br>Specific error focus |
| **Log Loss** | -[y log(p) + (1-y) log(1-p)]<br>Negative log-likelihood of predictions (Binary Cross-Entropy) | ✅ Measures probability quality<br>✅ Penalizes confident wrong predictions<br>✅ Differentiable | ❌ Hard to interpret<br>❌ Sensitive to outliers<br>❌ Unbounded above | Probability calibration matters<br>Ranking/scoring<br>Model optimization | Only care about labels<br>Need interpretability<br>Probabilities unreliable |
| **Brier Score** | Mean((predicted_prob - actual)²)<br>Mean squared error of predicted probabilities | ✅ MSE of probabilities<br>✅ Proper scoring rule<br>✅ Bounded [0, 1] | ❌ Less common<br>❌ Harder to interpret<br>❌ Less literature | Probability calibration critical<br>Weather forecasting<br>Risk assessment | Only care about classifications<br>Need interpretability<br>Common metric needed |

### 1.2 Multi-class Classification Metrics

```python
class MulticlassMetrics:
    """Comprehensive multi-class classification metrics."""
    
    def __init__(self):
        """Initialize Multiclass Metrics."""
        pass
    
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None,
        labels: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compute all multi-class classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            labels: Class labels (optional)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # 1. Overall Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # 2. Balanced Accuracy
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # 3. Top-k Accuracy (if probabilities provided)
        if y_pred_proba is not None:
            for k in [1, 3, 5]:
                if y_pred_proba.shape[1] >= k:
                    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
                    top_k_acc = np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
                    metrics[f'top_{k}_accuracy'] = top_k_acc
        
        # 4. Macro-averaged metrics (equal weight to each class)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # 5. Weighted-averaged metrics (weighted by class support)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 6. Micro-averaged metrics (aggregate contributions of all classes)
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # 7. Per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class_precision'] = per_class_precision.tolist()
        metrics['per_class_recall'] = per_class_recall.tolist()
        metrics['per_class_f1'] = per_class_f1.tolist()
        
        # 8. Cohen's Kappa
        metrics['cohens_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # 9. Matthews Correlation Coefficient
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # Probability-based metrics
        if y_pred_proba is not None:
            # 10. Log Loss (Categorical Cross-Entropy)
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            
            # 11. Multi-class ROC AUC (one-vs-rest)
            try:
                metrics['roc_auc_ovr'] = roc_auc_score(
                    y_true, y_pred_proba, multi_class='ovr', average='macro'
                )
            except ValueError:
                metrics['roc_auc_ovr'] = None
            
            # 12. Multi-class ROC AUC (one-vs-one)
            try:
                metrics['roc_auc_ovo'] = roc_auc_score(
                    y_true, y_pred_proba, multi_class='ovo', average='macro'
                )
            except ValueError:
                metrics['roc_auc_ovo'] = None
        
        # 13. Confusion Matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics


# Example usage
multiclass_metrics = MulticlassMetrics()

# Sample data (3 classes)
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1, 0])
y_pred_proba = np.array([
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8],
    [0.7, 0.2, 0.1],
    [0.2, 0.3, 0.5],
    [0.1, 0.2, 0.7],
    [0.9, 0.05, 0.05],
    [0.2, 0.7, 0.1],
    [0.3, 0.4, 0.3],
    [0.85, 0.1, 0.05]
])

metrics = multiclass_metrics.compute_all_metrics(y_true, y_pred, y_pred_proba)

print("\nMulti-class Classification Metrics:")
print(f"  Accuracy: {metrics['accuracy']:.4f}")
print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
print(f"  Top-3 Accuracy: {metrics.get('top_3_accuracy', 'N/A'):.4f}")
print(f"  Log Loss: {metrics['log_loss']:.4f}")
```

#### Metric Analysis: Multi-class Classification

| Metric | Definition | Pros | Cons | When to Use | When NOT to Use |
|--------|------------|------|------|-------------|-----------------|
| **Accuracy** | Correct predictions / Total predictions<br>Proportion of correctly classified samples | ✅ Simple and intuitive<br>✅ Single number<br>✅ Easy to communicate | ❌ Misleading with imbalance<br>❌ Treats all errors equally<br>❌ Ignores confidence | Balanced multi-class<br>Equal error costs<br>Quick baseline | Imbalanced classes<br>Different error costs<br>Need per-class insight |
| **Macro F1** | Average of per-class F1 scores<br>Unweighted mean F1 across all classes | ✅ Treats all classes equally<br>✅ Good for imbalanced data<br>✅ Highlights poor minority performance | ❌ May not reflect overall performance<br>❌ Weights rare classes heavily<br>❌ Can be low despite good majority | All classes equally important<br>Minority class matters<br>Medical diagnosis | Rare classes unimportant<br>Focus on common classes<br>Population-level metrics |
| **Weighted F1** | Weighted average of per-class F1<br>F1 scores weighted by class support | ✅ Reflects class distribution<br>✅ More realistic than macro<br>✅ Balances class importance | ❌ Dominated by majority classes<br>❌ May hide minority issues<br>❌ Sensitive to distribution | Real-world performance<br>Preserve class distribution<br>Production metrics | All classes equally critical<br>Minority class focus<br>Imbalance correction |
| **Micro F1** | F1 computed globally<br>Aggregate TP, FP, FN across classes then compute F1 | ✅ Overall performance across all<br>✅ Treats each instance equally<br>✅ Same as accuracy for multi-class | ❌ Dominated by frequent classes<br>❌ Identical to accuracy<br>❌ Less informative | Instance-level performance<br>Total error rate<br>Micro-level decisions | Class-level insights needed<br>Different than accuracy wanted<br>Per-class important |
| **Top-k Accuracy** | Correct if true label in top k predictions<br>Proportion where true class in top k ranked outputs | ✅ More forgiving metric<br>✅ Realistic for recommendations<br>✅ Useful for large # classes | ❌ Less strict than top-1<br>❌ May hide poor ranking<br>❌ Harder to interpret | Large number of classes<br>Recommendation systems<br>Search/retrieval | Few classes<br>Need exact prediction<br>Single answer required |
| **Log Loss** | -Σ y_true × log(y_pred)<br>Categorical cross-entropy across all classes | ✅ Penalizes confident errors<br>✅ Measures calibration<br>✅ Smooth and differentiable | ❌ Hard to interpret<br>❌ Unbounded above<br>❌ Sensitive to outliers | Probability quality matters<br>Well-calibrated models<br>Ranking applications | Only care about labels<br>Need interpretability<br>Probabilities not used |
| **ROC AUC OvR** | Average of per-class ROC AUC<br>One-vs-Rest AUC for each class then averaged | ✅ Handles multi-class<br>✅ Threshold independent<br>✅ Per-class discrimination | ❌ Complex to interpret<br>❌ May be optimistic<br>❌ Requires probabilities | Model comparison<br>Variable thresholds<br>Discrimination focus | Extreme imbalance<br>Fixed threshold<br>Need simplicity |

### 1.3 Multi-label Classification Metrics

```python
class MultilabelMetrics:
    """Comprehensive multi-label classification metrics."""
    
    def __init__(self):
        """Initialize Multilabel Metrics."""
        pass
    
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Compute all multi-label classification metrics.
        
        Args:
            y_true: True binary labels (n_samples, n_labels)
            y_pred: Predicted binary labels (n_samples, n_labels)
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # 1. Exact Match Ratio (Subset Accuracy)
        metrics['exact_match_ratio'] = accuracy_score(y_true, y_pred)
        
        # 2. Hamming Loss
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
        
        # 3. Hamming Score (1 - Hamming Loss)
        metrics['hamming_score'] = 1 - metrics['hamming_loss']
        
        # 4. Jaccard Score (Intersection over Union)
        metrics['jaccard_score_samples'] = jaccard_score(y_true, y_pred, average='samples')
        metrics['jaccard_score_macro'] = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['jaccard_score_weighted'] = jaccard_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 5. Precision, Recall, F1 (samples average)
        metrics['precision_samples'] = precision_score(y_true, y_pred, average='samples', zero_division=0)
        metrics['recall_samples'] = recall_score(y_true, y_pred, average='samples', zero_division=0)
        metrics['f1_samples'] = f1_score(y_true, y_pred, average='samples', zero_division=0)
        
        # 6. Precision, Recall, F1 (macro average)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # 7. Precision, Recall, F1 (micro average)
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # 8. Precision, Recall, F1 (weighted average)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Probability-based metrics
        if y_pred_proba is not None:
            # 9. Coverage Error
            metrics['coverage_error'] = coverage_error(y_true, y_pred_proba)
            
            # 10. Label Ranking Average Precision
            metrics['label_ranking_avg_precision'] = label_ranking_average_precision_score(y_true, y_pred_proba)
            
            # 11. Ranking Loss
            metrics['ranking_loss'] = label_ranking_loss(y_true, y_pred_proba)
            
            # 12. ROC AUC (samples average)
            try:
                metrics['roc_auc_samples'] = roc_auc_score(y_true, y_pred_proba, average='samples')
            except ValueError:
                metrics['roc_auc_samples'] = None
            
            # 13. ROC AUC (macro average)
            try:
                metrics['roc_auc_macro'] = roc_auc_score(y_true, y_pred_proba, average='macro')
            except ValueError:
                metrics['roc_auc_macro'] = None
        
        return metrics


# Example usage
multilabel_metrics = MultilabelMetrics()

# Sample data (multi-label: each sample can have multiple labels)
y_true = np.array([
    [1, 0, 1, 0],
    [0, 1, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1]
])
y_pred = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 1]
])
y_pred_proba = np.array([
    [0.9, 0.1, 0.8, 0.2],
    [0.3, 0.7, 0.4, 0.1],
    [0.85, 0.9, 0.2, 0.1],
    [0.1, 0.6, 0.8, 0.7],
    [0.95, 0.2, 0.1, 0.85]
])

metrics = multilabel_metrics.compute_all_metrics(y_true, y_pred, y_pred_proba)

print("\nMulti-label Classification Metrics:")
print(f"  Exact Match Ratio: {metrics['exact_match_ratio']:.4f}")
print(f"  Hamming Score: {metrics['hamming_score']:.4f}")
print(f"  F1 Micro: {metrics['f1_micro']:.4f}")
print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
print(f"  Jaccard Score (samples): {metrics['jaccard_score_samples']:.4f}")
```

#### Metric Analysis: Multi-label Classification

| Metric | Definition | Pros | Cons | When to Use | When NOT to Use |
|--------|------------|------|------|-------------|-----------------|
| **Exact Match Ratio** | Proportion of samples with all labels correct<br>Subset Accuracy: exact label set match | ✅ Strict evaluation<br>✅ Clear success criteria<br>✅ Easy to understand | ❌ Very strict (all must match)<br>❌ Often very low<br>❌ Ignores partial success | Perfect prediction required<br>Small label sets<br>High-stakes decisions | Large label sets<br>Partial credit acceptable<br>Learning/evaluation phase |
| **Hamming Loss** | Fraction of incorrect labels<br>Average proportion of wrong labels per sample | ✅ Label-wise accuracy<br>✅ Interpretable as error rate<br>✅ Considers all labels | ❌ Doesn't consider label correlation<br>❌ Can be misleading<br>❌ Equal weight to all labels | Independent labels<br>Error rate focus<br>Label-level performance | Correlated labels<br>Sample-level needed<br>Unequal label importance |
| **Jaccard Score** | |Predicted ∩ True| / |Predicted ∪ True|<br>Intersection over Union of label sets | ✅ Considers intersection and union<br>✅ Handles variable set sizes<br>✅ Set similarity metric | ❌ Less interpretable<br>❌ Sensitive to label count<br>❌ Range [0, 1] but varies | Tag/category prediction<br>Set similarity important<br>Flexible label count | Exact matches needed<br>Fixed label count<br>Need simple metric |
| **F1 Samples** | Average F1 score per sample<br>Compute F1 for each sample then average | ✅ Per-sample F1 average<br>✅ Balances precision/recall<br>✅ Sample-level focus | ❌ Sensitive to label count<br>❌ May be optimistic<br>❌ Complex to interpret | Sample-level performance<br>Variable labels per sample<br>Balanced P/R needed | Label-level focus<br>Fixed label count<br>Global metrics preferred |
| **F1 Macro** | Average F1 across all labels<br>Unweighted mean of per-label F1 scores | ✅ Equal weight per label<br>✅ Good for imbalanced labels<br>✅ Highlights rare labels | ❌ Ignores label frequency<br>❌ May not reflect real performance<br>❌ Dominated by rare labels | All labels equally important<br>Rare label focus<br>Fair evaluation | Frequent labels more important<br>Real-world distribution<br>Production metrics |
| **Coverage Error** | Average rank of true labels<br>Average number of top-ranked labels needed to cover all true labels | ✅ Measures ranking quality<br>✅ Lower is better<br>✅ Considers all relevant | ❌ Hard to interpret<br>❌ Requires probabilities<br>❌ Not widely used | Ranking quality matters<br>All relevant items needed<br>Threshold optimization | Only care about binary<br>No probabilities<br>Need common metric |
| **Ranking Loss** | Fraction of label pairs incorrectly ordered<br>Proportion of (relevant, irrelevant) pairs with relevant ranked lower | ✅ Measures label ordering<br>✅ Ranking-based<br>✅ Threshold independent | ❌ Complex to interpret<br>❌ Less intuitive<br>❌ Not standard | Label ranking important<br>Relative ordering matters<br>Multiple thresholds | Binary decisions only<br>No ranking needed<br>Interpretability critical |

---

## 2. Regression Metrics

```python
class RegressionMetrics:
    """Comprehensive regression metrics."""
    
    def __init__(self):
        """Initialize Regression Metrics."""
        pass
    
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # 1. Mean Absolute Error (MAE / L1 Loss)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # 2. Mean Squared Error (MSE / L2 Loss)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        
        # 3. Root Mean Squared Error (RMSE)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # 4. Mean Absolute Percentage Error (MAPE)
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['mape'] = None
        
        # 5. Symmetric Mean Absolute Percentage Error (SMAPE)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if mask.sum() > 0:
            metrics['smape'] = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        else:
            metrics['smape'] = None
        
        # 6. R² Score (Coefficient of Determination)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # 7. Adjusted R²
        n = len(y_true)
        p = 1  # number of features (assuming 1 for simplicity)
        if n > p + 1:
            metrics['adjusted_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)
        else:
            metrics['adjusted_r2'] = None
        
        # 8. Explained Variance Score
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
        
        # 9. Max Error
        metrics['max_error'] = max_error(y_true, y_pred)
        
        # 10. Mean Squared Log Error (MSLE)
        # Only for non-negative values
        if np.all(y_true >= 0) and np.all(y_pred >= 0):
            metrics['msle'] = mean_squared_log_error(y_true, y_pred)
            metrics['rmsle'] = np.sqrt(metrics['msle'])
        else:
            metrics['msle'] = None
            metrics['rmsle'] = None
        
        # 11. Median Absolute Error
        metrics['median_ae'] = median_absolute_error(y_true, y_pred)
        
        # 12. Mean Poisson Deviance
        metrics['mean_poisson_deviance'] = mean_poisson_deviance(y_true, y_pred)
        
        # 13. Mean Gamma Deviance
        metrics['mean_gamma_deviance'] = mean_gamma_deviance(y_true, y_pred)
        
        # 14. Mean Tweedie Deviance
        metrics['mean_tweedie_deviance'] = mean_tweedie_deviance(y_true, y_pred)
        
        # 15. Mean Absolute Scaled Error (MASE) - requires baseline
        # Using naive forecast (previous value) as baseline
        if len(y_true) > 1:
            naive_forecast_mae = np.mean(np.abs(np.diff(y_true)))
            if naive_forecast_mae != 0:
                metrics['mase'] = metrics['mae'] / naive_forecast_mae
            else:
                metrics['mase'] = None
        else:
            metrics['mase'] = None
        
        # 16. Quantile Loss (for different quantiles)
        for q in [0.1, 0.5, 0.9]:
            residual = y_true - y_pred
            metrics[f'quantile_loss_{q}'] = np.mean(
                np.maximum(q * residual, (q - 1) * residual)
            )
        
        # 17. Huber Loss
        delta = 1.0
        residual = y_true - y_pred
        is_small_error = np.abs(residual) <= delta
        squared_loss = 0.5 * residual ** 2
        linear_loss = delta * (np.abs(residual) - 0.5 * delta)
        metrics['huber_loss'] = np.mean(
            np.where(is_small_error, squared_loss, linear_loss)
        )
        
        # 18. Log-Cosh Loss
        metrics['log_cosh_loss'] = np.mean(np.log(np.cosh(y_pred - y_true)))
        
        # 19. Coefficient of Variation (CV) of RMSE
        mean_y_true = np.mean(y_true)
        if mean_y_true != 0:
            metrics['cv_rmse'] = (metrics['rmse'] / mean_y_true) * 100
        else:
            metrics['cv_rmse'] = None
        
        # 20. Normalized RMSE (by range)
        y_range = np.max(y_true) - np.min(y_true)
        if y_range != 0:
            metrics['nrmse_range'] = metrics['rmse'] / y_range
        else:
            metrics['nrmse_range'] = None
        
        # 21. Normalized RMSE (by mean)
        if mean_y_true != 0:
            metrics['nrmse_mean'] = metrics['rmse'] / mean_y_true
        else:
            metrics['nrmse_mean'] = None
        
        # 22. Mean Directional Accuracy (for time series)
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            metrics['mean_directional_accuracy'] = np.mean(true_direction == pred_direction)
        else:
            metrics['mean_directional_accuracy'] = None
        
        return metrics
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all regression metrics."""
        return {
            'mae': 'Mean Absolute Error. Average absolute difference. Same units as target',
            'mse': 'Mean Squared Error. Average squared difference. Penalizes large errors',
            'rmse': 'Root Mean Squared Error. Square root of MSE. Same units as target',
            'mape': 'Mean Absolute Percentage Error. Average percentage error. Scale-independent',
            'smape': 'Symmetric MAPE. More balanced version of MAPE',
            'r2': 'R-squared. Proportion of variance explained. Range: (-∞, 1]',
            'adjusted_r2': 'Adjusted R². R² adjusted for number of features',
            'explained_variance': 'Explained variance. Similar to R² but different formula',
            'max_error': 'Maximum absolute error. Worst-case prediction error',
            'msle': 'Mean Squared Log Error. Good for targets with exponential growth',
            'rmsle': 'Root MSLE. Square root of MSLE',
            'median_ae': 'Median Absolute Error. Robust to outliers',
            'huber_loss': 'Huber Loss. Combines MSE and MAE. Robust to outliers',
            'mase': 'Mean Absolute Scaled Error. Error relative to naive baseline'
        }


# Example usage
regression_metrics = RegressionMetrics()

# Sample data
y_true = np.array([3.0, -0.5, 2.0, 7.0, 4.2, 5.1, 2.8, 3.9, 6.5, 4.7])
y_pred = np.array([2.5, 0.0, 2.1, 7.8, 4.0, 5.5, 2.3, 4.2, 6.8, 4.5])

metrics = regression_metrics.compute_all_metrics(y_true, y_pred)

print("\nRegression Metrics:")
print(f"  MAE: {metrics['mae']:.4f}")
print(f"  RMSE: {metrics['rmse']:.4f}")
print(f"  R²: {metrics['r2']:.4f}")
print(f"  MAPE: {metrics['mape']:.2f}%")
print(f"  Median AE: {metrics['median_ae']:.4f}")
print(f"  Max Error: {metrics['max_error']:.4f}")
```

#### Metric Analysis: Regression

| Metric | Definition | Pros | Cons | When to Use | When NOT to Use |
|--------|------------|------|------|-------------|-----------------|
| **MAE** | Mean(|y_true - y_pred|)<br>Average absolute difference between predictions and actuals | ✅ Easy to interpret (same units)<br>✅ Robust to outliers<br>✅ Linear penalty | ❌ Doesn't penalize large errors heavily<br>❌ Not differentiable at zero<br>❌ May ignore worst cases | Outliers present<br>All errors equal weight<br>Interpretability important | Large errors very bad<br>Need smooth gradients<br>Emphasize worst cases |
| **MSE** | Mean((y_true - y_pred)²)<br>Average squared difference between predictions and actuals | ✅ Penalizes large errors<br>✅ Differentiable everywhere<br>✅ Mathematical properties | ❌ Not in original units<br>❌ Sensitive to outliers<br>❌ Hard to interpret | Large errors costly<br>Optimization focus<br>No outliers | Outliers present<br>Need interpretability<br>Scale-independent metric |
| **RMSE** | √MSE<br>Square root of Mean Squared Error | ✅ Same units as target<br>✅ Penalizes large errors<br>✅ More interpretable than MSE | ❌ Sensitive to outliers<br>❌ Harder to optimize<br>❌ Scale dependent | Same as MSE but need units<br>Standard reporting<br>Interpretability + large error penalty | Outliers present<br>Different scales compared<br>Robust metric needed |
| **MAPE** | Mean(|y_true - y_pred| / |y_true|) × 100<br>Average percentage error | ✅ Scale independent (%)<br>✅ Easy to interpret<br>✅ Intuitive percentage | ❌ Undefined for zero values<br>❌ Asymmetric (penalizes over-prediction)<br>❌ Skewed by small denominators | Percentage error natural<br>Cross-dataset comparison<br>Business reporting | Zero values present<br>Need symmetry<br>Small values in target |
| **SMAPE** | Mean(2×|y_true - y_pred| / (|y_true| + |y_pred|)) × 100<br>Symmetric version of MAPE | ✅ More symmetric than MAPE<br>✅ Bounded [0, 200]<br>✅ Handles zeros better | ❌ Still has issues with zeros<br>❌ Multiple definitions exist<br>❌ Less common | MAPE problems present<br>Need symmetry<br>Zero values possible | Standard MAPE works<br>Consistency needed<br>Simple interpretation |
| **R²** | 1 - (SS_residual / SS_total)<br>Proportion of variance in target explained by model | ✅ Scale independent<br>✅ Proportion of variance explained<br>✅ Widely understood | ❌ Can be negative<br>❌ Doesn't show error magnitude<br>❌ Misleading with non-linear | Model comparison<br>Goodness of fit<br>Linear relationships | Non-linear models<br>Need error magnitude<br>Can be gamed | 
| **Adjusted R²** | 1 - ((1-R²)(n-1) / (n-p-1))<br>R² adjusted for number of predictors | ✅ Accounts for # features<br>✅ Prevents overfitting<br>✅ Fair comparison | ❌ Still has R² issues<br>❌ More complex<br>❌ Less intuitive | Multiple features<br>Feature selection<br>Model complexity matters | Single feature<br>R² sufficient<br>Simplicity preferred |
| **Median AE** | Median(|y_true - y_pred|)<br>50th percentile of absolute errors | ✅ Very robust to outliers<br>✅ Interpretable<br>✅ Central tendency focus | ❌ Ignores many predictions<br>❌ Less sensitive<br>❌ Hard to optimize | Outliers dominant<br>Robust metric needed<br>Central performance | All predictions matter<br>Need sensitivity<br>Smooth optimization |
| **Max Error** | Max(|y_true - y_pred|)<br>Largest absolute error | ✅ Worst-case measure<br>✅ Safety critical<br>✅ Clear interpretation | ❌ Ignores most predictions<br>❌ Very sensitive to outliers<br>❌ Unstable | Worst case critical<br>Safety/reliability focus<br>Identify extremes | Typical performance matters<br>Outliers expected<br>Need stability |
| **MSLE** | Mean((log(1+y_true) - log(1+y_pred))²)<br>MSE in log space | ✅ Good for exponential growth<br>✅ Penalizes under-prediction<br>✅ Relative errors | ❌ Only for non-negative<br>❌ Asymmetric<br>❌ Less interpretable | Exponential targets<br>Under-prediction worse<br>Orders of magnitude vary | Negative values<br>Need symmetry<br>Linear scale sufficient |
| **Huber Loss** | MSE if |error| ≤ δ, else MAE<br>Quadratic for small errors, linear for large | ✅ Robust to outliers<br>✅ Differentiable<br>✅ Best of MAE and MSE | ❌ Extra hyperparameter (delta)<br>❌ More complex<br>❌ Less standard | Outliers present but matter<br>Need optimization<br>Balance robustness and sensitivity | No outliers<br>Simple metric needed<br>Standard reporting |
| **Quantile Loss** | q×(y-ŷ) if y≥ŷ, (q-1)×(y-ŷ) if y<ŷ<br>Asymmetric loss for specific quantiles | ✅ Asymmetric penalties<br>✅ Confidence intervals<br>✅ Flexible | ❌ Complex to interpret<br>❌ Need multiple quantiles<br>❌ Less common | Confidence bounds needed<br>Asymmetric costs<br>Risk assessment | Point estimates only<br>Symmetric costs<br>Simple metric needed |
| **MASE** | MAE / MAE_naive_forecast<br>MAE relative to naive baseline forecast | ✅ Scale independent<br>✅ Compared to baseline<br>✅ Time series standard | ❌ Requires baseline<br>❌ Undefined for constant series<br>❌ Less known outside TS | Time series<br>Need baseline comparison<br>Cross-series comparison | No baseline available<br>Not time series<br>Standard metrics work |

---

## 3. Ranking Metrics

```python
class RankingMetrics:
    """Comprehensive ranking and information retrieval metrics."""
    
    def __init__(self):
        """Initialize Ranking Metrics."""
        pass
    
    def compute_all_metrics(
        self,
        y_true: List[List[int]],
        y_pred_scores: List[List[float]],
        k_values: List[int] = [1, 3, 5, 10, 20]
    ) -> Dict[str, Any]:
        """
        Compute all ranking metrics.
        
        Args:
            y_true: List of relevant items (binary relevance) for each query
            y_pred_scores: List of predicted scores for each query
            k_values: K values for @k metrics
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        for k in k_values:
            metrics[f'precision@{k}'] = []
            metrics[f'recall@{k}'] = []
            metrics[f'f1@{k}'] = []
            metrics[f'map@{k}'] = []
            metrics[f'ndcg@{k}'] = []
            metrics[f'mrr@{k}'] = []
        
        # Compute per-query metrics
        for query_true, query_scores in zip(y_true, y_pred_scores):
            # Sort by scores (descending)
            sorted_indices = np.argsort(query_scores)[::-1]
            
            for k in k_values:
                # Get top k predictions
                top_k_indices = sorted_indices[:k]
                top_k_relevant = [query_true[i] if i < len(query_true) else 0 
                                 for i in top_k_indices]
                
                # 1. Precision@k
                precision_k = sum(top_k_relevant) / k if k > 0 else 0
                metrics[f'precision@{k}'].append(precision_k)
                
                # 2. Recall@k
                total_relevant = sum(query_true)
                recall_k = sum(top_k_relevant) / total_relevant if total_relevant > 0 else 0
                metrics[f'recall@{k}'].append(recall_k)
                
                # 3. F1@k
                if precision_k + recall_k > 0:
                    f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
                else:
                    f1_k = 0
                metrics[f'f1@{k}'].append(f1_k)
                
                # 4. Average Precision@k (AP@k)
                ap_k = 0
                num_relevant = 0
                for i in range(min(k, len(top_k_indices))):
                    if top_k_relevant[i]:
                        num_relevant += 1
                        ap_k += num_relevant / (i + 1)
                ap_k = ap_k / total_relevant if total_relevant > 0 else 0
                metrics[f'map@{k}'].append(ap_k)
                
                # 5. NDCG@k (Normalized Discounted Cumulative Gain)
                dcg_k = sum([top_k_relevant[i] / np.log2(i + 2) 
                            for i in range(len(top_k_relevant))])
                
                # Ideal DCG
                ideal_relevance = sorted(query_true, reverse=True)[:k]
                idcg_k = sum([ideal_relevance[i] / np.log2(i + 2) 
                             for i in range(len(ideal_relevance))])
                
                ndcg_k = dcg_k / idcg_k if idcg_k > 0 else 0
                metrics[f'ndcg@{k}'].append(ndcg_k)
                
                # 6. MRR@k (Mean Reciprocal Rank)
                mrr_k = 0
                for i in range(min(k, len(top_k_relevant))):
                    if top_k_relevant[i]:
                        mrr_k = 1 / (i + 1)
                        break
                metrics[f'mrr@{k}'].append(mrr_k)
        
        # Average across all queries
        for k in k_values:
            metrics[f'precision@{k}'] = np.mean(metrics[f'precision@{k}'])
            metrics[f'recall@{k}'] = np.mean(metrics[f'recall@{k}'])
            metrics[f'f1@{k}'] = np.mean(metrics[f'f1@{k}'])
            metrics[f'map@{k}'] = np.mean(metrics[f'map@{k}'])
            metrics[f'ndcg@{k}'] = np.mean(metrics[f'ndcg@{k}'])
            metrics[f'mrr@{k}'] = np.mean(metrics[f'mrr@{k}'])
        
        # 7. Hit Rate@k
        for k in k_values:
            hit_rate = []
            for query_true, query_scores in zip(y_true, y_pred_scores):
                sorted_indices = np.argsort(query_scores)[::-1][:k]
                has_relevant = any([query_true[i] if i < len(query_true) else 0 
                                   for i in sorted_indices])
                hit_rate.append(1 if has_relevant else 0)
            metrics[f'hit_rate@{k}'] = np.mean(hit_rate)
        
        return metrics


# Example usage
ranking_metrics = RankingMetrics()

# Sample data: 3 queries with relevance judgments and predicted scores
y_true = [
    [1, 0, 1, 0, 1],  # Query 1: items 0, 2, 4 are relevant
    [0, 1, 1, 0, 0],  # Query 2: items 1, 2 are relevant
    [1, 1, 0, 1, 0]   # Query 3: items 0, 1, 3 are relevant
]
y_pred_scores = [
    [0.9, 0.2, 0.8, 0.1, 0.7],  # Query 1 scores
    [0.3, 0.9, 0.8, 0.2, 0.1],  # Query 2 scores
    [0.95, 0.85, 0.4, 0.75, 0.2] # Query 3 scores
]

metrics = ranking_metrics.compute_all_metrics(y_true, y_pred_scores, k_values=[1, 3, 5])

print("\nRanking Metrics:")
print(f"  Precision@3: {metrics['precision@3']:.4f}")
print(f"  Recall@3: {metrics['recall@3']:.4f}")
print(f"  NDCG@3: {metrics['ndcg@3']:.4f}")
print(f"  MAP@3: {metrics['map@3']:.4f}")
print(f"  MRR@3: {metrics['mrr@3']:.4f}")
print(f"  Hit Rate@3: {metrics['hit_rate@3']:.4f}")
```

#### Metric Analysis: Ranking & Information Retrieval

| Metric | Definition | Pros | Cons | When to Use | When NOT to Use |
|--------|------------|------|------|-------------|-----------------|
| **Precision@k** | # relevant in top k / k<br>Proportion of top k results that are relevant | ✅ Easy to interpret<br>✅ User-focused (top k)<br>✅ Intuitive | ❌ Ignores ranking within top k<br>❌ Doesn't consider total relevant<br>❌ Sensitive to k choice | Fixed result set (e.g., top 10)<br>User sees limited results<br>Simple evaluation | Ranking order matters<br>Variable relevant items<br>Need complete picture |
| **Recall@k** | # relevant in top k / total relevant<br>Proportion of all relevant items found in top k | ✅ Measures coverage<br>✅ Shows completeness<br>✅ Complements precision | ❌ Depends on total relevant count<br>❌ Unfair across queries<br>❌ Ignores position | All relevant items important<br>Search completeness<br>Coverage focus | Unequal relevant counts<br>Position very important<br>Top results critical |
| **F1@k** | 2 × (P@k × R@k) / (P@k + R@k)<br>Harmonic mean of Precision@k and Recall@k | ✅ Balances P@k and R@k<br>✅ Single metric<br>✅ Standard balance | ❌ Equal weight to P and R<br>❌ May not match goals<br>❌ Still ignores position | Need P/R balance<br>No clear priority<br>General ranking | Different P/R importance<br>Position critical<br>Business metrics differ |
| **NDCG@k** | DCG@k / IDCG@k<br>Normalized Discounted Cumulative Gain: relevance discounted by position | ✅ Considers position/ranking<br>✅ Graded relevance<br>✅ Industry standard | ❌ More complex<br>❌ Requires relevance scores<br>❌ Less intuitive | Ranking order critical<br>Graded relevance available<br>Search/recommendations | Binary relevance only<br>Position doesn't matter<br>Need simplicity |
| **MAP@k** | Mean of Average Precision@k across queries<br>Average of precision at each relevant item position | ✅ Considers precision at all positions<br>✅ Rewards relevant items early<br>✅ Single score across queries | ❌ Complex calculation<br>❌ Less intuitive<br>❌ Binary relevance | Multiple queries<br>Position important<br>Information retrieval | Single query<br>Graded relevance<br>Simple metric needed |
| **MRR@k** | Mean(1 / rank of first relevant)<br>Average reciprocal rank of first relevant item | ✅ Focuses on first relevant<br>✅ Simple and interpretable<br>✅ Good for single answer | ❌ Only considers first match<br>❌ Ignores other relevant items<br>❌ Not for multiple answers | Single correct answer<br>First result critical<br>QA systems | Multiple relevant items<br>All results matter<br>Comprehensive evaluation |
| **Hit Rate@k** | # queries with ≥1 relevant in top k / # queries<br>Proportion of queries with at least one hit | ✅ Binary success measure<br>✅ Easy to understand<br>✅ User-centric | ❌ Doesn't consider position<br>❌ Binary (no partial credit)<br>❌ Ignores how many hits | At least one relevant needed<br>Exploration focus<br>Simple success metric | Position matters<br>Multiple relevant items<br>Fine-grained evaluation |

---

## 4. Clustering Metrics

```python
class ClusteringMetrics:
    """Comprehensive clustering evaluation metrics."""
    
    def __init__(self):
        """Initialize Clustering Metrics."""
        pass
    
    def compute_all_metrics(
        self,
        X: np.ndarray,
        labels_pred: np.ndarray,
        labels_true: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Compute all clustering metrics.
        
        Args:
            X: Feature matrix
            labels_pred: Predicted cluster labels
            labels_true: True labels (if available for supervised metrics)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Internal metrics (no ground truth needed)
        
        # 1. Silhouette Score
        if len(np.unique(labels_pred)) > 1:
            metrics['silhouette_score'] = silhouette_score(X, labels_pred)
        else:
            metrics['silhouette_score'] = None
        
        # 2. Calinski-Harabasz Index (Variance Ratio Criterion)
        if len(np.unique(labels_pred)) > 1:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels_pred)
        else:
            metrics['calinski_harabasz_score'] = None
        
        # 3. Davies-Bouldin Index
        if len(np.unique(labels_pred)) > 1:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels_pred)
        else:
            metrics['davies_bouldin_score'] = None
        
        # External metrics (ground truth required)
        if labels_true is not None:
            # 4. Adjusted Rand Index (ARI)
            metrics['adjusted_rand_score'] = adjusted_rand_score(labels_true, labels_pred)
            
            # 5. Normalized Mutual Information (NMI)
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(labels_true, labels_pred)
            
            # 6. Adjusted Mutual Information (AMI)
            metrics['adjusted_mutual_info'] = adjusted_mutual_info_score(labels_true, labels_pred)
            
            # 7. Fowlkes-Mallows Score
            metrics['fowlkes_mallows_score'] = fowlkes_mallows_score(labels_true, labels_pred)
            
            # 8. V-Measure
            metrics['v_measure_score'] = v_measure_score(labels_true, labels_pred)
            
            # 9. Homogeneity
            metrics['homogeneity_score'] = homogeneity_score(labels_true, labels_pred)
            
            # 10. Completeness
            metrics['completeness_score'] = completeness_score(labels_true, labels_pred)
            
            # 11. Rand Index (unadjusted)
            metrics['rand_score'] = rand_score(labels_true, labels_pred)
            
            # 12. Mutual Information
            metrics['mutual_info_score'] = mutual_info_score(labels_true, labels_pred)
        
        return metrics


# Example usage
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

clustering_metrics = ClusteringMetrics()

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

# Perform clustering
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X)

metrics = clustering_metrics.compute_all_metrics(X, y_pred, y_true)

print("\nClustering Metrics:")
print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
print(f"  Calinski-Harabasz: {metrics['calinski_harabasz_score']:.2f}")
print(f"  Davies-Bouldin: {metrics['davies_bouldin_score']:.4f}")
print(f"  Adjusted Rand Index: {metrics['adjusted_rand_score']:.4f}")
print(f"  NMI: {metrics['normalized_mutual_info']:.4f}")
print(f"  V-Measure: {metrics['v_measure_score']:.4f}")
```

#### Metric Analysis: Clustering

| Metric | Definition | Pros | Cons | When to Use | When NOT to Use |
|--------|------------|------|------|-------------|-----------------|
| **Silhouette Score** | (b-a) / max(a,b) per sample, averaged<br>a=mean intra-cluster distance, b=mean nearest-cluster distance | ✅ No ground truth needed<br>✅ Intuitive [-1, 1]<br>✅ Per-sample and global<br>✅ Geometric interpretation | ❌ Assumes convex clusters<br>❌ Expensive to compute (O(n²))<br>❌ Poor for complex shapes<br>❌ Distance metric dependent | Convex clusters<br>Need interpretable metric<br>Compare algorithms<br>Determine k | Non-convex shapes<br>Very large datasets<br>Density-based clusters<br>High dimensions |
| **Calinski-Harabasz** | (SS_between / SS_within) × ((n-k) / (k-1))<br>Ratio of between-cluster to within-cluster variance | ✅ Fast to compute<br>✅ Higher is better<br>✅ Works with any shape<br>✅ No assumptions | ❌ Unbounded above<br>❌ Hard to interpret absolute value<br>❌ Favors convex clusters<br>❌ No ground truth comparison | Large datasets<br>Speed important<br>Compare k values<br>Convex preference ok | Need interpretable scores<br>Complex cluster shapes<br>Ground truth available<br>Absolute quality needed |
| **Davies-Bouldin** | Mean of max((σ_i + σ_j) / d(c_i, c_j))<br>Average similarity between each cluster and its most similar one | ✅ Fast to compute<br>✅ Lower is better<br>✅ Intuitive (cluster separation)<br>✅ Bounded below (0) | ❌ Unbounded above<br>❌ Assumes convex clusters<br>❌ Sensitive to outliers<br>❌ Distance based | Convex clusters<br>Speed needed<br>Minimize overlap<br>Compare algorithms | Non-convex clusters<br>Outliers present<br>Complex shapes<br>Need interpretable scale |
| **Adjusted Rand Index** | (RI - Expected_RI) / (max(RI) - Expected_RI)<br>Rand Index corrected for chance | ✅ Ground truth comparison<br>✅ Corrects for chance<br>✅ Bounded [-1, 1]<br>✅ Widely used | ❌ Requires ground truth<br>❌ Can be negative<br>❌ Sensitive to cluster count<br>❌ Not directly interpretable | Ground truth available<br>Compare to reference<br>Algorithm evaluation<br>Benchmark datasets | No ground truth<br>Unsupervised only<br>Different cluster counts<br>Production use |
| **NMI (Normalized Mutual Info)** | MI(U,V) / mean(H(U), H(V))<br>Mutual information normalized by entropy | ✅ Information theoretic<br>✅ Normalized [0, 1]<br>✅ Handles different cluster counts<br>✅ Symmetric | ❌ Requires ground truth<br>❌ Complex to interpret<br>❌ Computation intensive<br>❌ Multiple normalization methods | Ground truth available<br>Different k values<br>Information focus<br>Compare partitions | No ground truth<br>Speed critical<br>Simple interpretation needed<br>Geometric focus preferred |
| **V-Measure** | 2 × (h × c) / (h + c)<br>Harmonic mean of homogeneity and completeness | ✅ Harmonic mean of homogeneity/completeness<br>✅ Interpretable components<br>✅ Normalized [0, 1]<br>✅ No assumptions | ❌ Requires ground truth<br>❌ Entropy-based complexity<br>❌ May favor more clusters | Ground truth available<br>Need homogeneity AND completeness<br>Balanced evaluation<br>Research/benchmarking | No ground truth<br>One aspect more important<br>Simple metric needed<br>Production monitoring |
| **Homogeneity** | 1 - (H(C|K) / H(C))<br>Each cluster contains only members of a single class | ✅ Single cluster contains one class<br>✅ Easy to interpret<br>✅ Independent metric | ❌ Requires ground truth<br>❌ Doesn't check completeness<br>❌ Perfect score easy with many clusters | Cluster purity critical<br>Ground truth available<br>Evaluate precision-like aspect | Completeness also matters<br>Need balanced view<br>No ground truth<br>May over-cluster |
| **Completeness** | 1 - (H(K|C) / H(K))<br>All members of a class are in the same cluster | ✅ All class members in same cluster<br>✅ Easy to interpret<br>✅ Recall-like for clustering | ❌ Requires ground truth<br>❌ Doesn't check homogeneity<br>❌ Perfect score easy with few clusters | All class members together critical<br>Evaluate recall aspect<br>Ground truth available | Homogeneity also matters<br>Need balanced view<br>No ground truth<br>May under-cluster |

---

## 5. Object Detection Metrics

```python
class ObjectDetectionMetrics:
    """Comprehensive object detection metrics."""
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize Object Detection Metrics.
        
        Args:
            iou_threshold: IOU threshold for considering a detection as correct
        """
        self.iou_threshold = iou_threshold
    
    def compute_iou(
        self,
        box1: List[float],
        box2: List[float]
    ) -> float:
        """
        Compute Intersection over Union (IoU) between two boxes.
        
        Args:
            box1: [x1, y1, x2, y2] format
            box2: [x1, y1, x2, y2] format
            
        Returns:
            IoU value
        """
        # Calculate intersection area
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def compute_all_metrics(
        self,
        pred_boxes: List[List[float]],
        pred_scores: List[float],
        pred_labels: List[int],
        true_boxes: List[List[float]],
        true_labels: List[int]
    ) -> Dict[str, float]:
        """
        Compute all object detection metrics.
        
        Args:
            pred_boxes: Predicted bounding boxes
            pred_scores: Confidence scores
            pred_labels: Predicted class labels
            true_boxes: Ground truth boxes
            true_labels: Ground truth labels
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Sort predictions by confidence
        sorted_indices = np.argsort(pred_scores)[::-1]
        pred_boxes = [pred_boxes[i] for i in sorted_indices]
        pred_scores = [pred_scores[i] for i in sorted_indices]
        pred_labels = [pred_labels[i] for i in sorted_indices]
        
        # Match predictions to ground truth
        tp = 0
        fp = 0
        matched_gt = set()
        
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (gt_box, gt_label) in enumerate(zip(true_boxes, true_labels)):
                if gt_label != pred_label:
                    continue
                
                iou = self.compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= self.iou_threshold and best_gt_idx not in matched_gt:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn = len(true_boxes) - len(matched_gt)
        
        # 1. Precision
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # 2. Recall
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # 3. F1 Score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (
                metrics['precision'] + metrics['recall']
            )
        else:
            metrics['f1_score'] = 0
        
        # 4. Average Precision (AP) - simplified version
        # In practice, would compute over multiple confidence thresholds
        metrics['ap'] = metrics['precision']  # Simplified
        
        # 5. True Positives, False Positives, False Negatives
        metrics['true_positives'] = tp
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        
        # 6. Mean IoU for matched boxes
        if len(matched_gt) > 0:
            iou_sum = 0
            for pred_box, pred_label in zip(pred_boxes[:len(matched_gt)], pred_labels[:len(matched_gt)]):
                for gt_box, gt_label in zip(true_boxes, true_labels):
                    if pred_label == gt_label:
                        iou_sum += self.compute_iou(pred_box, gt_box)
                        break
            metrics['mean_iou'] = iou_sum / len(matched_gt)
        else:
            metrics['mean_iou'] = 0
        
        return metrics


# Example usage
obj_detection_metrics = ObjectDetectionMetrics(iou_threshold=0.5)

# Sample data (simplified)
pred_boxes = [[10, 10, 50, 50], [60, 60, 100, 100], [20, 20, 55, 55]]
pred_scores = [0.9, 0.85, 0.7]
pred_labels = [1, 2, 1]

true_boxes = [[12, 12, 52, 52], [58, 58, 98, 98]]
true_labels = [1, 2]

metrics = obj_detection_metrics.compute_all_metrics(
    pred_boxes, pred_scores, pred_labels, true_boxes, true_labels
)

print("\nObject Detection Metrics:")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall: {metrics['recall']:.4f}")
print(f"  F1 Score: {metrics['f1_score']:.4f}")
print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
print(f"  TP/FP/FN: {metrics['true_positives']}/{metrics['false_positives']}/{metrics['false_negatives']}")
```

#### Metric Analysis: Object Detection

| Metric | Definition | Pros | Cons | When to Use | When NOT to Use |
|--------|------------|------|------|-------------|-----------------|
| **IoU (Intersection over Union)** | Area(Prediction ∩ Ground Truth) / Area(Prediction ∪ Ground Truth)<br>Overlap ratio of predicted and true boxes | ✅ Measures box overlap quality<br>✅ Intuitive geometric meaning<br>✅ Bounded [0, 1]<br>✅ Standard in object detection | ❌ Sensitive to box size<br>❌ Zero for non-overlapping boxes<br>❌ Doesn't consider class<br>❌ Treats all errors equally | Bounding box quality<br>Localization accuracy<br>Box regression<br>Standard detection | Position more important than size<br>Small objects (very sensitive)<br>Need distance metric<br>Class confusion matters |
| **Precision** | TP / (TP + FP)<br>Proportion of detections that are correct (IoU ≥ threshold) | ✅ Measures false positives<br>✅ Important for confidence<br>✅ Easy to interpret | ❌ Ignores missed detections<br>❌ Can be gamed by fewer predictions<br>❌ Threshold dependent | False positives costly<br>High confidence critical<br>Minimize false alarms<br>Security applications | Missing objects worse<br>Need complete detection<br>Recall critical | 
| **Recall** | TP / (TP + FN)<br>Proportion of ground truth objects detected | ✅ Measures missed detections<br>✅ Completeness metric<br>✅ Critical for safety | ❌ Ignores false positives<br>❌ Can be 100% with many boxes<br>❌ Threshold dependent | Missing objects critical<br>Safety applications<br>Medical imaging<br>Autonomous vehicles | False positives costly<br>Limited compute/time<br>Need high precision |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall)<br>Harmonic mean of detection precision and recall | ✅ Balances precision and recall<br>✅ Single metric<br>✅ Good for comparison | ❌ Equal weight to P and R<br>❌ May not match use case<br>❌ Threshold dependent | Need P/R balance<br>General comparison<br>No clear priority | Different P/R importance<br>Multiple thresholds<br>Need AP/mAP |
| **AP (Average Precision)** | Area under Precision-Recall curve<br>Integral of precision over all recall levels | ✅ Threshold independent<br>✅ Precision-recall curve area<br>✅ Single metric<br>✅ Robust to threshold | ❌ Complex calculation<br>❌ Harder to interpret<br>❌ Computationally expensive<br>❌ Assumes good ranking | Model comparison<br>Threshold will vary<br>Standard evaluation<br>Research/benchmarking | Single threshold sufficient<br>Need simple metric<br>Real-time constraints<br>Interpretability critical |
| **mAP (mean AP)** | Mean of AP across all classes<br>Average precision for each class then averaged | ✅ Average over all classes<br>✅ Single model quality metric<br>✅ Industry standard<br>✅ Comprehensive | ❌ Very complex<br>❌ Hides per-class performance<br>❌ Expensive to compute<br>❌ Less interpretable | Multi-class detection<br>Overall model quality<br>Benchmark comparison<br>Research standard | Per-class insights needed<br>Single class focus<br>Simple metric required<br>Quick evaluation |
| **mAP@IoU** | mAP at specific IoU threshold<br>Mean AP computed with detections counted as TP only if IoU ≥ threshold | ✅ Controls localization strictness<br>✅ mAP at specific IoU threshold<br>✅ Tunable strictness | ❌ Single threshold arbitrary<br>❌ Need to choose threshold<br>❌ May miss degradation | Fixed IoU requirement<br>Specific use case<br>Localization standard known | Multiple IoU levels needed<br>Comprehensive evaluation<br>Compare localization quality |
| **mAP@[.5:.95]** | Mean of mAP at IoU=[0.5, 0.55, ..., 0.95]<br>Average mAP across 10 IoU thresholds (COCO metric) | ✅ Average over IoU thresholds<br>✅ Comprehensive localization<br>✅ COCO standard<br>✅ Rewards good localization | ❌ Very expensive<br>❌ Complex to interpret<br>❌ May be too strict<br>❌ Slower to compute | Research/competition<br>Comprehensive evaluation<br>Localization quality critical<br>Standard benchmarking | Quick evaluation<br>Speed critical<br>Single IoU sufficient<br>Approximate quality ok |

---

## Summary: Quick Reference Guide

### Classification
- **Binary**: Accuracy, Precision, Recall, F1, ROC AUC, PR AUC, MCC
- **Multi-class**: Accuracy, Macro/Micro/Weighted F1, Top-k Accuracy, Log Loss
- **Multi-label**: Hamming Loss, Jaccard Score, Exact Match Ratio, Label Ranking

### Regression
- **Error-based**: MAE, MSE, RMSE, MAPE, Max Error
- **Correlation-based**: R², Adjusted R², Explained Variance
- **Robust**: Median AE, Huber Loss, Quantile Loss

### Ranking
- **@k Metrics**: Precision@k, Recall@k, NDCG@k, MAP@k, MRR@k
- **Coverage**: Hit Rate@k

### Clustering
- **Internal**: Silhouette Score, Calinski-Harabasz, Davies-Bouldin
- **External**: ARI, NMI, AMI, V-Measure, Fowlkes-Mallows

### Object Detection
- **Box-level**: IoU, Precision, Recall, F1, mAP, AP@IoU

---

## Best Practices and Decision Framework

### 1. Classification Task Selection Guide

#### Choose Binary Classification Metrics Based On:

**Imbalanced Data (Rare Positive Class)**
- ✅ **Use**: PR AUC, F1 Score, Precision, Recall, MCC
- ❌ **Avoid**: Accuracy, ROC AUC (can be optimistic)
- 📋 **Example**: Fraud detection (0.1% fraud rate)

**Balanced Data**
- ✅ **Use**: Accuracy, F1 Score, ROC AUC
- ❌ **Avoid**: Over-complicated metrics
- 📋 **Example**: Gender classification (50/50 split)

**False Positives Very Costly**
- ✅ **Use**: Precision, PPV, False Discovery Rate
- ❌ **Avoid**: Metrics that ignore FP
- 📋 **Example**: Spam detection (false positive = missed important email)

**False Negatives Very Costly**
- ✅ **Use**: Recall, Sensitivity, F2 Score
- ❌ **Avoid**: Precision-focused metrics
- 📋 **Example**: Disease screening (false negative = missed diagnosis)

**Probability Calibration Matters**
- ✅ **Use**: Log Loss, Brier Score, Calibration plots
- ❌ **Avoid**: Label-only metrics
- 📋 **Example**: Weather forecasting, risk scoring

**Need Single Balanced Metric**
- ✅ **Use**: F1 Score, MCC, Balanced Accuracy
- ❌ **Avoid**: One-sided metrics (precision OR recall only)
- 📋 **Example**: General binary classification with no clear priority

---

### 2. Regression Task Selection Guide

#### Choose Regression Metrics Based On:

**Outliers Present**
- ✅ **Use**: MAE, Median AE, Huber Loss
- ❌ **Avoid**: MSE, RMSE, Max Error
- 📋 **Example**: House prices with luxury mansions

**Large Errors Very Bad**
- ✅ **Use**: MSE, RMSE, Max Error
- ❌ **Avoid**: MAE, Median AE
- 📋 **Example**: Temperature control (overshoot dangerous)

**Need Scale Independence**
- ✅ **Use**: MAPE, R², SMAPE
- ❌ **Avoid**: MAE, RMSE (absolute metrics)
- 📋 **Example**: Comparing models across different price ranges

**Exponential/Multiplicative Growth**
- ✅ **Use**: MSLE, RMSLE, Log-transformed metrics
- ❌ **Avoid**: MAE, RMSE on raw scale
- 📋 **Example**: Population growth, viral spread

**Percentage Error Natural**
- ✅ **Use**: MAPE, SMAPE
- ❌ **Avoid**: Absolute error metrics
- 📋 **Example**: Sales forecasting, business metrics

**Zero Values Present**
- ✅ **Use**: MAE, RMSE, R²
- ❌ **Avoid**: MAPE, MSLE
- 📋 **Example**: Daily sales (some days have zero)

**Model Comparison**
- ✅ **Use**: R², Adjusted R², AIC, BIC
- ❌ **Avoid**: Absolute metrics alone
- 📋 **Example**: Feature selection, model selection

**Time Series**
- ✅ **Use**: MASE, MAE, Directional Accuracy
- ❌ **Avoid**: Metrics without baseline
- 📋 **Example**: Stock price forecasting

---

### 3. Ranking/Retrieval Selection Guide

#### Choose Ranking Metrics Based On:

**Search Engines**
- ✅ **Use**: NDCG@k, MAP, Precision@k
- ❌ **Avoid**: Metrics ignoring position
- 📋 **Example**: Google search, product search

**Recommendation Systems**
- ✅ **Use**: NDCG@k, Hit Rate@k, Precision@k
- ❌ **Avoid**: Metrics requiring all relevant items
- 📋 **Example**: Netflix recommendations, e-commerce

**Question Answering**
- ✅ **Use**: MRR, Precision@1, Hit Rate@1
- ❌ **Avoid**: Metrics considering all results
- 📋 **Example**: Chatbots, FAQ systems

**Information Retrieval**
- ✅ **Use**: MAP, NDCG, Recall@k
- ❌ **Avoid**: Single-answer metrics
- 📋 **Example**: Academic search, legal documents

**Binary Relevance Only**
- ✅ **Use**: Precision@k, Recall@k, MAP
- ❌ **Avoid**: NDCG (needs graded relevance)
- 📋 **Example**: Document classification

**Graded Relevance Available**
- ✅ **Use**: NDCG@k
- ❌ **Avoid**: Binary metrics
- 📋 **Example**: Search with 5-star relevance ratings

---

### 4. Clustering Selection Guide

#### Choose Clustering Metrics Based On:

**No Ground Truth**
- ✅ **Use**: Silhouette, Calinski-Harabasz, Davies-Bouldin
- ❌ **Avoid**: External metrics (need labels)
- 📋 **Example**: Customer segmentation, exploratory analysis

**Ground Truth Available**
- ✅ **Use**: ARI, NMI, V-Measure
- ❌ **Avoid**: Internal metrics alone
- 📋 **Example**: Algorithm benchmarking, supervised evaluation

**Large Dataset (>10,000 samples)**
- ✅ **Use**: Calinski-Harabasz, Davies-Bouldin
- ❌ **Avoid**: Silhouette (O(n²) complexity)
- 📋 **Example**: Big data clustering

**Convex Clusters Expected**
- ✅ **Use**: Silhouette, Calinski-Harabasz
- ❌ **Avoid**: Nothing specific, these work well
- 📋 **Example**: K-means, Gaussian mixture models

**Arbitrary Cluster Shapes**
- ✅ **Use**: DBSCAN metrics, Connectivity-based
- ❌ **Avoid**: Silhouette, Calinski-Harabasz
- 📋 **Example**: DBSCAN, HDBSCAN clustering

**Determine Optimal k**
- ✅ **Use**: Silhouette analysis, Elbow method with CH index
- ❌ **Avoid**: External metrics (need labels)
- 📋 **Example**: Choosing number of clusters

---

### 5. Object Detection Selection Guide

#### Choose Object Detection Metrics Based On:

**Research/Benchmarking**
- ✅ **Use**: mAP@[.5:.95], mAP@.5, mAP@.75
- ❌ **Avoid**: Simple precision/recall
- 📋 **Example**: COCO, Pascal VOC competitions

**Real-time Applications**
- ✅ **Use**: Precision/Recall at fixed threshold, F1
- ❌ **Avoid**: mAP (expensive to compute)
- 📋 **Example**: Video surveillance, autonomous driving inference

**Localization Quality Critical**
- ✅ **Use**: IoU, mAP@.75, mAP@[.5:.95]
- ❌ **Avoid**: mAP@.5 (too lenient)
- 📋 **Example**: Medical imaging, precise robotics

**Localization Less Critical**
- ✅ **Use**: mAP@.5, F1 at IoU=0.5
- ❌ **Avoid**: Strict IoU thresholds
- 📋 **Example**: General object presence detection

**Multi-class Detection**
- ✅ **Use**: mAP (mean over classes)
- ❌ **Avoid**: Single-class metrics
- 📋 **Example**: COCO (80 classes), general object detection

**Single Class Detection**
- ✅ **Use**: AP for that class, Precision, Recall
- ❌ **Avoid**: mAP (unnecessary averaging)
- 📋 **Example**: Face detection, pedestrian detection

**Missing Detections Very Bad**
- ✅ **Use**: Recall, AR (Average Recall)
- ❌ **Avoid**: Precision-focused metrics
- 📋 **Example**: Safety-critical detection

**False Alarms Very Bad**
- ✅ **Use**: Precision
- ❌ **Avoid**: Recall-focused metrics
- 📋 **Example**: Low-power edge devices with limited compute

---

## Common Pitfalls and How to Avoid Them

### ❌ Pitfall 1: Using Accuracy on Imbalanced Data
**Problem**: 99% accuracy when 99% of data is negative class (predicting all negative achieves this)

**Solution**: Use F1, PR AUC, or MCC instead

**Example**:
```python
# Bad: Accuracy on imbalanced fraud detection
accuracy = 0.99  # Looks great!
# But model predicts "not fraud" for everything

# Good: Check precision, recall, F1
precision = 0.05  # Only 5% of fraud predictions correct!
recall = 0.02     # Catching only 2% of actual fraud
f1 = 0.028        # Reveals poor performance
```

---

### ❌ Pitfall 2: Optimizing MSE with Outliers
**Problem**: Model focuses on reducing large errors from outliers, poor typical case performance

**Solution**: Use MAE, Huber Loss, or Median AE

**Example**:
```python
# Bad: MSE heavily weights outliers
# True values: [10, 10, 10, 10, 1000]
# Pred A: [9, 9, 9, 9, 900]     # MSE = 2000
# Pred B: [5, 5, 5, 5, 1000]    # MSE = 100
# Model B is "better" but terrible for typical cases!

# Good: MAE treats all errors more equally
# Model A: MAE = 20.8
# Model B: MAE = 5
```

---

### ❌ Pitfall 3: Using ROC AUC for Extreme Imbalance
**Problem**: ROC AUC can be optimistic when positive class is very rare

**Solution**: Use PR AUC instead

**Example**:
```python
# 0.1% fraud rate
# Model predicts random but slightly better than chance
roc_auc = 0.85    # Looks good!
pr_auc = 0.05     # Reveals poor performance on rare class
```

---

### ❌ Pitfall 4: MAPE with Zero or Near-Zero Values
**Problem**: Division by zero or unstable percentages

**Solution**: Use SMAPE, MAE, or filter zeros

**Example**:
```python
# True: [0, 10, 20]
# Pred: [1, 11, 21]
# MAPE: Undefined (division by zero) or infinite

# Better: SMAPE or MAE
```

---

### ❌ Pitfall 5: Comparing R² Across Different Problems
**Problem**: R² is relative to variance in specific dataset

**Solution**: Compare models on same data; use absolute metrics for cross-problem

**Example**:
```python
# Model A on easy problem: R² = 0.95
# Model B on hard problem: R² = 0.75
# Can't conclude A is better than B!
```

---

### ❌ Pitfall 6: Using Accuracy for Multi-label
**Problem**: Multi-label accuracy (exact match) is too strict

**Solution**: Use Hamming Score, F1 Macro/Micro, or Jaccard

**Example**:
```python
# True: [1, 1, 0, 0]
# Pred: [1, 1, 1, 0]  # 3 out of 4 correct!
# Exact match: 0.0    # Too harsh
# Hamming score: 0.75 # More reasonable
```

---

### ❌ Pitfall 7: Single Metric Obsession
**Problem**: Optimizing only one metric misses other important aspects

**Solution**: Track multiple complementary metrics

**Example**:
```python
# Track together:
# Classification: Precision + Recall + F1 + ROC AUC
# Regression: MAE + RMSE + R² + Max Error
# Detection: Precision + Recall + mAP + Inference Time
```

---

## Metric Selection Flowchart

```
START: What is your ML task?
│
├─ CLASSIFICATION
│  ├─ Binary?
│  │  ├─ Imbalanced? → PR AUC, F1, MCC
│  │  ├─ False Positives Costly? → Precision, FDR
│  │  ├─ False Negatives Costly? → Recall, F2
│  │  ├─ Need Probabilities? → Log Loss, Brier Score
│  │  └─ Balanced Data? → Accuracy, F1, ROC AUC
│  │
│  ├─ Multi-class?
│  │  ├─ All Classes Equal? → Macro F1
│  │  ├─ Weighted by Frequency? → Weighted F1
│  │  ├─ Large # Classes? → Top-k Accuracy
│  │  └─ Need Probabilities? → Log Loss
│  │
│  └─ Multi-label?
│     ├─ Perfect Prediction Required? → Exact Match Ratio
│     ├─ Label-wise Performance? → Hamming Score
│     ├─ Set Similarity? → Jaccard Score
│     └─ Sample-wise Performance? → F1 Samples
│
├─ REGRESSION
│  ├─ Outliers Present? → MAE, Median AE, Huber
│  ├─ Large Errors Bad? → MSE, RMSE, Max Error
│  ├─ Need Scale Independence? → MAPE, R², SMAPE
│  ├─ Exponential Growth? → MSLE, RMSLE
│  ├─ Zero Values? → Avoid MAPE, use MAE/RMSE
│  └─ Time Series? → MASE, MAE, Directional Accuracy
│
├─ RANKING
│  ├─ Search Engine? → NDCG@k, MAP
│  ├─ Recommendations? → NDCG@k, Hit Rate@k
│  ├─ Single Answer (QA)? → MRR, Precision@1
│  ├─ Binary Relevance? → Precision@k, MAP
│  └─ Graded Relevance? → NDCG@k
│
├─ CLUSTERING
│  ├─ No Ground Truth? → Silhouette, CH Index, DB Index
│  ├─ Ground Truth Available? → ARI, NMI, V-Measure
│  ├─ Large Dataset? → CH Index, DB Index
│  └─ Determine k? → Silhouette Analysis
│
└─ OBJECT DETECTION
   ├─ Research/Benchmark? → mAP@[.5:.95]
   ├─ Real-time? → Precision/Recall at threshold
   ├─ Precise Localization? → mAP@.75, IoU
   ├─ Multi-class? → mAP
   └─ Single Class? → AP, Precision, Recall
```

---

## Summary: Quick Reference Guide

### Classification
- **Binary**: Accuracy, Precision, Recall, F1, ROC AUC, PR AUC, MCC
- **Multi-class**: Accuracy, Macro/Micro/Weighted F1, Top-k Accuracy, Log Loss
- **Multi-label**: Hamming Loss, Jaccard Score, Exact Match Ratio, Label Ranking

### Regression
- **Error-based**: MAE, MSE, RMSE, MAPE, Max Error
- **Correlation-based**: R², Adjusted R², Explained Variance
- **Robust**: Median AE, Huber Loss, Quantile Loss

### Ranking
- **@k Metrics**: Precision@k, Recall@k, NDCG@k, MAP@k, MRR@k
- **Coverage**: Hit Rate@k

### Clustering
- **Internal**: Silhouette Score, Calinski-Harabasz, Davies-Bouldin
- **External**: ARI, NMI, AMI, V-Measure, Fowlkes-Mallows

### Object Detection
- **Box-level**: IoU, Precision, Recall, F1, mAP, AP@IoU

---

## Quick Decision Tables

### When Stakeholders Ask "How Good Is The Model?"

| Stakeholder | Task | Recommended Metrics | Why |
|-------------|------|---------------------|-----|
| **Business Executive** | Classification | Accuracy, F1 Score | Easy to understand, single number |
| **Business Executive** | Regression | MAPE, R² | Percentage error intuitive |
| **Data Scientist** | Classification | ROC AUC, PR AUC, MCC | Comprehensive, threshold-independent |
| **Data Scientist** | Regression | RMSE, MAE, R² | Standard, interpretable |
| **Product Manager** | Ranking | NDCG@10, Hit Rate@10 | User-facing top results |
| **Domain Expert (Medical)** | Classification | Recall, Sensitivity, NPV | Focus on not missing cases |
| **Domain Expert (Finance)** | Regression | Max Error, 95th Percentile Error | Worst-case scenarios |
| **QA/Testing** | Object Detection | mAP@.5, Precision, Recall | Standard benchmarks |
| **MLOps Engineer** | Any | Multiple + Inference Time | Production readiness |

---

## GCP-Specific Considerations

### Vertex AI AutoML Metrics

**AutoML Tables (Regression)**
- Primary: RMSE, MAE, RMSLE
- Feature Importance based on these metrics
- Use RMSLE for skewed targets

**AutoML Tables (Classification)**
- Primary: ROC AUC, Log Loss, Precision/Recall
- Confusion matrix provided
- Threshold tuning available

**AutoML Vision (Object Detection)**
- Primary: mAP@.5
- Per-class AP available
- IoU threshold configurable

**AutoML Natural Language**
- Classification: Precision, Recall, F1
- Entity Extraction: Precision, Recall per entity type

### BigQuery ML Metrics

**ML.EVALUATE for Classification**
```sql
-- Returns: precision, recall, accuracy, f1_score, log_loss, roc_auc
SELECT * FROM ML.EVALUATE(MODEL `project.dataset.model`)
```

**ML.EVALUATE for Regression**
```sql
-- Returns: mean_absolute_error, mean_squared_error, 
-- mean_squared_log_error, median_absolute_error, r2_score
SELECT * FROM ML.EVALUATE(MODEL `project.dataset.model`)
```

**ML.CONFUSION_MATRIX**
```sql
-- Visual confusion matrix for classification
SELECT * FROM ML.CONFUSION_MATRIX(MODEL `project.dataset.model`)
```

**ML.ROC_CURVE**
```sql
-- ROC curve data points for threshold tuning
SELECT * FROM ML.ROC_CURVE(MODEL `project.dataset.model`)
```

### Model Monitoring in Production (Vertex AI)

**Recommended Metrics to Track**
1. **Performance Metrics**: Same as training (F1, RMSE, etc.)
2. **Data Drift**: Feature distribution changes
3. **Prediction Drift**: Output distribution changes
4. **Serving Metrics**: Latency, throughput, errors

**Alert Thresholds**
- Performance drop > 5%: Warning
- Performance drop > 10%: Critical
- Data drift score > 0.3: Investigate
- Serving latency p99 > 500ms: Critical

---

## Certification Exam Tips

### Most Tested Concepts

1. **When accuracy is misleading** (imbalanced data)
2. **Precision vs Recall tradeoff** (understand both)
3. **ROC AUC vs PR AUC** (imbalanced data)
4. **MAE vs RMSE** (outliers, interpretability)
5. **R² interpretation** (variance explained)
6. **mAP for object detection** (COCO-style)
7. **NDCG for ranking** (position matters)
8. **Which metric for which scenario** (match metric to business goal)

### Common Exam Scenarios

**Scenario 1**: "0.1% fraud rate, model has 99% accuracy"
- ❌ Wrong: "Model is excellent"
- ✅ Right: "Check precision/recall, use PR AUC"

**Scenario 2**: "Predicting house prices, some luxury mansions"
- ❌ Wrong: "Use RMSE to penalize large errors"
- ✅ Right: "Use MAE or log-transform target, outliers skew RMSE"

**Scenario 3**: "Search engine ranking quality"
- ❌ Wrong: "Use accuracy or precision"
- ✅ Right: "Use NDCG@k, MAP"

**Scenario 4**: "Medical diagnosis, missing cancer very bad"
- ❌ Wrong: "Maximize accuracy or precision"
- ✅ Right: "Maximize recall/sensitivity, acceptable lower precision"

**Scenario 5**: "Compare two models on different datasets"
- ❌ Wrong: "Compare R² values directly"
- ✅ Right: "R² not comparable across datasets, use validation on same data"

---

## Additional Resources

### Python Libraries for Metrics
- **scikit-learn**: Most classification, regression, clustering metrics
- **tensorflow/keras**: Built-in metrics for deep learning
- **pycocotools**: Object detection mAP (COCO-style)
- **scipy**: Statistical tests, distance metrics
- **pandas**: Data manipulation for custom metrics

### Further Reading
- Scikit-learn Metrics Documentation
- COCO Evaluation Metrics
- Information Retrieval Metrics (TREC)
- Vertex AI Documentation (Evaluation Metrics)
- BigQuery ML Metrics Reference

---

This comprehensive guide covers the most important metrics for evaluating machine learning models across various tasks with practical guidance on when to use each metric!
