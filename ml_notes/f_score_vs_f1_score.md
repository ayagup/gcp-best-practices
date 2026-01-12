# F-Score vs F1-Score

## Quick Answer
**F1-Score is a specific type of F-Score** where β = 1, giving equal weight to precision and recall.

## F-Score (General Formula)

The **F-Score** (or F-beta score) is a weighted harmonic mean of precision and recall:

````python
F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
````

Where **β** controls the balance between precision and recall.

## F1-Score (Special Case)

**F1-Score** is when **β = 1**:

````python
F1 = 2 × (Precision × Recall) / (Precision + Recall)
````

Equal weight to both precision and recall.

## Different F-Scores

| Score | β Value | Emphasis | Use Case |
|-------|---------|----------|----------|
| **F0.5** | 0.5 | **Precision > Recall** | Spam detection (avoid false positives) |
| **F1** | 1.0 | **Equal balance** | General classification |
| **F2** | 2.0 | **Recall > Precision** | Disease detection (catch all cases) |

## Complete Python Example

````python
"""
F-Score vs F1-Score: Complete Comparison
"""

import numpy as np
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    fbeta_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

class FScoreComparison:
    """
    Demonstrates different F-scores and their use cases
    """
    
    def __init__(self):
        # Example predictions
        self.y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        self.y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
    
    def calculate_manual_metrics(self):
        """Calculate metrics manually from confusion matrix"""
        print("=" * 60)
        print("CONFUSION MATRIX BREAKDOWN")
        print("=" * 60 + "\n")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print("Confusion Matrix:")
        print(f"  True Negatives (TN):  {tn}")
        print(f"  False Positives (FP): {fp}")
        print(f"  False Negatives (FN): {fn}")
        print(f"  True Positives (TP):  {tp}\n")
        
        # Calculate metrics manually
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision:.4f}")
        print(f"Recall    = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall:.4f}\n")
        
        return precision, recall, tp, fp, fn, tn
    
    def calculate_f_scores(self, precision, recall):
        """Calculate different F-scores"""
        print("=" * 60)
        print("F-SCORE CALCULATIONS")
        print("=" * 60 + "\n")
        
        # F1 Score (β = 1)
        f1_manual = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_sklearn = f1_score(self.y_true, self.y_pred)
        
        print("F1-Score (β = 1) - Equal weight to Precision & Recall:")
        print(f"  Formula: 2 × (P × R) / (P + R)")
        print(f"  Manual:  2 × ({precision:.4f} × {recall:.4f}) / ({precision:.4f} + {recall:.4f})")
        print(f"  Result:  {f1_manual:.4f}")
        print(f"  Sklearn: {f1_sklearn:.4f}\n")
        
        # F0.5 Score (β = 0.5) - Precision weighted
        beta = 0.5
        f05 = fbeta_score(self.y_true, self.y_pred, beta=beta)
        f05_manual = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        
        print(f"F0.5-Score (β = 0.5) - Weights Precision 2× more:")
        print(f"  Formula: (1 + β²) × (P × R) / (β² × P + R)")
        print(f"  Manual:  (1 + {beta}²) × ({precision:.4f} × {recall:.4f}) / ({beta}² × {precision:.4f} + {recall:.4f})")
        print(f"  Result:  {f05_manual:.4f}")
        print(f"  Sklearn: {f05:.4f}\n")
        
        # F2 Score (β = 2) - Recall weighted
        beta = 2.0
        f2 = fbeta_score(self.y_true, self.y_pred, beta=beta)
        f2_manual = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        
        print(f"F2-Score (β = 2) - Weights Recall 2× more:")
        print(f"  Formula: (1 + β²) × (P × R) / (β² × P + R)")
        print(f"  Manual:  (1 + {beta}²) × ({precision:.4f} × {recall:.4f}) / ({beta}² × {precision:.4f} + {recall:.4f})")
        print(f"  Result:  {f2_manual:.4f}")
        print(f"  Sklearn: {f2:.4f}\n")
        
        return f05, f1_sklearn, f2
    
    def visualize_beta_impact(self):
        """Visualize how beta affects F-score"""
        print("=" * 60)
        print("VISUALIZING BETA IMPACT")
        print("=" * 60 + "\n")
        
        precision = precision_score(self.y_true, self.y_pred)
        recall = recall_score(self.y_true, self.y_pred)
        
        # Range of beta values
        beta_values = np.linspace(0.1, 3, 50)
        f_scores = [fbeta_score(self.y_true, self.y_pred, beta=b) for b in beta_values]
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(beta_values, f_scores, linewidth=2, label='F-Score')
        plt.axhline(y=precision, color='blue', linestyle='--', label=f'Precision = {precision:.3f}')
        plt.axhline(y=recall, color='red', linestyle='--', label=f'Recall = {recall:.3f}')
        plt.axvline(x=1, color='green', linestyle=':', alpha=0.5, label='β = 1 (F1)')
        
        # Mark specific points
        plt.scatter([0.5, 1, 2], 
                   [fbeta_score(self.y_true, self.y_pred, beta=0.5),
                    fbeta_score(self.y_true, self.y_pred, beta=1),
                    fbeta_score(self.y_true, self.y_pred, beta=2)],
                   color='red', s=100, zorder=5)
        
        plt.xlabel('Beta (β)', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Impact of Beta on F-Score\n(β < 1: favor Precision, β > 1: favor Recall)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('f_score_beta_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: f_score_beta_comparison.png\n")
    
    def use_case_examples(self):
        """Real-world use case examples"""
        print("=" * 60)
        print("REAL-WORLD USE CASES")
        print("=" * 60 + "\n")
        
        use_cases = [
            {
                'scenario': 'Spam Email Detection',
                'priority': 'Avoid marking legitimate emails as spam',
                'metric': 'F0.5 (favor Precision)',
                'beta': 0.5,
                'reason': 'False positives are costly - missing important emails'
            },
            {
                'scenario': 'Credit Card Fraud Detection',
                'priority': 'Catch all fraudulent transactions',
                'metric': 'F2 (favor Recall)',
                'beta': 2.0,
                'reason': 'False negatives are costly - missing fraud'
            },
            {
                'scenario': 'Disease Diagnosis (Cancer)',
                'priority': 'Catch all positive cases',
                'metric': 'F2 or F3 (favor Recall)',
                'beta': 2.0,
                'reason': 'Missing a case is much worse than false alarm'
            },
            {
                'scenario': 'Product Recommendations',
                'priority': 'Balance accuracy and coverage',
                'metric': 'F1 (balanced)',
                'beta': 1.0,
                'reason': 'Equal importance to precision and recall'
            },
            {
                'scenario': 'Content Moderation',
                'priority': 'Avoid removing good content',
                'metric': 'F0.5 (favor Precision)',
                'beta': 0.5,
                'reason': 'Censoring good content damages user experience'
            }
        ]
        
        for i, case in enumerate(use_cases, 1):
            print(f"{i}. {case['scenario']}")
            print(f"   Priority: {case['priority']}")
            print(f"   Best Metric: {case['metric']}")
            print(f"   Reason: {case['reason']}\n")
    
    def comparison_table(self):
        """Create comparison table"""
        print("=" * 60)
        print("QUICK REFERENCE TABLE")
        print("=" * 60 + "\n")
        
        precision = precision_score(self.y_true, self.y_pred)
        recall = recall_score(self.y_true, self.y_pred)
        
        data = {
            'Metric': ['Precision', 'Recall', 'F0.5', 'F1', 'F2'],
            'Value': [
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{fbeta_score(self.y_true, self.y_pred, beta=0.5):.4f}",
                f"{fbeta_score(self.y_true, self.y_pred, beta=1.0):.4f}",
                f"{fbeta_score(self.y_true, self.y_pred, beta=2.0):.4f}"
            ],
            'Emphasis': [
                'Correct positives',
                'Find all positives',
                'Precision (2×)',
                'Balanced',
                'Recall (2×)'
            ],
            'Use When': [
                'False positives costly',
                'False negatives costly',
                'Precision > Recall',
                'Equal importance',
                'Recall > Precision'
            ]
        }
        
        # Print formatted table
        print(f"{'Metric':<15} {'Value':<10} {'Emphasis':<20} {'Use When':<30}")
        print("-" * 75)
        for i in range(len(data['Metric'])):
            print(f"{data['Metric'][i]:<15} {data['Value'][i]:<10} {data['Emphasis'][i]:<20} {data['Use When'][i]:<30}")
        print()
    
    def run_complete_demo(self):
        """Run complete demonstration"""
        print("\n" + "=" * 60)
        print("F-SCORE vs F1-SCORE COMPLETE GUIDE")
        print("=" * 60 + "\n")
        
        print(f"Sample Data:")
        print(f"y_true: {self.y_true}")
        print(f"y_pred: {self.y_pred}\n")
        
        # Calculate metrics
        precision, recall, tp, fp, fn, tn = self.calculate_manual_metrics()
        
        # Calculate F-scores
        f05, f1, f2 = self.calculate_f_scores(precision, recall)
        
        # Visualize
        self.visualize_beta_impact()
        
        # Use cases
        self.use_case_examples()
        
        # Comparison table
        self.comparison_table()
        
        print("=" * 60)
        print("KEY TAKEAWAYS")
        print("=" * 60)
        print("• F1-Score is F-Score with β = 1 (equal weight)")
        print("• F0.5 favors Precision (use when false positives costly)")
        print("• F2 favors Recall (use when false negatives costly)")
        print("• Choose β based on business requirements")
        print("=" * 60 + "\n")


# Additional: Scikit-learn example
def sklearn_complete_example():
    """Complete sklearn example with all metrics"""
    print("=" * 60)
    print("SCIKIT-LEARN COMPLETE EXAMPLE")
    print("=" * 60 + "\n")
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate dataset
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate all metrics
    print("Classification Report (includes F1):")
    print(classification_report(y_test, y_pred))
    
    # Calculate different F-scores
    f05 = fbeta_score(y_test, y_pred, beta=0.5)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2.0)
    
    print(f"\nF0.5-Score (Precision-focused): {f05:.4f}")
    print(f"F1-Score (Balanced):            {f1:.4f}")
    print(f"F2-Score (Recall-focused):      {f2:.4f}\n")


if __name__ == "__main__":
    # Run main demo
    demo = FScoreComparison()
    demo.run_complete_demo()
    
    # Run sklearn example
    sklearn_complete_example()
````

## Summary

| Aspect | F-Score | F1-Score |
|--------|---------|----------|
| **Definition** | General formula with β parameter | Specific case where β = 1 |
| **Formula** | (1 + β²) × (P×R) / (β²×P + R) | 2 × (P×R) / (P + R) |
| **Flexibility** | Adjustable weight | Fixed equal weight |
| **When to use** | Need custom precision/recall balance | Need balanced metric |

**Choose your metric based on the cost of errors in your specific use case!**