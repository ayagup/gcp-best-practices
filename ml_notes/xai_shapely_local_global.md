# Explainable AI (XAI)

**Explainable AI** makes machine learning model predictions interpretable and understandable to humans.

## SHAP (SHapley Additive exPlanations)

**SHAP** uses Shapley values from game theory to explain individual predictions by attributing feature contributions.

### How SHAP Works
- Calculates the marginal contribution of each feature
- Based on **Shapley values**: fair distribution of "payout" among features
- Satisfies properties: consistency, local accuracy, missingness

````python
import shap
import xgboost as xgb

# Train model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Explain predictions using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize feature importance
shap.summary_plot(shap_values, X_test)
````

## Local vs Global Features

### **Local Features (Instance-Level)**
- Explain **individual predictions**
- "Why did the model predict this specific outcome for THIS user?"
- Different explanations for different instances

````python
# Local explanation for a single prediction
import shap

# Explain one instance
shap.force_plot(explainer.expected_value, 
                shap_values[0], 
                X_test.iloc[0])
````

### **Global Features (Model-Level)**
- Explain **overall model behavior**
- "Which features are most important across ALL predictions?"
- Aggregate view of feature importance

````python
# Global explanation
import shap

# Overall feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Or use built-in feature importance
import matplotlib.pyplot as plt
xgb.plot_importance(model)
plt.show()
````

## Comparison Table

| Aspect | Local Explanation | Global Explanation |
|--------|-------------------|-------------------|
| **Scope** | Single prediction | Entire model |
| **Question** | "Why this result?" | "What drives the model?" |
| **Example** | SHAP force plot, LIME | Feature importance, PDP |
| **Use Case** | Debug individual cases | Model audit, compliance |

## XAI Techniques

### Local Interpretability
- **SHAP** (SHapley values)
- **LIME** (Local Interpretable Model-agnostic Explanations)
- **Counterfactual explanations**

### Global Interpretability
- **Feature importance** (Gini, permutation)
- **Partial Dependence Plots (PDP)**
- **Accumulated Local Effects (ALE)**

## Google Cloud XAI

````python
# Vertex AI Explainable AI
from google.cloud import aiplatform

# Deploy model with explanations
endpoint = aiplatform.Endpoint.create(
    display_name="my-endpoint"
)

# Get predictions with explanations
response = endpoint.explain(instances=[instance])
print(response.explanations)  # SHAP-like feature attributions
````

## Regarding Your Note
````markdown
models have to be evaluated on precision and recall

# Plus Explainability for:
- Understanding WHY the model made incorrect predictions
- Identifying bias in features
- Meeting regulatory requirements (GDPR, CCPA)
- Building trust with stakeholders
````









# Complete Python Programs for Explainability: Local & Global Features

## Program 1: Complete XAI Pipeline with SHAP

````python
"""
Complete Explainable AI Demo: Local and Global Feature Explanations
Using SHAP (SHapley Additive exPlanations)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix
import shap

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class ExplainableAIDemo:
    """
    Demonstrates local and global explainability using SHAP
    """
    
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.explainer = None
        self.shap_values = None
        
    def load_and_prepare_data(self):
        """Load and split dataset"""
        print("=" * 60)
        print("STEP 1: Loading Data")
        print("=" * 60)
        
        # Load breast cancer dataset
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target
        
        self.feature_names = data.feature_names
        
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Classes: {data.target_names}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Train size: {len(self.X_train)}")
        print(f"Test size: {len(self.X_test)}\n")
        
        return X, y
    
    def train_model(self):
        """Train Random Forest model"""
        print("=" * 60)
        print("STEP 2: Training Model")
        print("=" * 60)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)
        
        print(f"Train Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}\n")
        
        # Detailed metrics
        y_pred = self.model.predict(self.X_test)
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        return self.model
    
    def initialize_shap(self):
        """Initialize SHAP explainer"""
        print("=" * 60)
        print("STEP 3: Initializing SHAP Explainer")
        print("=" * 60)
        
        # Use Tree explainer for tree-based models
        self.explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values for test set
        self.shap_values = self.explainer.shap_values(self.X_test)
        
        print("SHAP explainer initialized")
        print(f"SHAP values shape: {np.array(self.shap_values).shape}\n")
        
        return self.explainer, self.shap_values
    
    def global_feature_importance(self):
        """
        GLOBAL EXPLANATIONS: Overall feature importance
        """
        print("=" * 60)
        print("GLOBAL EXPLANATIONS: Feature Importance")
        print("=" * 60)
        
        # 1. Built-in feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features (Model's Built-in):")
        print(feature_importance.head(10))
        
        # Plot built-in feature importance
        plt.figure(figsize=(10, 6))
        top_features = feature_importance.head(15)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Feature Importance')
        plt.title('Global Feature Importance (Built-in)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('global_feature_importance_builtin.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: global_feature_importance_builtin.png")
        
        # 2. SHAP global feature importance
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values[1],  # For class 1 (malignant)
            self.X_test,
            plot_type="bar",
            show=False
        )
        plt.title('Global Feature Importance (SHAP)')
        plt.tight_layout()
        plt.savefig('global_feature_importance_shap.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: global_feature_importance_shap.png")
        
        # 3. SHAP summary plot (shows distribution)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values[1],
            self.X_test,
            show=False
        )
        plt.title('Global Feature Impact Distribution (SHAP)')
        plt.tight_layout()
        plt.savefig('global_shap_summary.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: global_shap_summary.png\n")
        
        return feature_importance
    
    def local_explanations(self, instance_idx=0):
        """
        LOCAL EXPLANATIONS: Individual prediction explanations
        """
        print("=" * 60)
        print(f"LOCAL EXPLANATIONS: Instance {instance_idx}")
        print("=" * 60)
        
        # Get instance
        instance = self.X_test.iloc[instance_idx]
        true_label = self.y_test.iloc[instance_idx]
        pred_label = self.model.predict([instance])[0]
        pred_proba = self.model.predict_proba([instance])[0]
        
        print(f"\nInstance {instance_idx} Details:")
        print(f"True Label: {'Malignant' if true_label == 1 else 'Benign'}")
        print(f"Predicted Label: {'Malignant' if pred_label == 1 else 'Benign'}")
        print(f"Prediction Probability: {pred_proba}")
        
        # 1. SHAP Force Plot (local explanation)
        print("\nGenerating SHAP force plot...")
        shap_value_instance = self.shap_values[1][instance_idx]
        
        plt.figure(figsize=(20, 3))
        shap.force_plot(
            self.explainer.expected_value[1],
            shap_value_instance,
            instance,
            matplotlib=True,
            show=False
        )
        plt.title(f'Local Explanation: Instance {instance_idx} (Force Plot)')
        plt.tight_layout()
        plt.savefig(f'local_explanation_force_{instance_idx}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: local_explanation_force_{instance_idx}.png")
        
        # 2. SHAP Waterfall Plot (more detailed local explanation)
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_value_instance,
                base_values=self.explainer.expected_value[1],
                data=instance.values,
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.title(f'Local Explanation: Instance {instance_idx} (Waterfall)')
        plt.tight_layout()
        plt.savefig(f'local_explanation_waterfall_{instance_idx}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: local_explanation_waterfall_{instance_idx}.png")
        
        # 3. Top contributing features for this instance
        feature_contributions = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_value_instance,
            'feature_value': instance.values
        }).sort_values('shap_value', key=abs, ascending=False)
        
        print("\nTop 10 Features Contributing to This Prediction:")
        print(feature_contributions.head(10))
        print()
        
        return feature_contributions
    
    def compare_multiple_instances(self, indices=[0, 1, 2]):
        """
        Compare local explanations for multiple instances
        """
        print("=" * 60)
        print("COMPARING MULTIPLE INSTANCES")
        print("=" * 60)
        
        fig, axes = plt.subplots(len(indices), 1, figsize=(20, 5*len(indices)))
        
        for i, idx in enumerate(indices):
            instance = self.X_test.iloc[idx]
            true_label = self.y_test.iloc[idx]
            pred_label = self.model.predict([instance])[0]
            
            ax = axes[i] if len(indices) > 1 else axes
            
            shap.force_plot(
                self.explainer.expected_value[1],
                self.shap_values[1][idx],
                instance,
                matplotlib=True,
                show=False,
                ax=ax
            )
            
            ax.set_title(f'Instance {idx}: True={true_label}, Pred={pred_label}')
        
        plt.tight_layout()
        plt.savefig('local_comparison_multiple.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: local_comparison_multiple.png\n")
    
    def partial_dependence_plots(self, features=['mean radius', 'mean texture']):
        """
        GLOBAL: Partial Dependence Plots
        Shows how features affect predictions on average
        """
        print("=" * 60)
        print("GLOBAL: Partial Dependence Plots")
        print("=" * 60)
        
        from sklearn.inspection import partial_dependence, PartialDependenceDisplay
        
        feature_indices = [list(self.feature_names).index(f) for f in features]
        
        fig, ax = plt.subplots(figsize=(14, 5))
        display = PartialDependenceDisplay.from_estimator(
            self.model,
            self.X_test,
            feature_indices,
            feature_names=self.feature_names,
            ax=ax
        )
        
        plt.suptitle('Partial Dependence Plots (Global Effect)')
        plt.tight_layout()
        plt.savefig('global_partial_dependence.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: global_partial_dependence.png\n")
    
    def run_complete_demo(self):
        """Run complete explainability demo"""
        print("\n" + "="*60)
        print("EXPLAINABLE AI COMPLETE DEMO")
        print("Local vs Global Feature Explanations")
        print("="*60 + "\n")
        
        # Load data
        self.load_and_prepare_data()
        
        # Train model
        self.train_model()
        
        # Initialize SHAP
        self.initialize_shap()
        
        # Global explanations
        self.global_feature_importance()
        self.partial_dependence_plots()
        
        # Local explanations for multiple instances
        print("Generating local explanations for specific instances...\n")
        self.local_explanations(instance_idx=0)
        self.local_explanations(instance_idx=5)
        self.local_explanations(instance_idx=10)
        
        # Compare multiple instances
        self.compare_multiple_instances([0, 5, 10])
        
        print("="*60)
        print("DEMO COMPLETE!")
        print("="*60)
        print("\nGenerated Files:")
        print("  • global_feature_importance_builtin.png")
        print("  • global_feature_importance_shap.png")
        print("  • global_shap_summary.png")
        print("  • global_partial_dependence.png")
        print("  • local_explanation_force_*.png")
        print("  • local_explanation_waterfall_*.png")
        print("  • local_comparison_multiple.png")


if __name__ == "__main__":
    # Run demo
    demo = ExplainableAIDemo()
    demo.run_complete_demo()
````

## Program 2: LIME for Local Explanations

````python
"""
Local Explanations using LIME (Local Interpretable Model-agnostic Explanations)
Alternative to SHAP for local interpretability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
import lime
import lime.lime_tabular

class LIMEExplainer:
    """
    Demonstrates LOCAL explanations using LIME
    """
    
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.class_names = None
        self.explainer = None
    
    def prepare_data(self):
        """Load and prepare data"""
        print("Loading data...")
        data = load_breast_cancer()
        X = data.data
        y = data.target
        
        self.feature_names = data.feature_names
        self.class_names = data.target_names
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Train: {len(self.X_train)}, Test: {len(self.X_test)}\n")
    
    def train_model(self):
        """Train model"""
        print("Training Gradient Boosting model...")
        self.model = GradientBoostingClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        accuracy = self.model.score(self.X_test, self.y_test)
        print(f"Test Accuracy: {accuracy:.4f}\n")
    
    def initialize_lime(self):
        """Initialize LIME explainer"""
        print("Initializing LIME explainer...")
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification',
            random_state=42
        )
        print("LIME explainer ready\n")
    
    def explain_instance(self, instance_idx=0):
        """
        LOCAL EXPLANATION: Explain a single prediction
        """
        print(f"="*60)
        print(f"LOCAL EXPLANATION: Instance {instance_idx}")
        print(f"="*60)
        
        instance = self.X_test[instance_idx]
        true_label = self.y_test[instance_idx]
        pred_label = self.model.predict([instance])[0]
        pred_proba = self.model.predict_proba([instance])[0]
        
        print(f"\nTrue: {self.class_names[true_label]}")
        print(f"Predicted: {self.class_names[pred_label]}")
        print(f"Probabilities: {pred_proba}")
        
        # Generate LIME explanation
        exp = self.explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=10
        )
        
        # Print explanation
        print("\nLIME Explanation (Top Features):")
        for feature, weight in exp.as_list():
            print(f"  {feature}: {weight:.4f}")
        
        # Save visualization
        fig = exp.as_pyplot_figure()
        plt.title(f'LIME Local Explanation: Instance {instance_idx}')
        plt.tight_layout()
        plt.savefig(f'lime_local_explanation_{instance_idx}.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: lime_local_explanation_{instance_idx}.png\n")
        
        return exp
    
    def compare_instances(self, indices=[0, 5, 10]):
        """Compare LIME explanations for multiple instances"""
        print(f"="*60)
        print("COMPARING MULTIPLE LOCAL EXPLANATIONS")
        print(f"="*60 + "\n")
        
        for idx in indices:
            self.explain_instance(idx)
    
    def run_demo(self):
        """Run complete LIME demo"""
        self.prepare_data()
        self.train_model()
        self.initialize_lime()
        self.compare_instances([0, 5, 10, 15])
        
        print("="*60)
        print("LIME DEMO COMPLETE!")
        print("="*60)


if __name__ == "__main__":
    lime_demo = LIMEExplainer()
    lime_demo.run_demo()
````

## Program 3: Custom Feature Importance Analysis

````python
"""
Custom Feature Importance and Explainability Analysis
Includes permutation importance and ablation studies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

class CustomExplainability:
    """
    Custom explainability methods
    """
    
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
    
    def setup(self):
        """Setup data and model"""
        data = load_breast_cancer()
        self.feature_names = data.feature_names
        
        X = pd.DataFrame(data.data, columns=self.feature_names)
        y = data.target
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        print(f"Model Accuracy: {self.model.score(self.X_test, self.y_test):.4f}\n")
    
    def permutation_importance_analysis(self):
        """
        GLOBAL: Permutation Feature Importance
        Measures importance by shuffling features
        """
        print("="*60)
        print("GLOBAL: Permutation Feature Importance")
        print("="*60 + "\n")
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.model,
            self.X_test,
            self.y_test,
            n_repeats=10,
            random_state=42
        )
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        print("Top 10 Features (Permutation Importance):")
        print(importance_df.head(10))
        
        # Plot
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(15)
        plt.barh(top_features['feature'], top_features['importance_mean'])
        plt.xlabel('Permutation Importance')
        plt.title('Global Feature Importance (Permutation)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('global_permutation_importance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: global_permutation_importance.png\n")
        
        return importance_df
    
    def feature_ablation_study(self):
        """
        GLOBAL: Feature Ablation Study
        Remove features one at a time and measure impact
        """
        print("="*60)
        print("GLOBAL: Feature Ablation Study")
        print("="*60 + "\n")
        
        baseline_score = self.model.score(self.X_test, self.y_test)
        print(f"Baseline Accuracy: {baseline_score:.4f}")
        
        ablation_results = []
        
        for feature in self.feature_names[:10]:  # Top 10 for speed
            # Remove feature
            X_test_ablated = self.X_test.drop(columns=[feature])
            X_train_ablated = self.X_train.drop(columns=[feature])
            
            # Retrain model
            model_ablated = RandomForestClassifier(n_estimators=100, random_state=42)
            model_ablated.fit(X_train_ablated, self.y_train)
            
            # Measure impact
            score_ablated = model_ablated.score(X_test_ablated, self.y_test)
            impact = baseline_score - score_ablated
            
            ablation_results.append({
                'feature': feature,
                'accuracy_drop': impact
            })
        
        ablation_df = pd.DataFrame(ablation_results).sort_values('accuracy_drop', ascending=False)
        
        print("\nFeature Ablation Results:")
        print(ablation_df)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(ablation_df['feature'], ablation_df['accuracy_drop'])
        plt.xlabel('Accuracy Drop (Higher = More Important)')
        plt.title('Feature Ablation Study')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('global_ablation_study.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: global_ablation_study.png\n")
        
        return ablation_df
    
    def local_feature_contribution(self, instance_idx=0):
        """
        LOCAL: Feature contributions for specific instance
        Using tree's decision path
        """
        print("="*60)
        print(f"LOCAL: Feature Contribution (Instance {instance_idx})")
        print("="*60 + "\n")
        
        instance = self.X_test.iloc[instance_idx:instance_idx+1]
        
        # Get prediction
        pred = self.model.predict(instance)[0]
        pred_proba = self.model.predict_proba(instance)[0]
        
        print(f"Prediction: {pred}")
        print(f"Probability: {pred_proba}")
        
        # Get feature values for this instance
        feature_values = instance.iloc[0]
        
        # Calculate simple contribution (feature_value * feature_importance)
        contributions = feature_values * self.model.feature_importances_
        contributions_df = pd.DataFrame({
            'feature': self.feature_names,
            'value': feature_values,
            'contribution': contributions
        }).sort_values('contribution', key=abs, ascending=False)
        
        print("\nTop 10 Feature Contributions:")
        print(contributions_df.head(10))
        
        # Plot
        plt.figure(figsize=(10, 6))
        top_contrib = contributions_df.head(15)
        colors = ['red' if x < 0 else 'green' for x in top_contrib['contribution']]
        plt.barh(top_contrib['feature'], top_contrib['contribution'], color=colors)
        plt.xlabel('Contribution to Prediction')
        plt.title(f'Local Feature Contributions (Instance {instance_idx})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'local_contributions_{instance_idx}.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: local_contributions_{instance_idx}.png\n")
        
        return contributions_df
    
    def run_complete_analysis(self):
        """Run all analyses"""
        print("\n" + "="*60)
        print("CUSTOM EXPLAINABILITY ANALYSIS")
        print("="*60 + "\n")
        
        self.setup()
        
        # Global analyses
        self.permutation_importance_analysis()
        self.feature_ablation_study()
        
        # Local analyses
        self.local_feature_contribution(0)
        self.local_feature_contribution(5)
        self.local_feature_contribution(10)
        
        print("="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)


if __name__ == "__main__":
    analyzer = CustomExplainability()
    analyzer.run_complete_analysis()
````

## Installation Requirements

````bash
# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn shap lime
````

## Run All Programs

````bash
python xai_local_global_shap.py
python xai_lime_local.py
python xai_custom_analysis.py
````

These programs demonstrate:
- **Global explanations**: Overall model behavior and feature importance
- **Local explanations**: Why specific predictions were made
- **Multiple methods**: SHAP, LIME, permutation importance, ablation studies