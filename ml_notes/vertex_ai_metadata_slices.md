# Vertex AI Model Evaluation - Sliced Metrics Feature

## Overview
**Model Evaluation Slicing** in Vertex AI allows you to evaluate model performance across specific subsets (slices) of your evaluation data. This helps identify performance disparities across different data segments, ensuring fairness and detecting bias.

## What are Sliced Metrics?

Sliced metrics break down model performance by specific feature values or feature combinations, revealing how well your model performs on different data segments.

### **Example Use Cases**
- **Image Classification**: Performance by image size, lighting conditions, object position
- **Tabular Data**: Performance by age group, geographic region, income bracket
- **Text Classification**: Performance by document length, language, topic
- **Fraud Detection**: Performance by transaction amount ranges, merchant types

## Key Concepts

### **1. Slice Specification**
Define which features to slice on:

```python
# Single feature slice
slice_spec = {
    "configs": {
        "age_group": {  # Feature name
            "value": {"float_value": [0, 18, 35, 50, 65, 100]}  # Boundaries
        }
    }
}

# Multiple feature slices
slice_spec = {
    "configs": {
        "age_group": {
            "value": {"float_value": [0, 18, 35, 50, 65, 100]}
        },
        "gender": {
            "value": {"string_value": ["male", "female", "other"]}
        }
    }
}
```

### **2. Slice Types**

#### **Categorical Slices**
```python
# Slice by categorical feature values
slice_spec = {
    "configs": {
        "region": {
            "value": {"string_value": ["north", "south", "east", "west"]}
        }
    }
}
```

#### **Numerical Slices (Binning)**
```python
# Slice by numerical ranges
slice_spec = {
    "configs": {
        "income": {
            "value": {"float_value": [0, 30000, 60000, 100000, 200000]}
        }
    }
}
```

#### **Cross-Feature Slices**
```python
# Combine multiple features
slice_spec = {
    "configs": {
        "age_group": {
            "value": {"float_value": [0, 35, 65, 100]}
        },
        "gender": {
            "value": {"string_value": ["male", "female"]}
        }
    }
}
# Creates slices: age(0-35)×gender(male), age(0-35)×gender(female), etc.
```

## Implementation

### **Method 1: Using Vertex AI SDK (AutoML)**

```python
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project="my-project", location="us-central1")

# Create dataset
dataset = aiplatform.TabularDataset.create(
    display_name="customer_dataset",
    gcs_source="gs://my-bucket/data.csv"
)

# Train AutoML model with evaluation slicing
job = aiplatform.AutoMLTabularTrainingJob(
    display_name="model_with_slicing",
    optimization_prediction_type="classification"
)

model = job.run(
    dataset=dataset,
    target_column="churn",
    training_fraction_split=0.8,
    validation_fraction_split=0.1,
    test_fraction_split=0.1,
    model_display_name="churn_model",
    # Define slicing configuration
    export_evaluated_data_items=True,
    export_evaluated_data_items_bigquery_destination_uri="bq://my-project.my_dataset.evaluation_table",
    # Slicing is automatically done for categorical features
)

# Access sliced metrics
evaluation = model.get_model_evaluation()
sliced_metrics = evaluation.metrics.get("slicedMetrics", [])

for slice_metric in sliced_metrics:
    print(f"Slice: {slice_metric['singleOutputSlicingSpec']}")
    print(f"Metrics: {slice_metric['metrics']}")
```

### **Method 2: Custom Training with TensorFlow Model Analysis (TFMA)**

```python
import tensorflow_model_analysis as tfma
from google.cloud import aiplatform

# Define slice specifications
slice_spec = [
    tfma.SlicingSpec(),  # Overall metrics
    tfma.SlicingSpec(feature_keys=['age_group']),  # By age group
    tfma.SlicingSpec(feature_keys=['gender']),  # By gender
    tfma.SlicingSpec(feature_keys=['age_group', 'gender']),  # Cross-slice
    tfma.SlicingSpec(
        feature_keys=['income'],
        feature_values={'income': ['low', 'medium', 'high']}
    )
]

# Define metrics
metrics_specs = tfma.MetricsSpec(
    metrics=[
        tfma.MetricConfig(class_name='BinaryAccuracy'),
        tfma.MetricConfig(class_name='AUC'),
        tfma.MetricConfig(class_name='Precision'),
        tfma.MetricConfig(class_name='Recall'),
        tfma.MetricConfig(
            class_name='FairnessIndicators',
            config='{"thresholds": [0.5]}'
        )
    ]
)

# Create evaluation configuration
eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='label')],
    slicing_specs=slice_spec,
    metrics_specs=[metrics_specs]
)

# Run evaluation
eval_result = tfma.run_model_analysis(
    eval_config=eval_config,
    eval_shared_model=tfma.default_eval_shared_model(
        eval_saved_model_path='gs://my-bucket/model',
        tags=[tf.saved_model.SERVING]
    ),
    data_location='gs://my-bucket/eval_data/*',
    output_path='gs://my-bucket/eval_results'
)

# Upload to Vertex AI
aiplatform.init(project="my-project", location="us-central1")

model = aiplatform.Model.upload(
    display_name="model_with_tfma",
    artifact_uri="gs://my-bucket/model",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest"
)

# Import evaluation
model_evaluation = model.import_model_evaluation(
    metrics=eval_result.slicing_metrics,
    dataset_type="test"
)
```

### **Method 3: Batch Prediction with Slicing**

```python
from google.cloud import aiplatform

# Initialize
aiplatform.init(project="my-project", location="us-central1")

# Get model
model = aiplatform.Model("projects/123/locations/us-central1/models/456")

# Create batch prediction job
batch_prediction_job = model.batch_predict(
    job_display_name="batch_prediction_with_slicing",
    gcs_source="gs://my-bucket/input_data.csv",
    gcs_destination_prefix="gs://my-bucket/predictions/",
    machine_type="n1-standard-4",
    # Enable explanation and slicing
    generate_explanation=True,
    model_parameters={
        "slicingConfig": {
            "featureKeys": ["age_group", "region"]
        }
    }
)

# Wait for completion
batch_prediction_job.wait()

# Analyze sliced predictions
from google.cloud import storage
import json

client = storage.Client()
bucket = client.bucket("my-bucket")
blobs = bucket.list_blobs(prefix="predictions/")

for blob in blobs:
    content = blob.download_as_text()
    for line in content.split('\n'):
        if line:
            result = json.loads(line)
            print(f"Prediction: {result['prediction']}")
            print(f"Slice: {result.get('slice', 'N/A')}")
```

## Analyzing Sliced Metrics

### **1. Access via Console**
- Navigate to Vertex AI → Models → Select Model → Evaluate
- View metrics breakdown by slices
- Compare performance across segments

### **2. Programmatic Access**

```python
from google.cloud import aiplatform

# Get model evaluations
model = aiplatform.Model("projects/123/locations/us-central1/models/456")
evaluations = model.list_model_evaluations()

for evaluation in evaluations:
    metrics = evaluation.metrics
    
    # Overall metrics
    print(f"Overall Accuracy: {metrics['accuracy']}")
    print(f"Overall AUC: {metrics['auRoc']}")
    
    # Sliced metrics
    if 'slicedMetrics' in metrics:
        for slice_metric in metrics['slicedMetrics']:
            slice_spec = slice_metric['singleOutputSlicingSpec']
            slice_metrics = slice_metric['metrics']
            
            print(f"\nSlice: {slice_spec}")
            print(f"  Accuracy: {slice_metrics.get('accuracy')}")
            print(f"  Precision: {slice_metrics.get('precision')}")
            print(f"  Recall: {slice_metrics.get('recall')}")
```

### **3. Export to BigQuery for Analysis**

```python
from google.cloud import bigquery

# Export evaluation metrics to BigQuery
bq_client = bigquery.Client()

# Query sliced metrics
query = """
SELECT 
    slice_key,
    metric_name,
    metric_value
FROM `my-project.my_dataset.evaluation_metrics`
WHERE slice_key IS NOT NULL
ORDER BY slice_key, metric_name
"""

results = bq_client.query(query).to_dataframe()

# Analyze performance disparities
import pandas as pd

pivot_table = results.pivot_table(
    index='slice_key',
    columns='metric_name',
    values='metric_value'
)

print(pivot_table)

# Identify underperforming slices
threshold_accuracy = 0.85
underperforming = pivot_table[pivot_table['accuracy'] < threshold_accuracy]
print(f"\nUnderperforming slices:\n{underperforming}")
```

## Fairness and Bias Detection

### **Fairness Indicators Integration**

```python
import tensorflow_model_analysis as tfma

# Define fairness-aware slicing
slice_spec = [
    tfma.SlicingSpec(),
    tfma.SlicingSpec(feature_keys=['sensitive_attribute'])  # e.g., race, gender
]

# Configure fairness metrics
metrics_specs = tfma.MetricsSpec(
    metrics=[
        tfma.MetricConfig(class_name='BinaryAccuracy'),
        tfma.MetricConfig(
            class_name='FairnessIndicators',
            config='{"thresholds": [0.3, 0.5, 0.7]}'
        ),
        # Fairness-specific metrics
        tfma.MetricConfig(class_name='FalsePositiveRate'),
        tfma.MetricConfig(class_name='FalseNegativeRate'),
        tfma.MetricConfig(class_name='TruePositiveRate'),
        tfma.MetricConfig(class_name='TrueNegativeRate')
    ],
    # Compare slices for disparities
    model_names=['baseline', 'candidate']
)

eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='label')],
    slicing_specs=slice_spec,
    metrics_specs=[metrics_specs]
)
```

### **Detecting Bias Metrics**

```python
# Calculate disparate impact ratio
def calculate_disparate_impact(metrics_dict, protected_group, reference_group):
    """
    Disparate Impact = (Positive Rate for Protected Group) / (Positive Rate for Reference Group)
    Ideal value: 1.0 (no disparity)
    Concern if < 0.8 or > 1.25
    """
    protected_tpr = metrics_dict[protected_group]['true_positive_rate']
    reference_tpr = metrics_dict[reference_group]['true_positive_rate']
    
    return protected_tpr / reference_tpr

# Example usage
slice_metrics = {
    'age_18_35': {'true_positive_rate': 0.75},
    'age_65_plus': {'true_positive_rate': 0.60}
}

di_ratio = calculate_disparate_impact(slice_metrics, 'age_65_plus', 'age_18_35')
print(f"Disparate Impact Ratio: {di_ratio:.2f}")

if di_ratio < 0.8:
    print("⚠️ Potential bias detected - underperforming on older age group")
```

## Visualization

### **Plot Sliced Metrics**

```python
import matplotlib.pyplot as plt
import pandas as pd

# Sample sliced metrics data
data = {
    'slice': ['age_18_35', 'age_35_50', 'age_50_65', 'age_65+'],
    'accuracy': [0.92, 0.89, 0.85, 0.78],
    'precision': [0.91, 0.87, 0.83, 0.75],
    'recall': [0.93, 0.90, 0.86, 0.80]
}

df = pd.DataFrame(data)

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(df))
width = 0.25

ax.bar([i - width for i in x], df['accuracy'], width, label='Accuracy')
ax.bar(x, df['precision'], width, label='Precision')
ax.bar([i + width for i in x], df['recall'], width, label='Recall')

ax.set_xlabel('Age Group')
ax.set_ylabel('Score')
ax.set_title('Model Performance by Age Group')
ax.set_xticks(x)
ax.set_xticklabels(df['slice'])
ax.legend()
ax.axhline(y=0.85, color='r', linestyle='--', label='Target Threshold')

plt.tight_layout()
plt.savefig('sliced_metrics.png')
plt.show()
```

### **Using TFMA Visualization**

```python
import tensorflow_model_analysis as tfma

# Load evaluation results
eval_result = tfma.load_eval_result('gs://my-bucket/eval_results')

# Render slicing metrics view
tfma.view.render_slicing_metrics(
    eval_result,
    slicing_spec=tfma.SlicingSpec(feature_keys=['age_group']),
    metric_name='accuracy'
)

# Render fairness indicators
tfma.addons.fairness.view.widget_view.render_fairness_indicator(
    eval_result
)
```

## Best Practices

### **1. Choose Meaningful Slices**
```python
# ✅ Good: Actionable slices
slice_spec = {
    "configs": {
        "customer_segment": {"value": {"string_value": ["premium", "standard", "basic"]}},
        "transaction_amount": {"value": {"float_value": [0, 100, 500, 1000, 10000]}}
    }
}

# ❌ Bad: Too many slices (combinatorial explosion)
# Avoid slicing on high-cardinality features (e.g., user_id)
```

### **2. Set Performance Thresholds**
```python
# Define acceptable performance ranges per slice
thresholds = {
    "accuracy": {"min": 0.85, "max": 1.0},
    "false_positive_rate": {"min": 0.0, "max": 0.05},
    "false_negative_rate": {"min": 0.0, "max": 0.10}
}

def validate_slice_performance(slice_metrics, thresholds):
    issues = []
    for metric, bounds in thresholds.items():
        value = slice_metrics.get(metric)
        if value < bounds['min'] or value > bounds['max']:
            issues.append(f"{metric} out of range: {value}")
    return issues
```

### **3. Monitor Slice Drift**
```python
# Compare slice performance over time
def detect_slice_drift(current_metrics, baseline_metrics, threshold=0.05):
    """Alert if slice performance degrades beyond threshold"""
    drifted_slices = []
    
    for slice_key in current_metrics.keys():
        if slice_key in baseline_metrics:
            current_acc = current_metrics[slice_key]['accuracy']
            baseline_acc = baseline_metrics[slice_key]['accuracy']
            
            drift = abs(current_acc - baseline_acc)
            if drift > threshold:
                drifted_slices.append({
                    'slice': slice_key,
                    'drift': drift,
                    'current': current_acc,
                    'baseline': baseline_acc
                })
    
    return drifted_slices
```

### **4. Automate Monitoring**
```python
from google.cloud import aiplatform_v1

# Create model monitoring job with slice-aware thresholds
monitoring_job = aiplatform_v1.ModelDeploymentMonitoringJob(
    display_name="slice_monitoring",
    model_deployment_monitoring_objective_configs=[
        {
            "deployed_model_id": "model_123",
            "objective_config": {
                "training_dataset": {
                    "gcs_source": {"uris": ["gs://bucket/train_data.csv"]}
                },
                "training_prediction_skew_detection_config": {
                    "skew_thresholds": {
                        "age_group": {"value": 0.1},
                        "gender": {"value": 0.1}
                    }
                }
            }
        }
    ],
    logging_sampling_strategy={"random_sample_config": {"sample_rate": 0.2}}
)
```

## Common Metrics Per Slice

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Accuracy** | Overall correctness | General performance |
| **Precision** | Positive predictive value | Minimize false positives |
| **Recall** | True positive rate | Minimize false negatives |
| **F1 Score** | Harmonic mean of precision/recall | Balanced metric |
| **AUC-ROC** | Area under ROC curve | Classification threshold-independent |
| **Confusion Matrix** | TP, FP, TN, FN breakdown | Detailed error analysis |
| **False Positive Rate** | FP / (FP + TN) | Fairness indicator |
| **False Negative Rate** | FN / (FN + TP) | Fairness indicator |

## Limitations

❌ **Combinatorial Explosion**: Too many slice combinations can be computationally expensive  
❌ **Small Sample Sizes**: Some slices may have insufficient data for reliable metrics  
❌ **Statistical Significance**: Need enough samples per slice for meaningful comparisons  
❌ **Feature Engineering**: Slicing quality depends on meaningful feature selection  

## Use Cases by Industry

### **Healthcare**
```python
# Slice by patient demographics for fairness
slice_spec = {
    "age_group": [0, 18, 45, 65, 100],
    "ethnicity": ["caucasian", "african_american", "hispanic", "asian", "other"],
    "insurance_type": ["private", "medicare", "medicaid", "uninsured"]
}
```

### **Finance**
```python
# Slice by transaction characteristics
slice_spec = {
    "transaction_amount": [0, 100, 500, 1000, 5000, 10000],
    "merchant_category": ["retail", "online", "restaurant", "travel"],
    "customer_tenure": [0, 1, 3, 5, 10]  # years
}
```

### **E-commerce**
```python
# Slice by customer behavior
slice_spec = {
    "purchase_frequency": ["first_time", "occasional", "frequent", "vip"],
    "product_category": ["electronics", "clothing", "home", "sports"],
    "geography": ["urban", "suburban", "rural"]
}
```

## Summary

**Sliced metrics** in Vertex AI enable:
- ✅ **Fairness analysis** across demographic groups
- ✅ **Bias detection** through performance disparities
- ✅ **Targeted improvements** for underperforming segments
- ✅ **Regulatory compliance** (e.g., equal credit opportunity)
- ✅ **Business insights** into customer segments

**Key Integration Points**:
- AutoML models (automatic slicing on categorical features)
- Custom training (via TensorFlow Model Analysis)
- Model Monitoring (slice-aware drift detection)
- Batch Predictions (slice-specific predictions)

---

**Pro Tip**: Always validate that slices have sufficient sample sizes (typically 100+ examples) before drawing conclusions about performance disparities.