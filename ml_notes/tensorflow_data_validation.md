# TensorFlow Data Validation (TFDV)

## Overview
**TensorFlow Data Validation (TFDV)** is a library for exploring and validating machine learning data at scale. It analyzes training and serving data to compute descriptive statistics, infer schemas, and detect data anomalies including distribution skew and drift.

## Key Features

### **1. Data Statistics Generation**
- Descriptive statistics for features
- Distribution visualization
- Missing value analysis
- Cardinality detection

### **2. Schema Inference**
- Automatic schema generation from data
- Data type inference
- Domain constraints discovery
- Feature presence requirements

### **3. Anomaly Detection**
- Training-serving skew detection
- Data drift detection
- Schema violations
- Distribution changes

### **4. Data Validation**
- Schema-based validation
- Custom validation rules
- Batch and streaming support

## Installation

```bash
# Install TFDV
pip install tensorflow-data-validation

# With specific TensorFlow version
pip install tensorflow-data-validation tensorflow==2.15.0

# For Google Cloud integration
pip install tensorflow-data-validation[gcp]

# For visualization support
pip install tensorflow-data-validation[visualization]
```

## Basic Workflow

### **1. Generate Statistics**

```python
import tensorflow_data_validation as tfdv
import pandas as pd

# From pandas DataFrame
data = pd.read_csv('train_data.csv')
stats = tfdv.generate_statistics_from_dataframe(data)

# From CSV file
stats = tfdv.generate_statistics_from_csv(
    data_location='gs://bucket/data.csv',
    stats_options=tfdv.StatsOptions()
)

# From TFRecord
stats = tfdv.generate_statistics_from_tfrecord(
    data_location='gs://bucket/data.tfrecord'
)

# Visualize statistics
tfdv.visualize_statistics(stats)
```

### **2. Infer Schema**

```python
# Infer schema from statistics
schema = tfdv.infer_schema(statistics=stats)

# Display schema
tfdv.display_schema(schema)

# Write schema to file
tfdv.write_schema_text(schema, 'schema.pbtxt')

# Load schema from file
schema = tfdv.load_schema_text('schema.pbtxt')
```

### **3. Validate Data**

```python
# Validate new data against schema
eval_stats = tfdv.generate_statistics_from_dataframe(eval_data)
anomalies = tfdv.validate_statistics(
    statistics=eval_stats,
    schema=schema
)

# Display anomalies
tfdv.display_anomalies(anomalies)

# Check if anomalies exist
if anomalies.anomaly_info:
    print("⚠️ Anomalies detected!")
    for feature_name, anomaly_info in anomalies.anomaly_info.items():
        print(f"Feature: {feature_name}")
        print(f"Description: {anomaly_info.description}")
else:
    print("✅ No anomalies detected")
```

## Detailed Examples

### **Example 1: Complete Data Validation Pipeline**

```python
import tensorflow_data_validation as tfdv
import pandas as pd
from datetime import datetime

# Step 1: Load training data
print("Loading training data...")
train_df = pd.read_csv('train_data.csv')
print(f"Training data shape: {train_df.shape}")

# Step 2: Generate statistics for training data
print("\nGenerating statistics...")
train_stats = tfdv.generate_statistics_from_dataframe(train_df)

# Step 3: Visualize statistics
print("Visualizing statistics...")
tfdv.visualize_statistics(train_stats)

# Step 4: Infer schema from training data
print("\nInferring schema...")
schema = tfdv.infer_schema(statistics=train_stats)

# Step 5: Display and save schema
tfdv.display_schema(schema)
tfdv.write_schema_text(schema, 'schema.pbtxt')
print("Schema saved to schema.pbtxt")

# Step 6: Load evaluation data
print("\nLoading evaluation data...")
eval_df = pd.read_csv('eval_data.csv')

# Step 7: Generate statistics for evaluation data
eval_stats = tfdv.generate_statistics_from_dataframe(eval_df)

# Step 8: Validate evaluation data against schema
print("\nValidating evaluation data...")
anomalies = tfdv.validate_statistics(
    statistics=eval_stats,
    schema=schema
)

# Step 9: Display anomalies
if anomalies.anomaly_info:
    print("⚠️ Anomalies found:")
    tfdv.display_anomalies(anomalies)
    
    # Log anomalies
    with open(f'anomalies_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'w') as f:
        for feature, info in anomalies.anomaly_info.items():
            f.write(f"Feature: {feature}\n")
            f.write(f"Description: {info.description}\n")
            f.write(f"Severity: {info.severity}\n\n")
else:
    print("✅ No anomalies detected")

# Step 10: Compare training and evaluation statistics
print("\nComparing distributions...")
tfdv.visualize_statistics(
    lhs_statistics=train_stats,
    rhs_statistics=eval_stats,
    lhs_name='TRAIN',
    rhs_name='EVAL'
)
```

### **Example 2: Schema Customization**

```python
import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2

# Load or infer initial schema
schema = tfdv.infer_schema(statistics=train_stats)

# Customize feature expectations

# 1. Set feature as required (cannot be missing)
tfdv.get_feature(schema, 'user_id').presence.min_fraction = 1.0

# 2. Set feature as optional
tfdv.get_feature(schema, 'optional_field').presence.min_fraction = 0.0

# 3. Set value range for numerical feature
tfdv.set_domain(
    schema,
    'age',
    schema_pb2.IntDomain(name='age', min=0, max=120)
)

# 4. Set categorical values (string domain)
tfdv.set_domain(
    schema,
    'country',
    schema_pb2.StringDomain(
        name='country',
        value=['US', 'UK', 'CA', 'AU', 'IN']
    )
)

# 5. Set expected data type
age_feature = tfdv.get_feature(schema, 'age')
age_feature.type = schema_pb2.INT

# 6. Allow feature to be deprecated (not present in new data)
tfdv.get_feature(schema, 'old_feature').lifecycle_stage = schema_pb2.DEPRECATED

# 7. Set distribution constraints
tfdv.get_feature(schema, 'price').distribution_constraints.min_domain_mass = 0.95

# 8. Freeze specific feature domains (don't auto-update)
tfdv.get_domain(schema, 'country').freeze = True

# Save customized schema
tfdv.write_schema_text(schema, 'customized_schema.pbtxt')

# Validate with customized schema
anomalies = tfdv.validate_statistics(
    statistics=new_stats,
    schema=schema
)
```

### **Example 3: Detecting Training-Serving Skew**

```python
import tensorflow_data_validation as tfdv

# Generate statistics for training data
train_stats = tfdv.generate_statistics_from_csv('training_data.csv')

# Generate statistics for serving data
serving_stats = tfdv.generate_statistics_from_csv('serving_data.csv')

# Infer schema from training data
schema = tfdv.infer_schema(train_stats)

# Set skew comparator for specific features
age_feature = tfdv.get_feature(schema, 'age')
age_feature.skew_comparator.infinity_norm.threshold = 0.1  # 10% threshold

category_feature = tfdv.get_feature(schema, 'category')
category_feature.skew_comparator.jensen_shannon_divergence.threshold = 0.01

# Detect skew
skew_anomalies = tfdv.validate_statistics(
    statistics=serving_stats,
    schema=schema,
    previous_statistics=train_stats,
    serving_statistics=serving_stats
)

# Display skew anomalies
if skew_anomalies.anomaly_info:
    print("⚠️ Training-serving skew detected!")
    tfdv.display_anomalies(skew_anomalies)
else:
    print("✅ No skew detected")

# Visualize comparison
tfdv.visualize_statistics(
    lhs_statistics=train_stats,
    rhs_statistics=serving_stats,
    lhs_name='TRAINING',
    rhs_name='SERVING'
)
```

### **Example 4: Detecting Data Drift**

```python
import tensorflow_data_validation as tfdv
from datetime import datetime, timedelta

# Generate statistics for baseline data (e.g., last week)
baseline_stats = tfdv.generate_statistics_from_csv('data_week1.csv')

# Generate statistics for current data (e.g., this week)
current_stats = tfdv.generate_statistics_from_csv('data_week2.csv')

# Infer schema
schema = tfdv.infer_schema(baseline_stats)

# Set drift comparator for features
for feature_name in ['price', 'quantity', 'user_age']:
    feature = tfdv.get_feature(schema, feature_name)
    # L-infinity norm: max absolute difference in normalized histograms
    feature.drift_comparator.infinity_norm.threshold = 0.1
    
    # Or use Jensen-Shannon divergence
    # feature.drift_comparator.jensen_shannon_divergence.threshold = 0.01

# Detect drift
drift_anomalies = tfdv.validate_statistics(
    statistics=current_stats,
    schema=schema,
    previous_statistics=baseline_stats
)

# Display drift anomalies
if drift_anomalies.anomaly_info:
    print("⚠️ Data drift detected!")
    tfdv.display_anomalies(drift_anomalies)
    
    # Send alert
    for feature_name, anomaly_info in drift_anomalies.anomaly_info.items():
        if 'drift' in anomaly_info.description.lower():
            print(f"ALERT: Drift detected in feature '{feature_name}'")
            print(f"Description: {anomaly_info.description}")
else:
    print("✅ No drift detected")

# Visualize drift
tfdv.visualize_statistics(
    lhs_statistics=baseline_stats,
    rhs_statistics=current_stats,
    lhs_name='BASELINE',
    rhs_name='CURRENT'
)
```

### **Example 5: Custom Statistics Options**

```python
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.statistics import stats_options

# Create custom stats options
options = stats_options.StatsOptions(
    # Sample size for large datasets
    sample_rate=0.1,  # Sample 10% of data
    
    # Number of top/bottom values to track
    num_top_values=20,
    num_rank_histogram_buckets=128,
    
    # Quantiles to compute
    num_quantiles_histogram_buckets=10,
    
    # Feature whitelist (only compute stats for these)
    feature_whitelist=['user_id', 'age', 'purchase_amount'],
    
    # Enable semantic domain inference
    infer_type_from_schema=True,
    
    # Schema for guided statistics
    schema=existing_schema,
    
    # Label feature (for classification tasks)
    label_feature='label',
    
    # Weight feature (for weighted statistics)
    weight_feature='sample_weight',
    
    # Enable mutual information computation
    enable_semantic_domain_stats=True,
)

# Generate statistics with custom options
stats = tfdv.generate_statistics_from_csv(
    data_location='data.csv',
    stats_options=options
)

# Visualize
tfdv.visualize_statistics(stats)
```

### **Example 6: Working with Sliced Data**

```python
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils import slicing_util

# Define slicing functions
def slice_by_country(example):
    """Slice data by country"""
    return [(('country', example['country']),)]

def slice_by_age_group(example):
    """Slice data by age group"""
    age = example['age']
    if age < 18:
        return [(('age_group', 'minor'),)]
    elif age < 65:
        return [(('age_group', 'adult'),)]
    else:
        return [(('age_group', 'senior'),)]

# Combine slicing functions
def combined_slice_fn(example):
    """Slice by both country and age group"""
    slices = []
    slices.extend(slice_by_country(example))
    slices.extend(slice_by_age_group(example))
    # Add overall slice
    slices.append(())
    return slices

# Generate sliced statistics
sliced_stats = tfdv.generate_statistics_from_dataframe(
    dataframe=df,
    stats_options=tfdv.StatsOptions(
        slice_functions=[combined_slice_fn]
    )
)

# Visualize specific slice
tfdv.visualize_statistics(
    sliced_stats,
    slicing_config=slicing_util.SlicingConfig(
        slicing_spec=[
            slicing_util.SlicingSpec(
                feature_keys=['country'],
                feature_values=[('country', 'US')]
            )
        ]
    )
)
```

## Integration with Apache Beam

### **Large-Scale Data Processing**

```python
import apache_beam as beam
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils import batch_util

# Define pipeline
def run_validation_pipeline():
    pipeline_options = beam.options.pipeline_options.PipelineOptions(
        runner='DataflowRunner',  # Or 'DirectRunner' for local
        project='my-gcp-project',
        region='us-central1',
        temp_location='gs://my-bucket/temp',
        staging_location='gs://my-bucket/staging'
    )
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        # Read data
        data = (
            pipeline
            | 'ReadData' >> beam.io.ReadFromText('gs://bucket/data.csv')
            | 'ParseCSV' >> beam.Map(parse_csv_line)
        )
        
        # Generate statistics
        stats = (
            data
            | 'GenerateStatistics' >> tfdv.GenerateStatistics(
                stats_options=tfdv.StatsOptions()
            )
        )
        
        # Write statistics
        _ = (
            stats
            | 'WriteStats' >> tfdv.WriteStatisticsToBinaryFile(
                'gs://bucket/stats/train_stats.pb'
            )
        )

# Run pipeline
run_validation_pipeline()

# Load statistics
stats = tfdv.load_statistics('gs://bucket/stats/train_stats.pb')
```

## Integration with TFX (TensorFlow Extended)

### **In a TFX Pipeline**

```python
from tfx import components
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

# Define TFX pipeline with TFDV components
def create_pipeline():
    # Data ingestion
    example_gen = components.CsvExampleGen(
        input_base='gs://bucket/data'
    )
    
    # Statistics generation (uses TFDV)
    statistics_gen = components.StatisticsGen(
        examples=example_gen.outputs['examples']
    )
    
    # Schema generation (uses TFDV)
    schema_gen = components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True
    )
    
    # Example validation (uses TFDV)
    example_validator = components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )
    
    # Transform component
    transform = components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file='transform_module.py'
    )
    
    # Trainer component
    trainer = components.Trainer(
        module_file='trainer_module.py',
        examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=components.TrainArgs(num_steps=10000),
        eval_args=components.EvalArgs(num_steps=1000)
    )
    
    return pipeline.Pipeline(
        pipeline_name='my_tfdv_pipeline',
        pipeline_root='gs://bucket/pipeline_root',
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            trainer
        ]
    )

# Run pipeline
BeamDagRunner().run(create_pipeline())
```

## Anomaly Types and Solutions

### **Common Anomalies**

```python
# 1. FEATURE_TYPE_MISMATCH
# Problem: Feature has different type than expected
# Solution: Fix data type or update schema
schema.feature[i].type = schema_pb2.FLOAT  # Update schema

# 2. FEATURE_MISSING
# Problem: Required feature is missing
# Solution: Add feature or mark as optional
tfdv.get_feature(schema, 'feature_name').presence.min_fraction = 0.0

# 3. UNEXPECTED_STRING_VALUES
# Problem: New categorical values not in domain
# Solution: Update domain or relax constraints
domain = tfdv.get_domain(schema, 'category')
domain.value.extend(['new_value1', 'new_value2'])

# 4. SCHEMA_TRAINING_SERVING_SKEW
# Problem: Distribution differs between training and serving
# Solution: Adjust threshold or fix data pipeline
feature.skew_comparator.infinity_norm.threshold = 0.2  # Relax threshold

# 5. COMPARATOR_L_INFTY_HIGH
# Problem: Data drift detected
# Solution: Retrain model or investigate data changes
feature.drift_comparator.infinity_norm.threshold = 0.15

# 6. FEATURE_COVERAGE_TOO_LOW
# Problem: Too many missing values
# Solution: Fix data or adjust coverage requirement
tfdv.get_feature(schema, 'feature').presence.min_count = 1000

# 7. UNEXPECTED_FEATURE
# Problem: New feature not in schema
# Solution: Add to schema or remove from data
# Option 1: Add to schema
new_feature = schema.feature.add()
new_feature.name = 'new_feature'
new_feature.type = schema_pb2.FLOAT

# Option 2: Mark as extra column to ignore
schema.string_domain.add(name='extra_features', value=['new_feature'])
```

## Advanced Features

### **1. Custom Validators**

```python
from tensorflow_metadata.proto.v0 import anomalies_pb2

def custom_validator(statistics, schema):
    """Custom validation logic"""
    anomalies = {}
    
    for feature_stats in statistics.datasets[0].features:
        feature_name = feature_stats.name
        
        # Custom check: Ensure no feature has > 50% missing values
        if feature_stats.HasField('num_stats'):
            total_count = feature_stats.num_stats.common_stats.num_non_missing
            missing_count = (
                statistics.datasets[0].num_examples - total_count
            )
            missing_ratio = missing_count / statistics.datasets[0].num_examples
            
            if missing_ratio > 0.5:
                anomalies[feature_name] = {
                    'description': f'High missing rate: {missing_ratio:.2%}',
                    'severity': anomalies_pb2.AnomalyInfo.ERROR
                }
    
    return anomalies

# Use custom validator
stats = tfdv.generate_statistics_from_dataframe(df)
custom_anomalies = custom_validator(stats, schema)

if custom_anomalies:
    print("Custom anomalies detected:")
    for feature, info in custom_anomalies.items():
        print(f"  {feature}: {info['description']}")
```

### **2. Environment-Specific Schemas**

```python
# Create base schema
base_schema = tfdv.infer_schema(train_stats)

# Define environment-specific schemas
# Training environment: all features required
training_schema = base_schema
for feature in training_schema.feature:
    feature.presence.min_fraction = 1.0

# Serving environment: some features optional
serving_schema = base_schema
tfdv.get_feature(serving_schema, 'optional_feature').presence.min_fraction = 0.0
tfdv.get_feature(serving_schema, 'label').lifecycle_stage = schema_pb2.DEPRECATED

# Save schemas
tfdv.write_schema_text(training_schema, 'schema_training.pbtxt')
tfdv.write_schema_text(serving_schema, 'schema_serving.pbtxt')

# Validate against appropriate schema
train_anomalies = tfdv.validate_statistics(
    train_stats,
    schema=training_schema
)

serving_anomalies = tfdv.validate_statistics(
    serving_stats,
    schema=serving_schema
)
```

### **3. Monitoring Pipeline**

```python
import tensorflow_data_validation as tfdv
from datetime import datetime
import json

class DataValidator:
    def __init__(self, schema_path, baseline_stats_path):
        self.schema = tfdv.load_schema_text(schema_path)
        self.baseline_stats = tfdv.load_statistics(baseline_stats_path)
        self.alert_log = []
    
    def validate_batch(self, data_path, batch_id):
        """Validate a batch of data"""
        # Generate statistics
        current_stats = tfdv.generate_statistics_from_csv(data_path)
        
        # Validate schema
        schema_anomalies = tfdv.validate_statistics(
            statistics=current_stats,
            schema=self.schema
        )
        
        # Check for drift
        drift_anomalies = tfdv.validate_statistics(
            statistics=current_stats,
            schema=self.schema,
            previous_statistics=self.baseline_stats
        )
        
        # Log results
        results = {
            'batch_id': batch_id,
            'timestamp': datetime.now().isoformat(),
            'schema_violations': len(schema_anomalies.anomaly_info),
            'drift_detected': len(drift_anomalies.anomaly_info),
            'anomalies': []
        }
        
        # Process anomalies
        for feature, info in schema_anomalies.anomaly_info.items():
            results['anomalies'].append({
                'type': 'schema_violation',
                'feature': feature,
                'description': info.description,
                'severity': info.severity
            })
        
        for feature, info in drift_anomalies.anomaly_info.items():
            if 'drift' in info.description.lower():
                results['anomalies'].append({
                    'type': 'drift',
                    'feature': feature,
                    'description': info.description,
                    'severity': info.severity
                })
        
        # Send alerts if needed
        if results['anomalies']:
            self.send_alert(results)
        
        return results
    
    def send_alert(self, results):
        """Send alert for anomalies"""
        alert = {
            'timestamp': results['timestamp'],
            'batch_id': results['batch_id'],
            'total_anomalies': len(results['anomalies']),
            'critical_count': sum(
                1 for a in results['anomalies'] 
                if a['severity'] == 'ERROR'
            )
        }
        
        self.alert_log.append(alert)
        
        # In production: send to monitoring system
        print(f"🚨 ALERT: {alert['total_anomalies']} anomalies detected")
        print(f"   Critical: {alert['critical_count']}")
        print(json.dumps(results, indent=2))
    
    def get_report(self):
        """Generate validation report"""
        return {
            'total_batches_validated': len(self.alert_log),
            'total_alerts': len(self.alert_log),
            'alerts': self.alert_log
        }

# Usage
validator = DataValidator(
    schema_path='schema.pbtxt',
    baseline_stats_path='baseline_stats.pb'
)

# Validate daily batches
for day in range(1, 8):
    results = validator.validate_batch(
        data_path=f'data/day_{day}.csv',
        batch_id=f'batch_{day}'
    )

# Get report
report = validator.get_report()
print(f"\nValidation Report:")
print(f"Total batches: {report['total_batches_validated']}")
print(f"Alerts raised: {report['total_alerts']}")
```

## Best Practices

### **1. Continuous Validation**

```python
# Set up continuous validation
def continuous_validation_pipeline():
    """Run validation on new data continuously"""
    
    # 1. Establish baseline
    baseline_stats = tfdv.generate_statistics_from_csv('baseline_data.csv')
    schema = tfdv.infer_schema(baseline_stats)
    
    # 2. Configure drift thresholds
    for feature_name in ['feature1', 'feature2', 'feature3']:
        feature = tfdv.get_feature(schema, feature_name)
        feature.drift_comparator.infinity_norm.threshold = 0.1
    
    # 3. Save baseline artifacts
    tfdv.write_schema_text(schema, 'schema.pbtxt')
    tfdv.write_statistics_binary(baseline_stats, 'baseline_stats.pb')
    
    # 4. Validate new data periodically
    while True:
        new_data_path = get_latest_data()  # Your data fetching logic
        
        current_stats = tfdv.generate_statistics_from_csv(new_data_path)
        anomalies = tfdv.validate_statistics(
            statistics=current_stats,
            schema=schema,
            previous_statistics=baseline_stats
        )
        
        if anomalies.anomaly_info:
            handle_anomalies(anomalies)
        
        # Update baseline periodically
        if should_update_baseline():
            baseline_stats = current_stats
            tfdv.write_statistics_binary(baseline_stats, 'baseline_stats.pb')
        
        time.sleep(3600)  # Check every hour
```

### **2. Schema Evolution**

```python
def evolve_schema(old_schema, new_stats, auto_update=True):
    """Handle schema evolution safely"""
    
    # Validate with current schema
    anomalies = tfdv.validate_statistics(new_stats, old_schema)
    
    if not anomalies.anomaly_info:
        return old_schema, []
    
    # Analyze anomalies
    updates_needed = []
    
    for feature, info in anomalies.anomaly_info.items():
        if 'UNEXPECTED_STRING_VALUES' in info.short_description:
            updates_needed.append({
                'feature': feature,
                'type': 'domain_expansion',
                'action': 'add_new_values'
            })
        elif 'FEATURE_MISSING' in info.short_description:
            updates_needed.append({
                'feature': feature,
                'type': 'new_feature',
                'action': 'add_to_schema'
            })
    
    if auto_update:
        # Apply updates
        new_schema = old_schema
        for update in updates_needed:
            if update['type'] == 'domain_expansion':
                # Relax domain constraints
                feature = tfdv.get_feature(new_schema, update['feature'])
                if hasattr(feature, 'string_domain'):
                    # Allow new values
                    pass  # Domain will auto-expand
            elif update['type'] == 'new_feature':
                # Add new feature
                pass  # Feature will be added
        
        return new_schema, updates_needed
    else:
        # Return for manual review
        return old_schema, updates_needed
```

### **3. Performance Optimization**

```python
# For large datasets
options = tfdv.StatsOptions(
    # Sample data
    sample_rate=0.1,  # 10% sample
    
    # Limit histogram buckets
    num_quantiles_histogram_buckets=10,
    num_rank_histogram_buckets=128,
    
    # Limit top values
    num_top_values=20,
    
    # Feature whitelist (only important features)
    feature_whitelist=critical_features,
    
    # Disable expensive computations
    enable_semantic_domain_stats=False,
)

# Use Apache Beam for distributed processing
stats = tfdv.generate_statistics_from_tfrecord(
    data_location='gs://bucket/large_dataset/*.tfrecord',
    stats_options=options,
    pipeline_options=beam_options
)
```

## Common Use Cases

### **1. Data Quality Monitoring**
```python
# Monitor data quality metrics
def monitor_data_quality(stats):
    quality_metrics = {}
    
    for feature in stats.datasets[0].features:
        if feature.HasField('num_stats'):
            common = feature.num_stats.common_stats
            quality_metrics[feature.name] = {
                'completeness': 1 - (common.num_missing / common.num_non_missing),
                'mean': feature.num_stats.mean,
                'std_dev': feature.num_stats.std_dev,
                'min': feature.num_stats.min,
                'max': feature.num_stats.max
            }
    
    return quality_metrics
```

### **2. Feature Engineering Validation**
```python
# Validate transformed features
original_stats = tfdv.generate_statistics_from_dataframe(original_df)
transformed_stats = tfdv.generate_statistics_from_dataframe(transformed_df)

# Compare distributions
tfdv.visualize_statistics(
    lhs_statistics=original_stats,
    rhs_statistics=transformed_stats,
    lhs_name='ORIGINAL',
    rhs_name='TRANSFORMED'
)
```

### **3. Model Retraining Triggers**
```python
# Decide when to retrain based on drift
def should_retrain(current_stats, baseline_stats, schema, threshold=3):
    anomalies = tfdv.validate_statistics(
        statistics=current_stats,
        schema=schema,
        previous_statistics=baseline_stats
    )
    
    # Count drift anomalies
    drift_count = sum(
        1 for feature, info in anomalies.anomaly_info.items()
        if 'drift' in info.description.lower()
    )
    
    return drift_count >= threshold
```

## Integration with Vertex AI

```python
from google.cloud import aiplatform
import tensorflow_data_validation as tfdv

# Generate statistics and schema
stats = tfdv.generate_statistics_from_csv('gs://bucket/train_data.csv')
schema = tfdv.infer_schema(stats)

# Save artifacts
tfdv.write_schema_text(schema, 'schema.pbtxt')
tfdv.write_statistics_binary(stats, 'train_stats.pb')

# Upload to GCS for use in Vertex AI pipeline
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('my-bucket')

bucket.blob('tfdv/schema.pbtxt').upload_from_filename('schema.pbtxt')
bucket.blob('tfdv/train_stats.pb').upload_from_filename('train_stats.pb')

# Use in Vertex AI Training
training_job = aiplatform.CustomTrainingJob(
    display_name='training_with_validation',
    script_path='train.py',
    container_uri='us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12:latest',
    requirements=['tensorflow-data-validation'],
    model_serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest'
)

model = training_job.run(
    dataset=dataset,
    args=[
        '--schema-path=gs://my-bucket/tfdv/schema.pbtxt',
        '--stats-path=gs://my-bucket/tfdv/train_stats.pb'
    ]
)
```

## Summary

### **Key Benefits**
- ✅ **Automated data quality checks**: Detect anomalies automatically
- ✅ **Schema management**: Version control for data expectations
- ✅ **Drift detection**: Monitor data distribution changes
- ✅ **Scalable**: Works with large datasets via Apache Beam
- ✅ **Integrated**: Works seamlessly with TFX and Vertex AI
- ✅ **Visualizations**: Built-in visualization tools

### **When to Use TFDV**
- Data quality validation before training
- Monitoring production data pipelines
- Detecting training-serving skew
- Continuous model monitoring
- Feature engineering validation
- Regulatory compliance and auditing

### **Best Practices**
1. Generate baseline statistics from representative training data
2. Customize schemas to match business requirements
3. Set appropriate thresholds for drift detection
4. Automate validation in CI/CD pipelines
5. Version control schemas alongside model code
6. Monitor and alert on critical anomalies
7. Regularly update baseline statistics

---

**Pro Tip**: Integrate TFDV early in your ML pipeline to catch data issues before they affect model training. Use it continuously in production to detect drift and trigger retraining when needed.
