# Vertex AI Best Practices

*Last Updated: December 26, 2025*

## Overview

Vertex AI is Google Cloud's unified machine learning platform that brings together all GCP ML services under one unified API, client library, and user interface. It simplifies the process of building, deploying, and scaling ML models with integrated tools for the entire ML workflow.

---

## 1. Project Organization and Setup

### Workspace Structure

**Best Practices:**
- Organize projects by business domain or team
- Use separate projects for development, staging, and production
- Enable required APIs systematically
- Set up proper IAM roles from the start

```bash
# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable notebooks.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Create service account for Vertex AI
gcloud iam service-accounts create vertex-ai-sa \
    --display-name="Vertex AI Service Account" \
    --project=my-project

# Grant necessary roles
gcloud projects add-iam-policy-binding my-project \
    --member="serviceAccount:vertex-ai-sa@my-project.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding my-project \
    --member="serviceAccount:vertex-ai-sa@my-project.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```

### Regional Configuration

**Choose Appropriate Regions:**
```python
from google.cloud import aiplatform

# Initialize Vertex AI with region and project
aiplatform.init(
    project='my-project',
    location='us-central1',  # Choose based on data residency and latency
    staging_bucket='gs://my-vertex-ai-bucket',
    experiment='my-experiment',
    experiment_tensorboard='projects/my-project/locations/us-central1/tensorboards/123'
)

# Available regions for Vertex AI
RECOMMENDED_REGIONS = {
    'us-central1': 'Iowa - Most features available',
    'us-east4': 'Northern Virginia',
    'us-west1': 'Oregon',
    'europe-west4': 'Netherlands',
    'asia-southeast1': 'Singapore',
}
```

---

## 2. Data Preparation and Management

### Dataset Creation

**Best Practices:**
- Use Vertex AI Datasets for versioning
- Validate data quality before training
- Use appropriate dataset types
- Implement data validation pipelines

```python
from google.cloud import aiplatform

def create_tabular_dataset(
    display_name,
    gcs_source,
    project,
    location
):
    """Create a tabular dataset in Vertex AI."""
    
    aiplatform.init(project=project, location=location)
    
    dataset = aiplatform.TabularDataset.create(
        display_name=display_name,
        gcs_source=gcs_source,
        labels={'env': 'production', 'team': 'ml-team'}
    )
    
    print(f"Dataset created: {dataset.resource_name}")
    print(f"Dataset ID: {dataset.name}")
    
    return dataset

def create_image_dataset(
    display_name,
    gcs_source,
    import_schema_uri,
    project,
    location
):
    """Create an image dataset in Vertex AI."""
    
    aiplatform.init(project=project, location=location)
    
    dataset = aiplatform.ImageDataset.create(
        display_name=display_name,
        gcs_source=gcs_source,
        import_schema_uri=import_schema_uri,
        labels={'type': 'image-classification'}
    )
    
    return dataset

def create_text_dataset(
    display_name,
    gcs_source,
    project,
    location
):
    """Create a text dataset in Vertex AI."""
    
    aiplatform.init(project=project, location=location)
    
    dataset = aiplatform.TextDataset.create(
        display_name=display_name,
        gcs_source=gcs_source,
        labels={'type': 'text-classification'}
    )
    
    return dataset

# Example usage
dataset = create_tabular_dataset(
    display_name='customer-churn-dataset',
    gcs_source=['gs://my-bucket/data/train.csv'],
    project='my-project',
    location='us-central1'
)
```

### Data Validation

**Implement Data Quality Checks:**
```python
import pandas as pd
from google.cloud import storage

def validate_training_data(gcs_path):
    """Validate training data before creating dataset."""
    
    # Read data from GCS
    df = pd.read_csv(gcs_path)
    
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
    }
    
    # Check for issues
    issues = []
    
    # Check for excessive missing values
    for col, missing_count in validation_results['missing_values'].items():
        missing_pct = (missing_count / len(df)) * 100
        if missing_pct > 20:
            issues.append(f"Column '{col}' has {missing_pct:.2f}% missing values")
    
    # Check for duplicates
    if validation_results['duplicate_rows'] > 0:
        issues.append(f"Found {validation_results['duplicate_rows']} duplicate rows")
    
    # Check minimum data size
    if len(df) < 1000:
        issues.append(f"Dataset too small: {len(df)} rows (minimum recommended: 1000)")
    
    if issues:
        print("Data Validation Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Data validation passed!")
        return True

# Validate before creating dataset
if validate_training_data('gs://my-bucket/data/train.csv'):
    dataset = create_tabular_dataset(...)
```

---

## 3. AutoML Training

### Tabular AutoML

**Best Practices:**
- Use AutoML for quick baseline models
- Set appropriate training budget
- Enable early stopping
- Use column transformations wisely

```python
from google.cloud import aiplatform

def train_automl_tabular(
    dataset,
    target_column,
    optimization_objective,
    budget_hours=1.0
):
    """Train AutoML tabular model."""
    
    # Define training job
    job = aiplatform.AutoMLTabularTrainingJob(
        display_name='automl-customer-churn',
        optimization_prediction_type='classification',
        optimization_objective=optimization_objective,  # 'maximize-au-prc', 'maximize-au-roc'
        column_specs={
            'customer_id': 'auto',  # Will be excluded automatically
            'age': 'numeric',
            'income': 'numeric',
            'tenure_months': 'numeric',
            'product_category': 'categorical',
            'region': 'categorical',
        },
        labels={'model-type': 'automl', 'use-case': 'churn-prediction'}
    )
    
    # Train model
    model = job.run(
        dataset=dataset,
        target_column=target_column,
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1,
        budget_milli_node_hours=budget_hours * 1000,
        model_display_name='customer-churn-automl-v1',
        disable_early_stopping=False,
        export_evaluated_data_items=True,
        export_evaluated_data_items_bigquery_destination_uri='bq://my-project.ml_evaluation.automl_results',
    )
    
    print(f"Model resource name: {model.resource_name}")
    print(f"Model ID: {model.name}")
    
    return model

# Example usage
model = train_automl_tabular(
    dataset=dataset,
    target_column='churned',
    optimization_objective='maximize-au-roc',
    budget_hours=2.0
)
```

### Image AutoML

**Image Classification:**
```python
def train_automl_image_classification(
    dataset,
    model_type='CLOUD',
    budget_hours=24
):
    """Train AutoML image classification model."""
    
    job = aiplatform.AutoMLImageTrainingJob(
        display_name='automl-image-classifier',
        prediction_type='classification',
        multi_label=False,
        model_type=model_type,  # 'CLOUD', 'CLOUD_HIGH_ACCURACY_1', 'CLOUD_LOW_LATENCY_1'
        base_model=None,  # Optional: use pre-trained model
        labels={'model-type': 'automl-image', 'task': 'classification'}
    )
    
    model = job.run(
        dataset=dataset,
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1,
        budget_milli_node_hours=budget_hours * 1000,
        model_display_name='image-classifier-v1',
        disable_early_stopping=False,
    )
    
    return model
```

### Text AutoML

**Text Classification:**
```python
def train_automl_text_classification(
    dataset,
    sentiment_max=None,
    multi_label=False
):
    """Train AutoML text classification model."""
    
    job = aiplatform.AutoMLTextTrainingJob(
        display_name='automl-text-classifier',
        prediction_type='classification',
        multi_label=multi_label,
        sentiment_max=sentiment_max,  # For sentiment analysis: 1-10
        labels={'model-type': 'automl-text'}
    )
    
    model = job.run(
        dataset=dataset,
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1,
        model_display_name='text-classifier-v1',
    )
    
    return model
```

---

## 4. Custom Training

### Custom Training Jobs

**Best Practices:**
- Use custom containers for flexibility
- Implement checkpointing
- Use distributed training for large datasets
- Monitor training metrics

```python
from google.cloud import aiplatform

def create_custom_training_job(
    display_name,
    container_uri,
    model_serving_container_uri,
    args,
    replica_count=1,
    machine_type='n1-standard-4',
    accelerator_type=None,
    accelerator_count=0
):
    """Create custom training job with container."""
    
    # Define training job
    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=container_uri,
        model_serving_container_image_uri=model_serving_container_uri,
        model_serving_container_predict_route='/predict',
        model_serving_container_health_route='/health',
        labels={'training-type': 'custom'}
    )
    
    # Run training
    model = job.run(
        args=args,
        replica_count=replica_count,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        base_output_dir='gs://my-bucket/model-output',
        model_display_name=f'{display_name}-model',
        environment_variables={
            'EPOCHS': '100',
            'BATCH_SIZE': '32',
            'LEARNING_RATE': '0.001',
        },
        sync=True,
    )
    
    return model

# Example with GPU
model = create_custom_training_job(
    display_name='tensorflow-training',
    container_uri='gcr.io/my-project/tensorflow-training:latest',
    model_serving_container_uri='gcr.io/my-project/tensorflow-serving:latest',
    args=['--training-data', 'gs://my-bucket/data/train.csv'],
    replica_count=1,
    machine_type='n1-standard-8',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1
)
```

### Distributed Training

**Multi-GPU Training:**
```python
def create_distributed_training_job(
    display_name,
    python_package_gcs_uri,
    python_module_name,
    args,
    worker_pool_specs
):
    """Create distributed custom training job."""
    
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=display_name,
        python_package_gcs_uri=python_package_gcs_uri,
        python_module_name=python_module_name,
        container_uri='us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12:latest',
        labels={'training-type': 'distributed'}
    )
    
    # Define worker pools
    # worker_pool_specs = [
    #     {
    #         'machine_spec': {
    #             'machine_type': 'n1-standard-8',
    #             'accelerator_type': 'NVIDIA_TESLA_V100',
    #             'accelerator_count': 2,
    #         },
    #         'replica_count': 1,
    #         'container_spec': {
    #             'image_uri': container_uri,
    #         },
    #     },
    # ]
    
    model = job.run(
        args=args,
        replica_count=4,  # Number of workers
        machine_type='n1-highmem-8',
        accelerator_type='NVIDIA_TESLA_V100',
        accelerator_count=2,
        base_output_dir='gs://my-bucket/distributed-output',
        model_display_name='distributed-model-v1',
        sync=True,
    )
    
    return model
```

### Hyperparameter Tuning

**Best Practices:**
- Use Bayesian optimization
- Start with coarse search, then fine-tune
- Set reasonable parameter ranges
- Use parallel trials for faster results

```python
from google.cloud.aiplatform import hyperparameter_tuning as hpt

def run_hyperparameter_tuning(
    display_name,
    container_uri,
    metrics_spec,
    parameter_spec
):
    """Run hyperparameter tuning job."""
    
    # Define custom training job
    custom_job = aiplatform.CustomJob.from_local_script(
        display_name=display_name,
        script_path='trainer/task.py',
        container_uri=container_uri,
        requirements=['tensorflow==2.12.0', 'pandas', 'scikit-learn'],
        replica_count=1,
        machine_type='n1-standard-4',
        accelerator_type='NVIDIA_TESLA_T4',
        accelerator_count=1,
    )
    
    # Define hyperparameter tuning job
    hp_job = aiplatform.HyperparameterTuningJob(
        display_name=f'{display_name}-hp-tuning',
        custom_job=custom_job,
        metric_spec=metrics_spec,
        parameter_spec=parameter_spec,
        max_trial_count=20,
        parallel_trial_count=5,
        max_failed_trial_count=3,
        search_algorithm='bayesian',  # 'grid', 'random', 'bayesian'
        labels={'job-type': 'hp-tuning'}
    )
    
    # Run tuning
    hp_job.run(
        sync=True,
        timeout=7200,  # 2 hours
    )
    
    # Get best trial
    best_trial = hp_job.trials[0]
    print(f"Best trial ID: {best_trial.id}")
    print(f"Best trial metrics: {best_trial.final_measurement.metrics}")
    print(f"Best trial parameters: {best_trial.parameters}")
    
    return hp_job

# Example usage
metrics_spec = {
    'accuracy': 'maximize',
    'loss': 'minimize',
}

parameter_spec = {
    'learning_rate': hpt.DoubleParameterSpec(min=0.0001, max=0.1, scale='log'),
    'batch_size': hpt.DiscreteParameterSpec(values=[16, 32, 64, 128], scale='linear'),
    'num_layers': hpt.IntegerParameterSpec(min=2, max=10, scale='linear'),
    'dropout_rate': hpt.DoubleParameterSpec(min=0.1, max=0.5, scale='linear'),
    'optimizer': hpt.CategoricalParameterSpec(values=['adam', 'sgd', 'rmsprop']),
}

hp_job = run_hyperparameter_tuning(
    display_name='model-hp-tuning',
    container_uri='gcr.io/my-project/trainer:latest',
    metrics_spec=metrics_spec,
    parameter_spec=parameter_spec
)
```

---

## 5. Model Evaluation

### Evaluation Metrics

**Best Practices:**
- Evaluate on multiple metrics
- Use holdout test set
- Compare against baseline
- Analyze confusion matrix and feature importance

```python
def evaluate_model(model, test_dataset):
    """Evaluate trained model."""
    
    # Get model evaluation
    model_evaluations = model.list_model_evaluations()
    
    for evaluation in model_evaluations:
        print(f"\nEvaluation ID: {evaluation.name}")
        print(f"Metrics: {evaluation.metrics}")
        
        # For classification
        if 'auPrc' in evaluation.metrics:
            print(f"  AU-PRC: {evaluation.metrics['auPrc']}")
            print(f"  AU-ROC: {evaluation.metrics['auRoc']}")
            print(f"  Log Loss: {evaluation.metrics['logLoss']}")
            
            # Confusion matrix
            if 'confusionMatrix' in evaluation.metrics:
                print(f"  Confusion Matrix: {evaluation.metrics['confusionMatrix']}")
        
        # For regression
        if 'meanAbsoluteError' in evaluation.metrics:
            print(f"  MAE: {evaluation.metrics['meanAbsoluteError']}")
            print(f"  RMSE: {evaluation.metrics['rootMeanSquaredError']}")
            print(f"  R²: {evaluation.metrics['rSquared']}")
    
    return model_evaluations

def get_feature_importance(model):
    """Get feature importance from model."""
    
    # Get model metadata
    metadata = model.to_dict()
    
    if 'explanationSpec' in metadata:
        explanations = metadata['explanationSpec']
        print("Feature Importance:")
        for feature in explanations.get('metadata', {}).get('outputs', {}):
            print(f"  {feature}")
    
    return metadata

# Batch prediction for evaluation
def batch_predict_for_evaluation(
    model,
    gcs_source,
    gcs_destination
):
    """Run batch prediction for model evaluation."""
    
    batch_prediction_job = model.batch_predict(
        job_display_name='evaluation-batch-prediction',
        gcs_source=gcs_source,
        gcs_destination_prefix=gcs_destination,
        machine_type='n1-standard-4',
        accelerator_type=None,
        accelerator_count=0,
        starting_replica_count=1,
        max_replica_count=10,
        sync=True,
    )
    
    print(f"Batch prediction job: {batch_prediction_job.resource_name}")
    print(f"Output location: {batch_prediction_job.output_info.gcs_output_directory}")
    
    return batch_prediction_job
```

---

## 6. Model Deployment

### Endpoint Creation and Deployment

**Best Practices:**
- Use appropriate machine types
- Enable auto-scaling
- Use traffic splitting for A/B testing
- Implement health checks

```python
def deploy_model_to_endpoint(
    model,
    endpoint_display_name,
    machine_type='n1-standard-4',
    min_replica_count=1,
    max_replica_count=3,
    accelerator_type=None,
    accelerator_count=0
):
    """Deploy model to Vertex AI endpoint."""
    
    # Create endpoint
    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_display_name,
        labels={'env': 'production', 'version': 'v1'}
    )
    
    # Deploy model to endpoint
    deployed_model = model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f'{model.display_name}-deployment',
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        traffic_percentage=100,
        traffic_split={'0': 100},  # All traffic to this model initially
        metadata={
            'deployed_by': 'ml-team',
            'deployment_date': '2025-01-15',
        },
        sync=True,
    )
    
    print(f"Endpoint: {endpoint.resource_name}")
    print(f"Endpoint URL: {endpoint.gca_resource.deployed_models[0].display_name}")
    
    return endpoint

# Deploy model
endpoint = deploy_model_to_endpoint(
    model=model,
    endpoint_display_name='customer-churn-endpoint',
    machine_type='n1-standard-4',
    min_replica_count=2,
    max_replica_count=10,
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1
)
```

### Traffic Splitting for A/B Testing

**Deploy Multiple Model Versions:**
```python
def deploy_model_version_with_traffic_split(
    endpoint,
    new_model,
    traffic_percentage=20
):
    """Deploy new model version with traffic splitting."""
    
    # Get existing traffic split
    existing_traffic = {}
    for deployed_model in endpoint.gca_resource.deployed_models:
        existing_traffic[deployed_model.id] = (100 - traffic_percentage)
    
    # Deploy new model with traffic split
    new_deployed_model = new_model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f'{new_model.display_name}-v2',
        machine_type='n1-standard-4',
        min_replica_count=1,
        max_replica_count=5,
        traffic_percentage=traffic_percentage,
        traffic_split=existing_traffic,
        sync=True,
    )
    
    print(f"New model deployed with {traffic_percentage}% traffic")
    print(f"Traffic split: {endpoint.traffic_split}")
    
    return new_deployed_model

# Gradually increase traffic to new model
def update_traffic_split(endpoint, deployed_model_id, new_percentage):
    """Update traffic split for deployed model."""
    
    # Calculate new traffic split
    traffic_split = {}
    for dm in endpoint.gca_resource.deployed_models:
        if dm.id == deployed_model_id:
            traffic_split[dm.id] = new_percentage
        else:
            traffic_split[dm.id] = (100 - new_percentage) / (len(endpoint.gca_resource.deployed_models) - 1)
    
    # Update endpoint
    endpoint.update(traffic_split=traffic_split)
    
    print(f"Updated traffic split: {endpoint.traffic_split}")
```

---

## 7. Prediction and Inference

### Online Prediction

**Best Practices:**
- Batch requests when possible
- Use appropriate timeout values
- Implement retry logic
- Monitor prediction latency

```python
def predict_online(endpoint, instances):
    """Make online predictions."""
    
    predictions = endpoint.predict(instances=instances)
    
    return predictions

def predict_with_explanations(endpoint, instances):
    """Get predictions with explanations."""
    
    predictions = endpoint.explain(instances=instances)
    
    print("Predictions:")
    for idx, prediction in enumerate(predictions.predictions):
        print(f"\nInstance {idx}:")
        print(f"  Prediction: {prediction}")
        
        if predictions.explanations:
            print(f"  Feature Attributions:")
            for attribution in predictions.explanations[idx].attributions:
                print(f"    {attribution}")
    
    return predictions

# Example usage
instances = [
    {
        'age': 35,
        'income': 75000,
        'tenure_months': 24,
        'product_category': 'premium',
        'region': 'west',
    },
    {
        'age': 42,
        'income': 95000,
        'tenure_months': 48,
        'product_category': 'standard',
        'region': 'east',
    }
]

predictions = predict_online(endpoint, instances)
print(f"Predictions: {predictions.predictions}")
```

### Batch Prediction

**For Large-Scale Inference:**
```python
def run_batch_prediction(
    model,
    gcs_source,
    gcs_destination,
    machine_type='n1-standard-4'
):
    """Run batch prediction job."""
    
    batch_prediction_job = model.batch_predict(
        job_display_name='batch-prediction-job',
        gcs_source=gcs_source,
        gcs_destination_prefix=gcs_destination,
        instances_format='jsonl',  # 'jsonl', 'csv', 'bigquery'
        predictions_format='jsonl',
        machine_type=machine_type,
        accelerator_type='NVIDIA_TESLA_T4',
        accelerator_count=1,
        starting_replica_count=1,
        max_replica_count=10,
        generate_explanation=True,
        sync=True,
    )
    
    print(f"Batch prediction completed")
    print(f"Output: {batch_prediction_job.output_info.gcs_output_directory}")
    
    return batch_prediction_job

# Batch predict from BigQuery
def batch_predict_from_bigquery(
    model,
    bigquery_source,
    bigquery_destination
):
    """Run batch prediction with BigQuery as source and destination."""
    
    batch_prediction_job = model.batch_predict(
        job_display_name='bq-batch-prediction',
        bigquery_source=bigquery_source,
        bigquery_destination_prefix=bigquery_destination,
        instances_format='bigquery',
        predictions_format='bigquery',
        machine_type='n1-standard-4',
        sync=True,
    )
    
    return batch_prediction_job

# Example
batch_job = run_batch_prediction(
    model=model,
    gcs_source='gs://my-bucket/batch-input/*.jsonl',
    gcs_destination='gs://my-bucket/batch-output/',
    machine_type='n1-highmem-4'
)
```

---

## 8. Feature Store

### Feature Store Setup

**Best Practices:**
- Use Feature Store for feature reuse
- Implement online and offline serving
- Version features properly
- Monitor feature drift

```python
from google.cloud.aiplatform import Feature, Featurestore, EntityType

def create_featurestore(
    featurestore_id,
    project,
    location,
    online_serving_config=None
):
    """Create a Vertex AI Feature Store."""
    
    aiplatform.init(project=project, location=location)
    
    # Configure online serving
    if online_serving_config is None:
        online_serving_config = {
            'fixed_node_count': 1,  # For low-latency serving
        }
    
    featurestore = Featurestore.create(
        featurestore_id=featurestore_id,
        online_store_fixed_node_count=1,
        labels={'env': 'production', 'team': 'ml'}
    )
    
    print(f"Feature Store created: {featurestore.resource_name}")
    
    return featurestore

def create_entity_type(
    featurestore,
    entity_type_id,
    description
):
    """Create an entity type in Feature Store."""
    
    entity_type = featurestore.create_entity_type(
        entity_type_id=entity_type_id,
        description=description,
        labels={'type': 'customer'}
    )
    
    print(f"Entity type created: {entity_type.resource_name}")
    
    return entity_type

def create_features(entity_type, feature_configs):
    """Create features in entity type."""
    
    features = []
    
    for config in feature_configs:
        feature = entity_type.create_feature(
            feature_id=config['feature_id'],
            value_type=config['value_type'],  # 'BOOL', 'DOUBLE', 'INT64', 'STRING'
            description=config.get('description', ''),
            labels=config.get('labels', {})
        )
        features.append(feature)
        print(f"Feature created: {feature.resource_name}")
    
    return features

# Example: Create Feature Store for customer features
featurestore = create_featurestore(
    featurestore_id='customer-features',
    project='my-project',
    location='us-central1'
)

entity_type = create_entity_type(
    featurestore=featurestore,
    entity_type_id='customer',
    description='Customer entity with features'
)

features = create_features(
    entity_type=entity_type,
    feature_configs=[
        {
            'feature_id': 'age',
            'value_type': 'INT64',
            'description': 'Customer age in years'
        },
        {
            'feature_id': 'lifetime_value',
            'value_type': 'DOUBLE',
            'description': 'Customer lifetime value'
        },
        {
            'feature_id': 'is_premium',
            'value_type': 'BOOL',
            'description': 'Premium customer flag'
        },
        {
            'feature_id': 'segment',
            'value_type': 'STRING',
            'description': 'Customer segment'
        }
    ]
)
```

### Ingesting Features

**Batch and Stream Ingestion:**
```python
def batch_ingest_features(entity_type, feature_ids, feature_time, gcs_source):
    """Batch ingest features from GCS."""
    
    entity_type.ingest_from_gcs(
        feature_ids=feature_ids,
        feature_time=feature_time,
        gcs_source_uris=[gcs_source],
        gcs_source_type='csv',
        entity_id_field='customer_id',
        worker_count=10,
    )
    
    print(f"Batch ingestion started from {gcs_source}")

def stream_ingest_features(entity_type, entity_id, feature_values):
    """Stream ingest features for a single entity."""
    
    entity_type.write_feature_values(
        instances=[
            {
                'entity_id': entity_id,
                'feature_values': feature_values
            }
        ]
    )
    
    print(f"Features ingested for entity {entity_id}")

# Batch ingest from CSV
batch_ingest_features(
    entity_type=entity_type,
    feature_ids=['age', 'lifetime_value', 'is_premium', 'segment'],
    feature_time='2025-01-15T00:00:00Z',
    gcs_source='gs://my-bucket/features/customer_features.csv'
)

# Stream ingest
stream_ingest_features(
    entity_type=entity_type,
    entity_id='customer-12345',
    feature_values={
        'age': 35,
        'lifetime_value': 50000.0,
        'is_premium': True,
        'segment': 'high-value'
    }
)
```

### Reading Features

**Online and Offline Serving:**
```python
def read_features_online(entity_type, entity_ids, feature_ids):
    """Read features for online serving."""
    
    feature_values = entity_type.read(
        entity_ids=entity_ids,
        feature_ids=feature_ids
    )
    
    return feature_values

def read_features_offline(entity_type, feature_ids, start_time, end_time):
    """Read features for offline training."""
    
    # Export to BigQuery for batch processing
    entity_type.batch_serve_to_bq(
        bq_destination_output_uri='bq://my-project.ml_features.customer_features',
        feature_ids=feature_ids,
        start_time=start_time,
        end_time=end_time,
        read_instances_uri='gs://my-bucket/entity-ids.csv',
    )
    
    print("Features exported to BigQuery")

# Online read
features = read_features_online(
    entity_type=entity_type,
    entity_ids=['customer-12345', 'customer-67890'],
    feature_ids=['age', 'lifetime_value', 'is_premium']
)

print("Features:", features)
```

---

## 9. MLOps and Model Management

### Model Versioning

**Best Practices:**
- Use model registry
- Tag models with metadata
- Track model lineage
- Implement model governance

```python
def register_model(model, version_aliases=None, version_description=None):
    """Register model in Model Registry."""
    
    if version_aliases is None:
        version_aliases = ['champion']
    
    # Upload model to registry
    uploaded_model = aiplatform.Model.upload(
        display_name=model.display_name,
        artifact_uri=model.uri,
        serving_container_image_uri=model.container_spec.image_uri,
        labels={
            'framework': 'tensorflow',
            'version': 'v1',
            'team': 'ml-team'
        },
        version_aliases=version_aliases,
        version_description=version_description,
    )
    
    print(f"Model registered: {uploaded_model.resource_name}")
    
    return uploaded_model

def get_model_version(model_name, version_alias='champion'):
    """Get specific model version."""
    
    models = aiplatform.Model.list(
        filter=f'display_name="{model_name}"'
    )
    
    for model in models:
        if version_alias in model.version_aliases:
            print(f"Found model version: {model.resource_name}")
            return model
    
    return None
```

### Model Monitoring

**Monitor Model Performance:**
```python
from google.cloud.aiplatform import model_monitoring

def create_model_monitoring_job(
    endpoint,
    display_name,
    emails,
    skew_thresholds=None,
    drift_thresholds=None
):
    """Create model monitoring job."""
    
    if skew_thresholds is None:
        skew_thresholds = {'age': 0.1, 'income': 0.1}
    
    if drift_thresholds is None:
        drift_thresholds = {'age': 0.1, 'income': 0.1}
    
    monitoring_job = model_monitoring.ModelDeploymentMonitoringJob.create(
        display_name=display_name,
        logging_sampling_strategy=model_monitoring.RandomSampleConfig(sample_rate=0.2),
        schedule_config=model_monitoring.ScheduleConfig(monitor_interval=3600),  # 1 hour
        alert_config=model_monitoring.EmailAlertConfig(
            user_emails=emails
        ),
        objective_configs=[
            model_monitoring.ObjectiveConfig(
                training_dataset=model_monitoring.TrainingDatasetConfig(
                    gcs_source='gs://my-bucket/training-data/*.csv'
                ),
                training_prediction_skew_detection_config=model_monitoring.TrainingPredictionSkewDetectionConfig(
                    skew_thresholds=skew_thresholds
                ),
                prediction_drift_detection_config=model_monitoring.PredictionDriftDetectionConfig(
                    drift_thresholds=drift_thresholds
                ),
            )
        ],
        endpoint=endpoint,
    )
    
    print(f"Monitoring job created: {monitoring_job.resource_name}")
    
    return monitoring_job

# Create monitoring
monitoring_job = create_model_monitoring_job(
    endpoint=endpoint,
    display_name='customer-churn-monitoring',
    emails=['ml-team@example.com'],
    skew_thresholds={'age': 0.1, 'income': 0.15},
    drift_thresholds={'age': 0.1, 'income': 0.15}
)
```

---

## 10. Pipelines with Vertex AI

### Pipeline Creation

**Best Practices:**
- Use Kubeflow Pipelines SDK
- Modularize pipeline components
- Use parameters for flexibility
- Implement error handling

```python
from kfp.v2 import dsl, compiler
from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics

@component(
    packages_to_install=['google-cloud-aiplatform', 'pandas', 'scikit-learn']
)
def prepare_data_component(
    input_data: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    test_size: float = 0.2
):
    """Prepare and split data."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Read input data
    df = pd.read_csv(input_data.path)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    # Save splits
    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

@component(
    packages_to_install=['google-cloud-aiplatform', 'pandas', 'scikit-learn']
)
def train_model_component(
    train_data: Input[Dataset],
    model_output: Output[Model],
    metrics: Output[Metrics],
    learning_rate: float = 0.01,
    n_estimators: int = 100
):
    """Train model."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    import joblib
    
    # Load training data
    df = pd.read_csv(train_data.path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    
    # Calculate metrics
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    # Save model
    joblib.dump(model, model_output.path)
    
    # Log metrics
    metrics.log_metric('accuracy', accuracy)
    
    print(f"Model trained with accuracy: {accuracy}")

@component(
    packages_to_install=['google-cloud-aiplatform']
)
def deploy_model_component(
    model: Input[Model],
    project: str,
    location: str,
    endpoint_display_name: str
):
    """Deploy model to endpoint."""
    from google.cloud import aiplatform
    
    aiplatform.init(project=project, location=location)
    
    # Upload model
    uploaded_model = aiplatform.Model.upload(
        display_name='pipeline-trained-model',
        artifact_uri=model.uri,
        serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest',
    )
    
    # Create endpoint
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)
    
    # Deploy model
    uploaded_model.deploy(
        endpoint=endpoint,
        machine_type='n1-standard-4',
        min_replica_count=1,
        max_replica_count=3,
    )
    
    print(f"Model deployed to endpoint: {endpoint.resource_name}")

@dsl.pipeline(
    name='ml-training-pipeline',
    description='End-to-end ML training and deployment pipeline'
)
def ml_pipeline(
    project: str,
    location: str,
    input_data_path: str,
    endpoint_display_name: str,
    test_size: float = 0.2,
    learning_rate: float = 0.01,
    n_estimators: int = 100
):
    """Define ML pipeline."""
    
    # Step 1: Prepare data
    prepare_data_task = prepare_data_component(
        input_data=input_data_path,
        test_size=test_size
    )
    
    # Step 2: Train model
    train_model_task = train_model_component(
        train_data=prepare_data_task.outputs['train_data'],
        learning_rate=learning_rate,
        n_estimators=n_estimators
    )
    
    # Step 3: Deploy model
    deploy_model_task = deploy_model_component(
        model=train_model_task.outputs['model_output'],
        project=project,
        location=location,
        endpoint_display_name=endpoint_display_name
    )

# Compile pipeline
compiler.Compiler().compile(
    pipeline_func=ml_pipeline,
    package_path='ml_pipeline.json'
)

# Run pipeline
def run_pipeline(project, location, pipeline_root):
    """Execute pipeline."""
    
    aiplatform.init(project=project, location=location)
    
    job = aiplatform.PipelineJob(
        display_name='ml-training-pipeline-run',
        template_path='ml_pipeline.json',
        pipeline_root=pipeline_root,
        parameter_values={
            'project': project,
            'location': location,
            'input_data_path': 'gs://my-bucket/data/train.csv',
            'endpoint_display_name': 'pipeline-endpoint',
            'test_size': 0.2,
            'learning_rate': 0.01,
            'n_estimators': 100,
        },
        enable_caching=True,
    )
    
    job.run(sync=True)
    
    print(f"Pipeline completed: {job.resource_name}")
    
    return job

# Execute
pipeline_job = run_pipeline(
    project='my-project',
    location='us-central1',
    pipeline_root='gs://my-bucket/pipeline-root'
)
```

---

## 11. Cost Optimization

### Best Practices

**Reduce Training Costs:**
```python
# Use preemptible VMs
job = aiplatform.CustomTrainingJob(...)
model = job.run(
    replica_count=4,
    machine_type='n1-standard-8',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    reduction_server_replica_count=1,
    reduction_server_machine_type='n1-highcpu-16',
    base_output_dir='gs://my-bucket/output',
    # Use Spot VMs (preemptible)
    enable_web_access=False,
    timeout=7200,
)

# Use appropriate machine types
MACHINE_TYPE_RECOMMENDATIONS = {
    'small-model': 'n1-standard-4',
    'medium-model': 'n1-standard-8',
    'large-model': 'n1-highmem-16',
    'gpu-training': 'n1-standard-8',  # With GPU accelerator
}

# Optimize batch prediction
batch_job = model.batch_predict(
    instances_format='jsonl',
    machine_type='n1-standard-4',  # Use smaller machines
    starting_replica_count=1,
    max_replica_count=10,  # Scale as needed
)
```

**Monitor Costs:**
```sql
-- Query billing data in BigQuery
SELECT
    DATE(usage_start_time) AS usage_date,
    service.description AS service,
    sku.description AS sku,
    SUM(cost) AS total_cost,
    SUM(usage.amount) AS usage_amount,
    usage.unit AS usage_unit
FROM
    `project.dataset.gcp_billing_export_v1_XXXXX`
WHERE
    service.description LIKE '%Vertex AI%'
    AND DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY
    usage_date,
    service,
    sku,
    usage_unit
ORDER BY
    usage_date DESC,
    total_cost DESC;
```

---

## 12. Common Anti-Patterns

### ❌ Anti-Pattern 1: No Data Validation
**Problem:** Training on bad data
**Solution:** Implement comprehensive data validation

### ❌ Anti-Pattern 2: Ignoring Model Monitoring
**Problem:** Model degradation goes unnoticed
**Solution:** Set up continuous monitoring and alerts

### ❌ Anti-Pattern 3: Over-provisioning Resources
**Problem:** Unnecessary costs
**Solution:** Right-size machine types and use auto-scaling

### ❌ Anti-Pattern 4: No Model Versioning
**Problem:** Cannot rollback to previous versions
**Solution:** Use Model Registry with version aliases

### ❌ Anti-Pattern 5: Hardcoded Parameters
**Problem:** Inflexible pipelines
**Solution:** Use pipeline parameters and config files

---

## 13. Quick Reference Checklist

### Setup
- [ ] Enable Vertex AI API
- [ ] Set up IAM roles and service accounts
- [ ] Create Cloud Storage buckets
- [ ] Configure regional settings
- [ ] Set up billing alerts

### Data Preparation
- [ ] Validate data quality
- [ ] Create Vertex AI datasets
- [ ] Split data appropriately
- [ ] Document data schema
- [ ] Version datasets

### Training
- [ ] Choose appropriate algorithm
- [ ] Set training budget
- [ ] Configure hyperparameters
- [ ] Enable checkpointing
- [ ] Monitor training progress

### Evaluation
- [ ] Evaluate on test set
- [ ] Compare multiple metrics
- [ ] Analyze feature importance
- [ ] Check for bias
- [ ] Validate against baseline

### Deployment
- [ ] Create endpoint
- [ ] Configure auto-scaling
- [ ] Enable monitoring
- [ ] Test predictions
- [ ] Implement A/B testing

### MLOps
- [ ] Register models
- [ ] Set up monitoring
- [ ] Create pipelines
- [ ] Implement CI/CD
- [ ] Document workflows

---

*Best Practices for Google Cloud Data Engineer Certification*
