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

## 13. Explainable AI (XAI)

### Feature Attributions

**Best Practices:**
- Enable explanations for model predictions
- Use appropriate explanation methods
- Visualize feature importance
- Document explanation methodology

```python
from google.cloud.aiplatform import explain

def deploy_model_with_explanations(
    model,
    endpoint_display_name,
    explanation_metadata,
    explanation_parameters
):
    """Deploy model with explanation configuration."""
    
    # Create endpoint
    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_display_name,
        labels={'explainable': 'true'}
    )
    
    # Deploy with explanations enabled
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f'{model.display_name}-explained',
        machine_type='n1-standard-4',
        min_replica_count=1,
        max_replica_count=3,
        explanation_metadata=explanation_metadata,
        explanation_parameters=explanation_parameters,
        sync=True,
    )
    
    return endpoint

# Configure explanations
explanation_metadata = aiplatform.explain.ExplanationMetadata(
    inputs={
        'age': {'input_tensor_name': 'age'},
        'income': {'input_tensor_name': 'income'},
        'tenure_months': {'input_tensor_name': 'tenure_months'},
        'product_category': {'input_tensor_name': 'product_category'},
    },
    outputs={
        'prediction': {'output_tensor_name': 'prediction'}
    }
)

explanation_parameters = aiplatform.explain.ExplanationParameters(
    {
        'sampled_shapley_attribution': {
            'path_count': 10
        }
    }
)

# Alternative: Integrated Gradients
explanation_parameters_ig = aiplatform.explain.ExplanationParameters(
    {
        'integrated_gradients_attribution': {
            'step_count': 50,
            'smooth_grad_config': {
                'noise_sigma': 0.1,
                'noisy_sample_count': 10
            }
        }
    }
)

# Deploy with explanations
endpoint = deploy_model_with_explanations(
    model=model,
    endpoint_display_name='explainable-model-endpoint',
    explanation_metadata=explanation_metadata,
    explanation_parameters=explanation_parameters
)
```

### Understanding Explanations

**Interpret Feature Attributions:**
```python
def get_predictions_with_explanations(endpoint, instances):
    """Get predictions with detailed explanations."""
    
    response = endpoint.explain(instances=instances)
    
    for idx, (prediction, explanation) in enumerate(zip(response.predictions, response.explanations)):
        print(f"\n=== Instance {idx + 1} ===")
        print(f"Prediction: {prediction}")
        
        # Get feature attributions
        if explanation.attributions:
            print("\nFeature Attributions:")
            attributions = explanation.attributions[0]
            
            # Sort by absolute attribution value
            feature_attributions = []
            for feature_name, attribution in attributions.feature_attributions.items():
                feature_attributions.append((feature_name, attribution))
            
            feature_attributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Display top features
            for feature_name, attribution in feature_attributions[:10]:
                direction = "↑" if attribution > 0 else "↓"
                print(f"  {direction} {feature_name}: {attribution:.4f}")
    
    return response

# Example usage
instances = [
    {'age': 35, 'income': 75000, 'tenure_months': 24, 'product_category': 'premium'}
]

explanations = get_predictions_with_explanations(endpoint, instances)
```

### Visualization of Explanations

**Create Explanation Visualizations:**
```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_feature_attributions(feature_names, attributions, title='Feature Attributions'):
    """Visualize feature attributions as bar chart."""
    
    # Sort features by absolute attribution
    sorted_indices = np.argsort(np.abs(attributions))[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_attributions = [attributions[i] for i in sorted_indices]
    
    # Create bar chart
    colors = ['green' if x > 0 else 'red' for x in sorted_attributions]
    
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features[:15], sorted_attributions[:15], color=colors)
    plt.xlabel('Attribution Value')
    plt.title(title)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('feature_attributions.png')
    plt.show()
    
    print(f"Visualization saved as 'feature_attributions.png'")
```

---

## 14. Model Cards and Governance

### Model Card Creation

**Best Practices:**
- Document model details
- Include performance metrics
- Describe limitations
- Specify intended use cases

```python
from google.cloud import aiplatform

def create_model_card(
    model,
    model_card_data
):
    """Create comprehensive model card."""
    
    model_card = {
        'model_details': {
            'name': model.display_name,
            'version': model_card_data.get('version', '1.0'),
            'description': model_card_data.get('description', ''),
            'owners': model_card_data.get('owners', []),
            'license': model_card_data.get('license', 'Proprietary'),
            'references': model_card_data.get('references', []),
        },
        'model_parameters': {
            'model_architecture': model_card_data.get('architecture', ''),
            'input_format': model_card_data.get('input_format', {}),
            'output_format': model_card_data.get('output_format', {}),
        },
        'considerations': {
            'users': model_card_data.get('intended_users', []),
            'use_cases': model_card_data.get('use_cases', []),
            'limitations': model_card_data.get('limitations', []),
            'tradeoffs': model_card_data.get('tradeoffs', []),
        },
        'training_data': {
            'description': model_card_data.get('training_data_description', ''),
            'size': model_card_data.get('training_data_size', 0),
            'preprocessing': model_card_data.get('preprocessing_steps', []),
        },
        'evaluation_data': {
            'description': model_card_data.get('eval_data_description', ''),
            'size': model_card_data.get('eval_data_size', 0),
        },
        'metrics': model_card_data.get('metrics', {}),
        'ethical_considerations': model_card_data.get('ethical_considerations', []),
        'caveats_recommendations': model_card_data.get('caveats', []),
    }
    
    # Save model card as metadata
    model.update(
        description=f"{model.description}\n\nModel Card: {model_card}",
        labels={
            **model.labels,
            'has_model_card': 'true',
            'version': model_card['model_details']['version']
        }
    )
    
    return model_card

# Example model card
model_card_data = {
    'version': '1.0',
    'description': 'Customer churn prediction model using Random Forest',
    'owners': ['ML Team', 'ml-team@example.com'],
    'license': 'Proprietary',
    'architecture': 'Random Forest Classifier',
    'intended_users': ['Business Analysts', 'Marketing Teams'],
    'use_cases': ['Predict customer churn', 'Identify at-risk customers'],
    'limitations': [
        'Model trained on US data only',
        'May not generalize to international customers',
        'Requires minimum 6 months of customer history'
    ],
    'training_data_description': 'Historical customer data from 2022-2024',
    'training_data_size': 500000,
    'metrics': {
        'accuracy': 0.87,
        'precision': 0.85,
        'recall': 0.82,
        'f1_score': 0.83,
        'auc_roc': 0.91
    },
    'ethical_considerations': [
        'Potential bias in historical data',
        'Regular monitoring for fairness required',
        'Should not be sole factor in business decisions'
    ],
}

model_card = create_model_card(model, model_card_data)
```

### Model Governance

**Implement Governance Policies:**
```python
def implement_model_governance(project, location):
    """Set up model governance policies."""
    
    aiplatform.init(project=project, location=location)
    
    # Define governance policies
    governance_policies = {
        'approval_required': True,
        'reviewers': ['senior-ml-engineer@example.com', 'ml-lead@example.com'],
        'performance_threshold': {
            'accuracy': 0.80,
            'auc_roc': 0.85,
        },
        'monitoring_required': True,
        'documentation_required': True,
        'bias_testing_required': True,
    }
    
    return governance_policies

def approve_model_for_production(model, approver_email, approval_notes):
    """Approve model for production deployment."""
    
    # Add approval metadata
    model.update(
        labels={
            **model.labels,
            'production_approved': 'true',
            'approved_by': approver_email.split('@')[0],
            'approval_date': '2025-01-15'
        },
        description=f"{model.description}\n\nApproval Notes: {approval_notes}"
    )
    
    print(f"Model approved for production by {approver_email}")
    
    return model
```

---

## 15. Security Best Practices

### IAM and Access Control

**Best Practices:**
- Use principle of least privilege
- Separate service accounts by function
- Implement VPC Service Controls
- Enable audit logging

```python
def setup_secure_vertex_ai(project_id, service_account_email):
    """Configure secure Vertex AI setup."""
    
    # Required IAM roles for different functions
    iam_roles = {
        'data_scientist': [
            'roles/aiplatform.user',
            'roles/storage.objectViewer',
            'roles/bigquery.dataViewer',
        ],
        'ml_engineer': [
            'roles/aiplatform.user',
            'roles/aiplatform.serviceAgent',
            'roles/storage.objectAdmin',
            'roles/bigquery.dataEditor',
        ],
        'deployment_account': [
            'roles/aiplatform.admin',
            'roles/storage.objectAdmin',
            'roles/iam.serviceAccountUser',
        ],
    }
    
    # Grant roles using gcloud
    for role_type, roles in iam_roles.items():
        print(f"\nRoles for {role_type}:")
        for role in roles:
            print(f"  - {role}")
            print(f"    gcloud projects add-iam-policy-binding {project_id} \\")
            print(f"      --member='serviceAccount:{service_account_email}' \\")
            print(f"      --role='{role}'")
    
    return iam_roles

# VPC Service Controls
def configure_vpc_service_controls():
    """Configure VPC Service Controls for Vertex AI."""
    
    vpc_config = {
        'service_perimeter': 'vertex-ai-perimeter',
        'restricted_services': [
            'aiplatform.googleapis.com',
            'storage.googleapis.com',
            'bigquery.googleapis.com',
        ],
        'access_levels': [
            'require_corp_network',
            'require_device_policy',
        ],
        'ingress_policies': [
            {
                'from': 'allowed_projects',
                'to': 'vertex_ai_operations',
            }
        ],
    }
    
    print("VPC Service Controls Configuration:")
    print(f"  Service Perimeter: {vpc_config['service_perimeter']}")
    print(f"  Restricted Services: {', '.join(vpc_config['restricted_services'])}")
    
    return vpc_config

# Example usage
iam_config = setup_secure_vertex_ai(
    project_id='my-project',
    service_account_email='vertex-ai-sa@my-project.iam.gserviceaccount.com'
)
```

### Data Encryption

**Encryption at Rest and in Transit:**
```python
def create_encrypted_dataset(
    display_name,
    gcs_source,
    encryption_spec_key_name,
    project,
    location
):
    """Create dataset with customer-managed encryption key."""
    
    aiplatform.init(project=project, location=location)
    
    # Create encryption spec
    encryption_spec = aiplatform.gapic.EncryptionSpec(
        kms_key_name=encryption_spec_key_name
    )
    
    # Create dataset with encryption
    dataset = aiplatform.TabularDataset.create(
        display_name=display_name,
        gcs_source=gcs_source,
        encryption_spec=encryption_spec,
        labels={'encrypted': 'true', 'compliance': 'required'}
    )
    
    print(f"Encrypted dataset created: {dataset.resource_name}")
    
    return dataset

def deploy_model_with_encryption(
    model,
    endpoint_display_name,
    encryption_spec_key_name
):
    """Deploy model with encryption."""
    
    # Create encryption spec
    encryption_spec = aiplatform.gapic.EncryptionSpec(
        kms_key_name=encryption_spec_key_name
    )
    
    # Create endpoint with encryption
    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_display_name,
        encryption_spec=encryption_spec,
        labels={'encrypted': 'true'}
    )
    
    # Deploy model
    model.deploy(
        endpoint=endpoint,
        machine_type='n1-standard-4',
        min_replica_count=1,
        max_replica_count=3,
        sync=True,
    )
    
    return endpoint

# Example with CMEK
kms_key = 'projects/my-project/locations/us-central1/keyRings/vertex-ai-keyring/cryptoKeys/vertex-ai-key'

dataset = create_encrypted_dataset(
    display_name='encrypted-dataset',
    gcs_source=['gs://my-bucket/data/train.csv'],
    encryption_spec_key_name=kms_key,
    project='my-project',
    location='us-central1'
)
```

### Audit Logging

**Enable and Monitor Audit Logs:**
```python
def setup_audit_logging(project_id):
    """Configure audit logging for Vertex AI."""
    
    audit_config = {
        'service': 'aiplatform.googleapis.com',
        'audit_log_configs': [
            {
                'log_type': 'ADMIN_READ',
                'exempted_members': [],
            },
            {
                'log_type': 'DATA_READ',
                'exempted_members': [],
            },
            {
                'log_type': 'DATA_WRITE',
                'exempted_members': [],
            },
        ],
    }
    
    print("Audit Logging Configuration:")
    print(f"  Service: {audit_config['service']}")
    print(f"  Log Types: ADMIN_READ, DATA_READ, DATA_WRITE")
    
    return audit_config

# Query audit logs in BigQuery
def query_vertex_ai_audit_logs():
    """Query Vertex AI audit logs."""
    
    query = """
    SELECT
        timestamp,
        protoPayload.authenticationInfo.principalEmail AS user,
        protoPayload.methodName AS method,
        protoPayload.resourceName AS resource,
        protoPayload.request AS request_details,
        severity
    FROM
        `my-project.logs.cloudaudit_googleapis_com_activity_*`
    WHERE
        resource.type = 'aiplatform.googleapis.com/Model'
        OR resource.type = 'aiplatform.googleapis.com/Endpoint'
        OR resource.type = 'aiplatform.googleapis.com/TrainingPipeline'
    ORDER BY
        timestamp DESC
    LIMIT 100;
    """
    
    return query
```

---

## 16. Performance Optimization

### Model Optimization Techniques

**Best Practices:**
- Use model quantization
- Implement model pruning
- Optimize inference batch size
- Use TensorFlow Lite for edge deployment

```python
import tensorflow as tf

def optimize_tensorflow_model(saved_model_path, optimized_model_path):
    """Optimize TensorFlow model for inference."""
    
    # Load model
    model = tf.saved_model.load(saved_model_path)
    
    # Convert to TFLite with optimization
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    
    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Quantization
    converter.target_spec.supported_types = [tf.float16]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save optimized model
    with open(f'{optimized_model_path}/model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"Optimized model saved to {optimized_model_path}")
    
    # Compare sizes
    import os
    original_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(saved_model_path)
        for filename in filenames
    )
    optimized_size = os.path.getsize(f'{optimized_model_path}/model.tflite')
    
    print(f"Original model size: {original_size / 1024 / 1024:.2f} MB")
    print(f"Optimized model size: {optimized_size / 1024 / 1024:.2f} MB")
    print(f"Reduction: {(1 - optimized_size / original_size) * 100:.2f}%")

def quantize_model_for_deployment(model_path, quantization_type='dynamic'):
    """Apply quantization to model."""
    
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    
    if quantization_type == 'dynamic':
        # Dynamic range quantization (easiest)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    elif quantization_type == 'float16':
        # Float16 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    elif quantization_type == 'int8':
        # Int8 quantization (requires representative dataset)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    return tflite_model
```

### Batch Prediction Optimization

**Optimize Batch Inference:**
```python
def optimize_batch_prediction(
    model,
    input_data,
    batch_size=32,
    use_gpu=True
):
    """Optimize batch prediction configuration."""
    
    # Calculate optimal batch size based on data size
    total_instances = len(input_data)
    optimal_batch_size = min(batch_size, total_instances // 10)
    
    # Calculate optimal number of replicas
    if total_instances < 1000:
        num_replicas = 1
        machine_type = 'n1-standard-4'
    elif total_instances < 10000:
        num_replicas = 2
        machine_type = 'n1-standard-8'
    else:
        num_replicas = 5
        machine_type = 'n1-highmem-8'
    
    # Configure batch prediction
    config = {
        'machine_type': machine_type,
        'starting_replica_count': num_replicas,
        'max_replica_count': num_replicas * 2,
        'batch_size': optimal_batch_size,
        'accelerator_type': 'NVIDIA_TESLA_T4' if use_gpu else None,
        'accelerator_count': 1 if use_gpu else 0,
    }
    
    print(f"Optimized Batch Prediction Config:")
    print(f"  Instances: {total_instances}")
    print(f"  Batch Size: {optimal_batch_size}")
    print(f"  Machine Type: {machine_type}")
    print(f"  Replicas: {num_replicas}")
    
    return config
```

### Caching Strategies

**Implement Prediction Caching:**
```python
from google.cloud import redis
import hashlib
import json

class PredictionCache:
    """Cache predictions to reduce inference costs."""
    
    def __init__(self, redis_host, redis_port):
        self.client = redis.Client(host=redis_host, port=redis_port)
        self.ttl = 3600  # 1 hour
    
    def get_cache_key(self, instance):
        """Generate cache key from instance."""
        instance_str = json.dumps(instance, sort_keys=True)
        return hashlib.md5(instance_str.encode()).hexdigest()
    
    def get_prediction(self, instance):
        """Get prediction from cache."""
        key = self.get_cache_key(instance)
        cached = self.client.get(key)
        
        if cached:
            return json.loads(cached)
        return None
    
    def set_prediction(self, instance, prediction):
        """Store prediction in cache."""
        key = self.get_cache_key(instance)
        self.client.setex(
            key,
            self.ttl,
            json.dumps(prediction)
        )
    
    def predict_with_cache(self, endpoint, instance):
        """Get prediction with caching."""
        # Check cache first
        cached_prediction = self.get_prediction(instance)
        if cached_prediction:
            print("Cache hit!")
            return cached_prediction
        
        # Get prediction from endpoint
        print("Cache miss - calling endpoint")
        prediction = endpoint.predict([instance])
        
        # Store in cache
        self.set_prediction(instance, prediction.predictions[0])
        
        return prediction.predictions[0]

# Usage
cache = PredictionCache(
    redis_host='10.0.0.3',
    redis_port=6379
)

instance = {'age': 35, 'income': 75000}
prediction = cache.predict_with_cache(endpoint, instance)
```

---

## 17. Integration Patterns

### BigQuery Integration

**Best Practices:**
- Use BigQuery for feature engineering
- Store predictions in BigQuery
- Query historical predictions

```python
from google.cloud import bigquery

def create_features_from_bigquery(
    project,
    location,
    query,
    destination_table
):
    """Create features from BigQuery query."""
    
    bq_client = bigquery.Client(project=project, location=location)
    
    # Run feature engineering query
    job_config = bigquery.QueryJobConfig(
        destination=destination_table,
        write_disposition='WRITE_TRUNCATE',
    )
    
    query_job = bq_client.query(query, job_config=job_config)
    query_job.result()
    
    print(f"Features created in {destination_table}")
    
    return destination_table

def export_predictions_to_bigquery(
    predictions,
    project,
    dataset,
    table_name
):
    """Export predictions to BigQuery."""
    
    bq_client = bigquery.Client(project=project)
    
    table_id = f"{project}.{dataset}.{table_name}"
    
    # Define schema
    schema = [
        bigquery.SchemaField("prediction_id", "STRING"),
        bigquery.SchemaField("prediction_value", "FLOAT64"),
        bigquery.SchemaField("prediction_timestamp", "TIMESTAMP"),
        bigquery.SchemaField("model_version", "STRING"),
    ]
    
    # Create table if not exists
    table = bigquery.Table(table_id, schema=schema)
    table = bq_client.create_table(table, exists_ok=True)
    
    # Insert predictions
    errors = bq_client.insert_rows_json(table, predictions)
    
    if errors:
        print(f"Errors inserting rows: {errors}")
    else:
        print(f"Predictions exported to {table_id}")

# Feature engineering query example
feature_query = """
WITH customer_features AS (
    SELECT
        customer_id,
        DATE_DIFF(CURRENT_DATE(), first_purchase_date, DAY) AS days_since_first_purchase,
        COUNT(DISTINCT order_id) AS total_orders,
        SUM(order_amount) AS total_spent,
        AVG(order_amount) AS avg_order_value,
        MAX(order_date) AS last_order_date,
        DATE_DIFF(CURRENT_DATE(), MAX(order_date), DAY) AS days_since_last_order
    FROM
        `my-project.analytics.orders`
    GROUP BY
        customer_id, first_purchase_date
)
SELECT
    c.*,
    f.*
FROM
    `my-project.analytics.customers` c
JOIN
    customer_features f
ON
    c.customer_id = f.customer_id;
"""

features_table = create_features_from_bigquery(
    project='my-project',
    location='us-central1',
    query=feature_query,
    destination_table='my-project.ml_features.customer_features'
)
```

### Cloud Functions Integration

**Trigger Predictions from Events:**
```python
import functions_framework
from google.cloud import aiplatform

@functions_framework.http
def predict_on_request(request):
    """Cloud Function to handle prediction requests."""
    
    # Initialize Vertex AI
    aiplatform.init(
        project='my-project',
        location='us-central1'
    )
    
    # Get endpoint
    endpoint = aiplatform.Endpoint('projects/123/locations/us-central1/endpoints/456')
    
    # Get request data
    request_json = request.get_json()
    instances = request_json.get('instances', [])
    
    # Make prediction
    predictions = endpoint.predict(instances=instances)
    
    # Return response
    return {
        'predictions': predictions.predictions,
        'model_version': 'v1.0',
        'timestamp': '2025-01-15T10:30:00Z'
    }

@functions_framework.cloud_event
def predict_on_pubsub(cloud_event):
    """Trigger predictions from Pub/Sub messages."""
    
    import base64
    
    # Decode Pub/Sub message
    pubsub_message = base64.b64decode(cloud_event.data["message"]["data"])
    message_data = json.loads(pubsub_message)
    
    # Initialize Vertex AI
    aiplatform.init(project='my-project', location='us-central1')
    
    # Get endpoint
    endpoint = aiplatform.Endpoint('projects/123/locations/us-central1/endpoints/456')
    
    # Make prediction
    prediction = endpoint.predict(instances=[message_data])
    
    # Publish results
    from google.cloud import pubsub_v1
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path('my-project', 'prediction-results')
    
    result_data = json.dumps({
        'input': message_data,
        'prediction': prediction.predictions[0],
        'timestamp': '2025-01-15T10:30:00Z'
    })
    
    publisher.publish(topic_path, result_data.encode('utf-8'))
```

### Dataflow Integration

**Batch Processing with Dataflow:**
```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from google.cloud import aiplatform

class PredictDoFn(beam.DoFn):
    """DoFn for making predictions in Dataflow."""
    
    def __init__(self, project, location, endpoint_id):
        self.project = project
        self.location = location
        self.endpoint_id = endpoint_id
        self.endpoint = None
    
    def setup(self):
        """Initialize endpoint once per worker."""
        aiplatform.init(project=self.project, location=self.location)
        self.endpoint = aiplatform.Endpoint(self.endpoint_id)
    
    def process(self, element):
        """Make prediction for element."""
        prediction = self.endpoint.predict(instances=[element])
        
        yield {
            'input': element,
            'prediction': prediction.predictions[0],
        }

def run_dataflow_predictions(
    input_pattern,
    output_table,
    project,
    location,
    endpoint_id
):
    """Run batch predictions using Dataflow."""
    
    options = PipelineOptions([
        f'--project={project}',
        f'--region={location}',
        '--runner=DataflowRunner',
        '--temp_location=gs://my-bucket/temp',
        '--staging_location=gs://my-bucket/staging',
    ])
    
    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | 'Read Input' >> beam.io.ReadFromText(input_pattern)
            | 'Parse JSON' >> beam.Map(json.loads)
            | 'Predict' >> beam.ParDo(PredictDoFn(project, location, endpoint_id))
            | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
                output_table,
                schema='input:STRING,prediction:FLOAT64',
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
            )
        )
```

---

## 18. Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Training Job Fails

**Symptoms:**
- Training job fails with resource errors
- Out of memory errors

**Solutions:**
```python
# Solution 1: Increase machine resources
model = job.run(
    replica_count=1,
    machine_type='n1-highmem-16',  # More memory
    accelerator_type='NVIDIA_TESLA_V100',  # Better GPU
    accelerator_count=2,
)

# Solution 2: Reduce batch size in training script
# training_args = {
#     'batch_size': 16,  # Reduce from 32
#     'gradient_accumulation_steps': 2,
# }

# Solution 3: Enable gradient checkpointing
# model.gradient_checkpointing_enable()
```

#### Issue 2: Slow Predictions

**Symptoms:**
- High prediction latency
- Timeouts

**Solutions:**
```python
# Solution 1: Scale up endpoint
endpoint.update(
    min_replica_count=3,  # Increase minimum replicas
    max_replica_count=10,
)

# Solution 2: Use GPU for inference
model.deploy(
    endpoint=endpoint,
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
)

# Solution 3: Batch predictions
# Instead of: predictions = [endpoint.predict([x]) for x in instances]
# Use: predictions = endpoint.predict(instances)  # Batch all together
```

#### Issue 3: Model Accuracy Degradation

**Symptoms:**
- Model performance drops over time
- Increased prediction errors

**Solutions:**
```python
# Solution 1: Set up monitoring
monitoring_job = create_model_monitoring_job(
    endpoint=endpoint,
    display_name='model-monitoring',
    emails=['ml-team@example.com'],
    drift_thresholds={'feature_1': 0.1, 'feature_2': 0.1},
)

# Solution 2: Retrain model regularly
def schedule_retraining():
    """Schedule model retraining."""
    from google.cloud import scheduler_v1
    
    client = scheduler_v1.CloudSchedulerClient()
    
    job = {
        'name': 'projects/my-project/locations/us-central1/jobs/retrain-model',
        'schedule': '0 0 * * 0',  # Weekly on Sunday
        'http_target': {
            'uri': 'https://my-function-url.cloudfunctions.net/retrain',
            'http_method': 'POST',
        },
    }
    
    return job

# Solution 3: Implement A/B testing
new_model.deploy(
    endpoint=endpoint,
    traffic_percentage=10,  # Send 10% traffic to new model
)
```

#### Issue 4: High Costs

**Symptoms:**
- Unexpected billing charges
- Resource over-provisioning

**Solutions:**
```python
# Solution 1: Use auto-scaling efficiently
model.deploy(
    endpoint=endpoint,
    min_replica_count=1,  # Reduce minimum
    max_replica_count=5,  # Set reasonable maximum
    machine_type='n1-standard-4',  # Right-size machines
)

# Solution 2: Clean up unused resources
def cleanup_unused_resources(project, location, days_old=30):
    """Delete unused models and endpoints."""
    aiplatform.init(project=project, location=location)
    
    import datetime
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_old)
    
    # List and delete old models
    models = aiplatform.Model.list()
    for model in models:
        if model.create_time < cutoff_date:
            # Check if model is deployed
            if not model.gca_resource.deployed_models:
                print(f"Deleting unused model: {model.display_name}")
                model.delete()
    
    # List and delete unused endpoints
    endpoints = aiplatform.Endpoint.list()
    for endpoint in endpoints:
        if not endpoint.gca_resource.deployed_models:
            print(f"Deleting empty endpoint: {endpoint.display_name}")
            endpoint.delete()

# Solution 3: Use batch prediction instead of online
batch_job = model.batch_predict(
    gcs_source='gs://my-bucket/input.jsonl',
    gcs_destination_prefix='gs://my-bucket/output/',
)
```

### Debugging Tools

**Enable Detailed Logging:**
```python
import logging
from google.cloud import logging as cloud_logging

# Setup Cloud Logging
logging_client = cloud_logging.Client()
logging_client.setup_logging(log_level=logging.DEBUG)

# Log training details
logger = logging.getLogger('vertex-ai-training')
logger.setLevel(logging.DEBUG)

def log_training_progress(epoch, loss, accuracy):
    """Log training progress."""
    logger.info(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")

# View logs
def query_vertex_logs(project_id, hours=24):
    """Query recent Vertex AI logs."""
    
    query = f"""
    resource.type="aiplatform.googleapis.com/TrainingJob"
    timestamp >= "{hours}h"
    """
    
    entries = logging_client.list_entries(filter_=query, max_results=100)
    
    for entry in entries:
        print(f"{entry.timestamp}: {entry.payload}")
    
    return entries
```

---

## 19. Advanced Topics

### Custom Prediction Routines

**Implement Custom Preprocessing:**
```python
from google.cloud.aiplatform.prediction import LocalModel
from google.cloud.aiplatform.prediction.handler import PredictionHandler
import pickle

class CustomHandler(PredictionHandler):
    """Custom prediction handler with preprocessing."""
    
    def __init__(self):
        pass
    
    def load(self, artifacts_uri):
        """Load model and preprocessing artifacts."""
        # Load model
        with open(f'{artifacts_uri}/model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        # Load preprocessor
        with open(f'{artifacts_uri}/preprocessor.pkl', 'rb') as f:
            self.preprocessor = pickle.load(f)
    
    def preprocess(self, prediction_input):
        """Custom preprocessing logic."""
        instances = prediction_input.instances
        
        # Apply preprocessing
        processed = self.preprocessor.transform(instances)
        
        return processed
    
    def predict(self, instances):
        """Make predictions."""
        predictions = self.model.predict(instances)
        return predictions
    
    def postprocess(self, prediction_results):
        """Custom postprocessing logic."""
        # Add confidence scores, thresholds, etc.
        processed_results = []
        
        for pred in prediction_results:
            processed_results.append({
                'prediction': pred,
                'confidence': abs(pred),
                'threshold': 0.5,
            })
        
        return processed_results

# Deploy custom prediction routine
def deploy_custom_prediction_routine(
    model_artifacts_uri,
    handler_class,
    requirements
):
    """Deploy model with custom prediction routine."""
    
    local_model = LocalModel.build_cpr_model(
        model_artifacts_uri,
        'us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest',
        handler_class=handler_class,
        requirements_path=requirements,
    )
    
    # Upload and deploy
    uploaded_model = local_model.upload(
        display_name='custom-prediction-model'
    )
    
    endpoint = uploaded_model.deploy(
        machine_type='n1-standard-4',
        min_replica_count=1,
        max_replica_count=3,
    )
    
    return endpoint
```

### Multi-Model Endpoints

**Deploy Multiple Models to One Endpoint:**
```python
def create_multi_model_endpoint(
    models,
    endpoint_display_name,
    traffic_split
):
    """Deploy multiple models with traffic splitting."""
    
    # Create endpoint
    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_display_name,
        labels={'type': 'multi-model'}
    )
    
    # Deploy each model
    for idx, (model, traffic_pct) in enumerate(zip(models, traffic_split)):
        model.deploy(
            endpoint=endpoint,
            deployed_model_display_name=f'model-v{idx+1}',
            machine_type='n1-standard-4',
            traffic_percentage=traffic_pct,
            sync=True,
        )
    
    print(f"Multi-model endpoint created: {endpoint.resource_name}")
    print(f"Traffic split: {endpoint.traffic_split}")
    
    return endpoint

# Example: Champion-Challenger setup
models = [champion_model, challenger_model]
traffic_split = [90, 10]  # 90% champion, 10% challenger

endpoint = create_multi_model_endpoint(
    models=models,
    endpoint_display_name='champion-challenger-endpoint',
    traffic_split=traffic_split
)
```

### Private Endpoints

**Deploy in Private VPC:**
```python
def create_private_endpoint(
    model,
    endpoint_display_name,
    network,
    enable_private_service_connect=True
):
    """Create private endpoint for secure access."""
    
    # Create endpoint with VPC configuration
    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_display_name,
        network=network,
        labels={'access': 'private'},
    )
    
    # Deploy model to private endpoint
    model.deploy(
        endpoint=endpoint,
        machine_type='n1-standard-4',
        min_replica_count=1,
        max_replica_count=3,
        sync=True,
    )
    
    print(f"Private endpoint created: {endpoint.resource_name}")
    print(f"Network: {network}")
    
    return endpoint

# Example
private_endpoint = create_private_endpoint(
    model=model,
    endpoint_display_name='private-inference-endpoint',
    network='projects/my-project/global/networks/my-vpc',
)
```

---

## 20. Quick Reference Checklist

### Setup
- [ ] Enable Vertex AI API
- [ ] Set up IAM roles and service accounts
- [ ] Create Cloud Storage buckets
- [ ] Configure regional settings
- [ ] Set up billing alerts
- [ ] Enable audit logging
- [ ] Configure VPC Service Controls (if required)

### Data Preparation
- [ ] Validate data quality
- [ ] Create Vertex AI datasets
- [ ] Split data appropriately (train/val/test)
- [ ] Document data schema
- [ ] Version datasets
- [ ] Implement data encryption (if required)
- [ ] Set up Feature Store (for production)

### Training
- [ ] Choose appropriate algorithm
- [ ] Set training budget
- [ ] Configure hyperparameters
- [ ] Enable checkpointing
- [ ] Monitor training progress
- [ ] Implement distributed training (for large models)
- [ ] Run hyperparameter tuning
- [ ] Track experiments

### Evaluation
- [ ] Evaluate on test set
- [ ] Compare multiple metrics
- [ ] Analyze feature importance
- [ ] Check for bias and fairness
- [ ] Validate against baseline
- [ ] Generate model explanations
- [ ] Create model card

### Deployment
- [ ] Create endpoint
- [ ] Configure auto-scaling
- [ ] Enable monitoring
- [ ] Test predictions
- [ ] Implement A/B testing
- [ ] Set up private endpoints (if required)
- [ ] Configure caching strategy
- [ ] Enable explainability

### MLOps
- [ ] Register models in registry
- [ ] Set up continuous monitoring
- [ ] Create ML pipelines
- [ ] Implement CI/CD
- [ ] Document workflows
- [ ] Configure alerts
- [ ] Implement governance policies
- [ ] Schedule retraining

### Security
- [ ] Apply least privilege IAM
- [ ] Enable encryption (CMEK if required)
- [ ] Configure VPC Service Controls
- [ ] Enable audit logging
- [ ] Implement access controls
- [ ] Review security best practices

### Cost Optimization
- [ ] Right-size machine types
- [ ] Enable auto-scaling
- [ ] Use batch prediction for bulk inference
- [ ] Clean up unused resources
- [ ] Monitor costs regularly
- [ ] Use preemptible VMs for training
- [ ] Implement prediction caching

### Production Readiness
- [ ] Load test endpoints
- [ ] Set up monitoring and alerting
- [ ] Create runbooks
- [ ] Implement disaster recovery
- [ ] Document API contracts
- [ ] Set SLAs and SLOs
- [ ] Create incident response plan

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
