# Vertex AI Pipelines Best Practices

*Last Updated: January 4, 2026*

## Overview

Vertex AI Pipelines orchestrates ML workflows with automated, reproducible pipelines using Kubeflow Pipelines (KFP) and TensorFlow Extended (TFX), enabling end-to-end MLOps automation.

---

## 1. Pipeline Architecture

### Design Principles

**Best Practices:**
- Use modular, reusable components
- Implement clear data lineage
- Design for reproducibility
- Handle failures gracefully
- Version control pipeline definitions

```python
from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics
from google.cloud import aiplatform

# Define reusable component
@component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'scikit-learn']
)
def preprocess_data(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
    test_size: float = 0.2
):
    """Preprocess and split dataset."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Load data
    df = pd.read_csv(input_data.path)
    
    # Preprocessing
    df = df.dropna()
    
    # Save processed data
    df.to_csv(output_data.path, index=False)
    
    print(f"Processed {len(df)} records")

@component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'scikit-learn', 'joblib']
)
def train_model(
    training_data: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    max_depth: int = 5
):
    """Train machine learning model."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    import joblib
    
    # Load data
    df = pd.read_csv(training_data.path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train model
    clf = RandomForestClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X, y)
    
    # Evaluate
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    
    # Log metrics
    metrics.log_metric('accuracy', accuracy)
    metrics.log_metric('precision', precision)
    metrics.log_metric('recall', recall)
    
    # Save model
    joblib.dump(clf, model.path)
    
    print(f"Model trained - Accuracy: {accuracy:.4f}")

@component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'scikit-learn', 'joblib']
)
def evaluate_model(
    model: Input[Model],
    test_data: Input[Dataset],
    metrics: Output[Metrics],
    threshold: float = 0.85
) -> str:
    """Evaluate model on test data."""
    import pandas as pd
    from sklearn.metrics import accuracy_score
    import joblib
    
    # Load model and data
    clf = joblib.load(model.path)
    df = pd.read_csv(test_data.path)
    X_test = df.drop('target', axis=1)
    y_test = df['target']
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log metrics
    metrics.log_metric('test_accuracy', accuracy)
    
    # Determine if model passes threshold
    status = 'PASS' if accuracy >= threshold else 'FAIL'
    
    print(f"Test Accuracy: {accuracy:.4f} - Status: {status}")
    
    return status

# Define pipeline
@dsl.pipeline(
    name='ml-training-pipeline',
    description='End-to-end ML training pipeline',
    pipeline_root='gs://my-bucket/pipeline-root'
)
def ml_pipeline(
    input_data_uri: str,
    test_size: float = 0.2,
    max_depth: int = 5,
    accuracy_threshold: float = 0.85
):
    """Complete ML training pipeline."""
    
    # Import data
    import_data_op = dsl.importer(
        artifact_uri=input_data_uri,
        artifact_class=Dataset,
        reimport=False
    )
    
    # Preprocess
    preprocess_task = preprocess_data(
        input_data=import_data_op.output,
        test_size=test_size
    )
    
    # Train
    train_task = train_model(
        training_data=preprocess_task.outputs['output_data'],
        max_depth=max_depth
    )
    
    # Evaluate
    evaluate_task = evaluate_model(
        model=train_task.outputs['model'],
        test_data=preprocess_task.outputs['output_data'],
        threshold=accuracy_threshold
    )
    
    # Conditional deployment based on evaluation
    with dsl.Condition(
        evaluate_task.output == 'PASS',
        name='deploy-condition'
    ):
        deploy_task = deploy_model(
            model=train_task.outputs['model']
        )

# Compile and run pipeline
from kfp.v2 import compiler

compiler.Compiler().compile(
    pipeline_func=ml_pipeline,
    package_path='ml_pipeline.json'
)

# Submit pipeline
aiplatform.init(
    project='my-project',
    location='us-central1',
    staging_bucket='gs://my-bucket/staging'
)

job = aiplatform.PipelineJob(
    display_name='ml-training-pipeline',
    template_path='ml_pipeline.json',
    parameter_values={
        'input_data_uri': 'gs://my-bucket/data/training_data.csv',
        'test_size': 0.2,
        'max_depth': 10,
        'accuracy_threshold': 0.85
    }
)

job.run(sync=True)
```

---

## 2. Component Development

### Create Reusable Components

```python
from typing import NamedTuple

@component(
    base_image='python:3.9',
    packages_to_install=['google-cloud-bigquery', 'pandas']
)
def load_data_from_bigquery(
    project_id: str,
    dataset_id: str,
    table_id: str,
    output_dataset: Output[Dataset]
) -> NamedTuple('Outputs', [('row_count', int), ('column_count', int)]):
    """Load data from BigQuery."""
    from google.cloud import bigquery
    import pandas as pd
    from collections import namedtuple
    
    client = bigquery.Client(project=project_id)
    
    query = f"""
        SELECT *
        FROM `{project_id}.{dataset_id}.{table_id}`
    """
    
    df = client.query(query).to_dataframe()
    
    # Save to artifact
    df.to_csv(output_dataset.path, index=False)
    
    # Return metadata
    outputs = namedtuple('Outputs', ['row_count', 'column_count'])
    return outputs(len(df), len(df.columns))

@component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'google-cloud-storage']
)
def save_to_gcs(
    input_dataset: Input[Dataset],
    gcs_path: str
):
    """Save dataset to Cloud Storage."""
    from google.cloud import storage
    import pandas as pd
    
    # Load data
    df = pd.read_csv(input_dataset.path)
    
    # Parse GCS path
    bucket_name = gcs_path.split('/')[2]
    blob_name = '/'.join(gcs_path.split('/')[3:])
    
    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    blob.upload_from_string(df.to_csv(index=False), content_type='text/csv')
    
    print(f"Saved to {gcs_path}")

# Component with dependencies
@component(
    base_image='gcr.io/deeplearning-platform-release/tf2-gpu.2-11',
    packages_to_install=['tensorflow-datasets']
)
def train_deep_learning_model(
    training_data: Input[Dataset],
    model: Output[Model],
    epochs: int = 10,
    batch_size: int = 32
):
    """Train deep learning model with TensorFlow."""
    import tensorflow as tf
    import pandas as pd
    
    # Load data
    df = pd.read_csv(training_data.path)
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # Build model
    model_nn = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model_nn.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    history = model_nn.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    # Save model
    model_nn.save(model.path)
    
    print(f"Training completed - Final accuracy: {history.history['accuracy'][-1]:.4f}")
```

---

## 3. Pipeline Execution

### Schedule and Monitor Pipelines

```python
class PipelineManager:
    """Manage pipeline execution and monitoring."""
    
    def __init__(self, project_id, location, pipeline_root):
        self.project_id = project_id
        self.location = location
        self.pipeline_root = pipeline_root
        
        aiplatform.init(
            project=project_id,
            location=location,
            staging_bucket=pipeline_root
        )
    
    def run_pipeline(self, template_path, parameters, display_name=None):
        """Execute pipeline job."""
        
        job = aiplatform.PipelineJob(
            display_name=display_name or 'pipeline-job',
            template_path=template_path,
            parameter_values=parameters,
            enable_caching=True
        )
        
        print(f"Starting pipeline: {display_name}")
        
        job.run(
            sync=False,
            service_account='pipeline-sa@project.iam.gserviceaccount.com'
        )
        
        print(f"Pipeline job created: {job.resource_name}")
        
        return job
    
    def schedule_pipeline(self, template_path, parameters, cron_schedule):
        """Schedule recurring pipeline execution."""
        
        from google.cloud import scheduler_v1
        from google.protobuf import json_format
        import json
        
        client = scheduler_v1.CloudSchedulerClient()
        
        parent = f"projects/{self.project_id}/locations/{self.location}"
        
        # Create job configuration
        job_config = {
            'name': f"{parent}/jobs/ml-pipeline-schedule",
            'schedule': cron_schedule,  # e.g., '0 2 * * *' for daily at 2 AM
            'time_zone': 'America/New_York',
            'httpTarget': {
                'uri': f'https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/pipelineJobs',
                'httpMethod': 'POST',
                'body': json.dumps({
                    'displayName': 'scheduled-pipeline',
                    'runtimeConfig': {
                        'gcsOutputDirectory': self.pipeline_root,
                        'parameterValues': parameters
                    },
                    'templateUri': template_path
                }).encode(),
                'headers': {
                    'Content-Type': 'application/json'
                }
            }
        }
        
        job = json_format.ParseDict(job_config, scheduler_v1.Job())
        
        try:
            response = client.create_job(parent=parent, job=job)
            print(f"✓ Scheduled pipeline: {response.name}")
            return response
        except Exception as e:
            print(f"✗ Error scheduling pipeline: {e}")
            return None
    
    def monitor_pipeline(self, job_resource_name):
        """Monitor pipeline execution."""
        
        job = aiplatform.PipelineJob.get(job_resource_name)
        
        print(f"\nPipeline Status: {job.state.name}")
        print(f"Create Time: {job.create_time}")
        
        if job.state.name == 'PIPELINE_STATE_SUCCEEDED':
            print("✓ Pipeline completed successfully")
        elif job.state.name == 'PIPELINE_STATE_FAILED':
            print("✗ Pipeline failed")
            print(f"Error: {job.error}")
        elif job.state.name in ['PIPELINE_STATE_RUNNING', 'PIPELINE_STATE_PENDING']:
            print(f"Pipeline is {job.state.name.lower()}")
        
        return job
    
    def list_pipeline_jobs(self, filter_str=None, limit=10):
        """List recent pipeline jobs."""
        
        jobs = aiplatform.PipelineJob.list(
            filter=filter_str,
            order_by='create_time desc',
            limit=limit
        )
        
        print(f"\nRecent Pipeline Jobs ({len(jobs)}):\n")
        
        for job in jobs:
            print(f"Name: {job.display_name}")
            print(f"  Resource: {job.resource_name}")
            print(f"  State: {job.state.name}")
            print(f"  Created: {job.create_time}")
            print()
        
        return jobs
    
    def cancel_pipeline(self, job_resource_name):
        """Cancel running pipeline."""
        
        job = aiplatform.PipelineJob.get(job_resource_name)
        
        if job.state.name in ['PIPELINE_STATE_RUNNING', 'PIPELINE_STATE_PENDING']:
            job.cancel()
            print(f"✓ Cancelled pipeline: {job_resource_name}")
        else:
            print(f"Pipeline is not running: {job.state.name}")
        
        return job

# Example usage
manager = PipelineManager(
    project_id='my-project',
    location='us-central1',
    pipeline_root='gs://my-bucket/pipeline-root'
)

# Run pipeline
job = manager.run_pipeline(
    template_path='ml_pipeline.json',
    parameters={
        'input_data_uri': 'gs://my-bucket/data/training_data.csv',
        'max_depth': 10
    },
    display_name='ml-training-job'
)

# Schedule pipeline (daily at 2 AM)
# manager.schedule_pipeline(
#     template_path='ml_pipeline.json',
#     parameters={'input_data_uri': 'gs://my-bucket/data/training_data.csv'},
#     cron_schedule='0 2 * * *'
# )

# Monitor pipeline
# manager.monitor_pipeline(job.resource_name)
```

---

## 4. Artifact Management

### Track Data and Model Lineage

```python
@component
def log_artifact_metadata(
    artifact: Input[Dataset],
    metadata_key: str,
    metadata_value: str
):
    """Add metadata to artifact."""
    from google.cloud import aiplatform
    
    # Get artifact
    artifact_resource = aiplatform.Artifact(artifact.uri)
    
    # Update metadata
    artifact_resource.update(
        metadata={metadata_key: metadata_value}
    )
    
    print(f"Updated artifact metadata: {metadata_key}={metadata_value}")

@dsl.pipeline(
    name='data-lineage-pipeline',
    pipeline_root='gs://my-bucket/pipeline-root'
)
def lineage_tracking_pipeline(
    data_source: str
):
    """Pipeline with comprehensive lineage tracking."""
    
    # Import raw data
    raw_data = dsl.importer(
        artifact_uri=data_source,
        artifact_class=Dataset,
        reimport=False,
        metadata={'data_source': 'production', 'version': '1.0'}
    )
    
    # Preprocess with metadata
    processed_task = preprocess_data(input_data=raw_data.output)
    processed_task.set_display_name('Data Preprocessing')
    
    # Log preprocessing metadata
    log_metadata_task = log_artifact_metadata(
        artifact=processed_task.outputs['output_data'],
        metadata_key='preprocessing_date',
        metadata_value='2026-01-04'
    )
    
    # Train model
    train_task = train_model(
        training_data=processed_task.outputs['output_data']
    )
    train_task.set_display_name('Model Training')
    train_task.after(log_metadata_task)
    
    return train_task.outputs['model']
```

---

## 5. Error Handling and Retries

```python
@component(
    base_image='python:3.9',
    packages_to_install=['tenacity']
)
def resilient_component(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
    max_retries: int = 3
):
    """Component with retry logic."""
    from tenacity import retry, stop_after_attempt, wait_exponential
    import pandas as pd
    
    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def process_with_retry():
        """Process data with automatic retry."""
        try:
            df = pd.read_csv(input_data.path)
            
            # Simulate processing
            df_processed = df.dropna()
            
            if len(df_processed) == 0:
                raise ValueError("No valid data after processing")
            
            df_processed.to_csv(output_data.path, index=False)
            
            return len(df_processed)
            
        except Exception as e:
            print(f"Error processing data: {e}")
            raise
    
    try:
        row_count = process_with_retry()
        print(f"✓ Processed {row_count} rows successfully")
    except Exception as e:
        print(f"✗ Failed after {max_retries} retries: {e}")
        raise

@dsl.pipeline(
    name='resilient-pipeline',
    pipeline_root='gs://my-bucket/pipeline-root'
)
def pipeline_with_error_handling(data_uri: str):
    """Pipeline with comprehensive error handling."""
    
    import_task = dsl.importer(
        artifact_uri=data_uri,
        artifact_class=Dataset
    )
    
    # Component with retries
    process_task = resilient_component(
        input_data=import_task.output,
        max_retries=3
    )
    
    # Set retry policy for task
    process_task.set_retry(
        num_retries=3,
        backoff_duration='60s',
        backoff_factor=2.0,
        backoff_max_duration='3600s'
    )
    
    # Set timeout
    process_task.set_cpu_limit('4')
    process_task.set_memory_limit('16G')
    
    # Exit handler for cleanup
    with dsl.ExitHandler(cleanup_task):
        train_task = train_model(
            training_data=process_task.outputs['output_data']
        )
    
    return train_task.outputs['model']

@component
def cleanup_task():
    """Cleanup resources after pipeline completion."""
    print("Running cleanup operations...")
    # Add cleanup logic here
```

---

## 6. Performance Optimization

### Caching and Parallelization

```python
@dsl.pipeline(
    name='optimized-pipeline',
    pipeline_root='gs://my-bucket/pipeline-root'
)
def optimized_pipeline(data_uris: list):
    """Optimized pipeline with caching and parallelization."""
    
    # Enable caching for expensive operations
    with dsl.ParallelFor(data_uris) as data_uri:
        # Process multiple datasets in parallel
        preprocess_task = preprocess_data(input_data=data_uri)
        preprocess_task.set_caching_options(True)  # Enable caching
        
        # Resource optimization
        preprocess_task.set_cpu_limit('4')
        preprocess_task.set_memory_limit('16G')
        preprocess_task.set_gpu_limit('1')
    
    # Aggregate results
    aggregate_task = aggregate_datasets(
        datasets=dsl.Collected(preprocess_task.outputs['output_data'])
    )
    
    # Train model on aggregated data
    train_task = train_model(
        training_data=aggregate_task.outputs['combined_data']
    )
    
    # Use preemptible instances for cost savings
    train_task.add_node_selector_constraint(
        'cloud.google.com/gke-preemptible', 'true'
    )
    
    return train_task.outputs['model']

@component
def aggregate_datasets(
    datasets: Input[list],
    combined_data: Output[Dataset]
):
    """Aggregate multiple datasets."""
    import pandas as pd
    
    dfs = []
    for dataset in datasets:
        df = pd.read_csv(dataset.path)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(combined_data.path, index=False)
    
    print(f"Combined {len(dfs)} datasets into {len(combined_df)} rows")
```

---

## 7. CI/CD Integration

```python
# pipeline_cicd.py - Pipeline deployment automation

import os
from kfp.v2 import compiler
from google.cloud import aiplatform

def compile_pipeline(pipeline_func, output_path):
    """Compile pipeline definition."""
    compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=output_path
    )
    print(f"✓ Compiled pipeline to {output_path}")

def validate_pipeline(template_path):
    """Validate pipeline template."""
    import json
    
    with open(template_path, 'r') as f:
        pipeline_spec = json.load(f)
    
    # Validation checks
    assert 'pipelineSpec' in pipeline_spec, "Missing pipelineSpec"
    assert 'components' in pipeline_spec['pipelineSpec'], "Missing components"
    
    print("✓ Pipeline validation passed")
    
    return True

def deploy_pipeline(template_path, environment='staging'):
    """Deploy pipeline to environment."""
    
    # Environment-specific configuration
    configs = {
        'staging': {
            'project': 'my-project-staging',
            'location': 'us-central1',
            'bucket': 'gs://staging-bucket/pipelines'
        },
        'production': {
            'project': 'my-project-prod',
            'location': 'us-central1',
            'bucket': 'gs://prod-bucket/pipelines'
        }
    }
    
    config = configs[environment]
    
    # Upload template to environment bucket
    from google.cloud import storage
    
    client = storage.Client(project=config['project'])
    bucket_name = config['bucket'].replace('gs://', '').split('/')[0]
    blob_name = f"pipelines/{os.path.basename(template_path)}"
    
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(template_path)
    
    deployed_uri = f"{config['bucket']}/{blob_name}"
    
    print(f"✓ Deployed pipeline to {environment}: {deployed_uri}")
    
    return deployed_uri

# CI/CD workflow
if __name__ == '__main__':
    # Compile
    compile_pipeline(ml_pipeline, 'ml_pipeline.json')
    
    # Validate
    validate_pipeline('ml_pipeline.json')
    
    # Deploy to staging
    staging_uri = deploy_pipeline('ml_pipeline.json', 'staging')
    
    # Run tests in staging
    # If tests pass, deploy to production
    # prod_uri = deploy_pipeline('ml_pipeline.json', 'production')
```

---

## 8. Quick Reference Checklist

### Setup
- [ ] Enable Vertex AI API
- [ ] Configure service account with proper permissions
- [ ] Set up Cloud Storage bucket for pipeline artifacts
- [ ] Install KFP SDK: `pip install kfp google-cloud-aiplatform`
- [ ] Configure project and location

### Development
- [ ] Design modular, reusable components
- [ ] Implement proper error handling and retries
- [ ] Add comprehensive logging
- [ ] Version control pipeline definitions
- [ ] Document component inputs and outputs

### Optimization
- [ ] Enable caching for expensive operations
- [ ] Use parallel execution where possible
- [ ] Optimize resource allocation (CPU, memory, GPU)
- [ ] Consider preemptible instances for cost savings
- [ ] Implement incremental data processing

### Production
- [ ] Set up pipeline monitoring and alerting
- [ ] Implement CI/CD for pipeline deployment
- [ ] Configure scheduled execution
- [ ] Track artifacts and lineage
- [ ] Regular review and optimization

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
