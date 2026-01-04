# AI Platform (Legacy) Best Practices

*Last Updated: December 26, 2025*

## Overview

AI Platform (Legacy) was Google Cloud's managed machine learning service before Vertex AI. While it's being replaced by Vertex AI, understanding AI Platform is still important for maintaining existing systems and for the certification exam. This document covers best practices for AI Platform with notes on migration to Vertex AI.

**Important:** AI Platform is now legacy. For new projects, use **Vertex AI**. This guide is for maintaining existing AI Platform deployments and understanding the evolution to Vertex AI.

---

## 1. Migration to Vertex AI

### Migration Path

**Best Practices:**
- Plan migration to Vertex AI for new projects
- Understand feature parity
- Use migration tools
- Maintain both systems during transition

```bash
# Check AI Platform resources
gcloud ai-platform models list
gcloud ai-platform versions list --model=MODEL_NAME
gcloud ai-platform jobs list

# Migrate model to Vertex AI
gcloud ai models upload \
    --region=us-central1 \
    --display-name=migrated-model \
    --artifact-uri=gs://my-bucket/model/ \
    --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest

# Feature mapping
# AI Platform -> Vertex AI
# - Training Jobs -> CustomJob / AutoMLJob
# - Models -> Model Registry
# - Versions -> Model Versions
# - Predictions -> Endpoints / Batch Predictions
# - Notebooks -> Workbench
```

### Key Differences

**AI Platform vs Vertex AI:**
```python
# AI Platform (Legacy)
from googleapiclient import discovery
from googleapiclient import errors

ml = discovery.build('ml', 'v1')
project_id = 'my-project'
model_name = 'my_model'

request = ml.projects().models().get(
    name=f'projects/{project_id}/models/{model_name}'
)
response = request.execute()

# Vertex AI (Modern)
from google.cloud import aiplatform

aiplatform.init(project='my-project', location='us-central1')

model = aiplatform.Model.list(
    filter='display_name="my_model"'
)[0]

print(f"Model resource: {model.resource_name}")
```

---

## 2. Training Jobs (Legacy)

### Custom Training Jobs

**Best Practices:**
- Package code properly
- Use Cloud Storage for data
- Monitor job progress
- Implement checkpointing

```python
from googleapiclient import discovery
from googleapiclient import errors
from oauth2client.client import GoogleCredentials

def submit_training_job(
    project_id,
    job_id,
    training_input,
    package_uris,
    python_module,
    args
):
    """Submit AI Platform training job."""
    
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)
    
    request_body = {
        'jobId': job_id,
        'trainingInput': {
            'scaleTier': training_input.get('scale_tier', 'BASIC'),
            'packageUris': package_uris,
            'pythonModule': python_module,
            'args': args,
            'region': training_input.get('region', 'us-central1'),
            'jobDir': training_input.get('job_dir'),
            'runtimeVersion': training_input.get('runtime_version', '2.11'),
            'pythonVersion': training_input.get('python_version', '3.7'),
        }
    }
    
    # Add machine configuration if custom tier
    if training_input.get('scale_tier') == 'CUSTOM':
        request_body['trainingInput']['masterType'] = training_input.get('master_type', 'n1-standard-4')
        request_body['trainingInput']['workerType'] = training_input.get('worker_type', 'n1-standard-4')
        request_body['trainingInput']['workerCount'] = training_input.get('worker_count', 2)
        request_body['trainingInput']['parameterServerType'] = training_input.get('ps_type', 'n1-standard-4')
        request_body['trainingInput']['parameterServerCount'] = training_input.get('ps_count', 1)
    
    # Add GPU configuration
    if training_input.get('use_gpu'):
        request_body['trainingInput']['masterConfig'] = {
            'acceleratorConfig': {
                'count': training_input.get('gpu_count', 1),
                'type': training_input.get('gpu_type', 'NVIDIA_TESLA_K80')
            }
        }
    
    project_name = f'projects/{project_id}'
    request = ml.projects().jobs().create(
        parent=project_name,
        body=request_body
    )
    
    try:
        response = request.execute()
        print(f"Job submitted: {response}")
        return response
    except errors.HttpError as err:
        print(f"Error: {err}")
        return None

# Example usage
training_config = {
    'scale_tier': 'CUSTOM',
    'master_type': 'n1-highmem-8',
    'worker_type': 'n1-standard-4',
    'worker_count': 3,
    'ps_type': 'n1-standard-4',
    'ps_count': 1,
    'region': 'us-central1',
    'job_dir': 'gs://my-bucket/jobs/training-001',
    'runtime_version': '2.11',
    'python_version': '3.7',
    'use_gpu': True,
    'gpu_count': 2,
    'gpu_type': 'NVIDIA_TESLA_V100'
}

submit_training_job(
    project_id='my-project',
    job_id='training_job_001',
    training_input=training_config,
    package_uris=['gs://my-bucket/packages/trainer-0.1.tar.gz'],
    python_module='trainer.task',
    args=[
        '--train-files', 'gs://my-bucket/data/train.csv',
        '--eval-files', 'gs://my-bucket/data/eval.csv',
        '--epochs', '100',
        '--batch-size', '32'
    ]
)
```

### Package Training Code

**Create Trainer Package:**
```bash
# Directory structure
trainer/
├── __init__.py
├── task.py
├── model.py
└── setup.py

# setup.py
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tensorflow==2.11.0',
    'pandas>=1.3.0',
    'scikit-learn>=1.0.0',
]

setup(
    name='trainer',
    version='0.1',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    description='AI Platform training package'
)

# Build package
python setup.py sdist
gsutil cp dist/trainer-0.1.tar.gz gs://my-bucket/packages/

# Submit job with gcloud
gcloud ai-platform jobs submit training training_job_001 \
    --region=us-central1 \
    --module-name=trainer.task \
    --package-path=trainer/ \
    --job-dir=gs://my-bucket/jobs/training-001 \
    --runtime-version=2.11 \
    --python-version=3.7 \
    --scale-tier=CUSTOM \
    --master-machine-type=n1-highmem-8 \
    --worker-machine-type=n1-standard-4 \
    --worker-count=3 \
    --parameter-server-machine-type=n1-standard-4 \
    --parameter-server-count=1 \
    -- \
    --train-files=gs://my-bucket/data/train.csv \
    --eval-files=gs://my-bucket/data/eval.csv \
    --epochs=100 \
    --batch-size=32
```

### Training Code Structure

**Example trainer/task.py:**
```python
import argparse
import os
import tensorflow as tf
from tensorflow import keras
from . import model

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument(
        '--train-files',
        required=True,
        type=str,
        help='Training data path'
    )
    parser.add_argument(
        '--eval-files',
        required=True,
        type=str,
        help='Evaluation data path'
    )
    
    # Model arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    # AI Platform arguments
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='Job output directory'
    )
    
    args = parser.parse_args()
    return args

def train_and_evaluate(args):
    """Train and evaluate model."""
    
    # Load data
    train_dataset = tf.data.experimental.make_csv_dataset(
        args.train_files,
        batch_size=args.batch_size,
        label_name='label',
        num_epochs=1,
        shuffle=True
    )
    
    eval_dataset = tf.data.experimental.make_csv_dataset(
        args.eval_files,
        batch_size=args.batch_size,
        label_name='label',
        num_epochs=1,
        shuffle=False
    )
    
    # Build model
    model_instance = model.create_model(
        learning_rate=args.learning_rate
    )
    
    # Configure callbacks
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=os.path.join(args.job_dir, 'logs'),
        histogram_freq=1
    )
    
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.job_dir, 'checkpoints', 'model_{epoch:02d}.h5'),
        save_best_only=True,
        monitor='val_loss'
    )
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    history = model_instance.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=eval_dataset,
        callbacks=[tensorboard_callback, checkpoint_callback, early_stopping]
    )
    
    # Save final model
    export_path = os.path.join(args.job_dir, 'model')
    model_instance.save(export_path)
    
    print(f"Model exported to: {export_path}")
    
    return history

def main():
    """Main training function."""
    args = get_args()
    
    # Configure distributed training if needed
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    
    with strategy.scope():
        train_and_evaluate(args)

if __name__ == '__main__':
    main()
```

---

## 3. Hyperparameter Tuning (Legacy)

### HyperTune Configuration

**Best Practices:**
- Define search space carefully
- Use Bayesian optimization
- Set appropriate trial counts
- Monitor tuning progress

```python
def submit_hyperparameter_tuning_job(
    project_id,
    job_id,
    training_input,
    package_uris,
    python_module,
    args,
    hyperparameters
):
    """Submit hyperparameter tuning job."""
    
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)
    
    request_body = {
        'jobId': job_id,
        'trainingInput': {
            'scaleTier': training_input.get('scale_tier', 'STANDARD_1'),
            'packageUris': package_uris,
            'pythonModule': python_module,
            'args': args,
            'region': training_input.get('region', 'us-central1'),
            'jobDir': training_input.get('job_dir'),
            'runtimeVersion': training_input.get('runtime_version', '2.11'),
            'pythonVersion': training_input.get('python_version', '3.7'),
            'hyperparameters': hyperparameters
        }
    }
    
    project_name = f'projects/{project_id}'
    request = ml.projects().jobs().create(
        parent=project_name,
        body=request_body
    )
    
    response = request.execute()
    return response

# Hyperparameter configuration
hyperparameter_config = {
    'goal': 'MAXIMIZE',
    'hyperparameterMetricTag': 'accuracy',
    'maxTrials': 20,
    'maxParallelTrials': 5,
    'maxFailedTrials': 2,
    'algorithm': 'ALGORITHM_UNSPECIFIED',  # Bayesian optimization
    'params': [
        {
            'parameterName': 'learning_rate',
            'type': 'DOUBLE',
            'minValue': 0.0001,
            'maxValue': 0.1,
            'scaleType': 'UNIT_LOG_SCALE'
        },
        {
            'parameterName': 'batch_size',
            'type': 'DISCRETE',
            'discreteValues': [16, 32, 64, 128]
        },
        {
            'parameterName': 'num_layers',
            'type': 'INTEGER',
            'minValue': 2,
            'maxValue': 10,
            'scaleType': 'UNIT_LINEAR_SCALE'
        },
        {
            'parameterName': 'dropout_rate',
            'type': 'DOUBLE',
            'minValue': 0.1,
            'maxValue': 0.5,
            'scaleType': 'UNIT_LINEAR_SCALE'
        },
        {
            'parameterName': 'optimizer',
            'type': 'CATEGORICAL',
            'categoricalValues': ['adam', 'sgd', 'rmsprop']
        }
    ]
}

# Submit tuning job
submit_hyperparameter_tuning_job(
    project_id='my-project',
    job_id='hp_tuning_job_001',
    training_input={
        'scale_tier': 'STANDARD_1',
        'region': 'us-central1',
        'job_dir': 'gs://my-bucket/jobs/hp-tuning-001',
        'runtime_version': '2.11',
        'python_version': '3.7'
    },
    package_uris=['gs://my-bucket/packages/trainer-0.1.tar.gz'],
    python_module='trainer.task',
    args=['--train-files', 'gs://my-bucket/data/train.csv'],
    hyperparameters=hyperparameter_config
)
```

### Report Metrics for HyperTune

**In Training Code:**
```python
import hypertune

def train_with_hypertune(args):
    """Train model and report metrics to HyperTune."""
    
    # Train model
    model = create_model(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        optimizer=args.optimizer
    )
    
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=eval_dataset
    )
    
    # Report metric to HyperTune
    hpt = hypertune.HyperTune()
    
    # Report final validation accuracy
    final_accuracy = history.history['val_accuracy'][-1]
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='accuracy',
        metric_value=final_accuracy,
        global_step=args.epochs
    )
    
    # Can report multiple metrics
    final_loss = history.history['val_loss'][-1]
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='loss',
        metric_value=final_loss,
        global_step=args.epochs
    )
    
    return model

# requirements.txt should include:
# cloudml-hypertune
```

---

## 4. Model Deployment (Legacy)

### Create and Deploy Model

**Best Practices:**
- Version models systematically
- Use appropriate machine types
- Enable auto-scaling
- Monitor predictions

```python
def create_model(project_id, model_name, regions=None):
    """Create AI Platform model."""
    
    if regions is None:
        regions = ['us-central1']
    
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)
    
    request_body = {
        'name': model_name,
        'description': 'Production model for predictions',
        'regions': regions,
        'onlinePredictionLogging': True,
        'onlinePredictionConsoleLogging': True,
        'labels': {
            'env': 'production',
            'team': 'ml-team'
        }
    }
    
    project_name = f'projects/{project_id}'
    request = ml.projects().models().create(
        parent=project_name,
        body=request_body
    )
    
    response = request.execute()
    print(f"Model created: {response}")
    return response

def create_version(
    project_id,
    model_name,
    version_name,
    deployment_uri,
    runtime_version='2.11',
    machine_type='n1-standard-4',
    framework='TENSORFLOW'
):
    """Create model version."""
    
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)
    
    request_body = {
        'name': version_name,
        'deploymentUri': deployment_uri,
        'runtimeVersion': runtime_version,
        'pythonVersion': '3.7',
        'framework': framework,
        'machineType': machine_type,
        'autoScaling': {
            'minNodes': 1,
            'maxNodes': 10,
            'metrics': [
                {
                    'name': 'aiplatform.googleapis.com/prediction/online/cpu/utilization',
                    'target': 60
                }
            ]
        },
        'description': 'Model version with auto-scaling',
        'labels': {
            'version': 'v1',
            'status': 'production'
        }
    }
    
    model_path = f'projects/{project_id}/models/{model_name}'
    request = ml.projects().models().versions().create(
        parent=model_path,
        body=request_body
    )
    
    response = request.execute()
    print(f"Version created: {response}")
    return response

# Example usage
create_model(
    project_id='my-project',
    model_name='customer_churn_model',
    regions=['us-central1']
)

create_version(
    project_id='my-project',
    model_name='customer_churn_model',
    version_name='v1',
    deployment_uri='gs://my-bucket/model/export/final',
    runtime_version='2.11',
    machine_type='n1-standard-4',
    framework='TENSORFLOW'
)
```

### Set Default Version

**Manage Model Versions:**
```python
def set_default_version(project_id, model_name, version_name):
    """Set default model version."""
    
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)
    
    version_path = f'projects/{project_id}/models/{model_name}/versions/{version_name}'
    request = ml.projects().models().versions().setDefault(name=version_path)
    
    response = request.execute()
    print(f"Default version set: {response}")
    return response

def list_versions(project_id, model_name):
    """List all versions of a model."""
    
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)
    
    model_path = f'projects/{project_id}/models/{model_name}'
    request = ml.projects().models().versions().list(parent=model_path)
    
    response = request.execute()
    
    if 'versions' in response:
        for version in response['versions']:
            print(f"Version: {version['name']}")
            print(f"  State: {version.get('state', 'UNKNOWN')}")
            print(f"  Created: {version.get('createTime', 'N/A')}")
            print(f"  Is Default: {version.get('isDefault', False)}")
    
    return response

# Usage with gcloud
"""
# List models
gcloud ai-platform models list

# List versions
gcloud ai-platform versions list --model=customer_churn_model

# Set default version
gcloud ai-platform versions set-default v2 --model=customer_churn_model

# Delete old version
gcloud ai-platform versions delete v1 --model=customer_churn_model
"""
```

---

## 5. Predictions (Legacy)

### Online Predictions

**Best Practices:**
- Batch requests when possible
- Handle errors gracefully
- Monitor latency
- Implement retry logic

```python
def predict_online(project_id, model_name, version_name, instances):
    """Make online predictions."""
    
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)
    
    name = f'projects/{project_id}/models/{model_name}'
    
    if version_name:
        name += f'/versions/{version_name}'
    
    request_body = {
        'instances': instances
    }
    
    request = ml.projects().predict(name=name, body=request_body)
    
    try:
        response = request.execute()
        
        if 'predictions' in response:
            return response['predictions']
        else:
            print(f"Error: {response.get('error', 'Unknown error')}")
            return None
            
    except errors.HttpError as err:
        print(f"HTTP Error: {err}")
        return None

# Example usage
instances = [
    {
        'age': 35,
        'income': 75000,
        'tenure_months': 24,
        'product_category': 'premium',
        'region': 'west'
    },
    {
        'age': 42,
        'income': 95000,
        'tenure_months': 48,
        'product_category': 'standard',
        'region': 'east'
    }
]

predictions = predict_online(
    project_id='my-project',
    model_name='customer_churn_model',
    version_name='v1',
    instances=instances
)

print(f"Predictions: {predictions}")

# Using gcloud
"""
# Create request file (instances.json)
[
  {"age": 35, "income": 75000, "tenure_months": 24},
  {"age": 42, "income": 95000, "tenure_months": 48}
]

# Make prediction
gcloud ai-platform predict \
    --model=customer_churn_model \
    --version=v1 \
    --json-instances=instances.json
"""
```

### Batch Predictions

**Large-Scale Inference:**
```python
def submit_batch_prediction_job(
    project_id,
    job_id,
    model_name,
    version_name,
    input_paths,
    output_path,
    region='us-central1',
    data_format='JSON'
):
    """Submit batch prediction job."""
    
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)
    
    model_path = f'projects/{project_id}/models/{model_name}/versions/{version_name}'
    
    request_body = {
        'jobId': job_id,
        'predictionInput': {
            'modelName': model_path,
            'inputPaths': input_paths,
            'outputPath': output_path,
            'region': region,
            'dataFormat': data_format,
            'batchSize': 64,
            'maxWorkerCount': 10
        }
    }
    
    project_name = f'projects/{project_id}'
    request = ml.projects().jobs().create(
        parent=project_name,
        body=request_body
    )
    
    response = request.execute()
    print(f"Batch prediction job submitted: {response}")
    return response

def check_job_status(project_id, job_id):
    """Check batch prediction job status."""
    
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)
    
    job_name = f'projects/{project_id}/jobs/{job_id}'
    request = ml.projects().jobs().get(name=job_name)
    
    response = request.execute()
    
    print(f"Job state: {response.get('state', 'UNKNOWN')}")
    
    if response.get('state') == 'SUCCEEDED':
        print(f"Output path: {response['predictionOutput']['outputPath']}")
    elif response.get('state') == 'FAILED':
        print(f"Error: {response.get('errorMessage', 'Unknown error')}")
    
    return response

# Example usage
submit_batch_prediction_job(
    project_id='my-project',
    job_id='batch_prediction_001',
    model_name='customer_churn_model',
    version_name='v1',
    input_paths=['gs://my-bucket/batch-input/*.json'],
    output_path='gs://my-bucket/batch-output/',
    region='us-central1',
    data_format='JSON'
)

# Using gcloud
"""
gcloud ai-platform jobs submit prediction batch_prediction_001 \
    --model=customer_churn_model \
    --version=v1 \
    --input-paths=gs://my-bucket/batch-input/*.json \
    --output-path=gs://my-bucket/batch-output/ \
    --region=us-central1 \
    --data-format=JSON
"""
```

---

## 6. Notebooks (Legacy)

### AI Platform Notebooks

**Best Practices:**
- Use managed notebooks for development
- Install required packages
- Save work regularly
- Use version control

```bash
# Create AI Platform Notebook instance
gcloud notebooks instances create my-notebook \
    --vm-image-project=deeplearning-platform-release \
    --vm-image-family=tf2-latest-gpu \
    --machine-type=n1-standard-4 \
    --location=us-central1-b \
    --accelerator-type=NVIDIA_TESLA_T4 \
    --accelerator-core-count=1 \
    --install-gpu-driver

# List notebook instances
gcloud notebooks instances list --location=us-central1-b

# Stop instance when not in use
gcloud notebooks instances stop my-notebook --location=us-central1-b

# Start instance
gcloud notebooks instances start my-notebook --location=us-central1-b

# Delete instance
gcloud notebooks instances delete my-notebook --location=us-central1-b
```

### Migrate to Vertex AI Workbench

**Migration Steps:**
```bash
# Vertex AI Workbench (replacement for AI Platform Notebooks)
gcloud workbench instances create my-workbench \
    --location=us-central1-b \
    --machine-type=n1-standard-4 \
    --accelerator-type=NVIDIA_TESLA_T4 \
    --accelerator-core-count=1 \
    --metadata="framework=TensorFlow,version=2.11"

# List Workbench instances
gcloud workbench instances list --location=us-central1-b
```

---

## 7. Monitoring and Logging (Legacy)

### Prediction Logging

**Best Practices:**
- Enable prediction logging
- Monitor prediction latency
- Track prediction distribution
- Set up alerts

```python
# Enable logging when creating version
request_body = {
    'name': version_name,
    'deploymentUri': deployment_uri,
    'runtimeVersion': '2.11',
    'machineType': 'n1-standard-4',
    'requestLoggingConfig': {
        'samplingPercentage': 1.0  # Log all predictions
    },
    'explanationConfig': {
        'integratedGradientsAttribution': {
            'numIntegralSteps': 50
        }
    }
}

# Query prediction logs in BigQuery
"""
SELECT
    model,
    model_version,
    predict.instances,
    predict.predictions,
    timestamp
FROM
    `my-project.prediction_logs.predictions_*`
WHERE
    DATE(timestamp) = CURRENT_DATE()
    AND model = 'customer_churn_model'
ORDER BY
    timestamp DESC
LIMIT 100;
"""
```

### Monitor with Cloud Monitoring

**Set Up Monitoring:**
```python
from google.cloud import monitoring_v3

def create_prediction_alert(project_id):
    """Create alert for prediction errors."""
    
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"
    
    alert_policy = monitoring_v3.AlertPolicy(
        display_name="AI Platform Prediction Errors",
        conditions=[
            monitoring_v3.AlertPolicy.Condition(
                display_name="High error rate",
                condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                    filter='resource.type="ml_model" '
                           'metric.type="ml.googleapis.com/prediction/error_count"',
                    comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                    threshold_value=10,
                    duration={"seconds": 300},
                ),
            )
        ],
        notification_channels=[],  # Add notification channels
    )
    
    created_policy = client.create_alert_policy(
        name=project_name,
        alert_policy=alert_policy
    )
    
    return created_policy
```

---

## 8. Common Anti-Patterns

### ❌ Anti-Pattern 1: Not Migrating to Vertex AI
**Problem:** Using legacy service for new projects
**Solution:** Use Vertex AI for all new ML workloads

### ❌ Anti-Pattern 2: Hardcoded Credentials
**Problem:** Security vulnerabilities
**Solution:** Use Application Default Credentials

### ❌ Anti-Pattern 3: No Version Management
**Problem:** Cannot rollback deployments
**Solution:** Maintain multiple model versions

### ❌ Anti-Pattern 4: Ignoring Costs
**Problem:** Unnecessary expenses
**Solution:** Stop instances when not in use, use appropriate machine types

### ❌ Anti-Pattern 5: No Monitoring
**Problem:** Issues go unnoticed
**Solution:** Enable logging and set up alerts

---

## 9. Quick Reference Checklist

### Migration Planning
- [ ] Inventory AI Platform resources
- [ ] Review Vertex AI feature parity
- [ ] Plan migration timeline
- [ ] Test migrations in dev environment
- [ ] Update documentation

### Training (Legacy)
- [ ] Package code properly
- [ ] Use Cloud Storage for data
- [ ] Configure appropriate machine types
- [ ] Enable checkpointing
- [ ] Monitor training progress

### Deployment (Legacy)
- [ ] Create model versions
- [ ] Configure auto-scaling
- [ ] Enable prediction logging
- [ ] Test predictions
- [ ] Set up monitoring

### Maintenance
- [ ] Monitor prediction latency
- [ ] Track model performance
- [ ] Clean up old versions
- [ ] Review costs regularly
- [ ] Plan Vertex AI migration

---

## 10. Vertex AI Migration Commands

### Quick Migration Reference

```bash
# Export AI Platform model
gcloud ai-platform models describe MODEL_NAME > model_config.yaml

# Download model artifacts
gsutil -m cp -r gs://bucket/model-path ./local-model

# Upload to Vertex AI
gcloud ai models upload \
    --region=us-central1 \
    --display-name=MODEL_NAME \
    --artifact-uri=gs://bucket/model-path \
    --container-image-uri=CONTAINER_IMAGE

# Create Vertex AI endpoint
gcloud ai endpoints create \
    --region=us-central1 \
    --display-name=ENDPOINT_NAME

# Deploy to endpoint
gcloud ai endpoints deploy-model ENDPOINT_ID \
    --region=us-central1 \
    --model=MODEL_ID \
    --display-name=DEPLOYMENT_NAME \
    --machine-type=n1-standard-4 \
    --min-replica-count=1 \
    --max-replica-count=3
```

---

**Important Note:** This document covers AI Platform (Legacy) for maintaining existing deployments and understanding the service for certification purposes. **For all new machine learning projects, use Vertex AI.** Refer to the Vertex AI best practices document for modern ML workflows on Google Cloud.

---

*Best Practices for Google Cloud Data Engineer Certification*
