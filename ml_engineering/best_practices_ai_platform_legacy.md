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

## 10. Custom Prediction Routines (Legacy)

### Custom Prediction Code

**Best Practices:**
- Implement custom preprocessing
- Handle prediction logic
- Add postprocessing
- Include error handling

```python
# predictor.py
import os
import pickle
import numpy as np
from google.cloud import storage

class MyPredictor(object):
    """Custom prediction routine."""
    
    def __init__(self, model, preprocessor):
        """Initialize predictor with model and preprocessor."""
        self._model = model
        self._preprocessor = preprocessor
    
    def predict(self, instances, **kwargs):
        """
        Perform prediction.
        
        Args:
            instances: List of input instances
            **kwargs: Additional arguments
            
        Returns:
            List of predictions
        """
        # Preprocess inputs
        preprocessed = self._preprocessor.transform(instances)
        
        # Make predictions
        predictions = self._model.predict(preprocessed)
        
        # Postprocess outputs
        outputs = self._postprocess(predictions)
        
        return outputs
    
    def _postprocess(self, predictions):
        """Apply postprocessing to predictions."""
        # Convert to list and add metadata
        outputs = []
        for pred in predictions:
            outputs.append({
                'prediction': float(pred),
                'confidence': float(abs(pred)),
                'threshold': 0.5
            })
        return outputs
    
    @classmethod
    def from_path(cls, model_dir):
        """Load predictor from model directory."""
        # Load model
        model_path = os.path.join(model_dir, 'model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load preprocessor
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        return cls(model, preprocessor)

# Export with custom code
def export_with_custom_predictor(model, preprocessor, export_path):
    """Export model with custom predictor."""
    import pickle
    
    # Save model
    model_path = os.path.join(export_path, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save preprocessor
    preprocessor_path = os.path.join(export_path, 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Copy predictor code
    import shutil
    shutil.copy('predictor.py', os.path.join(export_path, 'predictor.py'))
    
    print(f"Model with custom predictor exported to {export_path}")

# setup.py for custom prediction
"""
from setuptools import setup

setup(
    name='my_custom_predictor',
    version='0.1',
    scripts=['predictor.py']
)
"""
```

### Deploy Custom Predictor

**Deployment Configuration:**
```python
def create_version_with_custom_predictor(
    project_id,
    model_name,
    version_name,
    deployment_uri,
    runtime_version='2.11',
    prediction_class='predictor.MyPredictor',
    package_uris=None
):
    """Create version with custom prediction routine."""
    
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)
    
    request_body = {
        'name': version_name,
        'deploymentUri': deployment_uri,
        'runtimeVersion': runtime_version,
        'pythonVersion': '3.7',
        'framework': 'SCIKIT_LEARN',
        'predictionClass': prediction_class,
        'machineType': 'n1-standard-4',
        'autoScaling': {
            'minNodes': 1,
            'maxNodes': 5
        }
    }
    
    if package_uris:
        request_body['packageUris'] = package_uris
    
    model_path = f'projects/{project_id}/models/{model_name}'
    request = ml.projects().models().versions().create(
        parent=model_path,
        body=request_body
    )
    
    response = request.execute()
    return response

# Deploy
create_version_with_custom_predictor(
    project_id='my-project',
    model_name='custom_model',
    version_name='v1',
    deployment_uri='gs://my-bucket/custom-model/',
    prediction_class='predictor.MyPredictor',
    package_uris=['gs://my-bucket/packages/custom-predictor-0.1.tar.gz']
)
```

---

## 11. Cost Optimization (Legacy)

### Training Cost Optimization

**Best Practices:**
- Use appropriate scale tiers
- Stop jobs when complete
- Use preemptible instances
- Optimize data loading

```python
# Cost-effective training configurations
SCALE_TIER_COSTS = {
    'BASIC': 'Lowest cost, single machine',
    'STANDARD_1': '1 master, 4 workers, 3 parameter servers',
    'PREMIUM_1': '1 master, 19 workers, 11 parameter servers',
    'BASIC_GPU': 'Single machine with 1 GPU',
    'BASIC_TPU': 'Single machine with TPU v2',
    'CUSTOM': 'Configure your own - most flexible'
}

def submit_cost_optimized_training(
    project_id,
    job_id,
    training_config
):
    """Submit training job with cost optimization."""
    
    # Use CUSTOM tier for fine-grained control
    request_body = {
        'jobId': job_id,
        'trainingInput': {
            'scaleTier': 'CUSTOM',
            # Use smaller master for coordination only
            'masterType': 'n1-standard-4',
            'masterConfig': {
                'acceleratorConfig': {
                    'count': 0,  # No GPU on master saves cost
                    'type': 'ACCELERATOR_TYPE_UNSPECIFIED'
                }
            },
            # Workers do the heavy lifting
            'workerType': 'n1-highmem-8',
            'workerCount': 3,
            'workerConfig': {
                'acceleratorConfig': {
                    'count': 1,
                    'type': 'NVIDIA_TESLA_K80'  # Most cost-effective GPU
                }
            },
            # Minimal parameter servers
            'parameterServerType': 'n1-standard-4',
            'parameterServerCount': 1,
            'region': 'us-central1',
            'jobDir': training_config['job_dir'],
            'runtimeVersion': '2.11',
            'pythonVersion': '3.7',
            'packageUris': training_config['package_uris'],
            'pythonModule': training_config['python_module'],
            'args': training_config['args']
        }
    }
    
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)
    
    project_name = f'projects/{project_id}'
    request = ml.projects().jobs().create(
        parent=project_name,
        body=request_body
    )
    
    response = request.execute()
    return response

# Cost monitoring query
"""
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
    service.description LIKE '%AI Platform%'
    AND DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY
    usage_date,
    service,
    sku,
    usage_unit
ORDER BY
    usage_date DESC,
    total_cost DESC;
"""
```

### Prediction Cost Optimization

**Optimize Serving Costs:**
```python
def create_cost_optimized_version(
    project_id,
    model_name,
    version_name,
    deployment_uri
):
    """Create model version with cost-optimized settings."""
    
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)
    
    request_body = {
        'name': version_name,
        'deploymentUri': deployment_uri,
        'runtimeVersion': '2.11',
        'pythonVersion': '3.7',
        'framework': 'TENSORFLOW',
        'machineType': 'mls1-c4-m2',  # Cost-optimized machine type
        'autoScaling': {
            'minNodes': 1,  # Minimize idle resources
            'maxNodes': 10,  # Cap maximum to control costs
            'metrics': [
                {
                    'name': 'aiplatform.googleapis.com/prediction/online/cpu/utilization',
                    'target': 70  # Higher target = fewer nodes = lower cost
                }
            ]
        },
        'requestLoggingConfig': {
            'samplingPercentage': 0.1  # Sample 10% to reduce logging costs
        }
    }
    
    model_path = f'projects/{project_id}/models/{model_name}'
    request = ml.projects().models().versions().create(
        parent=model_path,
        body=request_body
    )
    
    response = request.execute()
    return response

# Clean up old resources
def cleanup_old_resources(project_id, days_old=30):
    """Delete old models and job artifacts."""
    
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)
    
    import datetime
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_old)
    
    # List and clean up old jobs
    project_name = f'projects/{project_id}'
    jobs_request = ml.projects().jobs().list(parent=project_name)
    jobs_response = jobs_request.execute()
    
    for job in jobs_response.get('jobs', []):
        create_time = job.get('createTime', '')
        job_date = datetime.datetime.fromisoformat(create_time.replace('Z', '+00:00'))
        
        if job_date < cutoff_date and job.get('state') in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            print(f"Consider cleaning up artifacts for old job: {job['jobId']}")
            # Delete job directory from GCS
            job_dir = job.get('trainingInput', {}).get('jobDir', '')
            if job_dir:
                print(f"  Delete: {job_dir}")
```

---

## 12. Security Best Practices (Legacy)

### IAM and Access Control

**Best Practices:**
- Use principle of least privilege
- Create service accounts per function
- Enable audit logging
- Rotate credentials regularly

```python
# Required IAM roles for AI Platform
IAM_ROLES = {
    'ml_developer': [
        'roles/ml.developer',  # Full access to AI Platform
        'roles/storage.objectAdmin',  # GCS access
        'roles/logging.viewer'  # View logs
    ],
    'ml_operator': [
        'roles/ml.operationManager',  # Manage operations
        'roles/ml.modelUser',  # Use models for predictions
        'roles/storage.objectViewer'  # Read GCS
    ],
    'prediction_service': [
        'roles/ml.modelUser',  # Make predictions only
        'roles/logging.logWriter'  # Write prediction logs
    ]
}

# Grant roles using gcloud
"""
# Create service account
gcloud iam service-accounts create ai-platform-sa \
    --display-name="AI Platform Service Account"

# Grant necessary roles
gcloud projects add-iam-policy-binding my-project \
    --member="serviceAccount:ai-platform-sa@my-project.iam.gserviceaccount.com" \
    --role="roles/ml.developer"

gcloud projects add-iam-policy-binding my-project \
    --member="serviceAccount:ai-platform-sa@my-project.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Use service account for training
gcloud ai-platform jobs submit training training_job_001 \
    --service-account=ai-platform-sa@my-project.iam.gserviceaccount.com \
    --region=us-central1 \
    --module-name=trainer.task \
    --package-path=trainer/ \
    --job-dir=gs://my-bucket/jobs/training-001
"""
```

### Data Security

**Protect Training Data:**
```python
from google.cloud import storage
from google.cloud import kms

def encrypt_training_data(
    bucket_name,
    source_file,
    destination_file,
    kms_key_name
):
    """Encrypt training data using Cloud KMS."""
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_file)
    
    # Set encryption key
    blob.kms_key_name = kms_key_name
    
    # Upload with encryption
    blob.upload_from_filename(source_file)
    
    print(f"File {source_file} uploaded and encrypted to {destination_file}")

# Use encrypted data in training
def submit_training_with_encryption(
    project_id,
    job_id,
    training_config,
    kms_key_name
):
    """Submit training job with encrypted data."""
    
    # Ensure service account has KMS decryptor role
    # gcloud kms keys add-iam-policy-binding KEY_NAME \
    #     --location=LOCATION \
    #     --keyring=KEYRING_NAME \
    #     --member="serviceAccount:SA_EMAIL" \
    #     --role="roles/cloudkms.cryptoKeyDecrypter"
    
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)
    
    request_body = {
        'jobId': job_id,
        'trainingInput': training_config
    }
    
    project_name = f'projects/{project_id}'
    request = ml.projects().jobs().create(
        parent=project_name,
        body=request_body
    )
    
    response = request.execute()
    return response

# Enable audit logging
"""
# Create audit config
{
  "auditConfigs": [
    {
      "service": "ml.googleapis.com",
      "auditLogConfigs": [
        {
          "logType": "ADMIN_READ"
        },
        {
          "logType": "DATA_READ"
        },
        {
          "logType": "DATA_WRITE"
        }
      ]
    }
  ]
}
"""
```

### VPC Service Controls

**Secure with VPC Perimeters:**
```bash
# Create service perimeter for AI Platform
gcloud access-context-manager perimeters create ai_platform_perimeter \
    --title="AI Platform Perimeter" \
    --resources=projects/PROJECT_NUMBER \
    --restricted-services=ml.googleapis.com,storage.googleapis.com \
    --policy=POLICY_ID

# Add access level
gcloud access-context-manager perimeters update ai_platform_perimeter \
    --add-access-levels=ACCESS_LEVEL_NAME \
    --policy=POLICY_ID

# View perimeter details
gcloud access-context-manager perimeters describe ai_platform_perimeter \
    --policy=POLICY_ID
```

---

## 13. Troubleshooting Guide (Legacy)

### Common Training Issues

**Issue 1: Training Job Fails to Start**

```python
# Check job status
def diagnose_job_failure(project_id, job_id):
    """Diagnose why a job failed."""
    
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)
    
    job_name = f'projects/{project_id}/jobs/{job_id}'
    request = ml.projects().jobs().get(name=job_name)
    response = request.execute()
    
    print(f"Job State: {response.get('state')}")
    print(f"Error Message: {response.get('errorMessage', 'None')}")
    
    # Common issues and solutions
    if 'PERMISSION_DENIED' in response.get('errorMessage', ''):
        print("\nSolution: Check IAM permissions")
        print("  1. Verify service account has ml.developer role")
        print("  2. Verify service account has storage.objectAdmin role")
        print("  3. Check bucket permissions")
    
    elif 'NOT_FOUND' in response.get('errorMessage', ''):
        print("\nSolution: Check resource paths")
        print("  1. Verify package URI exists in GCS")
        print("  2. Verify training data paths are correct")
        print("  3. Check job directory path")
    
    elif 'INVALID_ARGUMENT' in response.get('errorMessage', ''):
        print("\nSolution: Check configuration")
        print("  1. Verify runtime version is supported")
        print("  2. Check machine type specifications")
        print("  3. Validate hyperparameter configuration")
    
    return response

# View training logs
"""
# Using gcloud
gcloud ai-platform jobs stream-logs JOB_ID

# Using Cloud Logging
gcloud logging read "resource.type=ml_job AND 
                     resource.labels.job_id=JOB_ID" \
    --limit=50 \
    --format=json
"""
```

**Issue 2: Out of Memory Errors**

```python
# Solutions for OOM errors
MEMORY_SOLUTIONS = {
    'increase_machine_memory': {
        'description': 'Use larger machine type',
        'example': {
            'masterType': 'n1-highmem-16',  # More memory
            'workerType': 'n1-highmem-8'
        }
    },
    'reduce_batch_size': {
        'description': 'Decrease batch size in training code',
        'example': {
            'batch_size': 16  # Reduce from 32 or 64
        }
    },
    'gradient_accumulation': {
        'description': 'Accumulate gradients over multiple batches',
        'example': {
            'batch_size': 16,
            'gradient_accumulation_steps': 4  # Effective batch size = 64
        }
    },
    'mixed_precision': {
        'description': 'Use mixed precision training (fp16)',
        'code': """
# In training code
import tensorflow as tf
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
        """
    }
}
```

### Common Prediction Issues

**Issue 3: Slow Predictions**

```python
def optimize_prediction_latency(project_id, model_name, version_name):
    """Optimize model version for lower latency."""
    
    # Update version configuration
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)
    
    version_path = f'projects/{project_id}/models/{model_name}/versions/{version_name}'
    
    # Get current configuration
    request = ml.projects().models().versions().get(name=version_path)
    current_config = request.execute()
    
    print("Current Configuration:")
    print(f"  Machine Type: {current_config.get('machineType')}")
    print(f"  Min Nodes: {current_config.get('autoScaling', {}).get('minNodes')}")
    
    print("\nOptimization Suggestions:")
    print("1. Increase minimum nodes to reduce cold starts")
    print("2. Use faster machine type (e.g., n1-highcpu-16)")
    print("3. Optimize model size (quantization, pruning)")
    print("4. Batch predictions when possible")
    print("5. Use caching for repeated requests")
    
    # Optimization command
    print("\nUpdate command:")
    print(f"""
gcloud ai-platform versions update {version_name} \\
    --model={model_name} \\
    --min-nodes=3 \\
    --machine-type=n1-highcpu-16
    """)

# Implement request batching
def batch_predictions(instances, batch_size=32):
    """Batch prediction requests for better throughput."""
    
    results = []
    
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i + batch_size]
        
        predictions = predict_online(
            project_id='my-project',
            model_name='my_model',
            version_name='v1',
            instances=batch
        )
        
        results.extend(predictions)
    
    return results
```

**Issue 4: Prediction Errors**

```python
def handle_prediction_errors(project_id, model_name, version_name, instances):
    """Robust prediction with error handling."""
    
    import time
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            predictions = predict_online(
                project_id=project_id,
                model_name=model_name,
                version_name=version_name,
                instances=instances
            )
            
            return predictions
            
        except errors.HttpError as err:
            if err.resp.status in [429, 503]:  # Rate limit or service unavailable
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limited. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} attempts")
                    raise
            elif err.resp.status == 400:
                print("Invalid request. Check input format:")
                print(f"  Instances: {instances}")
                raise
            elif err.resp.status == 404:
                print(f"Model or version not found: {model_name}/{version_name}")
                raise
            else:
                print(f"Unexpected error: {err}")
                raise
    
    return None

# Validate input before prediction
def validate_prediction_input(instances, expected_schema):
    """Validate input matches expected schema."""
    
    for idx, instance in enumerate(instances):
        for field, field_type in expected_schema.items():
            if field not in instance:
                raise ValueError(f"Instance {idx} missing required field: {field}")
            
            if not isinstance(instance[field], field_type):
                raise TypeError(
                    f"Instance {idx} field '{field}' has wrong type. "
                    f"Expected {field_type}, got {type(instance[field])}"
                )
    
    return True

# Example usage
expected_schema = {
    'age': (int, float),
    'income': (int, float),
    'tenure_months': int,
    'product_category': str
}

validate_prediction_input(instances, expected_schema)
predictions = handle_prediction_errors('my-project', 'my_model', 'v1', instances)
```

---

## 14. Advanced Training Configurations (Legacy)

### Distributed Training

**Multi-Worker Training:**
```python
def submit_distributed_training_job(
    project_id,
    job_id,
    training_config
):
    """Submit distributed training job."""
    
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)
    
    request_body = {
        'jobId': job_id,
        'trainingInput': {
            'scaleTier': 'CUSTOM',
            # Master coordinates training
            'masterType': 'n1-standard-8',
            'masterConfig': {
                'acceleratorConfig': {
                    'count': 1,
                    'type': 'NVIDIA_TESLA_V100'
                }
            },
            # Multiple workers for data parallelism
            'workerType': 'n1-highmem-16',
            'workerCount': 8,  # 8 workers
            'workerConfig': {
                'acceleratorConfig': {
                    'count': 2,  # 2 GPUs per worker
                    'type': 'NVIDIA_TESLA_V100'
                }
            },
            # Parameter servers for distributed gradients
            'parameterServerType': 'n1-highmem-8',
            'parameterServerCount': 3,
            'region': 'us-central1',
            'jobDir': training_config['job_dir'],
            'packageUris': training_config['package_uris'],
            'pythonModule': training_config['python_module'],
            'args': training_config['args'],
            'runtimeVersion': '2.11',
            'pythonVersion': '3.7'
        }
    }
    
    project_name = f'projects/{project_id}'
    request = ml.projects().jobs().create(
        parent=project_name,
        body=request_body
    )
    
    response = request.execute()
    return response

# Distributed training code
"""
# In trainer/task.py
import tensorflow as tf
import json
import os

def get_distribution_strategy():
    '''Get TensorFlow distribution strategy.'''
    
    # Get TF_CONFIG from environment (set by AI Platform)
    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
    
    if tf_config:
        # Multi-worker setup
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        print(f"Using MultiWorkerMirroredStrategy")
        print(f"Number of devices: {strategy.num_replicas_in_sync}")
    else:
        # Single worker
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy")
    
    return strategy

# Use strategy in training
strategy = get_distribution_strategy()

with strategy.scope():
    model = create_model()
    
    # Batch size should be per replica
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    
    model.fit(train_dataset, epochs=epochs)
"""
```

### TPU Training

**Using TPUs:**
```bash
# Submit TPU training job
gcloud ai-platform jobs submit training tpu_training_job_001 \
    --region=us-central1 \
    --module-name=trainer.task \
    --package-path=trainer/ \
    --job-dir=gs://my-bucket/jobs/tpu-training-001 \
    --runtime-version=2.11 \
    --python-version=3.7 \
    --scale-tier=BASIC_TPU \
    -- \
    --train-files=gs://my-bucket/data/train.tfrecords \
    --eval-files=gs://my-bucket/data/eval.tfrecords \
    --use-tpu=true

# TPU training code
"""
import tensorflow as tf

def get_tpu_strategy():
    '''Initialize TPU strategy.'''
    
    try:
        # Detect TPU
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        
        strategy = tf.distribute.TPUStrategy(resolver)
        print(f"Running on TPU with {strategy.num_replicas_in_sync} cores")
        
    except ValueError:
        # Fallback to CPU/GPU
        strategy = tf.distribute.MirroredStrategy()
        print("Running on CPU/GPU")
    
    return strategy

strategy = get_tpu_strategy()

with strategy.scope():
    model = create_model()
    model.fit(train_dataset, epochs=epochs)
"""
```

---

## 15. Integration Patterns (Legacy)

### BigQuery Integration

**Train from BigQuery:**
```python
def submit_training_from_bigquery(
    project_id,
    job_id,
    training_config,
    bigquery_table
):
    """Submit training job with BigQuery as data source."""
    
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)
    
    request_body = {
        'jobId': job_id,
        'trainingInput': {
            **training_config,
            'args': [
                '--bigquery-table', bigquery_table,
                '--output-path', training_config['job_dir']
            ]
        }
    }
    
    project_name = f'projects/{project_id}'
    request = ml.projects().jobs().create(
        parent=project_name,
        body=request_body
    )
    
    response = request.execute()
    return response

# Training code to read from BigQuery
"""
from google.cloud import bigquery
import pandas as pd

def load_data_from_bigquery(table_id):
    '''Load training data from BigQuery.'''
    
    client = bigquery.Client()
    
    query = f'''
        SELECT *
        FROM `{table_id}`
        WHERE DATE(timestamp) BETWEEN 
            DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY) 
            AND CURRENT_DATE()
    '''
    
    # Load into DataFrame
    df = client.query(query).to_dataframe()
    
    return df
"""
```

### Dataflow Integration

**Preprocessing with Dataflow:**
```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

def run_preprocessing_pipeline(
    input_pattern,
    output_path,
    project,
    region
):
    """Run Dataflow pipeline for preprocessing."""
    
    options = PipelineOptions([
        f'--project={project}',
        f'--region={region}',
        '--runner=DataflowRunner',
        '--temp_location=gs://my-bucket/temp',
        '--staging_location=gs://my-bucket/staging',
    ])
    
    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | 'Read' >> beam.io.ReadFromText(input_pattern)
            | 'Parse' >> beam.Map(parse_record)
            | 'Preprocess' >> beam.Map(preprocess_features)
            | 'Write' >> beam.io.WriteToText(
                output_path,
                file_name_suffix='.jsonl'
            )
        )

def parse_record(line):
    """Parse CSV line."""
    import json
    values = line.split(',')
    return {
        'age': int(values[0]),
        'income': float(values[1]),
        'tenure': int(values[2]),
        'label': int(values[3])
    }

def preprocess_features(record):
    """Apply feature preprocessing."""
    import json
    
    # Normalize features
    record['age_normalized'] = record['age'] / 100.0
    record['income_normalized'] = record['income'] / 100000.0
    
    return json.dumps(record)
```

---

## 16. Vertex AI Migration Commands

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

# Migrate training job to Vertex AI
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=TRAINING_JOB \
    --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=TRAINING_IMAGE \
    --args=--epochs=100,--batch-size=32
```

### Migration Checklist

```bash
# Pre-migration checklist
- [ ] Document current AI Platform resources
- [ ] Test Vertex AI equivalent features
- [ ] Update training code for Vertex AI SDK
- [ ] Update prediction code for Vertex AI endpoints
- [ ] Plan data migration strategy
- [ ] Update IAM roles and permissions
- [ ] Update monitoring and logging

# Migration steps
1. Create Vertex AI project structure
2. Migrate training pipelines
3. Migrate models to Model Registry
4. Create Vertex AI endpoints
5. Deploy models to endpoints
6. Update client applications
7. Test predictions
8. Monitor performance
9. Decommission AI Platform resources

# Post-migration
- [ ] Verify all predictions working
- [ ] Confirm monitoring is active
- [ ] Update documentation
- [ ] Train team on Vertex AI
- [ ] Clean up AI Platform resources
```

---

## 17. Quick Reference Checklist

### Migration Planning
- [ ] Inventory AI Platform resources
- [ ] Review Vertex AI feature parity
- [ ] Plan migration timeline
- [ ] Test migrations in dev environment
- [ ] Update documentation
- [ ] Train team on new platform

### Training (Legacy)
- [ ] Package code properly
- [ ] Use Cloud Storage for data
- [ ] Configure appropriate machine types
- [ ] Enable checkpointing
- [ ] Monitor training progress
- [ ] Implement error handling
- [ ] Use distributed training for large models

### Deployment (Legacy)
- [ ] Create model versions
- [ ] Configure auto-scaling
- [ ] Enable prediction logging
- [ ] Test predictions thoroughly
- [ ] Set up monitoring and alerts
- [ ] Implement error handling
- [ ] Document API contracts

### Security
- [ ] Apply least privilege IAM
- [ ] Use service accounts
- [ ] Enable audit logging
- [ ] Encrypt sensitive data
- [ ] Configure VPC Service Controls
- [ ] Rotate credentials regularly

### Cost Optimization
- [ ] Right-size machine types
- [ ] Use appropriate scale tiers
- [ ] Clean up old resources
- [ ] Monitor costs regularly
- [ ] Use batch predictions when possible
- [ ] Stop instances when not in use

### Maintenance
- [ ] Monitor prediction latency
- [ ] Track model performance
- [ ] Clean up old versions
- [ ] Review costs regularly
- [ ] Plan Vertex AI migration
- [ ] Keep runtime versions updated

---

**Important Note:** This document covers AI Platform (Legacy) for maintaining existing deployments and understanding the service for certification purposes. **For all new machine learning projects, use Vertex AI.** Refer to the Vertex AI best practices document for modern ML workflows on Google Cloud.

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
