# Vertex AI Model Registry Best Practices

*Last Updated: January 4, 2026*

## Overview

Vertex AI Model Registry provides centralized model versioning, metadata management, and lineage tracking for production ML models, enabling governance and collaboration across teams.

---

## 1. Model Registration

### Register Models

```python
from google.cloud import aiplatform

class ModelRegistry:
    """Manage models in Vertex AI Model Registry."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        
        aiplatform.init(project=project_id, location=location)
    
    def register_model(
        self,
        display_name,
        artifact_uri,
        serving_container_image_uri,
        description='',
        labels=None
    ):
        """Register a new model in the registry."""
        
        if labels is None:
            labels = {}
        
        model = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri=serving_container_image_uri,
            description=description,
            labels=labels
        )
        
        print(f"✓ Registered model: {model.display_name}")
        print(f"  Resource name: {model.resource_name}")
        print(f"  Version ID: {model.version_id}")
        
        return model
    
    def register_sklearn_model(
        self,
        display_name,
        model_path,
        sklearn_version='1.0'
    ):
        """Register scikit-learn model."""
        
        # Pre-built sklearn serving container
        serving_container = f'us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.{sklearn_version}:latest'
        
        model = self.register_model(
            display_name=display_name,
            artifact_uri=model_path,
            serving_container_image_uri=serving_container,
            description=f'Scikit-learn {sklearn_version} model',
            labels={'framework': 'sklearn', 'version': sklearn_version}
        )
        
        return model
    
    def register_tensorflow_model(
        self,
        display_name,
        saved_model_dir,
        tf_version='2.11'
    ):
        """Register TensorFlow SavedModel."""
        
        serving_container = f'us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.{tf_version}:latest'
        
        model = self.register_model(
            display_name=display_name,
            artifact_uri=saved_model_dir,
            serving_container_image_uri=serving_container,
            description=f'TensorFlow {tf_version} model',
            labels={'framework': 'tensorflow', 'version': tf_version}
        )
        
        return model
    
    def register_pytorch_model(
        self,
        display_name,
        model_path,
        pytorch_version='1.13'
    ):
        """Register PyTorch model."""
        
        serving_container = f'us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.{pytorch_version}:latest'
        
        model = self.register_model(
            display_name=display_name,
            artifact_uri=model_path,
            serving_container_image_uri=serving_container,
            description=f'PyTorch {pytorch_version} model',
            labels={'framework': 'pytorch', 'version': pytorch_version}
        )
        
        return model

# Example usage
registry = ModelRegistry(project_id='my-project')

# Register scikit-learn model
# model = registry.register_sklearn_model(
#     display_name='fraud-detection-model',
#     model_path='gs://my-bucket/models/fraud_detector',
#     sklearn_version='1.0'
# )
```

---

## 2. Model Versioning

### Manage Model Versions

```python
class ModelVersionManager:
    """Manage model versions and aliases."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def create_model_version(
        self,
        parent_model_name,
        artifact_uri,
        version_description=''
    ):
        """Create a new version of existing model."""
        
        # Get parent model
        parent_model = aiplatform.Model(parent_model_name)
        
        # Upload new version
        new_version = aiplatform.Model.upload(
            display_name=parent_model.display_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri=parent_model.container_spec.image_uri,
            description=version_description,
            parent_model=parent_model.resource_name,
            is_default_version=False
        )
        
        print(f"✓ Created model version: {new_version.version_id}")
        print(f"  Parent: {parent_model.display_name}")
        print(f"  Resource: {new_version.resource_name}")
        
        return new_version
    
    def list_model_versions(self, model_id):
        """List all versions of a model."""
        
        # Get base model
        models = aiplatform.Model.list(filter=f'display_name="{model_id}"')
        
        if not models:
            print(f"Model not found: {model_id}")
            return []
        
        base_model = models[0]
        
        # List all versions
        all_versions = aiplatform.Model.list(
            filter=f'model_id="{base_model.name.split("/")[-1]}"',
            order_by='create_time desc'
        )
        
        print(f"\nModel Versions for {model_id}:\n")
        
        for version in all_versions:
            is_default = ' (DEFAULT)' if version.version_id == base_model.version_id else ''
            print(f"Version {version.version_id}{is_default}")
            print(f"  Created: {version.create_time}")
            print(f"  Resource: {version.resource_name}")
            print(f"  Description: {version.description or 'N/A'}")
            print()
        
        return all_versions
    
    def set_default_version(self, model_resource_name):
        """Set a specific version as default."""
        
        model = aiplatform.Model(model_resource_name)
        model.update(is_default_version=True)
        
        print(f"✓ Set default version: {model.version_id}")
        
        return model
    
    def add_version_alias(self, model_resource_name, alias):
        """Add alias to model version."""
        
        model = aiplatform.Model(model_resource_name)
        
        current_aliases = list(model.version_aliases or [])
        
        if alias not in current_aliases:
            current_aliases.append(alias)
            model.update(version_aliases=current_aliases)
            print(f"✓ Added alias '{alias}' to version {model.version_id}")
        else:
            print(f"Alias '{alias}' already exists")
        
        return model
    
    def remove_version_alias(self, model_resource_name, alias):
        """Remove alias from model version."""
        
        model = aiplatform.Model(model_resource_name)
        
        current_aliases = list(model.version_aliases or [])
        
        if alias in current_aliases:
            current_aliases.remove(alias)
            model.update(version_aliases=current_aliases)
            print(f"✓ Removed alias '{alias}' from version {model.version_id}")
        else:
            print(f"Alias '{alias}' not found")
        
        return model
    
    def get_model_by_alias(self, model_id, alias):
        """Get model version by alias."""
        
        models = aiplatform.Model.list(filter=f'display_name="{model_id}"')
        
        for model in models:
            if model.version_aliases and alias in model.version_aliases:
                print(f"Found model with alias '{alias}':")
                print(f"  Version: {model.version_id}")
                print(f"  Resource: {model.resource_name}")
                return model
        
        print(f"No model found with alias '{alias}'")
        return None

# Example usage
version_manager = ModelVersionManager(project_id='my-project')

# Create new version
# new_version = version_manager.create_model_version(
#     parent_model_name='projects/.../locations/.../models/123',
#     artifact_uri='gs://my-bucket/models/v2',
#     version_description='Improved accuracy by 5%'
# )

# Add aliases
# version_manager.add_version_alias(new_version.resource_name, 'champion')
# version_manager.add_version_alias(new_version.resource_name, 'production')

# List versions
# versions = version_manager.list_model_versions('fraud-detection-model')
```

---

## 3. Model Metadata Management

### Track Model Information

```python
class ModelMetadataManager:
    """Manage model metadata and properties."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def add_model_metadata(
        self,
        model_resource_name,
        training_dataset=None,
        evaluation_metrics=None,
        hyperparameters=None,
        training_framework=None
    ):
        """Add comprehensive metadata to model."""
        
        model = aiplatform.Model(model_resource_name)
        
        # Build metadata dictionary
        metadata = {}
        
        if training_dataset:
            metadata['training_dataset'] = training_dataset
        
        if evaluation_metrics:
            for metric_name, metric_value in evaluation_metrics.items():
                metadata[f'metric_{metric_name}'] = str(metric_value)
        
        if hyperparameters:
            for param_name, param_value in hyperparameters.items():
                metadata[f'param_{param_name}'] = str(param_value)
        
        if training_framework:
            metadata['framework'] = training_framework
        
        # Update model with metadata
        current_labels = dict(model.labels or {})
        current_labels.update(metadata)
        
        model.update(labels=current_labels)
        
        print(f"✓ Updated metadata for model version {model.version_id}")
        print(f"  Added {len(metadata)} metadata fields")
        
        return model
    
    def get_model_metadata(self, model_resource_name):
        """Retrieve model metadata."""
        
        model = aiplatform.Model(model_resource_name)
        
        print(f"\nModel Metadata: {model.display_name} v{model.version_id}\n")
        
        print(f"Display Name: {model.display_name}")
        print(f"Version ID: {model.version_id}")
        print(f"Description: {model.description or 'N/A'}")
        print(f"Created: {model.create_time}")
        print(f"Updated: {model.update_time}")
        print(f"Artifact URI: {model.artifact_uri}")
        print(f"Container Image: {model.container_spec.image_uri}")
        
        if model.labels:
            print(f"\nLabels:")
            for key, value in model.labels.items():
                print(f"  {key}: {value}")
        
        if model.version_aliases:
            print(f"\nAliases: {', '.join(model.version_aliases)}")
        
        return {
            'display_name': model.display_name,
            'version_id': model.version_id,
            'labels': dict(model.labels or {}),
            'aliases': list(model.version_aliases or []),
            'artifact_uri': model.artifact_uri
        }
    
    def compare_model_versions(self, model_resource_names):
        """Compare metadata across model versions."""
        
        comparison = []
        
        for resource_name in model_resource_names:
            model = aiplatform.Model(resource_name)
            
            comparison.append({
                'version_id': model.version_id,
                'created': model.create_time,
                'metrics': {k: v for k, v in (model.labels or {}).items() if k.startswith('metric_')},
                'params': {k: v for k, v in (model.labels or {}).items() if k.startswith('param_')},
                'aliases': list(model.version_aliases or [])
            })
        
        print("\n=== Model Version Comparison ===\n")
        
        for item in comparison:
            print(f"Version {item['version_id']}:")
            print(f"  Created: {item['created']}")
            print(f"  Aliases: {', '.join(item['aliases']) or 'None'}")
            
            if item['metrics']:
                print(f"  Metrics:")
                for metric, value in item['metrics'].items():
                    print(f"    {metric.replace('metric_', '')}: {value}")
            
            if item['params']:
                print(f"  Parameters:")
                for param, value in item['params'].items():
                    print(f"    {param.replace('param_', '')}: {value}")
            print()
        
        return comparison

# Example usage
metadata_manager = ModelMetadataManager(project_id='my-project')

# Add metadata
# metadata_manager.add_model_metadata(
#     model_resource_name='projects/.../models/123/versions/1',
#     training_dataset='gs://my-bucket/data/train_v1.csv',
#     evaluation_metrics={
#         'accuracy': 0.95,
#         'precision': 0.93,
#         'recall': 0.94,
#         'f1_score': 0.935
#     },
#     hyperparameters={
#         'learning_rate': 0.001,
#         'batch_size': 32,
#         'epochs': 50
#     },
#     training_framework='tensorflow-2.11'
# )

# Get metadata
# metadata = metadata_manager.get_model_metadata('projects/.../models/123')
```

---

## 4. Model Lineage Tracking

### Track Model Provenance

```python
from google.cloud.aiplatform import metadata

class ModelLineageTracker:
    """Track model lineage and data provenance."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        
        metadata.init(project=project_id, location=location)
    
    def create_execution(self, execution_name, execution_type='Training'):
        """Create execution for tracking."""
        
        execution = metadata.Execution.create(
            schema_title=execution_type,
            display_name=execution_name,
            metadata={
                'start_time': str(pd.Timestamp.now()),
                'execution_type': execution_type
            }
        )
        
        print(f"✓ Created execution: {execution.display_name}")
        
        return execution
    
    def link_artifacts_to_execution(
        self,
        execution,
        input_artifacts,
        output_artifacts
    ):
        """Link input/output artifacts to execution."""
        
        # Add input artifacts
        for artifact in input_artifacts:
            execution.add_input_artifact(artifact)
        
        # Add output artifacts
        for artifact in output_artifacts:
            execution.add_output_artifact(artifact)
        
        print(f"✓ Linked {len(input_artifacts)} inputs and {len(output_artifacts)} outputs")
        
        return execution
    
    def create_dataset_artifact(self, dataset_uri, dataset_name):
        """Create dataset artifact for lineage."""
        
        artifact = metadata.Artifact.create(
            schema_title='system.Dataset',
            display_name=dataset_name,
            uri=dataset_uri,
            metadata={
                'dataset_type': 'training_data',
                'created_at': str(pd.Timestamp.now())
            }
        )
        
        print(f"✓ Created dataset artifact: {dataset_name}")
        
        return artifact
    
    def create_model_artifact(self, model_uri, model_name):
        """Create model artifact for lineage."""
        
        artifact = metadata.Artifact.create(
            schema_title='system.Model',
            display_name=model_name,
            uri=model_uri,
            metadata={
                'model_type': 'classification',
                'created_at': str(pd.Timestamp.now())
            }
        )
        
        print(f"✓ Created model artifact: {model_name}")
        
        return artifact
    
    def query_lineage(self, artifact_resource_name):
        """Query lineage for an artifact."""
        
        import pandas as pd
        
        artifact = metadata.Artifact(artifact_resource_name)
        
        print(f"\n=== Lineage for {artifact.display_name} ===\n")
        
        # Get executions
        executions = artifact.get_executions()
        
        print(f"Related Executions ({len(executions)}):")
        for execution in executions:
            print(f"  - {execution.display_name} ({execution.schema_title})")
        
        # Get upstream artifacts
        upstream = artifact.get_upstream_artifacts()
        print(f"\nUpstream Artifacts ({len(upstream)}):")
        for up_artifact in upstream:
            print(f"  - {up_artifact.display_name} ({up_artifact.schema_title})")
        
        # Get downstream artifacts
        downstream = artifact.get_downstream_artifacts()
        print(f"\nDownstream Artifacts ({len(downstream)}):")
        for down_artifact in downstream:
            print(f"  - {down_artifact.display_name} ({down_artifact.schema_title})")
        
        return {
            'executions': executions,
            'upstream': upstream,
            'downstream': downstream
        }

# Example: Complete lineage tracking
import pandas as pd

lineage_tracker = ModelLineageTracker(project_id='my-project')

# Create training execution
# execution = lineage_tracker.create_execution(
#     execution_name='fraud-model-training-v1',
#     execution_type='Training'
# )

# Create artifacts
# dataset_artifact = lineage_tracker.create_dataset_artifact(
#     dataset_uri='gs://my-bucket/data/train_v1.csv',
#     dataset_name='fraud-training-data-v1'
# )

# model_artifact = lineage_tracker.create_model_artifact(
#     model_uri='gs://my-bucket/models/fraud_detector_v1',
#     model_name='fraud-detection-model-v1'
# )

# Link artifacts
# lineage_tracker.link_artifacts_to_execution(
#     execution=execution,
#     input_artifacts=[dataset_artifact],
#     output_artifacts=[model_artifact]
# )

# Query lineage
# lineage = lineage_tracker.query_lineage(model_artifact.resource_name)
```

---

## 5. Model Governance

### Implement Model Approval Workflow

```python
class ModelGovernance:
    """Implement model governance and approval workflows."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def submit_for_approval(self, model_resource_name, reviewer_email):
        """Submit model for approval."""
        
        model = aiplatform.Model(model_resource_name)
        
        # Add approval metadata
        labels = dict(model.labels or {})
        labels['approval_status'] = 'pending'
        labels['reviewer'] = reviewer_email.replace('@', '_at_').replace('.', '_')
        labels['submission_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        model.update(labels=labels)
        
        print(f"✓ Submitted model for approval")
        print(f"  Model: {model.display_name} v{model.version_id}")
        print(f"  Reviewer: {reviewer_email}")
        
        # Send notification (implement email/slack notification)
        self._send_approval_notification(model, reviewer_email)
        
        return model
    
    def approve_model(self, model_resource_name, approver_email, comments=''):
        """Approve model for production."""
        
        model = aiplatform.Model(model_resource_name)
        
        labels = dict(model.labels or {})
        labels['approval_status'] = 'approved'
        labels['approver'] = approver_email.replace('@', '_at_').replace('.', '_')
        labels['approval_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        model.update(labels=labels)
        
        # Add production alias
        version_aliases = list(model.version_aliases or [])
        if 'production' not in version_aliases:
            version_aliases.append('production')
            model.update(version_aliases=version_aliases)
        
        print(f"✓ Approved model for production")
        print(f"  Model: {model.display_name} v{model.version_id}")
        print(f"  Approver: {approver_email}")
        
        return model
    
    def reject_model(self, model_resource_name, reviewer_email, reason=''):
        """Reject model."""
        
        model = aiplatform.Model(model_resource_name)
        
        labels = dict(model.labels or {})
        labels['approval_status'] = 'rejected'
        labels['reviewer'] = reviewer_email.replace('@', '_at_').replace('.', '_')
        labels['rejection_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
        labels['rejection_reason'] = reason[:63]  # Label value limit
        
        model.update(labels=labels)
        
        print(f"✗ Rejected model")
        print(f"  Model: {model.display_name} v{model.version_id}")
        print(f"  Reason: {reason}")
        
        return model
    
    def list_pending_approvals(self):
        """List models pending approval."""
        
        models = aiplatform.Model.list(
            filter='labels.approval_status="pending"'
        )
        
        print(f"\n=== Pending Approvals ({len(models)}) ===\n")
        
        for model in models:
            labels = model.labels or {}
            print(f"{model.display_name} v{model.version_id}")
            print(f"  Submitted: {labels.get('submission_date', 'N/A')}")
            print(f"  Reviewer: {labels.get('reviewer', 'N/A').replace('_at_', '@').replace('_', '.')}")
            print()
        
        return models
    
    def _send_approval_notification(self, model, reviewer_email):
        """Send approval notification (implement with email/Slack)."""
        print(f"Notification sent to {reviewer_email}")

# Example usage
import pandas as pd

governance = ModelGovernance(project_id='my-project')

# Submit for approval
# governance.submit_for_approval(
#     model_resource_name='projects/.../models/123',
#     reviewer_email='ml-lead@company.com'
# )

# Approve model
# governance.approve_model(
#     model_resource_name='projects/.../models/123',
#     approver_email='ml-lead@company.com',
#     comments='Meets accuracy threshold'
# )

# List pending
# pending = governance.list_pending_approvals()
```

---

## 6. Model Search and Discovery

```python
class ModelSearch:
    """Search and discover models in registry."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def search_models(self, filters=None, order_by='create_time desc', limit=10):
        """Search models with filters."""
        
        filter_str = ''
        
        if filters:
            filter_parts = []
            
            if 'framework' in filters:
                filter_parts.append(f'labels.framework="{filters["framework"]}"')
            
            if 'approval_status' in filters:
                filter_parts.append(f'labels.approval_status="{filters["approval_status"]}"')
            
            if 'alias' in filters:
                filter_parts.append(f'version_aliases="{filters["alias"]}"')
            
            filter_str = ' AND '.join(filter_parts)
        
        models = aiplatform.Model.list(
            filter=filter_str,
            order_by=order_by,
            limit=limit
        )
        
        print(f"\nFound {len(models)} models:\n")
        
        for model in models:
            print(f"{model.display_name} v{model.version_id}")
            print(f"  Created: {model.create_time}")
            print(f"  Aliases: {', '.join(model.version_aliases or []) or 'None'}")
            
            if model.labels:
                print(f"  Labels: {dict(model.labels)}")
            print()
        
        return models
    
    def find_production_models(self):
        """Find all models with production alias."""
        
        return self.search_models(
            filters={'alias': 'production'},
            order_by='update_time desc'
        )
    
    def find_models_by_framework(self, framework):
        """Find models by ML framework."""
        
        return self.search_models(
            filters={'framework': framework}
        )

# Example usage
search = ModelSearch(project_id='my-project')

# Find production models
# prod_models = search.find_production_models()

# Find TensorFlow models
# tf_models = search.find_models_by_framework('tensorflow')
```

---

## 7. Custom Containers for Model Serving

### 7.1 Overview of Custom Containers

```python
"""
WHEN TO USE CUSTOM CONTAINERS:

1. Custom Preprocessing Logic
   - Complex feature transformations
   - Business-specific data validation
   - Multi-step preprocessing pipelines

2. Custom Model Frameworks
   - Unsupported ML frameworks
   - Custom model architectures
   - Legacy model formats

3. Custom Dependencies
   - Specific library versions
   - Proprietary libraries
   - System-level dependencies

4. Custom Serving Logic
   - A/B testing logic
   - Ensemble models
   - Multi-model serving
   - Custom response formatting

PRE-BUILT CONTAINERS AVAILABLE:
- TensorFlow: us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.{version}
- PyTorch: us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.{version}
- Scikit-learn: us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.{version}
- XGBoost: us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.{version}
"""
```

### 7.2 Building Custom Serving Containers

#### Basic Custom Container Structure

```dockerfile
# Dockerfile for custom serving container
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts and code
COPY model /app/model
COPY predictor.py /app/
COPY preprocessing.py /app/

# Expose port for health check and predictions
EXPOSE 8080

# Set environment variables
ENV AIP_HTTP_PORT=8080
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/predict

# Run the prediction server
CMD ["python", "predictor.py"]
```

#### Custom Predictor Implementation

```python
# predictor.py - Custom prediction server
import os
import json
import logging
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class CustomPredictor:
    """Custom predictor with preprocessing and model serving."""
    
    def __init__(self, model_path):
        """Load model and preprocessors."""
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load model and preprocessing artifacts."""
        try:
            # Load model
            model_file = os.path.join(self.model_path, 'model.joblib')
            self.model = joblib.load(model_file)
            logger.info(f"Model loaded from {model_file}")
            
            # Load preprocessor
            preprocessor_file = os.path.join(self.model_path, 'preprocessor.joblib')
            if os.path.exists(preprocessor_file):
                self.preprocessor = joblib.load(preprocessor_file)
                logger.info(f"Preprocessor loaded from {preprocessor_file}")
            
            # Load feature names
            features_file = os.path.join(self.model_path, 'feature_names.json')
            if os.path.exists(features_file):
                with open(features_file, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info(f"Feature names loaded: {len(self.feature_names)} features")
            
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise
    
    def preprocess(self, instances):
        """Apply preprocessing to input instances."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(instances)
            
            # Validate features
            if self.feature_names:
                missing_features = set(self.feature_names) - set(df.columns)
                if missing_features:
                    raise ValueError(f"Missing features: {missing_features}")
                
                # Reorder columns
                df = df[self.feature_names]
            
            # Apply preprocessing
            if self.preprocessor:
                processed = self.preprocessor.transform(df)
            else:
                processed = df.values
            
            return processed
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise
    
    def predict(self, instances):
        """Make predictions on instances."""
        try:
            # Preprocess
            processed_data = self.preprocess(instances)
            
            # Predict
            predictions = self.model.predict(processed_data)
            
            # Get prediction probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_data)
                return predictions, probabilities
            
            return predictions, None
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

# Initialize predictor
MODEL_PATH = os.environ.get('AIP_STORAGE_URI', '/app/model')
predictor = CustomPredictor(MODEL_PATH)

@app.route(os.environ.get('AIP_HEALTH_ROUTE', '/health'), methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

@app.route(os.environ.get('AIP_PREDICT_ROUTE', '/predict'), methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        # Parse request
        request_json = request.get_json()
        
        if 'instances' not in request_json:
            return jsonify({'error': 'Missing instances in request'}), 400
        
        instances = request_json['instances']
        
        # Make predictions
        predictions, probabilities = predictor.predict(instances)
        
        # Format response
        response = {'predictions': predictions.tolist()}
        
        if probabilities is not None:
            response['probabilities'] = probabilities.tolist()
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction request failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('AIP_HTTP_PORT', 8080))
    app.run(host='0.0.0.0', port=port)
```

#### requirements.txt

```text
# requirements.txt
flask==2.3.0
scikit-learn==1.3.0
joblib==1.3.0
pandas==2.0.0
numpy==1.24.0
gunicorn==21.2.0
```

### 7.3 Advanced Custom Container Patterns

#### Multi-Model Serving Container

```python
# multi_model_predictor.py
import os
import json
import logging
from flask import Flask, request, jsonify
import joblib

logger = logging.getLogger(__name__)
app = Flask(__name__)

class MultiModelPredictor:
    """Serve multiple models from single container."""
    
    def __init__(self, models_path):
        """Load all models from directory."""
        self.models_path = models_path
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all model variants."""
        try:
            # Load model A (e.g., fraud detection)
            self.models['fraud_detector'] = joblib.load(
                os.path.join(self.models_path, 'fraud_model.joblib')
            )
            
            # Load model B (e.g., risk scoring)
            self.models['risk_scorer'] = joblib.load(
                os.path.join(self.models_path, 'risk_model.joblib')
            )
            
            logger.info(f"Loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def predict(self, model_name, instances):
        """Predict using specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        predictions = model.predict(instances)
        
        return predictions

# Initialize
predictor = MultiModelPredictor(os.environ.get('AIP_STORAGE_URI', '/app/models'))

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint with model selection."""
    try:
        request_json = request.get_json()
        
        model_name = request_json.get('model', 'fraud_detector')
        instances = request_json.get('instances')
        
        predictions = predictor.predict(model_name, instances)
        
        return jsonify({
            'model': model_name,
            'predictions': predictions.tolist()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('AIP_HTTP_PORT', 8080))
    app.run(host='0.0.0.0', port=port)
```

#### A/B Testing Container

```python
# ab_testing_predictor.py
import os
import random
import logging
from flask import Flask, request, jsonify
import joblib

logger = logging.getLogger(__name__)
app = Flask(__name__)

class ABTestingPredictor:
    """A/B testing with multiple model versions."""
    
    def __init__(self, models_path):
        """Load model versions."""
        self.models_path = models_path
        self.model_a = None
        self.model_b = None
        self.ab_ratio = 0.5  # 50/50 split
        
        self.load_models()
    
    def load_models(self):
        """Load both model versions."""
        try:
            self.model_a = joblib.load(
                os.path.join(self.models_path, 'model_a.joblib')
            )
            self.model_b = joblib.load(
                os.path.join(self.models_path, 'model_b.joblib')
            )
            
            logger.info("Loaded models A and B for A/B testing")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def predict(self, instances, user_id=None):
        """Predict with A/B testing logic."""
        # Determine which model to use
        if user_id:
            # Consistent assignment based on user_id
            use_model_a = hash(user_id) % 100 < (self.ab_ratio * 100)
        else:
            # Random assignment
            use_model_a = random.random() < self.ab_ratio
        
        # Select model
        model = self.model_a if use_model_a else self.model_b
        model_version = 'A' if use_model_a else 'B'
        
        # Predict
        predictions = model.predict(instances)
        
        return predictions, model_version

# Initialize
predictor = ABTestingPredictor(os.environ.get('AIP_STORAGE_URI', '/app/models'))

@app.route('/predict', methods=['POST'])
def predict():
    """A/B testing prediction endpoint."""
    try:
        request_json = request.get_json()
        
        instances = request_json.get('instances')
        user_id = request_json.get('user_id')
        
        predictions, model_version = predictor.predict(instances, user_id)
        
        return jsonify({
            'predictions': predictions.tolist(),
            'model_version': model_version
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('AIP_HTTP_PORT', 8080))
    app.run(host='0.0.0.0', port=port)
```

### 7.4 Building and Deploying Custom Containers

```python
# build_and_deploy.py
import os
import subprocess
from google.cloud import aiplatform

class CustomContainerDeployer:
    """Build and deploy custom serving containers."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        self.registry_host = f'{location}-docker.pkg.dev'
        
        aiplatform.init(project=project_id, location=location)
    
    def build_container(
        self,
        dockerfile_path,
        image_name,
        image_tag='latest'
    ):
        """Build Docker container."""
        
        full_image_name = f'{self.registry_host}/{self.project_id}/models/{image_name}:{image_tag}'
        
        print(f"Building container: {full_image_name}")
        
        # Build command
        build_cmd = [
            'docker', 'build',
            '-t', full_image_name,
            '-f', dockerfile_path,
            '.'
        ]
        
        try:
            subprocess.run(build_cmd, check=True)
            print(f"✓ Container built successfully")
            return full_image_name
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Container build failed: {e}")
            raise
    
    def push_container(self, image_name):
        """Push container to Artifact Registry."""
        
        print(f"Pushing container: {image_name}")
        
        # Configure Docker for Artifact Registry
        auth_cmd = [
            'gcloud', 'auth', 'configure-docker',
            self.registry_host,
            '--quiet'
        ]
        
        subprocess.run(auth_cmd, check=True)
        
        # Push command
        push_cmd = ['docker', 'push', image_name]
        
        try:
            subprocess.run(push_cmd, check=True)
            print(f"✓ Container pushed successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Container push failed: {e}")
            raise
    
    def register_model_with_custom_container(
        self,
        model_display_name,
        artifact_uri,
        container_image_uri,
        description='',
        health_route='/health',
        predict_route='/predict',
        ports=[8080]
    ):
        """Register model with custom container."""
        
        model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri=container_image_uri,
            serving_container_health_route=health_route,
            serving_container_predict_route=predict_route,
            serving_container_ports=ports,
            description=description
        )
        
        print(f"✓ Model registered with custom container")
        print(f"  Model: {model.display_name}")
        print(f"  Resource: {model.resource_name}")
        print(f"  Container: {container_image_uri}")
        
        return model
    
    def deploy_to_endpoint(
        self,
        model,
        endpoint_display_name,
        machine_type='n1-standard-4',
        min_replica_count=1,
        max_replica_count=3
    ):
        """Deploy model to endpoint."""
        
        # Create endpoint
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name
        )
        
        # Deploy model
        endpoint.deploy(
            model=model,
            deployed_model_display_name=f'{model.display_name}-deployment',
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            traffic_percentage=100
        )
        
        print(f"✓ Model deployed to endpoint")
        print(f"  Endpoint: {endpoint.display_name}")
        print(f"  Resource: {endpoint.resource_name}")
        
        return endpoint

# Complete deployment workflow
deployer = CustomContainerDeployer(project_id='my-project')

# 1. Build container
# image_uri = deployer.build_container(
#     dockerfile_path='./Dockerfile',
#     image_name='fraud-detection-custom',
#     image_tag='v1.0'
# )

# 2. Push to registry
# deployer.push_container(image_uri)

# 3. Register model
# model = deployer.register_model_with_custom_container(
#     model_display_name='fraud-detection-custom-v1',
#     artifact_uri='gs://my-bucket/models/fraud_detector',
#     container_image_uri=image_uri,
#     description='Custom fraud detection with preprocessing'
# )

# 4. Deploy to endpoint
# endpoint = deployer.deploy_to_endpoint(
#     model=model,
#     endpoint_display_name='fraud-detection-endpoint',
#     machine_type='n1-standard-4',
#     min_replica_count=1,
#     max_replica_count=5
# )
```

---

## 8. Custom Preprocessing Containers

### 8.1 Preprocessing Container Architecture

```python
"""
PREPROCESSING CONTAINER PATTERNS:

1. Pre-serving Preprocessing
   - Transform raw input before prediction
   - Feature engineering
   - Data validation
   - Format conversion

2. Batch Preprocessing
   - Large-scale data transformation
   - Feature extraction
   - Data cleaning
   - Format standardization

3. Streaming Preprocessing
   - Real-time transformations
   - Event-driven processing
   - Low-latency requirements
"""
```

### 8.2 Complex Preprocessing Container

```dockerfile
# Dockerfile for preprocessing container
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_preprocessing.txt .
RUN pip install --no-cache-dir -r requirements_preprocessing.txt

# Copy preprocessing code
COPY preprocessing/ /app/preprocessing/
COPY config/ /app/config/

# Expose port
EXPOSE 8080

# Run preprocessing server
CMD ["python", "-m", "preprocessing.server"]
```

#### Advanced Preprocessing Implementation

```python
# preprocessing/transformer.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import json
import logging

logger = logging.getLogger(__name__)

class FeatureTransformer:
    """Complex feature transformation pipeline."""
    
    def __init__(self, config_path):
        """Initialize transformer with configuration."""
        self.config = self.load_config(config_path)
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
        self.initialize_transformers()
    
    def load_config(self, config_path):
        """Load feature configuration."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def initialize_transformers(self):
        """Initialize transformation components."""
        try:
            # Load saved transformers
            for feature, settings in self.config['numerical_features'].items():
                if settings.get('scaler'):
                    scaler_path = settings['scaler']
                    self.scalers[feature] = joblib.load(scaler_path)
            
            for feature, settings in self.config['categorical_features'].items():
                if settings.get('encoder'):
                    encoder_path = settings['encoder']
                    self.encoders[feature] = joblib.load(encoder_path)
            
            logger.info("Transformers initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing transformers: {e}")
            raise
    
    def transform_numerical(self, df, feature_name):
        """Transform numerical feature."""
        try:
            settings = self.config['numerical_features'].get(feature_name, {})
            
            # Handle missing values
            if settings.get('impute'):
                if feature_name not in self.imputers:
                    self.imputers[feature_name] = SimpleImputer(
                        strategy=settings.get('impute_strategy', 'mean')
                    )
                df[feature_name] = self.imputers[feature_name].fit_transform(
                    df[[feature_name]]
                )
            
            # Apply transformations
            if settings.get('log_transform'):
                df[feature_name] = np.log1p(df[feature_name])
            
            if settings.get('clip'):
                lower, upper = settings['clip']
                df[feature_name] = df[feature_name].clip(lower, upper)
            
            # Scale
            if feature_name in self.scalers:
                df[feature_name] = self.scalers[feature_name].transform(
                    df[[feature_name]]
                )
            
            return df
            
        except Exception as e:
            logger.error(f"Error transforming {feature_name}: {e}")
            raise
    
    def transform_categorical(self, df, feature_name):
        """Transform categorical feature."""
        try:
            settings = self.config['categorical_features'].get(feature_name, {})
            
            # Handle missing
            if settings.get('fill_missing'):
                df[feature_name].fillna(
                    settings['fill_missing'],
                    inplace=True
                )
            
            # Encode
            if feature_name in self.encoders:
                # Handle unseen categories
                encoder = self.encoders[feature_name]
                known_classes = set(encoder.classes_)
                
                def safe_encode(val):
                    if val in known_classes:
                        return encoder.transform([val])[0]
                    else:
                        return -1  # Unknown category
                
                df[feature_name] = df[feature_name].apply(safe_encode)
            
            return df
            
        except Exception as e:
            logger.error(f"Error encoding {feature_name}: {e}")
            raise
    
    def create_features(self, df):
        """Create derived features."""
        try:
            feature_defs = self.config.get('derived_features', {})
            
            for feature_name, definition in feature_defs.items():
                if definition['type'] == 'ratio':
                    numerator = definition['numerator']
                    denominator = definition['denominator']
                    df[feature_name] = df[numerator] / (df[denominator] + 1e-8)
                
                elif definition['type'] == 'difference':
                    df[feature_name] = df[definition['feature1']] - df[definition['feature2']]
                
                elif definition['type'] == 'product':
                    df[feature_name] = df[definition['feature1']] * df[definition['feature2']]
                
                elif definition['type'] == 'polynomial':
                    df[feature_name] = df[definition['feature']] ** definition['degree']
                
                elif definition['type'] == 'interaction':
                    features = definition['features']
                    df[feature_name] = df[features].prod(axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating derived features: {e}")
            raise
    
    def validate(self, df):
        """Validate input data."""
        errors = []
        
        # Check required features
        required_features = set(self.config['numerical_features'].keys()) | \
                          set(self.config['categorical_features'].keys())
        
        missing_features = required_features - set(df.columns)
        if missing_features:
            errors.append(f"Missing features: {missing_features}")
        
        # Check data types
        for feature, settings in self.config['numerical_features'].items():
            if feature in df.columns:
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    errors.append(f"Feature {feature} must be numeric")
        
        # Check value ranges
        for feature, settings in self.config.get('validation_rules', {}).items():
            if feature in df.columns:
                if 'min' in settings:
                    if df[feature].min() < settings['min']:
                        errors.append(f"{feature} has values below minimum {settings['min']}")
                
                if 'max' in settings:
                    if df[feature].max() > settings['max']:
                        errors.append(f"{feature} has values above maximum {settings['max']}")
        
        if errors:
            raise ValueError(f"Validation errors: {'; '.join(errors)}")
    
    def transform(self, data):
        """Apply complete transformation pipeline."""
        try:
            # Convert to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # Validate
            self.validate(df)
            
            # Transform numerical features
            for feature in self.config['numerical_features'].keys():
                if feature in df.columns:
                    df = self.transform_numerical(df, feature)
            
            # Transform categorical features
            for feature in self.config['categorical_features'].keys():
                if feature in df.columns:
                    df = self.transform_categorical(df, feature)
            
            # Create derived features
            df = self.create_features(df)
            
            # Select final features
            final_features = self.config.get('final_features', df.columns.tolist())
            df = df[final_features]
            
            return df
            
        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            raise
```

#### Preprocessing Server

```python
# preprocessing/server.py
import os
import logging
from flask import Flask, request, jsonify
from preprocessing.transformer import FeatureTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize transformer
CONFIG_PATH = os.environ.get('PREPROCESSING_CONFIG', '/app/config/features.json')
transformer = FeatureTransformer(CONFIG_PATH)

@app.route('/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({'status': 'healthy'}), 200

@app.route('/preprocess', methods=['POST'])
def preprocess():
    """Preprocessing endpoint."""
    try:
        request_json = request.get_json()
        
        if 'instances' not in request_json:
            return jsonify({'error': 'Missing instances'}), 400
        
        instances = request_json['instances']
        
        # Transform
        transformed = transformer.transform(instances)
        
        # Convert to list of dicts
        result = transformed.to_dict('records')
        
        return jsonify({'processed_instances': result}), 200
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/validate', methods=['POST'])
def validate():
    """Validation endpoint."""
    try:
        request_json = request.get_json()
        instances = request_json.get('instances', [])
        
        # Validate
        transformer.validate(pd.DataFrame(instances))
        
        return jsonify({'valid': True}), 200
        
    except ValueError as e:
        return jsonify({'valid': False, 'errors': str(e)}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
```

#### Feature Configuration

```json
# config/features.json
{
  "numerical_features": {
    "age": {
      "scaler": "gs://my-bucket/preprocessors/age_scaler.joblib",
      "impute": true,
      "impute_strategy": "median",
      "clip": [18, 100]
    },
    "income": {
      "scaler": "gs://my-bucket/preprocessors/income_scaler.joblib",
      "log_transform": true,
      "impute": true,
      "impute_strategy": "median"
    },
    "credit_score": {
      "scaler": "gs://my-bucket/preprocessors/credit_scaler.joblib",
      "clip": [300, 850]
    }
  },
  "categorical_features": {
    "occupation": {
      "encoder": "gs://my-bucket/preprocessors/occupation_encoder.joblib",
      "fill_missing": "unknown"
    },
    "education": {
      "encoder": "gs://my-bucket/preprocessors/education_encoder.joblib",
      "fill_missing": "unknown"
    },
    "marital_status": {
      "encoder": "gs://my-bucket/preprocessors/marital_encoder.joblib",
      "fill_missing": "unknown"
    }
  },
  "derived_features": {
    "income_per_age": {
      "type": "ratio",
      "numerator": "income",
      "denominator": "age"
    },
    "income_age_interaction": {
      "type": "interaction",
      "features": ["income", "age"]
    },
    "credit_score_squared": {
      "type": "polynomial",
      "feature": "credit_score",
      "degree": 2
    }
  },
  "validation_rules": {
    "age": {
      "min": 18,
      "max": 100
    },
    "income": {
      "min": 0,
      "max": 10000000
    },
    "credit_score": {
      "min": 300,
      "max": 850
    }
  },
  "final_features": [
    "age",
    "income",
    "credit_score",
    "occupation",
    "education",
    "marital_status",
    "income_per_age",
    "income_age_interaction",
    "credit_score_squared"
  ]
}
```

### 8.3 Batch Preprocessing Container

```python
# batch_preprocessing.py
import os
import logging
from google.cloud import storage, bigquery
import pandas as pd
from preprocessing.transformer import FeatureTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchPreprocessor:
    """Batch preprocessing for large datasets."""
    
    def __init__(self, project_id, config_path):
        self.project_id = project_id
        self.storage_client = storage.Client(project=project_id)
        self.bq_client = bigquery.Client(project=project_id)
        self.transformer = FeatureTransformer(config_path)
    
    def preprocess_from_bigquery(
        self,
        source_table,
        destination_table,
        batch_size=10000
    ):
        """Preprocess data from BigQuery."""
        try:
            logger.info(f"Starting batch preprocessing from {source_table}")
            
            # Query data in batches
            query = f"""
            SELECT *
            FROM `{source_table}`
            """
            
            query_job = self.bq_client.query(query)
            
            # Process in batches
            total_rows = 0
            processed_rows = []
            
            for row in query_job:
                row_dict = dict(row)
                processed_rows.append(row_dict)
                
                if len(processed_rows) >= batch_size:
                    # Transform batch
                    df = pd.DataFrame(processed_rows)
                    transformed = self.transformer.transform(df)
                    
                    # Write to destination
                    self._write_to_bigquery(transformed, destination_table)
                    
                    total_rows += len(processed_rows)
                    logger.info(f"Processed {total_rows} rows")
                    
                    processed_rows = []
            
            # Process remaining rows
            if processed_rows:
                df = pd.DataFrame(processed_rows)
                transformed = self.transformer.transform(df)
                self._write_to_bigquery(transformed, destination_table)
                total_rows += len(processed_rows)
            
            logger.info(f"✓ Batch preprocessing complete: {total_rows} rows")
            
        except Exception as e:
            logger.error(f"Batch preprocessing failed: {e}")
            raise
    
    def preprocess_from_gcs(
        self,
        source_bucket,
        source_prefix,
        destination_bucket,
        destination_prefix
    ):
        """Preprocess CSV files from Cloud Storage."""
        try:
            logger.info(f"Processing files from gs://{source_bucket}/{source_prefix}")
            
            bucket = self.storage_client.bucket(source_bucket)
            blobs = bucket.list_blobs(prefix=source_prefix)
            
            for blob in blobs:
                if blob.name.endswith('.csv'):
                    logger.info(f"Processing {blob.name}")
                    
                    # Download and read
                    content = blob.download_as_string()
                    df = pd.read_csv(pd.io.common.BytesIO(content))
                    
                    # Transform
                    transformed = self.transformer.transform(df)
                    
                    # Upload processed file
                    output_name = blob.name.replace(source_prefix, destination_prefix)
                    output_blob = bucket.blob(output_name)
                    output_blob.upload_from_string(
                        transformed.to_csv(index=False),
                        content_type='text/csv'
                    )
                    
                    logger.info(f"✓ Processed {blob.name} -> {output_name}")
            
            logger.info("✓ All files processed")
            
        except Exception as e:
            logger.error(f"GCS preprocessing failed: {e}")
            raise
    
    def _write_to_bigquery(self, df, table_id):
        """Write DataFrame to BigQuery."""
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            schema_update_options=[
                bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
            ]
        )
        
        job = self.bq_client.load_table_from_dataframe(
            df, table_id, job_config=job_config
        )
        job.result()

# Usage
if __name__ == '__main__':
    preprocessor = BatchPreprocessor(
        project_id='my-project',
        config_path='/app/config/features.json'
    )
    
    # Preprocess from BigQuery
    preprocessor.preprocess_from_bigquery(
        source_table='my-project.dataset.raw_data',
        destination_table='my-project.dataset.processed_data',
        batch_size=10000
    )
```

### 8.4 Testing Custom Containers

```python
# test_custom_container.py
import requests
import json
import time

class ContainerTester:
    """Test custom serving and preprocessing containers."""
    
    def __init__(self, container_url):
        self.container_url = container_url
    
    def test_health(self):
        """Test health endpoint."""
        try:
            response = requests.get(f'{self.container_url}/health')
            
            if response.status_code == 200:
                print("✓ Health check passed")
                return True
            else:
                print(f"✗ Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Health check error: {e}")
            return False
    
    def test_prediction(self, instances):
        """Test prediction endpoint."""
        try:
            payload = {'instances': instances}
            
            start_time = time.time()
            response = requests.post(
                f'{self.container_url}/predict',
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Prediction successful (latency: {latency:.2f}ms)")
                print(f"  Response: {result}")
                return True, result
            else:
                print(f"✗ Prediction failed: {response.status_code}")
                print(f"  Error: {response.text}")
                return False, None
                
        except Exception as e:
            print(f"✗ Prediction error: {e}")
            return False, None
    
    def test_preprocessing(self, instances):
        """Test preprocessing endpoint."""
        try:
            payload = {'instances': instances}
            
            response = requests.post(
                f'{self.container_url}/preprocess',
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✓ Preprocessing successful")
                print(f"  Processed: {len(result['processed_instances'])} instances")
                return True, result
            else:
                print(f"✗ Preprocessing failed: {response.status_code}")
                return False, None
                
        except Exception as e:
            print(f"✗ Preprocessing error: {e}")
            return False, None
    
    def load_test(self, instances, num_requests=100):
        """Perform load testing."""
        print(f"\nRunning load test with {num_requests} requests...")
        
        latencies = []
        successes = 0
        failures = 0
        
        start_time = time.time()
        
        for i in range(num_requests):
            try:
                request_start = time.time()
                response = requests.post(
                    f'{self.container_url}/predict',
                    json={'instances': instances},
                    timeout=10
                )
                latency = (time.time() - request_start) * 1000
                
                if response.status_code == 200:
                    successes += 1
                    latencies.append(latency)
                else:
                    failures += 1
                    
            except Exception as e:
                failures += 1
        
        total_time = time.time() - start_time
        
        print(f"\n=== Load Test Results ===")
        print(f"Total requests: {num_requests}")
        print(f"Successes: {successes}")
        print(f"Failures: {failures}")
        print(f"Success rate: {successes/num_requests*100:.1f}%")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {num_requests/total_time:.2f} req/s")
        
        if latencies:
            print(f"\nLatency Statistics:")
            print(f"  Mean: {np.mean(latencies):.2f}ms")
            print(f"  Median: {np.median(latencies):.2f}ms")
            print(f"  P95: {np.percentile(latencies, 95):.2f}ms")
            print(f"  P99: {np.percentile(latencies, 99):.2f}ms")
            print(f"  Min: {np.min(latencies):.2f}ms")
            print(f"  Max: {np.max(latencies):.2f}ms")

# Example usage
import numpy as np

tester = ContainerTester(container_url='http://localhost:8080')

# Test health
tester.test_health()

# Test prediction
test_instances = [
    {'age': 35, 'income': 50000, 'credit_score': 720},
    {'age': 42, 'income': 75000, 'credit_score': 680}
]

success, result = tester.test_prediction(test_instances)

# Load test
tester.load_test(test_instances, num_requests=100)
```

---

## 9. Quick Reference Checklist

### Setup
- [ ] Enable Vertex AI API
- [ ] Configure IAM permissions
- [ ] Set up artifact storage in Cloud Storage
- [ ] Install SDK: `pip install google-cloud-aiplatform`
- [ ] Initialize project and location

### Registration
- [ ] Register models with descriptive names
- [ ] Add comprehensive metadata
- [ ] Specify serving container
- [ ] Add relevant labels
- [ ] Document model purpose

### Versioning
- [ ] Use semantic versioning strategy
- [ ] Create version aliases (dev, staging, production)
- [ ] Set appropriate default version
- [ ] Document version changes
- [ ] Track version lineage

### Governance
- [ ] Implement approval workflows
- [ ] Track model provenance
- [ ] Monitor model lifecycle
- [ ] Regular audit and cleanup
- [ ] Document governance policies

### Custom Containers
- [ ] Build Dockerfile with proper base image
- [ ] Implement health check endpoint
- [ ] Implement prediction endpoint
- [ ] Add comprehensive error handling
- [ ] Test container locally
- [ ] Push to Artifact Registry
- [ ] Verify container security
- [ ] Document container requirements

### Preprocessing
- [ ] Define preprocessing configuration
- [ ] Implement feature transformations
- [ ] Add data validation
- [ ] Test preprocessing logic
- [ ] Version preprocessing artifacts
- [ ] Document preprocessing steps
- [ ] Monitor preprocessing performance

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
