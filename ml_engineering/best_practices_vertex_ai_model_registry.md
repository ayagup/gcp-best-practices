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

## 7. Quick Reference Checklist

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

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
