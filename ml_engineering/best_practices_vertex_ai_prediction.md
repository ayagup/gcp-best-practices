# Vertex AI Prediction Best Practices

*Last Updated: January 4, 2026*

## Overview

Vertex AI Prediction provides scalable model serving with support for online (real-time) and batch predictions, auto-scaling, model versioning, traffic splitting for A/B testing, GPU acceleration, and private endpoints with VPC Service Controls.

---

## 1. Online Prediction (Real-Time)

### Deploy Model to Endpoint

```python
from google.cloud import aiplatform

class OnlinePredictionManager:
    """Manage online prediction endpoints."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        
        aiplatform.init(project=project_id, location=location)
    
    def create_endpoint(self, display_name, description=''):
        """Create prediction endpoint."""
        
        endpoint = aiplatform.Endpoint.create(
            display_name=display_name,
            description=description
        )
        
        print(f"✓ Created endpoint: {display_name}")
        print(f"  Resource name: {endpoint.resource_name}")
        print(f"  Endpoint ID: {endpoint.name}")
        
        return endpoint
    
    def deploy_model_to_endpoint(
        self,
        endpoint_id,
        model_id,
        deployed_model_display_name,
        machine_type='n1-standard-4',
        min_replica_count=1,
        max_replica_count=3,
        accelerator_type=None,
        accelerator_count=0
    ):
        """Deploy model to endpoint."""
        
        endpoint = aiplatform.Endpoint(endpoint_id)
        model = aiplatform.Model(model_id)
        
        endpoint.deploy(
            model=model,
            deployed_model_display_name=deployed_model_display_name,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            traffic_percentage=100,
            sync=True
        )
        
        print(f"✓ Deployed model to endpoint")
        print(f"  Model: {model.display_name}")
        print(f"  Endpoint: {endpoint.display_name}")
        print(f"  Machine type: {machine_type}")
        print(f"  Replicas: {min_replica_count}-{max_replica_count}")
        
        return endpoint
    
    def predict(self, endpoint_id, instances):
        """Make online prediction."""
        
        endpoint = aiplatform.Endpoint(endpoint_id)
        
        predictions = endpoint.predict(instances=instances)
        
        print(f"✓ Received {len(predictions.predictions)} predictions")
        
        return predictions.predictions
    
    def predict_with_explanation(self, endpoint_id, instances):
        """Get predictions with explanations."""
        
        endpoint = aiplatform.Endpoint(endpoint_id)
        
        response = endpoint.explain(instances=instances)
        
        print(f"✓ Received predictions with explanations")
        
        return {
            'predictions': response.predictions,
            'explanations': response.explanations
        }
    
    def batch_predict_online(self, endpoint_id, instances_batch):
        """Send batch of instances to online endpoint."""
        
        endpoint = aiplatform.Endpoint(endpoint_id)
        
        all_predictions = []
        
        # Process in batches
        batch_size = 100
        
        for i in range(0, len(instances_batch), batch_size):
            batch = instances_batch[i:i+batch_size]
            predictions = endpoint.predict(instances=batch)
            all_predictions.extend(predictions.predictions)
        
        print(f"✓ Processed {len(all_predictions)} predictions")
        
        return all_predictions

# Example usage
prediction_manager = OnlinePredictionManager(project_id='my-project')

# Create endpoint
# endpoint = prediction_manager.create_endpoint(
#     display_name='fraud-detection-endpoint',
#     description='Real-time fraud detection predictions'
# )

# Deploy model
# endpoint = prediction_manager.deploy_model_to_endpoint(
#     endpoint_id='projects/.../endpoints/123',
#     model_id='projects/.../models/456',
#     deployed_model_display_name='fraud-model-v1',
#     machine_type='n1-standard-4',
#     min_replica_count=2,
#     max_replica_count=10
# )

# Make prediction
# instances = [
#     {'transaction_amount': 125.50, 'user_age': 35, 'merchant_category': 'electronics'},
#     {'transaction_amount': 45.00, 'user_age': 28, 'merchant_category': 'grocery'}
# ]
# predictions = prediction_manager.predict('projects/.../endpoints/123', instances)
```

---

## 2. Batch Prediction

### Run Batch Predictions

```python
class BatchPredictionManager:
    """Manage batch prediction jobs."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def run_batch_prediction(
        self,
        job_display_name,
        model_id,
        input_uri,
        output_uri_prefix,
        instances_format='jsonl',
        predictions_format='jsonl',
        machine_type='n1-standard-4',
        max_replica_count=10
    ):
        """Run batch prediction job."""
        
        model = aiplatform.Model(model_id)
        
        batch_prediction_job = model.batch_predict(
            job_display_name=job_display_name,
            gcs_source=input_uri,
            gcs_destination_prefix=output_uri_prefix,
            instances_format=instances_format,
            predictions_format=predictions_format,
            machine_type=machine_type,
            max_replica_count=max_replica_count,
            sync=True
        )
        
        print(f"✓ Batch prediction completed")
        print(f"  Job: {batch_prediction_job.display_name}")
        print(f"  State: {batch_prediction_job.state}")
        print(f"  Output: {output_uri_prefix}")
        
        return batch_prediction_job
    
    def run_batch_prediction_from_bigquery(
        self,
        job_display_name,
        model_id,
        bigquery_source_uri,
        bigquery_destination_uri,
        machine_type='n1-standard-4'
    ):
        """Run batch prediction with BigQuery source."""
        
        model = aiplatform.Model(model_id)
        
        batch_prediction_job = model.batch_predict(
            job_display_name=job_display_name,
            bigquery_source=bigquery_source_uri,
            bigquery_destination_prefix=bigquery_destination_uri,
            machine_type=machine_type,
            sync=True
        )
        
        print(f"✓ Batch prediction from BigQuery completed")
        print(f"  Source: {bigquery_source_uri}")
        print(f"  Destination: {bigquery_destination_uri}")
        
        return batch_prediction_job
    
    def schedule_batch_predictions(
        self,
        model_id,
        input_uri_pattern,
        output_uri_prefix,
        schedule='0 2 * * *'  # Daily at 2 AM
    ):
        """Schedule recurring batch predictions."""
        
        from google.cloud import scheduler_v1
        
        client = scheduler_v1.CloudSchedulerClient()
        
        parent = f"projects/{self.project_id}/locations/{self.location}"
        
        # Create scheduled job
        job = {
            'name': f"{parent}/jobs/batch-prediction-{model_id}",
            'schedule': schedule,
            'time_zone': 'UTC',
            'http_target': {
                'uri': f'https://{self.location}-aiplatform.googleapis.com/v1/...',
                'http_method': 'POST',
                'headers': {
                    'Content-Type': 'application/json'
                }
            }
        }
        
        response = client.create_job(parent=parent, job=job)
        
        print(f"✓ Scheduled batch predictions")
        print(f"  Schedule: {schedule}")
        print(f"  Job: {response.name}")
        
        return response
    
    def monitor_batch_job(self, job_id):
        """Monitor batch prediction job."""
        
        job = aiplatform.BatchPredictionJob(job_id)
        
        print(f"\nBatch Prediction Job: {job.display_name}")
        print(f"  State: {job.state}")
        print(f"  Created: {job.create_time}")
        print(f"  Started: {job.start_time}")
        print(f"  Ended: {job.end_time}")
        
        if job.error:
            print(f"  Error: {job.error.message}")
        
        return job

# Example usage
batch_manager = BatchPredictionManager(project_id='my-project')

# Run batch prediction from GCS
# job = batch_manager.run_batch_prediction(
#     job_display_name='daily-fraud-predictions',
#     model_id='projects/.../models/456',
#     input_uri='gs://my-bucket/input/transactions.jsonl',
#     output_uri_prefix='gs://my-bucket/output/',
#     machine_type='n1-standard-8',
#     max_replica_count=20
# )

# Run from BigQuery
# job = batch_manager.run_batch_prediction_from_bigquery(
#     job_display_name='bigquery-batch-prediction',
#     model_id='projects/.../models/456',
#     bigquery_source_uri='bq://my-project.dataset.input_table',
#     bigquery_destination_uri='bq://my-project.dataset.predictions',
#     machine_type='n1-standard-8'
# )
```

---

## 3. Auto-Scaling Configuration

### Configure Auto-Scaling

```python
class AutoScalingManager:
    """Manage endpoint auto-scaling."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def configure_autoscaling(
        self,
        endpoint_id,
        deployed_model_id,
        min_replica_count=1,
        max_replica_count=10,
        target_cpu_utilization=60
    ):
        """Configure auto-scaling for deployed model."""
        
        from google.cloud.aiplatform_v1 import EndpointServiceClient
        from google.cloud.aiplatform_v1.types import DeployedModel
        
        client = EndpointServiceClient()
        
        # Get endpoint
        endpoint = aiplatform.Endpoint(endpoint_id)
        
        # Update deployed model with auto-scaling
        deployed_model = DeployedModel(
            id=deployed_model_id,
            automatic_resources={
                'min_replica_count': min_replica_count,
                'max_replica_count': max_replica_count
            }
        )
        
        print(f"✓ Configured auto-scaling")
        print(f"  Min replicas: {min_replica_count}")
        print(f"  Max replicas: {max_replica_count}")
        print(f"  Target CPU: {target_cpu_utilization}%")
        
        return endpoint
    
    def monitor_scaling_metrics(self, endpoint_id):
        """Monitor scaling metrics."""
        
        from google.cloud import monitoring_v3
        
        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{self.project_id}"
        
        # Query replica count metric
        query = f"""
        fetch aiplatform.googleapis.com/Endpoint
        | metric 'aiplatform.googleapis.com/endpoint/replica_count'
        | filter resource.endpoint_id == '{endpoint_id}'
        | group_by 1m, [value_replica_count_mean: mean(value.replica_count)]
        | within 1h
        """
        
        print(f"Monitoring endpoint: {endpoint_id}")
        print(f"  Query: {query}")
        
        # Note: Actual implementation would use monitoring API
        
        return query
    
    def set_scaling_policy(
        self,
        endpoint_id,
        policy='balanced'
    ):
        """Set scaling policy.
        
        Args:
            policy: 'conservative', 'balanced', or 'aggressive'
        """
        
        policies = {
            'conservative': {
                'min_replicas': 2,
                'max_replicas': 5,
                'target_cpu': 70
            },
            'balanced': {
                'min_replicas': 1,
                'max_replicas': 10,
                'target_cpu': 60
            },
            'aggressive': {
                'min_replicas': 1,
                'max_replicas': 50,
                'target_cpu': 50
            }
        }
        
        config = policies.get(policy, policies['balanced'])
        
        print(f"✓ Applied {policy} scaling policy")
        print(f"  Configuration: {config}")
        
        return config

# Example usage
autoscaling = AutoScalingManager(project_id='my-project')

# Configure auto-scaling
# autoscaling.configure_autoscaling(
#     endpoint_id='projects/.../endpoints/123',
#     deployed_model_id='456',
#     min_replica_count=2,
#     max_replica_count=20,
#     target_cpu_utilization=60
# )

# Set scaling policy
# config = autoscaling.set_scaling_policy(
#     endpoint_id='projects/.../endpoints/123',
#     policy='aggressive'
# )
```

---

## 4. Traffic Splitting & A/B Testing

### Manage Model Versions with Traffic Splitting

```python
class TrafficSplittingManager:
    """Manage traffic splitting for A/B testing."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def deploy_multiple_models(
        self,
        endpoint_id,
        model_configs
    ):
        """Deploy multiple models with traffic split.
        
        Args:
            model_configs: List of dicts with model_id, traffic_percentage, display_name
        """
        
        endpoint = aiplatform.Endpoint(endpoint_id)
        
        for config in model_configs:
            model = aiplatform.Model(config['model_id'])
            
            endpoint.deploy(
                model=model,
                deployed_model_display_name=config['display_name'],
                machine_type='n1-standard-4',
                min_replica_count=1,
                max_replica_count=5,
                traffic_percentage=config['traffic_percentage'],
                sync=True
            )
            
            print(f"✓ Deployed {config['display_name']}: {config['traffic_percentage']}% traffic")
    
    def update_traffic_split(
        self,
        endpoint_id,
        traffic_split
    ):
        """Update traffic split across deployed models.
        
        Args:
            traffic_split: Dict mapping deployed_model_id to traffic percentage
                Example: {'model-v1': 80, 'model-v2': 20}
        """
        
        endpoint = aiplatform.Endpoint(endpoint_id)
        
        # Validate percentages sum to 100
        total = sum(traffic_split.values())
        
        if total != 100:
            raise ValueError(f"Traffic percentages must sum to 100, got {total}")
        
        # Update traffic split
        endpoint.update(traffic_split=traffic_split)
        
        print(f"✓ Updated traffic split:")
        for model_id, percentage in traffic_split.items():
            print(f"  {model_id}: {percentage}%")
        
        return endpoint
    
    def canary_deployment(
        self,
        endpoint_id,
        new_model_id,
        canary_percentage=10
    ):
        """Deploy new model with canary traffic."""
        
        endpoint = aiplatform.Endpoint(endpoint_id)
        model = aiplatform.Model(new_model_id)
        
        # Deploy with small traffic percentage
        endpoint.deploy(
            model=model,
            deployed_model_display_name='canary-model',
            machine_type='n1-standard-4',
            min_replica_count=1,
            max_replica_count=3,
            traffic_percentage=canary_percentage,
            sync=True
        )
        
        print(f"✓ Canary deployment complete")
        print(f"  Canary traffic: {canary_percentage}%")
        
        return endpoint
    
    def gradual_rollout(
        self,
        endpoint_id,
        old_model_id,
        new_model_id,
        steps=5,
        wait_minutes=30
    ):
        """Gradually roll out new model."""
        
        import time
        
        endpoint = aiplatform.Endpoint(endpoint_id)
        
        # Calculate traffic increments
        increment = 100 // steps
        
        for i in range(1, steps + 1):
            new_traffic = increment * i
            old_traffic = 100 - new_traffic
            
            traffic_split = {
                old_model_id: old_traffic,
                new_model_id: new_traffic
            }
            
            endpoint.update(traffic_split=traffic_split)
            
            print(f"✓ Step {i}/{steps}:")
            print(f"  Old model: {old_traffic}%")
            print(f"  New model: {new_traffic}%")
            
            if i < steps:
                print(f"  Waiting {wait_minutes} minutes...")
                time.sleep(wait_minutes * 60)
        
        print(f"✓ Rollout complete: 100% traffic to new model")
        
        return endpoint

# Example usage
traffic_manager = TrafficSplittingManager(project_id='my-project')

# Deploy multiple models with traffic split
# model_configs = [
#     {
#         'model_id': 'projects/.../models/v1',
#         'display_name': 'model-v1',
#         'traffic_percentage': 80
#     },
#     {
#         'model_id': 'projects/.../models/v2',
#         'display_name': 'model-v2',
#         'traffic_percentage': 20
#     }
# ]
# traffic_manager.deploy_multiple_models('projects/.../endpoints/123', model_configs)

# Update traffic split
# traffic_manager.update_traffic_split(
#     endpoint_id='projects/.../endpoints/123',
#     traffic_split={'model-v1-id': 50, 'model-v2-id': 50}
# )

# Canary deployment
# traffic_manager.canary_deployment(
#     endpoint_id='projects/.../endpoints/123',
#     new_model_id='projects/.../models/v3',
#     canary_percentage=10
# )
```

---

## 5. Private Endpoints & VPC Service Controls

### Deploy with Private Endpoints

```python
class PrivateEndpointManager:
    """Manage private prediction endpoints."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def create_private_endpoint(
        self,
        display_name,
        network,
        description=''
    ):
        """Create private endpoint within VPC."""
        
        endpoint = aiplatform.Endpoint.create(
            display_name=display_name,
            description=description,
            network=network
        )
        
        print(f"✓ Created private endpoint: {display_name}")
        print(f"  Network: {network}")
        print(f"  Resource: {endpoint.resource_name}")
        
        return endpoint
    
    def deploy_with_private_service_connect(
        self,
        endpoint_id,
        model_id,
        enable_private_service_connect=True
    ):
        """Deploy model with Private Service Connect."""
        
        endpoint = aiplatform.Endpoint(endpoint_id)
        model = aiplatform.Model(model_id)
        
        endpoint.deploy(
            model=model,
            deployed_model_display_name='private-model',
            machine_type='n1-standard-4',
            min_replica_count=2,
            max_replica_count=5,
            enable_access_logging=True,
            sync=True
        )
        
        print(f"✓ Deployed with private endpoint")
        print(f"  Private Service Connect: {enable_private_service_connect}")
        
        return endpoint
    
    def configure_vpc_sc(
        self,
        access_policy_name,
        perimeter_name,
        restricted_services
    ):
        """Configure VPC Service Controls."""
        
        from google.cloud import accesscontextmanager_v1
        
        client = accesscontextmanager_v1.AccessContextManagerClient()
        
        print(f"✓ VPC Service Controls configured")
        print(f"  Policy: {access_policy_name}")
        print(f"  Perimeter: {perimeter_name}")
        print(f"  Restricted services: {len(restricted_services)}")
        
        # Note: Actual implementation would configure VPC-SC
        
        return perimeter_name

# Example usage
private_manager = PrivateEndpointManager(project_id='my-project')

# Create private endpoint
# endpoint = private_manager.create_private_endpoint(
#     display_name='private-fraud-detection',
#     network='projects/my-project/global/networks/my-vpc',
#     description='Private endpoint for internal predictions'
# )
```

---

## 6. GPU-Accelerated Predictions

```python
class GPUPredictionManager:
    """Manage GPU-accelerated predictions."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def deploy_with_gpu(
        self,
        endpoint_id,
        model_id,
        gpu_type='NVIDIA_TESLA_T4',
        gpu_count=1,
        machine_type='n1-standard-8'
    ):
        """Deploy model with GPU acceleration."""
        
        endpoint = aiplatform.Endpoint(endpoint_id)
        model = aiplatform.Model(model_id)
        
        endpoint.deploy(
            model=model,
            deployed_model_display_name='gpu-model',
            machine_type=machine_type,
            accelerator_type=gpu_type,
            accelerator_count=gpu_count,
            min_replica_count=1,
            max_replica_count=5,
            sync=True
        )
        
        print(f"✓ Deployed with GPU acceleration")
        print(f"  GPU: {gpu_count}x {gpu_type}")
        print(f"  Machine: {machine_type}")
        
        return endpoint

# Example usage
gpu_manager = GPUPredictionManager(project_id='my-project')

# Deploy with GPU
# endpoint = gpu_manager.deploy_with_gpu(
#     endpoint_id='projects/.../endpoints/123',
#     model_id='projects/.../models/456',
#     gpu_type='NVIDIA_TESLA_V100',
#     gpu_count=1,
#     machine_type='n1-standard-8'
# )
```

---

## 7. Quick Reference Checklist

### Online Prediction
- [ ] Create endpoint
- [ ] Deploy model with appropriate resources
- [ ] Configure auto-scaling
- [ ] Test predictions
- [ ] Monitor latency and throughput
- [ ] Set up request/response logging

### Batch Prediction
- [ ] Prepare input data in GCS or BigQuery
- [ ] Run batch prediction job
- [ ] Monitor job progress
- [ ] Validate output
- [ ] Schedule recurring jobs
- [ ] Optimize batch size and resources

### Traffic Management
- [ ] Deploy multiple model versions
- [ ] Configure traffic split
- [ ] Implement canary deployment
- [ ] Monitor A/B test metrics
- [ ] Gradual rollout strategy
- [ ] Rollback plan

### Security
- [ ] Use private endpoints for internal traffic
- [ ] Configure VPC Service Controls
- [ ] Enable access logging
- [ ] Set up IAM permissions
- [ ] Use service accounts
- [ ] Implement authentication

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
