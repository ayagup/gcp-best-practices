# Best Practices for AI Infrastructure on Google Cloud

## Overview

AI Infrastructure on Google Cloud encompasses the compute, storage, networking, and orchestration resources needed to build, train, and deploy machine learning models at scale. This guide covers resource management, scaling strategies, cost optimization, and infrastructure best practices for AI workloads.

## 1. Compute Resource Management

### 1.1 Resource Selection and Allocation

```python
from google.cloud import aiplatform
from google.cloud import compute_v1
from typing import Dict, Any, List, Optional
import json

class AIInfrastructureManager:
    """Manager for AI infrastructure resources."""
    
    def __init__(
        self,
        project_id: str,
        location: str = 'us-central1'
    ):
        """
        Initialize AI Infrastructure Manager.
        
        Args:
            project_id: GCP project ID
            location: GCP region
        """
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def calculate_compute_requirements(
        self,
        model_params: int,
        dataset_size_gb: float,
        training_hours: float,
        batch_size: int
    ) -> Dict[str, Any]:
        """
        Calculate compute requirements for training.
        
        Args:
            model_params: Number of model parameters
            dataset_size_gb: Dataset size in GB
            training_hours: Estimated training time in hours
            batch_size: Training batch size
            
        Returns:
            Dictionary with compute recommendations
        """
        # Estimate memory requirements
        model_memory_gb = (model_params * 4) / (1024**3)  # 4 bytes per param
        activation_memory_gb = (batch_size * 1024 * 4) / (1024**3)  # Rough estimate
        total_memory_gb = model_memory_gb + activation_memory_gb + dataset_size_gb
        
        recommendations = {
            'memory_requirements': {
                'model_memory_gb': model_memory_gb,
                'activation_memory_gb': activation_memory_gb,
                'dataset_memory_gb': dataset_size_gb,
                'total_memory_gb': total_memory_gb,
                'recommended_memory_gb': total_memory_gb * 1.5  # 50% buffer
            },
            'compute_options': []
        }
        
        # Recommend compute options
        if total_memory_gb <= 16:
            recommendations['compute_options'].append({
                'type': 'GPU',
                'instance': 'n1-standard-8 + 1x T4',
                'estimated_cost_per_hour': 0.95,
                'use_case': 'Small models, prototyping'
            })
        
        if total_memory_gb <= 32:
            recommendations['compute_options'].append({
                'type': 'GPU',
                'instance': 'n1-standard-16 + 1x V100',
                'estimated_cost_per_hour': 2.98,
                'use_case': 'Medium models, faster training'
            })
        
        if total_memory_gb > 32 or model_params > 10_000_000_000:
            recommendations['compute_options'].append({
                'type': 'TPU',
                'instance': 'TPU v3-8',
                'estimated_cost_per_hour': 8.00,
                'use_case': 'Large models, optimal performance'
            })
        
        # Calculate estimated costs
        for option in recommendations['compute_options']:
            option['estimated_total_cost'] = option['estimated_cost_per_hour'] * training_hours
        
        return recommendations
    
    def create_managed_resource_pool(
        self,
        pool_name: str,
        machine_spec: Dict[str, Any],
        min_replicas: int = 1,
        max_replicas: int = 10
    ) -> str:
        """
        Create a managed resource pool for training.
        
        Args:
            pool_name: Name for the resource pool
            machine_spec: Machine specification
            min_replicas: Minimum number of replicas
            max_replicas: Maximum number of replicas
            
        Returns:
            Resource pool name
        """
        # This is a conceptual implementation
        # In practice, you'd use GKE with autoscaling or Vertex AI custom training
        
        pool_config = {
            'name': pool_name,
            'machine_spec': machine_spec,
            'autoscaling': {
                'min_replicas': min_replicas,
                'max_replicas': max_replicas
            },
            'created_at': aiplatform.utils.get_timestamp()
        }
        
        print(f"Resource pool '{pool_name}' configuration:")
        print(json.dumps(pool_config, indent=2))
        
        return pool_name
    
    def optimize_resource_utilization(
        self,
        current_utilization: float,
        target_utilization: float = 0.80
    ) -> Dict[str, Any]:
        """
        Get recommendations for resource optimization.
        
        Args:
            current_utilization: Current resource utilization (0-1)
            target_utilization: Target utilization (0-1)
            
        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {
            'current_utilization': current_utilization,
            'target_utilization': target_utilization,
            'actions': []
        }
        
        if current_utilization < target_utilization * 0.5:
            recommendations['actions'].append({
                'action': 'SCALE_DOWN',
                'reason': 'Low utilization detected',
                'recommendation': 'Reduce instance count or switch to smaller instances'
            })
        elif current_utilization > target_utilization * 1.2:
            recommendations['actions'].append({
                'action': 'SCALE_UP',
                'reason': 'High utilization detected',
                'recommendation': 'Increase instance count or upgrade to larger instances'
            })
        
        if current_utilization < 0.3:
            recommendations['actions'].append({
                'action': 'USE_PREEMPTIBLE',
                'reason': 'Non-critical workload with low utilization',
                'recommendation': 'Consider using preemptible instances for cost savings'
            })
        
        return recommendations


# Example usage
infra_manager = AIInfrastructureManager(
    project_id='my-project',
    location='us-central1'
)

# Calculate requirements
requirements = infra_manager.calculate_compute_requirements(
    model_params=1_000_000_000,  # 1B parameters
    dataset_size_gb=100,
    training_hours=24,
    batch_size=32
)

print("Compute Requirements:")
print(f"Total Memory: {requirements['memory_requirements']['total_memory_gb']:.2f} GB")
print("\nRecommended Options:")
for option in requirements['compute_options']:
    print(f"- {option['instance']}: ${option['estimated_total_cost']:.2f} total")
```

## 2. Storage Infrastructure

### 2.1 Storage Selection and Optimization

```python
from google.cloud import storage
from google.cloud import bigquery
from datetime import datetime, timedelta

class StorageManager:
    """Manager for AI storage infrastructure."""
    
    def __init__(self, project_id: str):
        """
        Initialize Storage Manager.
        
        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.storage_client = storage.Client(project=project_id)
        self.bq_client = bigquery.Client(project=project_id)
    
    def recommend_storage_tier(
        self,
        access_frequency: str,
        data_size_tb: float,
        retention_days: int
    ) -> Dict[str, Any]:
        """
        Recommend storage tier based on access patterns.
        
        Args:
            access_frequency: 'frequent', 'infrequent', or 'archive'
            data_size_tb: Data size in TB
            retention_days: Data retention period in days
            
        Returns:
            Dictionary with storage recommendations
        """
        recommendations = []
        
        # Storage tier pricing (approximate per GB per month)
        pricing = {
            'standard': 0.020,
            'nearline': 0.010,
            'coldline': 0.004,
            'archive': 0.0012
        }
        
        data_size_gb = data_size_tb * 1024
        
        if access_frequency == 'frequent':
            recommendations.append({
                'tier': 'STANDARD',
                'monthly_cost': data_size_gb * pricing['standard'],
                'use_case': 'Active training data, frequent access',
                'retrieval_cost': 'Free',
                'min_storage_duration': 'None'
            })
        
        elif access_frequency == 'infrequent':
            if retention_days >= 30:
                recommendations.append({
                    'tier': 'NEARLINE',
                    'monthly_cost': data_size_gb * pricing['nearline'],
                    'use_case': 'Backup data, monthly access',
                    'retrieval_cost': '$0.01/GB',
                    'min_storage_duration': '30 days'
                })
            
            if retention_days >= 90:
                recommendations.append({
                    'tier': 'COLDLINE',
                    'monthly_cost': data_size_gb * pricing['coldline'],
                    'use_case': 'Archived models, quarterly access',
                    'retrieval_cost': '$0.02/GB',
                    'min_storage_duration': '90 days'
                })
        
        else:  # archive
            if retention_days >= 365:
                recommendations.append({
                    'tier': 'ARCHIVE',
                    'monthly_cost': data_size_gb * pricing['archive'],
                    'use_case': 'Long-term compliance, rare access',
                    'retrieval_cost': '$0.05/GB',
                    'min_storage_duration': '365 days'
                })
        
        return {
            'data_size_tb': data_size_tb,
            'access_frequency': access_frequency,
            'recommendations': recommendations
        }
    
    def setup_lifecycle_policy(
        self,
        bucket_name: str,
        rules: List[Dict[str, Any]]
    ):
        """
        Set up lifecycle management policy for storage bucket.
        
        Args:
            bucket_name: GCS bucket name
            rules: List of lifecycle rules
        """
        bucket = self.storage_client.bucket(bucket_name)
        
        # Example rules:
        # - Move to Nearline after 30 days
        # - Move to Coldline after 90 days
        # - Delete after 365 days
        
        lifecycle_rules = []
        
        for rule in rules:
            lifecycle_rules.append(storage.bucket.LifecycleRuleConditions(
                age=rule.get('age_days'),
                created_before=rule.get('created_before'),
                matches_storage_class=rule.get('matches_class')
            ))
        
        bucket.lifecycle_rules = lifecycle_rules
        bucket.patch()
        
        print(f"Lifecycle policy configured for bucket: {bucket_name}")
    
    def optimize_data_access(
        self,
        bucket_name: str,
        enable_cdn: bool = True,
        enable_requester_pays: bool = False
    ):
        """
        Optimize data access patterns.
        
        Args:
            bucket_name: GCS bucket name
            enable_cdn: Enable Cloud CDN
            enable_requester_pays: Enable requester pays
        """
        bucket = self.storage_client.bucket(bucket_name)
        
        # Enable uniform bucket-level access
        bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        
        # Configure requester pays
        if enable_requester_pays:
            bucket.requester_pays = True
        
        bucket.patch()
        
        print(f"Storage optimizations applied to: {bucket_name}")


# Example usage
storage_manager = StorageManager(project_id='my-project')

# Get storage recommendations
recommendations = storage_manager.recommend_storage_tier(
    access_frequency='infrequent',
    data_size_tb=10,
    retention_days=90
)

print("\nStorage Recommendations:")
for rec in recommendations['recommendations']:
    print(f"- {rec['tier']}: ${rec['monthly_cost']:.2f}/month")
    print(f"  Use case: {rec['use_case']}")

# Setup lifecycle policy
storage_manager.setup_lifecycle_policy(
    bucket_name='my-ml-data',
    rules=[
        {'age_days': 30, 'action': 'SetStorageClass', 'storage_class': 'NEARLINE'},
        {'age_days': 90, 'action': 'SetStorageClass', 'storage_class': 'COLDLINE'},
        {'age_days': 365, 'action': 'Delete'}
    ]
)
```

## 3. Autoscaling and Resource Optimization

### 3.1 Autoscaling Configuration

```python
from google.cloud import container_v1
from kubernetes import client, config

class AutoscalingManager:
    """Manager for autoscaling AI workloads."""
    
    def __init__(
        self,
        project_id: str,
        cluster_name: str,
        zone: str = 'us-central1-a'
    ):
        """
        Initialize Autoscaling Manager.
        
        Args:
            project_id: GCP project ID
            cluster_name: GKE cluster name
            zone: GCP zone
        """
        self.project_id = project_id
        self.cluster_name = cluster_name
        self.zone = zone
    
    def configure_horizontal_pod_autoscaling(
        self,
        deployment_name: str,
        namespace: str = 'default',
        min_replicas: int = 1,
        max_replicas: int = 10,
        target_cpu_utilization: int = 70
    ) -> Dict[str, Any]:
        """
        Configure Horizontal Pod Autoscaler (HPA).
        
        Args:
            deployment_name: Kubernetes deployment name
            namespace: Kubernetes namespace
            min_replicas: Minimum number of replicas
            max_replicas: Maximum number of replicas
            target_cpu_utilization: Target CPU utilization percentage
            
        Returns:
            HPA configuration
        """
        hpa_config = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': f'{deployment_name}-hpa',
                'namespace': namespace
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': deployment_name
                },
                'minReplicas': min_replicas,
                'maxReplicas': max_replicas,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': target_cpu_utilization
                            }
                        }
                    },
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'memory',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 80
                            }
                        }
                    }
                ]
            }
        }
        
        return hpa_config
    
    def configure_cluster_autoscaling(
        self,
        node_pool_name: str,
        min_nodes: int = 1,
        max_nodes: int = 10
    ) -> Dict[str, Any]:
        """
        Configure GKE Cluster Autoscaler.
        
        Args:
            node_pool_name: Node pool name
            min_nodes: Minimum number of nodes
            max_nodes: Maximum number of nodes
            
        Returns:
            Autoscaling configuration
        """
        autoscaling_config = {
            'enabled': True,
            'min_node_count': min_nodes,
            'max_node_count': max_nodes,
            'autoscaling_profile': 'OPTIMIZE_UTILIZATION'  # or 'BALANCED'
        }
        
        print(f"Cluster autoscaling configured for node pool: {node_pool_name}")
        print(f"Min nodes: {min_nodes}, Max nodes: {max_nodes}")
        
        return autoscaling_config
    
    def configure_vertical_pod_autoscaling(
        self,
        deployment_name: str,
        namespace: str = 'default',
        update_mode: str = 'Auto'
    ) -> Dict[str, Any]:
        """
        Configure Vertical Pod Autoscaler (VPA).
        
        Args:
            deployment_name: Kubernetes deployment name
            namespace: Kubernetes namespace
            update_mode: 'Off', 'Initial', 'Recreate', or 'Auto'
            
        Returns:
            VPA configuration
        """
        vpa_config = {
            'apiVersion': 'autoscaling.k8s.io/v1',
            'kind': 'VerticalPodAutoscaler',
            'metadata': {
                'name': f'{deployment_name}-vpa',
                'namespace': namespace
            },
            'spec': {
                'targetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': deployment_name
                },
                'updatePolicy': {
                    'updateMode': update_mode
                },
                'resourcePolicy': {
                    'containerPolicies': [
                        {
                            'containerName': '*',
                            'minAllowed': {
                                'cpu': '100m',
                                'memory': '100Mi'
                            },
                            'maxAllowed': {
                                'cpu': '8',
                                'memory': '32Gi'
                            }
                        }
                    ]
                }
            }
        }
        
        return vpa_config


# Example usage
autoscaling_mgr = AutoscalingManager(
    project_id='my-project',
    cluster_name='ml-cluster',
    zone='us-central1-a'
)

# Configure HPA
hpa_config = autoscaling_mgr.configure_horizontal_pod_autoscaling(
    deployment_name='model-server',
    min_replicas=2,
    max_replicas=20,
    target_cpu_utilization=70
)

# Configure cluster autoscaling
cluster_config = autoscaling_mgr.configure_cluster_autoscaling(
    node_pool_name='gpu-pool',
    min_nodes=0,
    max_nodes=10
)
```

## 4. Cost Optimization

### 4.1 Cost Management

```python
from google.cloud import billing_v1
from google.cloud import monitoring_v3
from datetime import datetime, timedelta

class CostOptimizationManager:
    """Manager for cost optimization."""
    
    def __init__(self, project_id: str):
        """
        Initialize Cost Optimization Manager.
        
        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
    
    def get_cost_recommendations(
        self,
        current_monthly_cost: float,
        workload_type: str
    ) -> List[Dict[str, Any]]:
        """
        Get cost optimization recommendations.
        
        Args:
            current_monthly_cost: Current monthly spend
            workload_type: Type of workload
            
        Returns:
            List of cost optimization recommendations
        """
        recommendations = []
        
        # Committed use discounts
        if current_monthly_cost > 1000:
            savings = current_monthly_cost * 0.57  # Up to 57% savings
            recommendations.append({
                'category': 'Committed Use Discounts',
                'action': 'Purchase 1-year or 3-year commitments',
                'potential_savings_monthly': savings,
                'effort': 'Low',
                'risk': 'Low'
            })
        
        # Preemptible instances
        if workload_type in ['training', 'batch']:
            savings = current_monthly_cost * 0.60  # Up to 60% savings
            recommendations.append({
                'category': 'Preemptible Instances',
                'action': 'Use preemptible VMs for fault-tolerant workloads',
                'potential_savings_monthly': savings,
                'effort': 'Medium',
                'risk': 'Low'
            })
        
        # Sustained use discounts (automatic)
        recommendations.append({
            'category': 'Sustained Use Discounts',
            'action': 'Automatically applied for instances running >25% of month',
            'potential_savings_monthly': current_monthly_cost * 0.30,
            'effort': 'None',
            'risk': 'None'
        })
        
        # Right-sizing
        recommendations.append({
            'category': 'Right-sizing',
            'action': 'Analyze utilization and resize underutilized instances',
            'potential_savings_monthly': current_monthly_cost * 0.25,
            'effort': 'Medium',
            'risk': 'Low'
        })
        
        # Storage optimization
        recommendations.append({
            'category': 'Storage Lifecycle',
            'action': 'Move infrequently accessed data to cheaper storage tiers',
            'potential_savings_monthly': current_monthly_cost * 0.15,
            'effort': 'Low',
            'risk': 'Low'
        })
        
        return recommendations
    
    def setup_cost_alerts(
        self,
        budget_amount: float,
        alert_thresholds: List[float] = [0.5, 0.9, 1.0]
    ):
        """
        Set up cost budget alerts.
        
        Args:
            budget_amount: Monthly budget amount
            alert_thresholds: List of threshold percentages
        """
        print(f"Setting up budget alerts for ${budget_amount}/month")
        
        for threshold in alert_thresholds:
            alert_amount = budget_amount * threshold
            print(f"- Alert at ${alert_amount:.2f} ({threshold*100}% of budget)")
    
    def analyze_resource_utilization(
        self,
        resource_type: str,
        time_window_days: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze resource utilization.
        
        Args:
            resource_type: Type of resource to analyze
            time_window_days: Analysis time window
            
        Returns:
            Dictionary with utilization analysis
        """
        # This would query Cloud Monitoring for actual metrics
        analysis = {
            'resource_type': resource_type,
            'time_window_days': time_window_days,
            'metrics': {
                'average_utilization': 0.65,
                'peak_utilization': 0.95,
                'idle_time_percentage': 0.15
            },
            'recommendations': []
        }
        
        if analysis['metrics']['average_utilization'] < 0.50:
            analysis['recommendations'].append(
                'Consider scaling down or using smaller instances'
            )
        
        if analysis['metrics']['idle_time_percentage'] > 0.20:
            analysis['recommendations'].append(
                'High idle time detected - consider autoscaling or scheduled shutdown'
            )
        
        return analysis


# Example usage
cost_optimizer = CostOptimizationManager(project_id='my-project')

# Get recommendations
recommendations = cost_optimizer.get_cost_recommendations(
    current_monthly_cost=5000,
    workload_type='training'
)

print("Cost Optimization Recommendations:")
for rec in recommendations:
    print(f"\n{rec['category']}:")
    print(f"  Action: {rec['action']}")
    print(f"  Potential Savings: ${rec['potential_savings_monthly']:.2f}/month")
    print(f"  Effort: {rec['effort']}, Risk: {rec['risk']}")

# Setup budget alerts
cost_optimizer.setup_cost_alerts(
    budget_amount=10000,
    alert_thresholds=[0.5, 0.75, 0.9, 1.0]
)
```

## 5. Quick Reference Checklist

### Compute Resources
- [ ] Right-size instances based on workload
- [ ] Use appropriate accelerators (GPU/TPU)
- [ ] Enable autoscaling for variable workloads
- [ ] Use preemptible instances for fault-tolerant tasks
- [ ] Monitor resource utilization
- [ ] Implement resource quotas

### Storage Management
- [ ] Choose appropriate storage tier
- [ ] Implement lifecycle policies
- [ ] Enable regional/multi-regional replication
- [ ] Use compression for large datasets
- [ ] Monitor storage costs
- [ ] Clean up unused data regularly

### Networking
- [ ] Use VPC for secure communication
- [ ] Enable Private Google Access
- [ ] Implement VPC Service Controls
- [ ] Use Cloud CDN for data distribution
- [ ] Monitor network egress costs
- [ ] Optimize data transfer patterns

### Autoscaling
- [ ] Configure HPA for pod-level scaling
- [ ] Enable cluster autoscaling
- [ ] Set appropriate min/max replicas
- [ ] Use VPA for resource optimization
- [ ] Monitor scaling metrics
- [ ] Test scaling under load

### Cost Optimization
- [ ] Purchase committed use discounts
- [ ] Use preemptible/spot instances
- [ ] Implement resource scheduling
- [ ] Set up cost budgets and alerts
- [ ] Regular cost analysis and optimization
- [ ] Use sustained use discounts

### Security
- [ ] Implement least privilege access (IAM)
- [ ] Enable encryption at rest and in transit
- [ ] Use Cloud KMS for key management
- [ ] Enable audit logging
- [ ] Implement VPC Service Controls
- [ ] Regular security reviews

### Monitoring
- [ ] Set up Cloud Monitoring dashboards
- [ ] Configure alerting policies
- [ ] Monitor resource utilization
- [ ] Track performance metrics
- [ ] Log analysis and debugging
- [ ] Capacity planning

### Best Practices
- [ ] Use infrastructure as code (Terraform)
- [ ] Implement CI/CD pipelines
- [ ] Regular backup and disaster recovery testing
- [ ] Document infrastructure architecture
- [ ] Implement tagging strategy
- [ ] Regular performance optimization reviews
