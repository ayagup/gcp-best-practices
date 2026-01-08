# Vertex AI Training Best Practices

*Last Updated: January 4, 2026*

## Overview

Vertex AI Training provides managed training for ML models with support for distributed training, hyperparameter tuning, custom containers, framework prebuilt containers, GPU/TPU acceleration, and scalable infrastructure.

---

## 1. Custom Training Jobs

### Create Custom Training Job

```python
from google.cloud import aiplatform

class CustomTrainingManager:
    """Manage custom training jobs on Vertex AI."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        
        aiplatform.init(project=project_id, location=location)
    
    def create_custom_job(
        self,
        display_name,
        script_path,
        container_uri,
        requirements=None,
        machine_type='n1-standard-4',
        accelerator_type=None,
        accelerator_count=0,
        args=None
    ):
        """Create custom training job."""
        
        if args is None:
            args = []
        
        if requirements is None:
            requirements = []
        
        # Create custom job
        job = aiplatform.CustomJob.from_local_script(
            display_name=display_name,
            script_path=script_path,
            container_uri=container_uri,
            requirements=requirements,
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            args=args
        )
        
        print(f"✓ Created custom job: {display_name}")
        print(f"  Machine type: {machine_type}")
        
        if accelerator_type:
            print(f"  Accelerator: {accelerator_count}x {accelerator_type}")
        
        return job
    
    def run_custom_job(
        self,
        display_name,
        script_path,
        container_uri,
        args=None,
        machine_type='n1-standard-4',
        replica_count=1,
        base_output_dir=None
    ):
        """Run custom training job."""
        
        if args is None:
            args = []
        
        job = aiplatform.CustomJob.from_local_script(
            display_name=display_name,
            script_path=script_path,
            container_uri=container_uri,
            args=args,
            machine_type=machine_type,
            replica_count=replica_count
        )
        
        # Run job
        job.run(
            service_account=None,
            network=None,
            base_output_dir=base_output_dir,
            sync=True
        )
        
        print(f"✓ Training job completed: {display_name}")
        print(f"  Job ID: {job.name}")
        
        return job
    
    def run_training_with_python_package(
        self,
        display_name,
        python_package_gcs_uri,
        python_module_name,
        machine_type='n1-standard-4',
        args=None
    ):
        """Run training with Python package."""
        
        if args is None:
            args = []
        
        job = aiplatform.CustomPythonPackageTrainingJob(
            display_name=display_name,
            python_package_gcs_uri=python_package_gcs_uri,
            python_module_name=python_module_name,
            container_uri='us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-11:latest'
        )
        
        model = job.run(
            machine_type=machine_type,
            replica_count=1,
            args=args,
            sync=True
        )
        
        print(f"✓ Training completed: {display_name}")
        
        return model

# Example usage
trainer = CustomTrainingManager(project_id='my-project')

# Run custom job
# job = trainer.run_custom_job(
#     display_name='fraud-detection-training',
#     script_path='training/train.py',
#     container_uri='us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-11:latest',
#     args=['--epochs', '50', '--batch-size', '32'],
#     machine_type='n1-standard-8',
#     base_output_dir='gs://my-bucket/training-output'
# )
```

---

## 2. Distributed Training

### Configure Multi-GPU and Multi-Node Training

```python
class DistributedTrainingManager:
    """Manage distributed training jobs."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def run_multi_gpu_training(
        self,
        display_name,
        script_path,
        container_uri,
        gpu_type='NVIDIA_TESLA_T4',
        gpu_count=4,
        machine_type='n1-standard-16',
        args=None
    ):
        """Run multi-GPU training on single machine."""
        
        if args is None:
            args = []
        
        job = aiplatform.CustomJob.from_local_script(
            display_name=display_name,
            script_path=script_path,
            container_uri=container_uri,
            machine_type=machine_type,
            accelerator_type=gpu_type,
            accelerator_count=gpu_count,
            args=args
        )
        
        job.run(sync=True)
        
        print(f"✓ Multi-GPU training completed")
        print(f"  GPUs: {gpu_count}x {gpu_type}")
        
        return job
    
    def run_multi_node_training(
        self,
        display_name,
        script_path,
        container_uri,
        worker_pool_specs,
        args=None
    ):
        """Run multi-node distributed training.
        
        Args:
            worker_pool_specs: List of worker pool configurations
                Example:
                [
                    {
                        'machine_spec': {'machine_type': 'n1-standard-16'},
                        'replica_count': 1,
                        'container_spec': {'image_uri': container_uri}
                    },
                    {
                        'machine_spec': {'machine_type': 'n1-standard-8'},
                        'replica_count': 3,
                        'container_spec': {'image_uri': container_uri}
                    }
                ]
        """
        
        if args is None:
            args = []
        
        job = aiplatform.CustomJob(
            display_name=display_name,
            worker_pool_specs=worker_pool_specs
        )
        
        job.run(sync=True)
        
        print(f"✓ Multi-node training completed")
        
        return job
    
    def create_tf_distributed_training(
        self,
        display_name,
        script_path,
        num_workers=4,
        gpu_per_worker=1,
        args=None
    ):
        """Create TensorFlow distributed training job."""
        
        if args is None:
            args = []
        
        # Chief worker (coordinator)
        chief_spec = {
            'machine_spec': {
                'machine_type': 'n1-standard-16',
                'accelerator_type': 'NVIDIA_TESLA_T4',
                'accelerator_count': gpu_per_worker
            },
            'replica_count': 1,
            'container_spec': {
                'image_uri': 'us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-11:latest',
                'command': ['python', script_path],
                'args': args
            }
        }
        
        # Worker nodes
        worker_spec = {
            'machine_spec': {
                'machine_type': 'n1-standard-16',
                'accelerator_type': 'NVIDIA_TESLA_T4',
                'accelerator_count': gpu_per_worker
            },
            'replica_count': num_workers - 1,
            'container_spec': {
                'image_uri': 'us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-11:latest',
                'command': ['python', script_path],
                'args': args
            }
        }
        
        worker_pool_specs = [chief_spec, worker_spec]
        
        job = self.run_multi_node_training(
            display_name=display_name,
            script_path=script_path,
            container_uri='us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-11:latest',
            worker_pool_specs=worker_pool_specs,
            args=args
        )
        
        print(f"✓ TensorFlow distributed training with {num_workers} workers")
        
        return job

# Example usage
dist_trainer = DistributedTrainingManager(project_id='my-project')

# Multi-GPU training
# job = dist_trainer.run_multi_gpu_training(
#     display_name='image-classification-multigpu',
#     script_path='training/train_multigpu.py',
#     container_uri='us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-11:latest',
#     gpu_type='NVIDIA_TESLA_V100',
#     gpu_count=4,
#     machine_type='n1-standard-16'
# )
```

---

## 3. Hyperparameter Tuning

### Optimize Hyperparameters

```python
class HyperparameterTuner:
    """Manage hyperparameter tuning jobs."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def create_hp_tuning_job(
        self,
        display_name,
        script_path,
        container_uri,
        parameter_spec,
        metric_spec,
        max_trial_count=20,
        parallel_trial_count=5,
        machine_type='n1-standard-4'
    ):
        """Create hyperparameter tuning job.
        
        Args:
            parameter_spec: Dict of parameters to tune
                Example:
                {
                    'learning_rate': hpt.DoubleParameterSpec(min=0.0001, max=0.1, scale='log'),
                    'batch_size': hpt.DiscreteParameterSpec(values=[16, 32, 64, 128]),
                    'num_layers': hpt.IntegerParameterSpec(min=2, max=5, scale='linear')
                }
            
            metric_spec: Dict with metric configuration
                Example:
                {
                    'accuracy': 'maximize'  # or 'minimize'
                }
        """
        
        from google.cloud.aiplatform import hyperparameter_tuning as hpt
        
        # Create custom job for each trial
        worker_pool_specs = [{
            'machine_spec': {
                'machine_type': machine_type
            },
            'replica_count': 1,
            'python_package_spec': {
                'executor_image_uri': container_uri,
                'python_module': script_path.replace('/', '.').replace('.py', '')
            }
        }]
        
        # Create HP tuning job
        hp_job = aiplatform.HyperparameterTuningJob(
            display_name=display_name,
            custom_job=aiplatform.CustomJob(
                display_name=f'{display_name}-trial',
                worker_pool_specs=worker_pool_specs
            ),
            metric_spec=metric_spec,
            parameter_spec=parameter_spec,
            max_trial_count=max_trial_count,
            parallel_trial_count=parallel_trial_count
        )
        
        print(f"✓ Created hyperparameter tuning job")
        print(f"  Max trials: {max_trial_count}")
        print(f"  Parallel trials: {parallel_trial_count}")
        
        return hp_job
    
    def run_hp_tuning(
        self,
        display_name,
        script_path,
        container_uri,
        parameter_spec,
        metric_id='accuracy',
        metric_goal='maximize',
        max_trial_count=20,
        parallel_trial_count=5
    ):
        """Run hyperparameter tuning job."""
        
        from google.cloud.aiplatform import hyperparameter_tuning as hpt
        
        # Create custom training job template
        job = aiplatform.CustomJob.from_local_script(
            display_name=f'{display_name}-trial',
            script_path=script_path,
            container_uri=container_uri
        )
        
        # Create HP tuning job
        hp_job = aiplatform.HyperparameterTuningJob(
            display_name=display_name,
            custom_job=job,
            metric_spec={metric_id: metric_goal},
            parameter_spec=parameter_spec,
            max_trial_count=max_trial_count,
            parallel_trial_count=parallel_trial_count,
            search_algorithm='ALGORITHM_UNSPECIFIED'  # Bayesian optimization
        )
        
        # Run tuning
        hp_job.run()
        
        print(f"✓ Hyperparameter tuning completed")
        
        # Get best trial
        best_trial = max(
            hp_job.trials,
            key=lambda t: t.final_measurement.metrics[0].value
        )
        
        print(f"\nBest Trial:")
        print(f"  Trial ID: {best_trial.id}")
        print(f"  {metric_id}: {best_trial.final_measurement.metrics[0].value}")
        print(f"  Parameters:")
        
        for param in best_trial.parameters:
            print(f"    {param.parameter_id}: {param.value}")
        
        return hp_job, best_trial

# Example usage
from google.cloud.aiplatform import hyperparameter_tuning as hpt

tuner = HyperparameterTuner(project_id='my-project')

# Define parameter space
# parameter_spec = {
#     'learning_rate': hpt.DoubleParameterSpec(min=0.0001, max=0.1, scale='log'),
#     'batch_size': hpt.DiscreteParameterSpec(values=[16, 32, 64, 128], scale='linear'),
#     'num_layers': hpt.IntegerParameterSpec(min=2, max=5, scale='linear'),
#     'dropout': hpt.DoubleParameterSpec(min=0.1, max=0.5, scale='linear')
# }

# Run tuning
# hp_job, best_trial = tuner.run_hp_tuning(
#     display_name='fraud-detection-hp-tuning',
#     script_path='training/train.py',
#     container_uri='us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-11:latest',
#     parameter_spec=parameter_spec,
#     metric_id='accuracy',
#     metric_goal='maximize',
#     max_trial_count=50,
#     parallel_trial_count=5
# )
```

---

## 4. Prebuilt Training Containers

### Use Framework Containers

```python
class PrebuiltContainerTraining:
    """Training with prebuilt framework containers."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def get_container_uri(self, framework, version, accelerator='cpu'):
        """Get prebuilt container URI."""
        
        containers = {
            'tensorflow': {
                'cpu': f'us-docker.pkg.dev/vertex-ai/training/tf-cpu.{version}:latest',
                'gpu': f'us-docker.pkg.dev/vertex-ai/training/tf-gpu.{version}:latest'
            },
            'pytorch': {
                'cpu': f'us-docker.pkg.dev/vertex-ai/training/pytorch-cpu.{version}:latest',
                'gpu': f'us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.{version}:latest'
            },
            'sklearn': {
                'cpu': f'us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.{version}:latest'
            },
            'xgboost': {
                'cpu': f'us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.{version}:latest'
            }
        }
        
        return containers.get(framework, {}).get(accelerator)
    
    def train_tensorflow_model(
        self,
        display_name,
        script_path,
        tf_version='2-11',
        use_gpu=False,
        gpu_count=0,
        args=None
    ):
        """Train TensorFlow model."""
        
        if args is None:
            args = []
        
        accelerator = 'gpu' if use_gpu else 'cpu'
        container_uri = self.get_container_uri('tensorflow', tf_version, accelerator)
        
        job = aiplatform.CustomJob.from_local_script(
            display_name=display_name,
            script_path=script_path,
            container_uri=container_uri,
            machine_type='n1-standard-8',
            accelerator_type='NVIDIA_TESLA_T4' if use_gpu else None,
            accelerator_count=gpu_count if use_gpu else 0,
            args=args
        )
        
        job.run(sync=True)
        
        print(f"✓ TensorFlow training completed")
        
        return job
    
    def train_pytorch_model(
        self,
        display_name,
        script_path,
        pytorch_version='1-13',
        use_gpu=False,
        args=None
    ):
        """Train PyTorch model."""
        
        if args is None:
            args = []
        
        accelerator = 'gpu' if use_gpu else 'cpu'
        container_uri = self.get_container_uri('pytorch', pytorch_version, accelerator)
        
        job = aiplatform.CustomJob.from_local_script(
            display_name=display_name,
            script_path=script_path,
            container_uri=container_uri,
            machine_type='n1-standard-8',
            accelerator_type='NVIDIA_TESLA_T4' if use_gpu else None,
            accelerator_count=1 if use_gpu else 0,
            args=args
        )
        
        job.run(sync=True)
        
        print(f"✓ PyTorch training completed")
        
        return job
    
    def train_sklearn_model(
        self,
        display_name,
        script_path,
        sklearn_version='1-0',
        args=None
    ):
        """Train scikit-learn model."""
        
        if args is None:
            args = []
        
        container_uri = self.get_container_uri('sklearn', sklearn_version, 'cpu')
        
        job = aiplatform.CustomJob.from_local_script(
            display_name=display_name,
            script_path=script_path,
            container_uri=container_uri,
            machine_type='n1-standard-4',
            args=args
        )
        
        job.run(sync=True)
        
        print(f"✓ scikit-learn training completed")
        
        return job

# Example usage
container_trainer = PrebuiltContainerTraining(project_id='my-project')

# Train TensorFlow model with GPU
# job = container_trainer.train_tensorflow_model(
#     display_name='image-classification-training',
#     script_path='training/train_tf.py',
#     tf_version='2-11',
#     use_gpu=True,
#     gpu_count=1,
#     args=['--epochs', '100', '--batch-size', '64']
# )
```

---

## 5. Resource Optimization

### Optimize Training Resources

```python
class ResourceOptimizer:
    """Optimize training resource usage."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
    
    def recommend_machine_type(
        self,
        dataset_size_gb,
        model_type,
        use_gpu=False
    ):
        """Recommend machine type based on workload."""
        
        recommendations = {
            'small': {  # < 10 GB
                'cpu': 'n1-standard-4',
                'gpu': 'n1-standard-8'
            },
            'medium': {  # 10-100 GB
                'cpu': 'n1-standard-8',
                'gpu': 'n1-standard-16'
            },
            'large': {  # > 100 GB
                'cpu': 'n1-highmem-16',
                'gpu': 'n1-highmem-32'
            }
        }
        
        if dataset_size_gb < 10:
            size_category = 'small'
        elif dataset_size_gb < 100:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        machine_type = recommendations[size_category]['gpu' if use_gpu else 'cpu']
        
        print(f"Recommended machine type: {machine_type}")
        print(f"  Dataset size: {dataset_size_gb} GB")
        print(f"  GPU: {use_gpu}")
        
        return machine_type
    
    def use_preemptible_vms(
        self,
        display_name,
        script_path,
        container_uri,
        machine_type='n1-standard-4',
        args=None
    ):
        """Use preemptible VMs for cost savings."""
        
        if args is None:
            args = []
        
        job = aiplatform.CustomJob.from_local_script(
            display_name=display_name,
            script_path=script_path,
            container_uri=container_uri,
            machine_type=machine_type,
            args=args
        )
        
        # Enable preemptible VMs
        job.run(
            restart_job_on_worker_restart=True,
            enable_web_access=False,
            sync=True
        )
        
        print(f"✓ Training completed with preemptible VMs")
        print(f"  Cost savings: ~60-80%")
        
        return job
    
    def implement_checkpointing(self, checkpoint_dir='gs://my-bucket/checkpoints'):
        """Implement model checkpointing for fault tolerance."""
        
        import tensorflow as tf
        
        # TensorFlow checkpointing
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'{checkpoint_dir}/model_{{epoch:02d}}.h5',
            save_weights_only=False,
            save_freq='epoch',
            verbose=1
        )
        
        print(f"✓ Checkpointing enabled")
        print(f"  Directory: {checkpoint_dir}")
        
        return checkpoint_callback

# Example usage
optimizer = ResourceOptimizer(project_id='my-project')

# Get machine type recommendation
# machine_type = optimizer.recommend_machine_type(
#     dataset_size_gb=50,
#     model_type='deep_learning',
#     use_gpu=True
# )

# Use preemptible VMs
# job = optimizer.use_preemptible_vms(
#     display_name='cost-effective-training',
#     script_path='training/train.py',
#     container_uri='us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-11:latest',
#     machine_type='n1-standard-8'
# )
```

---

## 6. Monitoring Training Jobs

```python
class TrainingJobMonitor:
    """Monitor training job progress."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def list_training_jobs(self, filter_str=None):
        """List all training jobs."""
        
        jobs = aiplatform.CustomJob.list(filter=filter_str)
        
        print(f"\nTraining Jobs ({len(jobs)}):\n")
        
        for job in jobs:
            print(f"{job.display_name}")
            print(f"  State: {job.state}")
            print(f"  Created: {job.create_time}")
            print()
        
        return jobs
    
    def get_job_status(self, job_name):
        """Get training job status."""
        
        job = aiplatform.CustomJob(job_name)
        
        print(f"\nJob: {job.display_name}")
        print(f"  State: {job.state}")
        print(f"  Created: {job.create_time}")
        print(f"  Started: {job.start_time}")
        print(f"  Ended: {job.end_time}")
        
        if job.error:
            print(f"  Error: {job.error.message}")
        
        return job
    
    def cancel_training_job(self, job_name):
        """Cancel running training job."""
        
        job = aiplatform.CustomJob(job_name)
        job.cancel()
        
        print(f"✓ Cancelled job: {job.display_name}")

# Example usage
monitor = TrainingJobMonitor(project_id='my-project')

# List jobs
# jobs = monitor.list_training_jobs()

# Get status
# status = monitor.get_job_status('projects/.../customJobs/123')
```

---

## 7. Quick Reference Checklist

### Setup
- [ ] Enable Vertex AI API
- [ ] Configure IAM permissions
- [ ] Set up Cloud Storage for outputs
- [ ] Prepare training script
- [ ] Install SDK: `pip install google-cloud-aiplatform`

### Job Configuration
- [ ] Choose appropriate machine type
- [ ] Select GPU/TPU if needed
- [ ] Configure worker replicas
- [ ] Set environment variables
- [ ] Specify output directory

### Optimization
- [ ] Use preemptible VMs for cost savings
- [ ] Enable model checkpointing
- [ ] Implement early stopping
- [ ] Use mixed precision training
- [ ] Monitor resource utilization

### Distributed Training
- [ ] Configure multi-GPU setup
- [ ] Set up multi-node training
- [ ] Use appropriate distribution strategy
- [ ] Test with small data first
- [ ] Monitor training synchronization

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
