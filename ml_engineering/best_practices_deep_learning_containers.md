# Best Practices for Deep Learning Containers on Google Cloud

## Overview

Deep Learning Containers provide pre-packaged, optimized container images for machine learning frameworks (TensorFlow, PyTorch, scikit-learn) that can be deployed on Google Kubernetes Engine (GKE), Cloud Run, Vertex AI, or any container runtime. These containers enable consistent, reproducible ML workloads across environments.

## 1. Container Image Selection

### 1.1 Available Container Images

```python
from typing import Dict, Any, List
import subprocess

class DeepLearningContainerManager:
    """Manager for Deep Learning Container operations."""
    
    def __init__(self, project_id: str, location: str = 'us-central1'):
        """
        Initialize Deep Learning Container Manager.
        
        Args:
            project_id: GCP project ID
            location: GCP region
        """
        self.project_id = project_id
        self.location = location
        self.registry = 'us-docker.pkg.dev/vertex-ai'
    
    def get_available_images(self) -> Dict[str, Dict[str, Any]]:
        """
        Get available Deep Learning Container images.
        
        Returns:
            Dictionary with container image information
        """
        images = {
            'tensorflow-training': {
                'registry': f'{self.registry}/training/tf-cpu.2-13',
                'framework': 'TensorFlow',
                'version': '2.13',
                'type': 'Training',
                'accelerator': 'CPU',
                'use_cases': ['Model training', 'Batch processing'],
                'base_os': 'Ubuntu 20.04'
            },
            'tensorflow-training-gpu': {
                'registry': f'{self.registry}/training/tf-gpu.2-13',
                'framework': 'TensorFlow',
                'version': '2.13',
                'type': 'Training',
                'accelerator': 'GPU',
                'cuda': '11.8',
                'use_cases': ['GPU training', 'Large models'],
                'base_os': 'Ubuntu 20.04'
            },
            'tensorflow-prediction': {
                'registry': f'{self.registry}/prediction/tf2-cpu.2-13',
                'framework': 'TensorFlow',
                'version': '2.13',
                'type': 'Prediction',
                'accelerator': 'CPU',
                'use_cases': ['Model serving', 'Online prediction'],
                'base_os': 'Debian 11'
            },
            'tensorflow-prediction-gpu': {
                'registry': f'{self.registry}/prediction/tf2-gpu.2-13',
                'framework': 'TensorFlow',
                'version': '2.13',
                'type': 'Prediction',
                'accelerator': 'GPU',
                'cuda': '11.8',
                'use_cases': ['GPU inference', 'Low latency serving'],
                'base_os': 'Debian 11'
            },
            'pytorch-training': {
                'registry': f'{self.registry}/training/pytorch-cpu.2-0',
                'framework': 'PyTorch',
                'version': '2.0',
                'type': 'Training',
                'accelerator': 'CPU',
                'use_cases': ['Model training', 'Research'],
                'base_os': 'Ubuntu 20.04'
            },
            'pytorch-training-gpu': {
                'registry': f'{self.registry}/training/pytorch-gpu.2-0',
                'framework': 'PyTorch',
                'version': '2.0',
                'type': 'Training',
                'accelerator': 'GPU',
                'cuda': '11.8',
                'use_cases': ['GPU training', 'Large models'],
                'base_os': 'Ubuntu 20.04'
            },
            'pytorch-prediction': {
                'registry': f'{self.registry}/prediction/pytorch-cpu.2-0',
                'framework': 'PyTorch',
                'version': '2.0',
                'type': 'Prediction',
                'accelerator': 'CPU',
                'use_cases': ['Model serving', 'TorchServe'],
                'base_os': 'Debian 11'
            },
            'sklearn-cpu': {
                'registry': f'{self.registry}/prediction/sklearn-cpu.1-0',
                'framework': 'scikit-learn',
                'version': '1.0',
                'type': 'Prediction',
                'accelerator': 'CPU',
                'use_cases': ['Traditional ML', 'Lightweight models'],
                'base_os': 'Debian 11'
            },
            'xgboost-cpu': {
                'registry': f'{self.registry}/prediction/xgboost-cpu.1-6',
                'framework': 'XGBoost',
                'version': '1.6',
                'type': 'Prediction',
                'accelerator': 'CPU',
                'use_cases': ['Gradient boosting', 'Tabular data'],
                'base_os': 'Debian 11'
            }
        }
        
        return images
    
    def recommend_container(
        self,
        framework: str,
        workload_type: str,
        use_gpu: bool = False
    ) -> List[str]:
        """
        Recommend container image based on requirements.
        
        Args:
            framework: ML framework ('tensorflow', 'pytorch', 'sklearn', 'xgboost')
            workload_type: 'training' or 'prediction'
            use_gpu: Whether GPU support is needed
            
        Returns:
            List of recommended container images
        """
        images = self.get_available_images()
        recommendations = []
        
        for image_name, image_info in images.items():
            if (framework.lower() in image_info['framework'].lower() and
                workload_type.lower() in image_info['type'].lower()):
                
                if use_gpu and image_info['accelerator'] == 'GPU':
                    recommendations.append(image_name)
                elif not use_gpu and image_info['accelerator'] == 'CPU':
                    recommendations.append(image_name)
        
        return recommendations


# Example usage
container_manager = DeepLearningContainerManager(
    project_id='my-project',
    location='us-central1'
)

# Get available images
images = container_manager.get_available_images()
print("Available Deep Learning Containers:")
for name, info in images.items():
    print(f"  {name}: {info['framework']} {info['version']} ({info['type']}, {info['accelerator']})")

# Get recommendations
recommendations = container_manager.recommend_container(
    framework='tensorflow',
    workload_type='training',
    use_gpu=True
)
print(f"\nRecommended containers: {recommendations}")
```

## 2. Container Deployment on GKE

### 2.1 GKE Deployment Configuration

```python
from kubernetes import client, config
from typing import Dict, Any

class GKEDeploymentManager:
    """Manager for deploying containers on GKE."""
    
    def __init__(self, namespace: str = 'default'):
        """
        Initialize GKE Deployment Manager.
        
        Args:
            namespace: Kubernetes namespace
        """
        self.namespace = namespace
    
    def create_training_job_spec(
        self,
        job_name: str,
        container_image: str,
        command: List[str],
        gpu_count: int = 0,
        memory_gb: int = 8,
        cpu_count: int = 4
    ) -> Dict[str, Any]:
        """
        Create Kubernetes Job spec for training.
        
        Args:
            job_name: Job name
            container_image: Container image URI
            command: Command to execute
            gpu_count: Number of GPUs
            memory_gb: Memory in GB
            cpu_count: Number of CPUs
            
        Returns:
            Kubernetes Job specification
        """
        job_spec = {
            'apiVersion': 'batch/v1',
            'kind': 'Job',
            'metadata': {
                'name': job_name,
                'namespace': self.namespace
            },
            'spec': {
                'template': {
                    'spec': {
                        'containers': [{
                            'name': 'trainer',
                            'image': container_image,
                            'command': command,
                            'resources': {
                                'requests': {
                                    'memory': f'{memory_gb}Gi',
                                    'cpu': str(cpu_count)
                                },
                                'limits': {
                                    'memory': f'{memory_gb}Gi',
                                    'cpu': str(cpu_count)
                                }
                            },
                            'env': [
                                {
                                    'name': 'GOOGLE_APPLICATION_CREDENTIALS',
                                    'value': '/var/secrets/google/key.json'
                                }
                            ],
                            'volumeMounts': [
                                {
                                    'name': 'gcp-credentials',
                                    'mountPath': '/var/secrets/google',
                                    'readOnly': True
                                }
                            ]
                        }],
                        'volumes': [
                            {
                                'name': 'gcp-credentials',
                                'secret': {
                                    'secretName': 'gcp-service-account-key'
                                }
                            }
                        ],
                        'restartPolicy': 'Never'
                    }
                },
                'backoffLimit': 3
            }
        }
        
        # Add GPU resources if needed
        if gpu_count > 0:
            job_spec['spec']['template']['spec']['containers'][0]['resources']['limits']['nvidia.com/gpu'] = str(gpu_count)
            
            # Add node selector for GPU nodes
            job_spec['spec']['template']['spec']['nodeSelector'] = {
                'cloud.google.com/gke-accelerator': 'nvidia-tesla-t4'
            }
        
        return job_spec
    
    def create_serving_deployment_spec(
        self,
        deployment_name: str,
        container_image: str,
        replicas: int = 2,
        port: int = 8080,
        memory_gb: int = 4,
        cpu_count: int = 2
    ) -> Dict[str, Any]:
        """
        Create Kubernetes Deployment spec for model serving.
        
        Args:
            deployment_name: Deployment name
            container_image: Container image URI
            replicas: Number of replicas
            port: Container port
            memory_gb: Memory in GB
            cpu_count: Number of CPUs
            
        Returns:
            Kubernetes Deployment specification
        """
        deployment_spec = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': deployment_name,
                'namespace': self.namespace,
                'labels': {
                    'app': deployment_name
                }
            },
            'spec': {
                'replicas': replicas,
                'selector': {
                    'matchLabels': {
                        'app': deployment_name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': deployment_name
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'model-server',
                            'image': container_image,
                            'ports': [{
                                'containerPort': port,
                                'protocol': 'TCP'
                            }],
                            'resources': {
                                'requests': {
                                    'memory': f'{memory_gb}Gi',
                                    'cpu': str(cpu_count)
                                },
                                'limits': {
                                    'memory': f'{memory_gb}Gi',
                                    'cpu': str(cpu_count)
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': port
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': port
                                },
                                'initialDelaySeconds': 10,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        return deployment_spec
    
    def create_service_spec(
        self,
        service_name: str,
        deployment_name: str,
        port: int = 8080,
        target_port: int = 8080,
        service_type: str = 'LoadBalancer'
    ) -> Dict[str, Any]:
        """
        Create Kubernetes Service spec.
        
        Args:
            service_name: Service name
            deployment_name: Target deployment name
            port: Service port
            target_port: Container port
            service_type: Service type (LoadBalancer, ClusterIP, NodePort)
            
        Returns:
            Kubernetes Service specification
        """
        service_spec = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': service_name,
                'namespace': self.namespace
            },
            'spec': {
                'type': service_type,
                'selector': {
                    'app': deployment_name
                },
                'ports': [{
                    'protocol': 'TCP',
                    'port': port,
                    'targetPort': target_port
                }]
            }
        }
        
        return service_spec


# Example usage
gke_manager = GKEDeploymentManager(namespace='ml-workloads')

# Create training job spec
training_job = gke_manager.create_training_job_spec(
    job_name='image-classification-training',
    container_image='us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-13:latest',
    command=['python', 'train.py', '--epochs=10'],
    gpu_count=1,
    memory_gb=16,
    cpu_count=8
)

print("Training Job Spec:")
print(f"  Job: {training_job['metadata']['name']}")
print(f"  Image: {training_job['spec']['template']['spec']['containers'][0]['image']}")

# Create serving deployment spec
serving_deployment = gke_manager.create_serving_deployment_spec(
    deployment_name='model-serving',
    container_image='us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-13:latest',
    replicas=3,
    port=8080
)

# Create service spec
service = gke_manager.create_service_spec(
    service_name='model-service',
    deployment_name='model-serving',
    port=80,
    target_port=8080
)
```

## 3. Custom Container Images

### 3.1 Building Custom Containers

```python
import subprocess
from typing import List

class CustomContainerBuilder:
    """Builder for custom Deep Learning containers."""
    
    def __init__(self, project_id: str, region: str = 'us-central1'):
        """
        Initialize Custom Container Builder.
        
        Args:
            project_id: GCP project ID
            region: Container registry region
        """
        self.project_id = project_id
        self.region = region
        self.registry = f'{region}-docker.pkg.dev/{project_id}'
    
    def generate_dockerfile(
        self,
        base_image: str,
        python_packages: List[str],
        system_packages: List[str] = None,
        custom_commands: List[str] = None
    ) -> str:
        """
        Generate Dockerfile for custom container.
        
        Args:
            base_image: Base container image
            python_packages: List of Python packages
            system_packages: List of system packages
            custom_commands: Additional commands
            
        Returns:
            Dockerfile content
        """
        dockerfile = f"""FROM {base_image}

# Set working directory
WORKDIR /app

# Install system packages
"""
        
        if system_packages:
            packages_str = ' '.join(system_packages)
            dockerfile += f"""RUN apt-get update && apt-get install -y \\
    {packages_str} \\
    && rm -rf /var/lib/apt/lists/*

"""
        
        # Install Python packages
        if python_packages:
            packages_str = ' '.join(python_packages)
            dockerfile += f"""# Install Python packages
RUN pip install --no-cache-dir \\
    {packages_str}

"""
        
        # Add custom commands
        if custom_commands:
            dockerfile += "# Custom commands\n"
            for cmd in custom_commands:
                dockerfile += f"RUN {cmd}\n"
            dockerfile += "\n"
        
        # Copy application code
        dockerfile += """# Copy application code
COPY . /app

# Set entrypoint
ENTRYPOINT ["python", "train.py"]
"""
        
        return dockerfile
    
    def build_and_push(
        self,
        dockerfile_path: str,
        image_name: str,
        image_tag: str = 'latest'
    ) -> str:
        """
        Build and push container image.
        
        Args:
            dockerfile_path: Path to Dockerfile
            image_name: Image name
            image_tag: Image tag
            
        Returns:
            Full image URI
        """
        full_image_uri = f'{self.registry}/{image_name}:{image_tag}'
        
        # Build image
        build_cmd = [
            'docker', 'build',
            '-t', full_image_uri,
            '-f', dockerfile_path,
            '.'
        ]
        
        print(f"Building image: {full_image_uri}")
        subprocess.run(build_cmd, check=True)
        
        # Push image
        push_cmd = ['docker', 'push', full_image_uri]
        
        print(f"Pushing image: {full_image_uri}")
        subprocess.run(push_cmd, check=True)
        
        return full_image_uri
    
    def build_with_cloud_build(
        self,
        source_path: str,
        image_name: str,
        image_tag: str = 'latest'
    ) -> str:
        """
        Build container using Cloud Build.
        
        Args:
            source_path: Path to source code
            image_name: Image name
            image_tag: Image tag
            
        Returns:
            Full image URI
        """
        full_image_uri = f'{self.registry}/{image_name}:{image_tag}'
        
        # Cloud Build command
        build_cmd = [
            'gcloud', 'builds', 'submit',
            '--tag', full_image_uri,
            source_path
        ]
        
        print(f"Building with Cloud Build: {full_image_uri}")
        subprocess.run(build_cmd, check=True)
        
        return full_image_uri


# Example usage
builder = CustomContainerBuilder(
    project_id='my-project',
    region='us-central1'
)

# Generate Dockerfile
dockerfile_content = builder.generate_dockerfile(
    base_image='us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-13:latest',
    python_packages=[
        'transformers==4.30.0',
        'datasets==2.12.0',
        'wandb==0.15.0',
        'mlflow==2.3.0'
    ],
    system_packages=['git', 'vim'],
    custom_commands=[
        'mkdir -p /app/models',
        'mkdir -p /app/data'
    ]
)

print("Generated Dockerfile:")
print(dockerfile_content)

# Build and push (using Cloud Build)
image_uri = builder.build_with_cloud_build(
    source_path='./training-code',
    image_name='custom-training',
    image_tag='v1.0'
)
print(f"\nBuilt image: {image_uri}")
```

## 4. Container Orchestration

### 4.1 Vertex AI Custom Training

```python
from google.cloud import aiplatform

class VertexAIContainerManager:
    """Manager for running containers on Vertex AI."""
    
    def __init__(self, project_id: str, location: str = 'us-central1'):
        """Initialize Vertex AI Container Manager."""
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def run_custom_training_job(
        self,
        display_name: str,
        container_uri: str,
        args: List[str] = None,
        replica_count: int = 1,
        machine_type: str = 'n1-standard-4',
        accelerator_type: str = None,
        accelerator_count: int = 0
    ) -> str:
        """
        Run custom training job with container.
        
        Args:
            display_name: Job display name
            container_uri: Container image URI
            args: Command arguments
            replica_count: Number of replicas
            machine_type: Machine type
            accelerator_type: Accelerator type (e.g., 'NVIDIA_TESLA_T4')
            accelerator_count: Number of accelerators
            
        Returns:
            Job resource name
        """
        # Create custom job
        job = aiplatform.CustomJob(
            display_name=display_name,
            worker_pool_specs=[{
                'machine_spec': {
                    'machine_type': machine_type,
                    'accelerator_type': accelerator_type,
                    'accelerator_count': accelerator_count
                } if accelerator_type else {
                    'machine_type': machine_type
                },
                'replica_count': replica_count,
                'container_spec': {
                    'image_uri': container_uri,
                    'args': args or []
                }
            }]
        )
        
        # Run job
        job.run(sync=False)
        
        print(f"Training job started: {job.resource_name}")
        return job.resource_name


# Example usage
vertex_manager = VertexAIContainerManager(
    project_id='my-project',
    location='us-central1'
)

# Run custom training job
job = vertex_manager.run_custom_training_job(
    display_name='image-classification-training',
    container_uri='us-docker.pkg.dev/my-project/ml/custom-training:v1.0',
    args=['--epochs=50', '--batch-size=32'],
    replica_count=1,
    machine_type='n1-standard-8',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1
)
```

## 5. Quick Reference Checklist

### Container Selection
- [ ] Choose appropriate base image
- [ ] Select correct framework version
- [ ] Determine training vs prediction needs
- [ ] Check GPU/CPU requirements
- [ ] Verify compatibility with target platform
- [ ] Review image size and layers

### Custom Containers
- [ ] Start with official base images
- [ ] Minimize image size
- [ ] Use multi-stage builds
- [ ] Install only required dependencies
- [ ] Set appropriate permissions
- [ ] Add health checks

### GKE Deployment
- [ ] Configure resource requests/limits
- [ ] Set up node pools (CPU/GPU)
- [ ] Enable cluster autoscaling
- [ ] Configure horizontal pod autoscaling
- [ ] Set up load balancing
- [ ] Enable monitoring and logging

### Security
- [ ] Use Artifact Registry
- [ ] Scan images for vulnerabilities
- [ ] Use service accounts
- [ ] Implement network policies
- [ ] Enable Binary Authorization
- [ ] Regular security updates

### Performance
- [ ] Use appropriate resource limits
- [ ] Enable GPU sharing (if applicable)
- [ ] Optimize container startup time
- [ ] Use init containers for setup
- [ ] Implement caching strategies
- [ ] Monitor resource utilization

### Best Practices
- [ ] Version all container images
- [ ] Use semantic versioning
- [ ] Implement CI/CD for containers
- [ ] Document container usage
- [ ] Test containers locally first
- [ ] Use environment variables for config
- [ ] Implement proper logging
- [ ] Set up alerting and monitoring
