# Best Practices for Deep Learning VM Images on Google Cloud

## Overview

Deep Learning VM Images are pre-configured virtual machine images optimized for machine learning and data science workloads. They come with popular frameworks (TensorFlow, PyTorch, JAX), CUDA drivers, and development tools pre-installed, enabling rapid development and deployment of ML models.

## 1. VM Image Selection

### 1.1 Available Framework Images

```python
from google.cloud import compute_v1
from typing import Dict, Any, List

class DeepLearningVMManager:
    """Manager for Deep Learning VM Images."""
    
    def __init__(self, project_id: str, zone: str = 'us-central1-a'):
        """
        Initialize Deep Learning VM Manager.
        
        Args:
            project_id: GCP project ID
            zone: GCP zone
        """
        self.project_id = project_id
        self.zone = zone
        self.compute_client = compute_v1.InstancesClient()
    
    def get_available_images(self) -> Dict[str, Dict[str, Any]]:
        """
        Get available Deep Learning VM images.
        
        Returns:
            Dictionary with image information
        """
        images = {
            'tf-latest-gpu': {
                'framework': 'TensorFlow',
                'version': '2.13',
                'python': '3.10',
                'cuda': '11.8',
                'accelerator': 'GPU',
                'pre_installed': [
                    'TensorFlow 2.13',
                    'Keras',
                    'Jupyter Lab',
                    'CUDA 11.8',
                    'cuDNN 8.6'
                ],
                'image_family': 'tf-latest-gpu',
                'project': 'deeplearning-platform-release'
            },
            'tf-latest-cpu': {
                'framework': 'TensorFlow',
                'version': '2.13',
                'python': '3.10',
                'cuda': None,
                'accelerator': 'CPU',
                'pre_installed': [
                    'TensorFlow 2.13 (CPU)',
                    'Keras',
                    'Jupyter Lab'
                ],
                'image_family': 'tf-latest-cpu',
                'project': 'deeplearning-platform-release'
            },
            'pytorch-latest-gpu': {
                'framework': 'PyTorch',
                'version': '2.0',
                'python': '3.10',
                'cuda': '11.8',
                'accelerator': 'GPU',
                'pre_installed': [
                    'PyTorch 2.0',
                    'torchvision',
                    'Jupyter Lab',
                    'CUDA 11.8',
                    'cuDNN 8.6'
                ],
                'image_family': 'pytorch-latest-gpu',
                'project': 'deeplearning-platform-release'
            },
            'pytorch-latest-cpu': {
                'framework': 'PyTorch',
                'version': '2.0',
                'python': '3.10',
                'cuda': None,
                'accelerator': 'CPU',
                'pre_installed': [
                    'PyTorch 2.0 (CPU)',
                    'torchvision',
                    'Jupyter Lab'
                ],
                'image_family': 'pytorch-latest-cpu',
                'project': 'deeplearning-platform-release'
            },
            'common-cu113': {
                'framework': 'Multi-framework',
                'version': 'Latest',
                'python': '3.10',
                'cuda': '11.3',
                'accelerator': 'GPU',
                'pre_installed': [
                    'TensorFlow',
                    'PyTorch',
                    'JAX',
                    'Jupyter Lab',
                    'CUDA 11.3'
                ],
                'image_family': 'common-cu113',
                'project': 'deeplearning-platform-release'
            },
            'common-cpu': {
                'framework': 'Multi-framework',
                'version': 'Latest',
                'python': '3.10',
                'cuda': None,
                'accelerator': 'CPU',
                'pre_installed': [
                    'TensorFlow (CPU)',
                    'PyTorch (CPU)',
                    'Jupyter Lab',
                    'scikit-learn',
                    'pandas'
                ],
                'image_family': 'common-cpu',
                'project': 'deeplearning-platform-release'
            }
        }
        
        return images
    
    def recommend_image(
        self,
        framework: str,
        use_gpu: bool = True,
        multi_framework: bool = False
    ) -> List[str]:
        """
        Recommend VM image based on requirements.
        
        Args:
            framework: Desired framework ('tensorflow', 'pytorch', 'jax')
            use_gpu: Whether GPU support is needed
            multi_framework: Whether multiple frameworks needed
            
        Returns:
            List of recommended image families
        """
        images = self.get_available_images()
        recommendations = []
        
        if multi_framework:
            if use_gpu:
                recommendations.append('common-cu113')
            else:
                recommendations.append('common-cpu')
        else:
            framework_lower = framework.lower()
            accelerator = 'gpu' if use_gpu else 'cpu'
            
            for image_name, image_info in images.items():
                if (framework_lower in image_info['framework'].lower() and
                    image_info['accelerator'].lower() == accelerator):
                    recommendations.append(image_name)
        
        return recommendations
    
    def create_deep_learning_vm(
        self,
        instance_name: str,
        image_family: str = 'tf-latest-gpu',
        machine_type: str = 'n1-standard-8',
        gpu_type: str = 'nvidia-tesla-t4',
        gpu_count: int = 1,
        boot_disk_size_gb: int = 100
    ) -> str:
        """
        Create a Deep Learning VM instance.
        
        Args:
            instance_name: Name for the instance
            image_family: Image family to use
            machine_type: Machine type
            gpu_type: GPU type (if using GPU image)
            gpu_count: Number of GPUs
            boot_disk_size_gb: Boot disk size
            
        Returns:
            Instance resource name
        """
        # Get latest image from family
        image_project = 'deeplearning-platform-release'
        source_image = f"projects/{image_project}/global/images/family/{image_family}"
        
        # Configure boot disk
        disk = compute_v1.AttachedDisk(
            initialize_params=compute_v1.AttachedDiskInitializeParams(
                source_image=source_image,
                disk_size_gb=boot_disk_size_gb
            ),
            boot=True,
            auto_delete=True
        )
        
        # Configure network
        network_interface = compute_v1.NetworkInterface(
            network="global/networks/default",
            access_configs=[compute_v1.AccessConfig(
                name="External NAT",
                type_="ONE_TO_ONE_NAT"
            )]
        )
        
        # Configure instance
        instance = compute_v1.Instance(
            name=instance_name,
            machine_type=f"zones/{self.zone}/machineTypes/{machine_type}",
            disks=[disk],
            network_interfaces=[network_interface]
        )
        
        # Add GPU if needed
        if 'gpu' in image_family:
            accelerator = compute_v1.AcceleratorConfig(
                accelerator_count=gpu_count,
                accelerator_type=f"projects/{self.project_id}/zones/{self.zone}/acceleratorTypes/{gpu_type}"
            )
            instance.guest_accelerators = [accelerator]
            instance.scheduling = compute_v1.Scheduling(
                on_host_maintenance="TERMINATE"
            )
        
        # Create instance
        request = compute_v1.InsertInstanceRequest(
            project=self.project_id,
            zone=self.zone,
            instance_resource=instance
        )
        
        print(f"Creating Deep Learning VM: {instance_name}")
        operation = self.compute_client.insert(request=request)
        
        return f"projects/{self.project_id}/zones/{self.zone}/instances/{instance_name}"


# Example usage
vm_manager = DeepLearningVMManager(
    project_id='my-project',
    zone='us-central1-a'
)

# Get available images
images = vm_manager.get_available_images()
print("Available Deep Learning VM Images:")
for image_name, info in images.items():
    print(f"  {image_name}: {info['framework']} {info['version']} ({info['accelerator']})")

# Get recommendations
recommendations = vm_manager.recommend_image(
    framework='tensorflow',
    use_gpu=True,
    multi_framework=False
)
print(f"\nRecommended images: {recommendations}")

# Create VM
instance = vm_manager.create_deep_learning_vm(
    instance_name='ml-training-vm',
    image_family='tf-latest-gpu',
    machine_type='n1-standard-8',
    gpu_type='nvidia-tesla-t4',
    gpu_count=1
)
```

## 2. VM Configuration and Setup

### 2.1 Post-Installation Configuration

```python
import subprocess
from typing import List

class VMConfigurationManager:
    """Manager for Deep Learning VM configuration."""
    
    @staticmethod
    def install_additional_packages(packages: List[str]):
        """
        Install additional Python packages.
        
        Args:
            packages: List of package names
        """
        for package in packages:
            try:
                subprocess.run(
                    ['pip', 'install', package],
                    check=True,
                    capture_output=True
                )
                print(f"✓ Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {package}: {e}")
    
    @staticmethod
    def configure_jupyter():
        """Configure Jupyter Lab settings."""
        config = """
# Jupyter Lab Configuration
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8080
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = ''
c.ServerApp.password = ''
"""
        
        config_path = '~/.jupyter/jupyter_lab_config.py'
        print(f"Jupyter Lab configuration saved to {config_path}")
        print("Run: jupyter lab --allow-root")
    
    @staticmethod
    def setup_ssh_tunnel(
        instance_name: str,
        zone: str,
        local_port: int = 8080,
        remote_port: int = 8080
    ) -> str:
        """
        Generate SSH tunnel command for Jupyter access.
        
        Args:
            instance_name: VM instance name
            zone: GCP zone
            local_port: Local port
            remote_port: Remote port
            
        Returns:
            SSH tunnel command
        """
        command = (
            f"gcloud compute ssh {instance_name} "
            f"--zone={zone} "
            f"-- -L {local_port}:localhost:{remote_port}"
        )
        
        return command
    
    @staticmethod
    def verify_gpu():
        """Verify GPU availability and configuration."""
        try:
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                text=True,
                check=True
            )
            print("GPU Configuration:")
            print(result.stdout)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("No GPU detected or nvidia-smi not available")
            return False
    
    @staticmethod
    def verify_cuda():
        """Verify CUDA installation."""
        try:
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            print("CUDA Version:")
            print(result.stdout)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("CUDA not found")
            return False


# Example usage
config_manager = VMConfigurationManager()

# Install additional packages
config_manager.install_additional_packages([
    'transformers',
    'datasets',
    'wandb',
    'mlflow'
])

# Configure Jupyter
config_manager.configure_jupyter()

# Generate SSH tunnel command
ssh_command = config_manager.setup_ssh_tunnel(
    instance_name='ml-training-vm',
    zone='us-central1-a',
    local_port=8080,
    remote_port=8080
)
print(f"\nSSH Tunnel Command:\n{ssh_command}")

# Verify GPU
config_manager.verify_gpu()

# Verify CUDA
config_manager.verify_cuda()
```

## 3. Development Workflow

### 3.1 Notebook-Based Development

```python
import tensorflow as tf
import torch
import jax
from typing import Dict, Any

class DevelopmentEnvironment:
    """Manager for development environment."""
    
    @staticmethod
    def check_framework_versions() -> Dict[str, str]:
        """
        Check installed framework versions.
        
        Returns:
            Dictionary with framework versions
        """
        versions = {}
        
        try:
            versions['tensorflow'] = tf.__version__
        except:
            versions['tensorflow'] = 'Not installed'
        
        try:
            versions['pytorch'] = torch.__version__
        except:
            versions['pytorch'] = 'Not installed'
        
        try:
            versions['jax'] = jax.__version__
        except:
            versions['jax'] = 'Not installed'
        
        return versions
    
    @staticmethod
    def check_gpu_availability() -> Dict[str, Any]:
        """
        Check GPU availability across frameworks.
        
        Returns:
            Dictionary with GPU availability info
        """
        gpu_info = {}
        
        # TensorFlow
        try:
            gpu_info['tensorflow'] = {
                'available': len(tf.config.list_physical_devices('GPU')) > 0,
                'devices': [device.name for device in tf.config.list_physical_devices('GPU')]
            }
        except:
            gpu_info['tensorflow'] = {'available': False}
        
        # PyTorch
        try:
            gpu_info['pytorch'] = {
                'available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count(),
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            }
        except:
            gpu_info['pytorch'] = {'available': False}
        
        # JAX
        try:
            from jax import devices
            gpu_info['jax'] = {
                'available': len([d for d in devices() if 'gpu' in str(d).lower()]) > 0,
                'devices': [str(d) for d in devices()]
            }
        except:
            gpu_info['jax'] = {'available': False}
        
        return gpu_info
    
    @staticmethod
    def setup_experiment_tracking(
        experiment_name: str,
        tracking_uri: str = None
    ):
        """
        Set up experiment tracking.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking server URI
        """
        try:
            import mlflow
            
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            
            mlflow.set_experiment(experiment_name)
            
            print(f"Experiment tracking configured: {experiment_name}")
            if tracking_uri:
                print(f"Tracking URI: {tracking_uri}")
        except ImportError:
            print("MLflow not installed. Run: pip install mlflow")


# Example usage
dev_env = DevelopmentEnvironment()

# Check framework versions
versions = dev_env.check_framework_versions()
print("Framework Versions:")
for framework, version in versions.items():
    print(f"  {framework}: {version}")

# Check GPU availability
gpu_info = dev_env.check_gpu_availability()
print("\nGPU Availability:")
for framework, info in gpu_info.items():
    print(f"  {framework}: {info['available']}")

# Setup experiment tracking
dev_env.setup_experiment_tracking(
    experiment_name='image-classification',
    tracking_uri='http://localhost:5000'
)
```

## 4. Data Management

### 4.1 Data Loading and Caching

```python
from google.cloud import storage
import os

class DataManager:
    """Manager for data operations on Deep Learning VMs."""
    
    def __init__(self, project_id: str):
        """
        Initialize Data Manager.
        
        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.storage_client = storage.Client(project=project_id)
    
    def download_dataset(
        self,
        gcs_path: str,
        local_path: str = '/data'
    ):
        """
        Download dataset from GCS.
        
        Args:
            gcs_path: GCS path (gs://bucket/path)
            local_path: Local directory path
        """
        # Parse GCS path
        path_parts = gcs_path.replace('gs://', '').split('/', 1)
        bucket_name = path_parts[0]
        prefix = path_parts[1] if len(path_parts) > 1 else ''
        
        bucket = self.storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        
        os.makedirs(local_path, exist_ok=True)
        
        for blob in blobs:
            local_file = os.path.join(local_path, blob.name.replace(prefix, '').lstrip('/'))
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            blob.download_to_filename(local_file)
            print(f"Downloaded: {blob.name}")
    
    def setup_gcsfuse(
        self,
        bucket_name: str,
        mount_point: str = '/gcs'
    ) -> str:
        """
        Set up Cloud Storage FUSE mount.
        
        Args:
            bucket_name: GCS bucket name
            mount_point: Local mount point
            
        Returns:
            Mount command
        """
        os.makedirs(mount_point, exist_ok=True)
        
        command = f"gcsfuse {bucket_name} {mount_point}"
        
        print(f"GCS FUSE mount command:")
        print(command)
        print(f"\nAccess bucket at: {mount_point}")
        
        return command


# Example usage
data_manager = DataManager(project_id='my-project')

# Download dataset
data_manager.download_dataset(
    gcs_path='gs://my-bucket/datasets/imagenet',
    local_path='/data/imagenet'
)

# Setup GCS FUSE
mount_cmd = data_manager.setup_gcsfuse(
    bucket_name='my-ml-data',
    mount_point='/gcs/ml-data'
)
```

## 5. Quick Reference Checklist

### VM Selection
- [ ] Choose appropriate image family
- [ ] Select correct framework version
- [ ] Determine GPU requirements
- [ ] Calculate storage needs
- [ ] Consider multi-framework needs
- [ ] Check CUDA/cuDNN compatibility

### Setup
- [ ] Create VM with appropriate resources
- [ ] Configure firewall rules
- [ ] Set up SSH access
- [ ] Install additional packages
- [ ] Configure Jupyter Lab
- [ ] Verify GPU availability

### Development
- [ ] Set up version control (Git)
- [ ] Configure experiment tracking
- [ ] Set up remote development (VS Code)
- [ ] Create virtual environments
- [ ] Install development tools
- [ ] Configure IDE settings

### Data Management
- [ ] Download datasets to local storage
- [ ] Set up GCS FUSE for large datasets
- [ ] Configure data pipeline
- [ ] Enable data versioning
- [ ] Implement caching strategy
- [ ] Monitor storage usage

### Security
- [ ] Use service accounts
- [ ] Configure IAM permissions
- [ ] Enable OS Login
- [ ] Set up private networking
- [ ] Regular security updates
- [ ] Enable audit logging

### Cost Optimization
- [ ] Use preemptible instances for development
- [ ] Stop VMs when not in use
- [ ] Use sustained use discounts
- [ ] Right-size machine types
- [ ] Monitor resource usage
- [ ] Set up budget alerts

### Best Practices
- [ ] Regular backups of notebooks
- [ ] Use startup scripts for configuration
- [ ] Save models to GCS
- [ ] Document dependencies
- [ ] Use metadata for instance info
- [ ] Implement checkpointing
