# Best Practices for GPU Support on Google Cloud

## Overview

Google Cloud provides GPU support for accelerating machine learning training and inference workloads. GPUs are well-suited for parallel processing tasks common in deep learning, computer vision, and scientific computing. This guide covers GPU selection, configuration, optimization, and best practices.

## 1. GPU Types and Selection

### 1.1 Available GPU Types

```python
from google.cloud import compute_v1
from typing import Dict, Any, List

class GPUManager:
    """Manager for GPU operations on GCP."""
    
    def __init__(self, project_id: str, zone: str = 'us-central1-a'):
        """
        Initialize GPU Manager.
        
        Args:
            project_id: GCP project ID
            zone: GCP zone
        """
        self.project_id = project_id
        self.zone = zone
    
    def get_gpu_specifications(self) -> Dict[str, Dict[str, Any]]:
        """
        Get specifications for available GPU types.
        
        Returns:
            Dictionary with GPU specifications
        """
        gpu_specs = {
            'nvidia-tesla-t4': {
                'memory_gb': 16,
                'cuda_cores': 2560,
                'tensor_cores': 320,
                'fp32_tflops': 8.1,
                'fp16_tflops': 65,
                'int8_tops': 130,
                'use_cases': ['Inference', 'Training (small models)', 'Mixed precision'],
                'cost_per_hour': 0.35,  # Approximate
                'recommended_for': 'Cost-effective inference and light training'
            },
            'nvidia-tesla-v100': {
                'memory_gb': 16,
                'cuda_cores': 5120,
                'tensor_cores': 640,
                'fp32_tflops': 15.7,
                'fp16_tflops': 125,
                'int8_tops': 250,
                'use_cases': ['Training', 'HPC', 'Large models'],
                'cost_per_hour': 2.48,  # Approximate
                'recommended_for': 'High-performance training'
            },
            'nvidia-tesla-p4': {
                'memory_gb': 8,
                'cuda_cores': 2560,
                'tensor_cores': 0,
                'fp32_tflops': 5.5,
                'fp16_tflops': 22,
                'int8_tops': 44,
                'use_cases': ['Inference', 'Video transcoding'],
                'cost_per_hour': 0.60,  # Approximate
                'recommended_for': 'Inference optimization'
            },
            'nvidia-tesla-p100': {
                'memory_gb': 16,
                'cuda_cores': 3584,
                'tensor_cores': 0,
                'fp32_tflops': 10.6,
                'fp16_tflops': 21.2,
                'use_cases': ['Training', 'HPC'],
                'cost_per_hour': 1.46,  # Approximate
                'recommended_for': 'General training workloads'
            },
            'nvidia-tesla-k80': {
                'memory_gb': 24,
                'cuda_cores': 4992,
                'tensor_cores': 0,
                'fp32_tflops': 8.73,
                'use_cases': ['Training (legacy)', 'HPC'],
                'cost_per_hour': 0.45,  # Approximate
                'recommended_for': 'Budget training (legacy)'
            },
            'nvidia-tesla-a100': {
                'memory_gb': 40,  # Also available in 80GB
                'cuda_cores': 6912,
                'tensor_cores': 432,
                'fp32_tflops': 19.5,
                'fp16_tflops': 312,
                'int8_tops': 624,
                'use_cases': ['Large model training', 'Multi-instance GPU', 'HPC'],
                'cost_per_hour': 3.67,  # Approximate
                'recommended_for': 'Largest models and fastest training'
            },
            'nvidia-l4': {
                'memory_gb': 24,
                'cuda_cores': 7424,
                'tensor_cores': 240,
                'fp32_tflops': 30,
                'fp16_tflops': 120,
                'int8_tops': 480,
                'use_cases': ['AI inference', 'Video AI', 'Graphics'],
                'cost_per_hour': 0.85,  # Approximate
                'recommended_for': 'Modern inference workloads'
            }
        }
        
        return gpu_specs
    
    def recommend_gpu(
        self,
        workload_type: str,
        model_size_gb: float,
        batch_size: int,
        budget_per_hour: float = None
    ) -> List[Dict[str, Any]]:
        """
        Recommend GPU based on workload requirements.
        
        Args:
            workload_type: 'training' or 'inference'
            model_size_gb: Model size in GB
            batch_size: Batch size
            budget_per_hour: Budget constraint per hour
            
        Returns:
            List of recommended GPUs
        """
        gpu_specs = self.get_gpu_specifications()
        recommendations = []
        
        for gpu_type, specs in gpu_specs.items():
            # Check memory requirements
            if specs['memory_gb'] < model_size_gb * 1.5:  # 1.5x for activations
                continue
            
            # Check workload type
            if workload_type.lower() not in [use.lower() for use in specs['use_cases']]:
                continue
            
            # Check budget
            if budget_per_hour and specs['cost_per_hour'] > budget_per_hour:
                continue
            
            score = 0
            
            # Score based on performance/cost ratio
            if workload_type == 'training':
                performance = specs['fp32_tflops']
            else:  # inference
                performance = specs.get('int8_tops', specs['fp16_tflops'])
            
            perf_cost_ratio = performance / specs['cost_per_hour']
            score += perf_cost_ratio
            
            # Bonus for tensor cores
            if specs['tensor_cores'] > 0:
                score += 10
            
            recommendations.append({
                'gpu_type': gpu_type,
                'score': score,
                'memory_gb': specs['memory_gb'],
                'cost_per_hour': specs['cost_per_hour'],
                'performance': performance,
                'recommended_for': specs['recommended_for']
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:3]  # Top 3 recommendations


# Example usage
gpu_manager = GPUManager(
    project_id='my-project',
    zone='us-central1-a'
)

# Get GPU specs
specs = gpu_manager.get_gpu_specifications()
print("Available GPUs:")
for gpu_type, spec in specs.items():
    print(f"{gpu_type}: {spec['memory_gb']}GB, {spec['fp32_tflops']} TFLOPS")

# Get recommendations
recommendations = gpu_manager.recommend_gpu(
    workload_type='training',
    model_size_gb=5.0,
    batch_size=32,
    budget_per_hour=2.0
)

print("\nTop Recommendations:")
for rec in recommendations:
    print(f"{rec['gpu_type']}: Score={rec['score']:.2f}, "
          f"Cost=${rec['cost_per_hour']}/hr")
```

## 2. GPU Instance Setup

### 2.1 Creating GPU Instances

```python
from google.cloud import aiplatform
from google.cloud import compute_v1
import time

class GPUInstanceManager:
    """Manager for GPU instance operations."""
    
    def __init__(self, project_id: str, zone: str = 'us-central1-a'):
        """
        Initialize GPU Instance Manager.
        
        Args:
            project_id: GCP project ID
            zone: GCP zone
        """
        self.project_id = project_id
        self.zone = zone
        self.instances_client = compute_v1.InstancesClient()
    
    def create_gpu_instance(
        self,
        instance_name: str,
        machine_type: str = 'n1-standard-8',
        gpu_type: str = 'nvidia-tesla-t4',
        gpu_count: int = 1,
        boot_disk_size_gb: int = 100,
        preemptible: bool = False
    ) -> str:
        """
        Create a GPU-enabled VM instance.
        
        Args:
            instance_name: Name for the instance
            machine_type: Machine type
            gpu_type: GPU type
            gpu_count: Number of GPUs
            boot_disk_size_gb: Boot disk size
            preemptible: Whether to use preemptible instance
            
        Returns:
            Instance resource name
        """
        # Configure boot disk
        disk = compute_v1.AttachedDisk(
            initialize_params=compute_v1.AttachedDiskInitializeParams(
                source_image='projects/ml-images/global/images/family/common-cu113',
                disk_size_gb=boot_disk_size_gb
            ),
            boot=True,
            auto_delete=True
        )
        
        # Configure GPU
        accelerator = compute_v1.AcceleratorConfig(
            accelerator_count=gpu_count,
            accelerator_type=f"projects/{self.project_id}/zones/{self.zone}/acceleratorTypes/{gpu_type}"
        )
        
        # Configure network interface
        network_interface = compute_v1.NetworkInterface(
            name="global/networks/default",
            access_configs=[compute_v1.AccessConfig(
                name="External NAT",
                type_="ONE_TO_ONE_NAT"
            )]
        )
        
        # Configure scheduling
        scheduling = compute_v1.Scheduling(
            preemptible=preemptible,
            on_host_maintenance="TERMINATE" if not preemptible else "TERMINATE"
        )
        
        # Create instance configuration
        instance = compute_v1.Instance(
            name=instance_name,
            machine_type=f"zones/{self.zone}/machineTypes/{machine_type}",
            disks=[disk],
            guest_accelerators=[accelerator],
            network_interfaces=[network_interface],
            scheduling=scheduling,
            metadata=compute_v1.Metadata(
                items=[
                    compute_v1.Items(
                        key="install-nvidia-driver",
                        value="True"
                    )
                ]
            )
        )
        
        # Create instance
        request = compute_v1.InsertInstanceRequest(
            project=self.project_id,
            zone=self.zone,
            instance_resource=instance
        )
        
        print(f"Creating GPU instance {instance_name}...")
        operation = self.instances_client.insert(request=request)
        
        # Wait for completion
        self._wait_for_operation(operation)
        
        print(f"Instance {instance_name} created successfully")
        return f"projects/{self.project_id}/zones/{self.zone}/instances/{instance_name}"
    
    def _wait_for_operation(self, operation, timeout: int = 300):
        """Wait for operation to complete."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if operation.status == compute_v1.Operation.Status.DONE:
                if operation.error:
                    raise Exception(f"Operation failed: {operation.error}")
                return
            time.sleep(5)
        raise TimeoutError("Operation timed out")


# Example usage
instance_manager = GPUInstanceManager(
    project_id='my-project',
    zone='us-central1-a'
)

# Create GPU instance
instance = instance_manager.create_gpu_instance(
    instance_name='gpu-training-vm',
    machine_type='n1-standard-8',
    gpu_type='nvidia-tesla-t4',
    gpu_count=1,
    preemptible=False
)
```

## 3. Multi-GPU Training

### 3.1 TensorFlow Multi-GPU Strategy

```python
import tensorflow as tf
from tensorflow import keras
from typing import Optional, List

class MultiGPUTrainer:
    """Trainer for multi-GPU operations."""
    
    def __init__(self):
        """Initialize Multi-GPU Trainer."""
        self.strategy = None
        self.num_gpus = 0
    
    def setup_gpu_strategy(
        self,
        strategy_type: str = 'mirrored'
    ) -> tf.distribute.Strategy:
        """
        Set up GPU distribution strategy.
        
        Args:
            strategy_type: 'mirrored' or 'multi_worker_mirrored'
            
        Returns:
            Distribution strategy
        """
        # List available GPUs
        gpus = tf.config.list_physical_devices('GPU')
        self.num_gpus = len(gpus)
        
        print(f"Number of GPUs available: {self.num_gpus}")
        
        if self.num_gpus == 0:
            print("No GPUs found, using default strategy")
            self.strategy = tf.distribute.get_strategy()
        elif self.num_gpus == 1:
            print("Single GPU found")
            self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            if strategy_type == 'mirrored':
                print(f"Using MirroredStrategy across {self.num_gpus} GPUs")
                self.strategy = tf.distribute.MirroredStrategy()
            else:
                print("Using MultiWorkerMirroredStrategy")
                self.strategy = tf.distribute.MultiWorkerMirroredStrategy()
        
        return self.strategy
    
    def configure_gpu_memory(self, memory_limit_mb: Optional[int] = None):
        """
        Configure GPU memory settings.
        
        Args:
            memory_limit_mb: Memory limit in MB (None for dynamic growth)
        """
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            try:
                for gpu in gpus:
                    if memory_limit_mb:
                        tf.config.set_logical_device_configuration(
                            gpu,
                            [tf.config.LogicalDeviceConfiguration(
                                memory_limit=memory_limit_mb
                            )]
                        )
                    else:
                        # Enable memory growth
                        tf.config.experimental.set_memory_growth(gpu, True)
                
                print(f"GPU memory configured for {len(gpus)} GPUs")
            except RuntimeError as e:
                print(f"Memory config error: {e}")
    
    def create_model_for_multi_gpu(
        self,
        input_shape: tuple,
        num_classes: int
    ) -> keras.Model:
        """
        Create model optimized for multi-GPU training.
        
        Args:
            input_shape: Input shape
            num_classes: Number of classes
            
        Returns:
            Keras model
        """
        if self.strategy is None:
            self.setup_gpu_strategy()
        
        with self.strategy.scope():
            # Use mixed precision for better GPU utilization
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            
            model = keras.Sequential([
                keras.layers.Conv2D(64, 3, activation='relu', input_shape=input_shape),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(),
                
                keras.layers.Conv2D(128, 3, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling2D(),
                
                keras.layers.Conv2D(256, 3, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.GlobalAveragePooling2D(),
                
                keras.layers.Dense(512, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(num_classes, dtype='float32')  # Keep output in float32
            ])
            
            # Scale learning rate with number of GPUs
            base_learning_rate = 0.001
            learning_rate = base_learning_rate * self.num_gpus
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
        
        return model
    
    def train_multi_gpu(
        self,
        model: keras.Model,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int = 10
    ):
        """
        Train model across multiple GPUs.
        
        Args:
            model: Keras model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs
        """
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath='./checkpoints/model_{epoch:02d}.h5',
                save_freq='epoch'
            ),
            keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        # Train
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        return history


# Example usage
trainer = MultiGPUTrainer()

# Configure GPU memory
trainer.configure_gpu_memory(memory_limit_mb=None)  # Dynamic growth

# Setup strategy
strategy = trainer.setup_gpu_strategy(strategy_type='mirrored')

# Create model
model = trainer.create_model_for_multi_gpu(
    input_shape=(224, 224, 3),
    num_classes=1000
)

# Prepare dataset
def create_dataset(file_pattern, batch_size):
    dataset = tf.data.TFRecordDataset(file_pattern)
    dataset = dataset.map(parse_function)
    dataset = dataset.batch(batch_size * trainer.num_gpus)  # Scale batch size
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset('train/*.tfrecord', batch_size=32)
val_dataset = create_dataset('val/*.tfrecord', batch_size=32)

# Train
history = trainer.train_multi_gpu(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=10
)
```

## 4. GPU Optimization

### 4.1 Performance Optimization

```python
import tensorflow as tf

class GPUOptimizer:
    """Optimizer for GPU performance."""
    
    @staticmethod
    def enable_mixed_precision():
        """Enable mixed precision training."""
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        print(f"Compute dtype: {policy.compute_dtype}")
        print(f"Variable dtype: {policy.variable_dtype}")
    
    @staticmethod
    def enable_xla_compilation():
        """Enable XLA (Accelerated Linear Algebra) compilation."""
        tf.config.optimizer.set_jit(True)
        print("XLA compilation enabled")
    
    @staticmethod
    def configure_gpu_options():
        """Configure optimal GPU options."""
        # Allow GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Enable tensor float 32 on Ampere GPUs
        tf.config.experimental.enable_tensor_float_32_execution(True)
        
        print("GPU options configured")
    
    @staticmethod
    def optimize_dataset_for_gpu(
        dataset: tf.data.Dataset,
        batch_size: int,
        prefetch_size: int = tf.data.AUTOTUNE
    ) -> tf.data.Dataset:
        """
        Optimize dataset for GPU training.
        
        Args:
            dataset: Input dataset
            batch_size: Batch size
            prefetch_size: Prefetch buffer size
            
        Returns:
            Optimized dataset
        """
        # Apply optimizations
        dataset = dataset.cache()
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_size)
        
        return dataset


# Example usage
optimizer = GPUOptimizer()

# Enable all optimizations
optimizer.enable_mixed_precision()
optimizer.enable_xla_compilation()
optimizer.configure_gpu_options()

# Optimize dataset
optimized_dataset = optimizer.optimize_dataset_for_gpu(
    dataset=raw_dataset,
    batch_size=128
)
```

## 5. Quick Reference Checklist

### GPU Selection
- [ ] Choose GPU based on workload (training vs inference)
- [ ] Consider memory requirements
- [ ] Evaluate cost vs performance
- [ ] Check GPU availability in target zone
- [ ] Consider preemptible GPUs for cost savings
- [ ] Verify CUDA/cuDNN compatibility

### Instance Setup
- [ ] Select appropriate machine type
- [ ] Configure adequate CPU and RAM
- [ ] Install NVIDIA drivers
- [ ] Install CUDA toolkit
- [ ] Install cuDNN library
- [ ] Set up monitoring

### Training Optimization
- [ ] Use mixed precision (float16)
- [ ] Enable XLA compilation
- [ ] Scale batch size with GPU count
- [ ] Use data parallelism
- [ ] Optimize data pipeline
- [ ] Monitor GPU utilization

### Multi-GPU Configuration
- [ ] Use MirroredStrategy for single-node
- [ ] Scale learning rate with GPU count
- [ ] Ensure batch size divisible by GPU count
- [ ] Synchronize batch normalization
- [ ] Monitor memory across GPUs
- [ ] Balance workload distribution

### Performance Monitoring
- [ ] Monitor GPU utilization (nvidia-smi)
- [ ] Track memory usage
- [ ] Monitor SM (Streaming Multiprocessor) activity
- [ ] Check for memory bottlenecks
- [ ] Profile kernel execution
- [ ] Monitor PCIe bandwidth

### Cost Optimization
- [ ] Use preemptible instances when possible
- [ ] Right-size GPU selection
- [ ] Clean up idle resources
- [ ] Use committed use discounts
- [ ] Consider spot instances
- [ ] Monitor usage and costs

### Best Practices
- [ ] Start with single GPU for debugging
- [ ] Use gradient checkpointing for large models
- [ ] Implement efficient data loading
- [ ] Use tf.function for graph optimization
- [ ] Enable automatic mixed precision
- [ ] Save checkpoints frequently
