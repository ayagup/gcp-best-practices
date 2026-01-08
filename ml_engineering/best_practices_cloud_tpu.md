# Best Practices for Cloud TPU on Google Cloud

## Overview

Cloud TPUs (Tensor Processing Units) are Google's custom-developed application-specific integrated circuits (ASICs) designed to accelerate machine learning workloads. TPUs provide exceptional performance for training and inference of large-scale neural networks, particularly for operations involving matrix multiplication and convolution.

## 1. TPU Architecture and Types

### 1.1 Understanding TPU Versions

```python
from google.cloud import aiplatform
from typing import Dict, Any, List
import tensorflow as tf

class TPUManager:
    """Manager for Cloud TPU operations."""
    
    def __init__(
        self,
        project_id: str,
        location: str = 'us-central1'
    ):
        """
        Initialize TPU Manager.
        
        Args:
            project_id: GCP project ID
            location: GCP region with TPU availability
        """
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def get_tpu_specifications(self) -> Dict[str, Dict[str, Any]]:
        """
        Get specifications for different TPU versions.
        
        Returns:
            Dictionary with TPU specifications
        """
        tpu_specs = {
            'v2': {
                'cores': 8,
                'memory_gb': 8,
                'peak_performance_tflops': 180,
                'memory_bandwidth_gb_s': 600,
                'use_cases': ['Training', 'Inference'],
                'recommended_for': 'General ML workloads'
            },
            'v3': {
                'cores': 8,
                'memory_gb': 16,
                'peak_performance_tflops': 420,
                'memory_bandwidth_gb_s': 900,
                'use_cases': ['Training', 'Inference'],
                'recommended_for': 'Large models and batch training'
            },
            'v4': {
                'cores': 8,
                'memory_gb': 32,
                'peak_performance_tflops': 275,  # Per chip
                'memory_bandwidth_gb_s': 1200,
                'use_cases': ['Training', 'Inference'],
                'recommended_for': 'Very large models and distributed training'
            }
        }
        
        return tpu_specs
    
    def calculate_tpu_pods(
        self,
        model_parameters: int,
        batch_size: int,
        sequence_length: int = 512
    ) -> Dict[str, Any]:
        """
        Calculate recommended TPU pod configuration.
        
        Args:
            model_parameters: Number of model parameters
            batch_size: Training batch size
            sequence_length: Sequence length for transformers
            
        Returns:
            Dictionary with pod recommendations
        """
        # Estimate memory requirements (rough approximation)
        param_memory_gb = (model_parameters * 4) / (1024**3)  # 4 bytes per param
        activation_memory_gb = (batch_size * sequence_length * 1024 * 4) / (1024**3)
        total_memory_gb = param_memory_gb + activation_memory_gb
        
        recommendations = {}
        
        # Check against TPU versions
        tpu_specs = self.get_tpu_specifications()
        
        for version, specs in tpu_specs.items():
            # Calculate number of cores needed
            cores_needed = max(8, int(total_memory_gb / specs['memory_gb']) * 8)
            
            # Round up to valid pod sizes (8, 32, 128, 256, 512, 1024, 2048)
            valid_sizes = [8, 32, 128, 256, 512, 1024, 2048]
            pod_size = next((size for size in valid_sizes if size >= cores_needed), 2048)
            
            recommendations[version] = {
                'pod_size': pod_size,
                'total_memory_gb': pod_size / 8 * specs['memory_gb'],
                'estimated_cost_per_hour': self._estimate_tpu_cost(version, pod_size)
            }
        
        return recommendations
    
    def _estimate_tpu_cost(self, version: str, pod_size: int) -> float:
        """
        Estimate TPU cost per hour.
        
        Args:
            version: TPU version
            pod_size: Pod size (number of cores)
            
        Returns:
            Estimated cost per hour in USD
        """
        # Approximate pricing (check current GCP pricing)
        base_costs = {
            'v2': 4.50,  # per 8 cores per hour
            'v3': 8.00,
            'v4': 10.00
        }
        
        base_cost = base_costs.get(version, 8.00)
        num_chips = pod_size / 8
        
        return base_cost * num_chips


# Example usage
tpu_manager = TPUManager(
    project_id='my-project',
    location='us-central1'
)

# Get specifications
specs = tpu_manager.get_tpu_specifications()
print(f"TPU v3 Performance: {specs['v3']['peak_performance_tflops']} TFLOPS")

# Calculate pod size
recommendations = tpu_manager.calculate_tpu_pods(
    model_parameters=175_000_000_000,  # 175B parameters (GPT-3 scale)
    batch_size=32,
    sequence_length=2048
)

for version, config in recommendations.items():
    print(f"\nTPU {version}:")
    print(f"  Pod Size: {config['pod_size']} cores")
    print(f"  Total Memory: {config['total_memory_gb']} GB")
    print(f"  Estimated Cost: ${config['estimated_cost_per_hour']:.2f}/hour")
```

## 2. TPU Setup and Configuration

### 2.1 Creating TPU VMs

```python
from google.cloud import tpu_v2
from google.api_core import operation
import time

class TPUSetupManager:
    """Manager for TPU setup operations."""
    
    def __init__(
        self,
        project_id: str,
        zone: str = 'us-central1-a'
    ):
        """
        Initialize TPU Setup Manager.
        
        Args:
            project_id: GCP project ID
            zone: GCP zone
        """
        self.project_id = project_id
        self.zone = zone
        self.client = tpu_v2.TpuClient()
    
    def create_tpu_vm(
        self,
        tpu_name: str,
        accelerator_type: str = 'v3-8',
        runtime_version: str = 'tpu-vm-tf-2.13.0',
        preemptible: bool = False
    ) -> str:
        """
        Create a TPU VM.
        
        Args:
            tpu_name: Name for the TPU
            accelerator_type: TPU type (e.g., 'v3-8', 'v3-32')
            runtime_version: TPU runtime version
            preemptible: Whether to use preemptible TPU
            
        Returns:
            TPU resource name
        """
        parent = f"projects/{self.project_id}/locations/{self.zone}"
        
        tpu_config = {
            'accelerator_type': accelerator_type,
            'runtime_version': runtime_version,
            'network_config': {
                'enable_external_ips': True
            }
        }
        
        if preemptible:
            tpu_config['scheduling_config'] = {
                'preemptible': True
            }
        
        node = tpu_v2.Node(tpu_config)
        
        request = tpu_v2.CreateNodeRequest(
            parent=parent,
            node_id=tpu_name,
            node=node
        )
        
        print(f"Creating TPU {tpu_name}...")
        operation_obj = self.client.create_node(request=request)
        
        # Wait for operation to complete
        result = operation_obj.result(timeout=600)
        
        print(f"TPU {tpu_name} created successfully")
        return result.name
    
    def list_tpu_vms(self) -> List[Dict[str, Any]]:
        """
        List all TPU VMs in the zone.
        
        Returns:
            List of TPU VM information
        """
        parent = f"projects/{self.project_id}/locations/{self.zone}"
        
        request = tpu_v2.ListNodesRequest(parent=parent)
        page_result = self.client.list_nodes(request=request)
        
        tpus = []
        for node in page_result:
            tpus.append({
                'name': node.name,
                'state': node.state.name,
                'accelerator_type': node.accelerator_type,
                'runtime_version': node.runtime_version,
                'cidr_block': node.cidr_block,
                'health': node.health.name if hasattr(node, 'health') else 'UNKNOWN'
            })
        
        return tpus
    
    def delete_tpu_vm(self, tpu_name: str):
        """
        Delete a TPU VM.
        
        Args:
            tpu_name: Name of the TPU to delete
        """
        name = f"projects/{self.project_id}/locations/{self.zone}/nodes/{tpu_name}"
        
        request = tpu_v2.DeleteNodeRequest(name=name)
        
        print(f"Deleting TPU {tpu_name}...")
        operation_obj = self.client.delete_node(request=request)
        operation_obj.result(timeout=600)
        
        print(f"TPU {tpu_name} deleted successfully")
    
    def get_tpu_info(self, tpu_name: str) -> Dict[str, Any]:
        """
        Get information about a specific TPU.
        
        Args:
            tpu_name: Name of the TPU
            
        Returns:
            Dictionary with TPU information
        """
        name = f"projects/{self.project_id}/locations/{self.zone}/nodes/{tpu_name}"
        
        request = tpu_v2.GetNodeRequest(name=name)
        node = self.client.get_node(request=request)
        
        return {
            'name': node.name,
            'state': node.state.name,
            'accelerator_type': node.accelerator_type,
            'runtime_version': node.runtime_version,
            'network_endpoints': [
                {
                    'ip_address': endpoint.ip_address,
                    'port': endpoint.port
                }
                for endpoint in node.network_endpoints
            ],
            'health': node.health.name if hasattr(node, 'health') else 'UNKNOWN'
        }


# Example usage
tpu_setup = TPUSetupManager(
    project_id='my-project',
    zone='us-central1-a'
)

# Create TPU VM
tpu_name = tpu_setup.create_tpu_vm(
    tpu_name='my-tpu-vm',
    accelerator_type='v3-8',
    runtime_version='tpu-vm-tf-2.13.0',
    preemptible=False
)

# List TPUs
tpus = tpu_setup.list_tpu_vms()
for tpu in tpus:
    print(f"TPU: {tpu['name']}, State: {tpu['state']}")
```

### 2.2 TPU Training Configuration

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Optional

class TPUTrainingManager:
    """Manager for TPU training operations."""
    
    def __init__(self, tpu_name: Optional[str] = None):
        """
        Initialize TPU Training Manager.
        
        Args:
            tpu_name: TPU name (if using TPU Pods)
        """
        self.tpu_name = tpu_name
        self.strategy = None
    
    def initialize_tpu_strategy(self) -> tf.distribute.TPUStrategy:
        """
        Initialize TPU distribution strategy.
        
        Returns:
            TPU distribution strategy
        """
        try:
            # Detect TPU
            if self.tpu_name:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
                    tpu=self.tpu_name
                )
            else:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            
            print(f"TPU detected: {tpu.cluster_spec().as_dict()}")
            
            # Connect to TPU
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            
            # Create strategy
            self.strategy = tf.distribute.TPUStrategy(tpu)
            
            print(f"Number of TPU cores: {self.strategy.num_replicas_in_sync}")
            
            return self.strategy
            
        except ValueError:
            print("No TPU detected, falling back to default strategy")
            self.strategy = tf.distribute.get_strategy()
            return self.strategy
    
    def create_model_for_tpu(
        self,
        input_shape: tuple,
        num_classes: int
    ) -> tf.keras.Model:
        """
        Create a model optimized for TPU training.
        
        Args:
            input_shape: Input shape
            num_classes: Number of output classes
            
        Returns:
            Keras model
        """
        if self.strategy is None:
            self.initialize_tpu_strategy()
        
        with self.strategy.scope():
            # Use bfloat16 for better TPU performance
            tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
            
            model = models.Sequential([
                layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
                layers.BatchNormalization(),
                layers.MaxPooling2D(),
                
                layers.Conv2D(64, 3, activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D(),
                
                layers.Conv2D(128, 3, activation='relu'),
                layers.BatchNormalization(),
                layers.GlobalAveragePooling2D(),
                
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(num_classes, dtype='float32')  # Output layer in float32
            ])
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
        
        return model
    
    def create_tpu_optimized_dataset(
        self,
        data_path: str,
        batch_size: int = 128,
        buffer_size: int = 10000
    ) -> tf.data.Dataset:
        """
        Create TPU-optimized dataset.
        
        Args:
            data_path: Path to training data
            batch_size: Batch size (should be divisible by 8)
            buffer_size: Shuffle buffer size
            
        Returns:
            TensorFlow dataset
        """
        # Ensure batch size is divisible by number of TPU cores
        if self.strategy:
            per_replica_batch_size = batch_size // self.strategy.num_replicas_in_sync
            global_batch_size = per_replica_batch_size * self.strategy.num_replicas_in_sync
        else:
            global_batch_size = batch_size
        
        # Load dataset (example with TFRecord files)
        dataset = tf.data.TFRecordDataset(
            tf.io.gfile.glob(f"{data_path}/*.tfrecord")
        )
        
        # Parse and preprocess
        def parse_function(example_proto):
            # Define your parsing logic here
            features = {
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),
            }
            parsed_features = tf.io.parse_single_example(example_proto, features)
            
            # Decode image
            image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
            image = tf.reshape(image, [224, 224, 3])
            image = tf.cast(image, tf.bfloat16) / 255.0
            
            label = parsed_features['label']
            
            return image, label
        
        # Optimize dataset for TPU
        dataset = dataset.map(
            parse_function,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(global_batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def train_on_tpu(
        self,
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int = 10,
        steps_per_epoch: Optional[int] = None
    ):
        """
        Train model on TPU.
        
        Args:
            model: Keras model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs
            steps_per_epoch: Steps per epoch (optional)
        """
        # Callbacks
        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                update_freq='batch'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='./checkpoints/model_{epoch:02d}.h5',
                save_freq='epoch'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
        
        # Train
        history = model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        return history


# Example usage
tpu_trainer = TPUTrainingManager()

# Initialize TPU strategy
strategy = tpu_trainer.initialize_tpu_strategy()

# Create model
model = tpu_trainer.create_model_for_tpu(
    input_shape=(224, 224, 3),
    num_classes=1000
)

# Create dataset
train_dataset = tpu_trainer.create_tpu_optimized_dataset(
    data_path='gs://my-bucket/training-data',
    batch_size=1024  # Large batch size for TPU
)

val_dataset = tpu_trainer.create_tpu_optimized_dataset(
    data_path='gs://my-bucket/validation-data',
    batch_size=1024
)

# Train
history = tpu_trainer.train_on_tpu(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=10
)
```

## 3. Performance Optimization

### 3.1 TPU Performance Best Practices

```python
import tensorflow as tf
from typing import Callable

class TPUOptimizer:
    """Optimizer for TPU performance."""
    
    @staticmethod
    def optimize_input_pipeline(
        dataset: tf.data.Dataset,
        batch_size: int,
        num_tpu_cores: int = 8
    ) -> tf.data.Dataset:
        """
        Optimize input pipeline for TPU.
        
        Args:
            dataset: Input dataset
            batch_size: Batch size
            num_tpu_cores: Number of TPU cores
            
        Returns:
            Optimized dataset
        """
        # Ensure batch size is multiple of TPU cores
        per_core_batch_size = batch_size // num_tpu_cores
        global_batch_size = per_core_batch_size * num_tpu_cores
        
        # Apply optimizations
        dataset = dataset.cache()  # Cache after expensive operations
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat()
        dataset = dataset.batch(global_batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    @staticmethod
    def use_bfloat16() -> None:
        """
        Enable bfloat16 precision for better TPU performance.
        """
        policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        print(f"Mixed precision policy: {policy.name}")
        print(f"Compute dtype: {policy.compute_dtype}")
        print(f"Variable dtype: {policy.variable_dtype}")
    
    @staticmethod
    def optimize_batch_size(
        model_size_gb: float,
        tpu_memory_gb: float = 16,
        target_utilization: float = 0.8
    ) -> int:
        """
        Calculate optimal batch size for TPU.
        
        Args:
            model_size_gb: Model size in GB
            tpu_memory_gb: TPU memory in GB
            target_utilization: Target memory utilization
            
        Returns:
            Recommended batch size
        """
        # Available memory for batch
        available_memory = (tpu_memory_gb - model_size_gb) * target_utilization
        
        # Estimate memory per sample (rough approximation)
        memory_per_sample_mb = 10  # Adjust based on input size
        memory_per_sample_gb = memory_per_sample_mb / 1024
        
        # Calculate batch size
        batch_size = int(available_memory / memory_per_sample_gb)
        
        # Round to multiple of 8 (TPU cores)
        batch_size = (batch_size // 8) * 8
        
        return max(8, batch_size)
    
    @staticmethod
    def create_custom_training_loop(
        model: tf.keras.Model,
        loss_fn: Callable,
        optimizer: tf.keras.optimizers.Optimizer,
        train_dataset: tf.data.Dataset,
        strategy: tf.distribute.TPUStrategy
    ):
        """
        Create custom training loop optimized for TPU.
        
        Args:
            model: Keras model
            loss_fn: Loss function
            optimizer: Optimizer
            train_dataset: Training dataset
            strategy: TPU strategy
        """
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy'
        )
        
        @tf.function
        def train_step(inputs):
            """Single training step."""
            images, labels = inputs
            
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_fn(labels, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_accuracy.update_state(labels, predictions)
            return loss
        
        @tf.function
        def distributed_train_step(dataset_inputs):
            """Distributed training step across TPU cores."""
            per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
            return strategy.reduce(
                tf.distribute.ReduceOp.SUM,
                per_replica_losses,
                axis=None
            )
        
        # Training loop
        for epoch in range(10):
            train_accuracy.reset_states()
            
            for step, inputs in enumerate(train_dataset):
                loss = distributed_train_step(inputs)
                
                if step % 100 == 0:
                    print(f"Epoch {epoch}, Step {step}: "
                          f"Loss = {loss:.4f}, "
                          f"Accuracy = {train_accuracy.result():.4f}")


# Example usage
optimizer = TPUOptimizer()

# Use bfloat16
optimizer.use_bfloat16()

# Calculate optimal batch size
batch_size = optimizer.optimize_batch_size(
    model_size_gb=2.5,
    tpu_memory_gb=16,
    target_utilization=0.8
)
print(f"Recommended batch size: {batch_size}")

# Optimize dataset
dataset = tf.data.Dataset.from_tensor_slices(...)
optimized_dataset = optimizer.optimize_input_pipeline(
    dataset=dataset,
    batch_size=batch_size,
    num_tpu_cores=8
)
```

## 4. Distributed Training on TPU Pods

### 4.1 Multi-Pod Training

```python
import tensorflow as tf
from tensorflow.python.distribute import distribution_strategy_context

class TPUPodTrainer:
    """Trainer for multi-pod TPU training."""
    
    def __init__(
        self,
        tpu_name: str,
        num_cores: int = 32
    ):
        """
        Initialize TPU Pod Trainer.
        
        Args:
            tpu_name: TPU Pod name
            num_cores: Number of TPU cores
        """
        self.tpu_name = tpu_name
        self.num_cores = num_cores
        self.strategy = None
    
    def setup_pod_strategy(self) -> tf.distribute.TPUStrategy:
        """
        Set up TPU Pod distribution strategy.
        
        Returns:
            TPU distribution strategy
        """
        # Resolve TPU
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=self.tpu_name
        )
        
        # Connect to cluster
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        
        # Create strategy
        self.strategy = tf.distribute.TPUStrategy(resolver)
        
        print(f"TPU Pod initialized with {self.strategy.num_replicas_in_sync} replicas")
        
        return self.strategy
    
    def create_distributed_dataset(
        self,
        file_pattern: str,
        batch_size: int,
        is_training: bool = True
    ) -> tf.data.Dataset:
        """
        Create distributed dataset for TPU Pod.
        
        Args:
            file_pattern: Pattern for data files
            batch_size: Global batch size
            is_training: Whether for training
            
        Returns:
            Distributed dataset
        """
        # Calculate per-replica batch size
        per_replica_batch_size = batch_size // self.strategy.num_replicas_in_sync
        
        def parse_fn(example):
            # Parse TFRecord example
            features = {
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),
            }
            parsed = tf.io.parse_single_example(example, features)
            
            image = tf.io.decode_raw(parsed['image'], tf.uint8)
            image = tf.reshape(image, [224, 224, 3])
            image = tf.cast(image, tf.bfloat16) / 255.0
            
            return image, parsed['label']
        
        # Get shard for each worker
        files = tf.io.gfile.glob(file_pattern)
        dataset = tf.data.Dataset.from_tensor_slices(files)
        
        # Shard across workers
        dataset = dataset.shard(
            num_shards=self.strategy.num_replicas_in_sync,
            index=distribution_strategy_context.get_replica_context().replica_id_in_sync_group
        )
        
        # Create dataset
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=4,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training:
            dataset = dataset.shuffle(10000)
            dataset = dataset.repeat()
        
        dataset = dataset.batch(per_replica_batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def train_on_pod(
        self,
        model: tf.keras.Model,
        train_files: str,
        val_files: str,
        epochs: int = 10,
        steps_per_epoch: int = 1000,
        batch_size: int = 2048
    ):
        """
        Train model on TPU Pod.
        
        Args:
            model: Keras model
            train_files: Training file pattern
            val_files: Validation file pattern
            epochs: Number of epochs
            steps_per_epoch: Steps per epoch
            batch_size: Global batch size
        """
        # Create datasets
        train_dataset = self.create_distributed_dataset(
            file_pattern=train_files,
            batch_size=batch_size,
            is_training=True
        )
        
        val_dataset = self.create_distributed_dataset(
            file_pattern=val_files,
            batch_size=batch_size,
            is_training=False
        )
        
        # Distribute datasets
        train_dist_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        val_dist_dataset = self.strategy.experimental_distribute_dataset(val_dataset)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath='gs://my-bucket/checkpoints/model_{epoch:02d}',
                save_freq='epoch'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='gs://my-bucket/logs',
                update_freq='batch'
            )
        ]
        
        # Train
        model.fit(
            train_dist_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dist_dataset,
            validation_steps=100,
            callbacks=callbacks
        )


# Example usage
pod_trainer = TPUPodTrainer(
    tpu_name='my-tpu-pod',
    num_cores=32
)

# Setup strategy
strategy = pod_trainer.setup_pod_strategy()

# Create model within strategy scope
with strategy.scope():
    model = tf.keras.applications.ResNet50(
        weights=None,
        classes=1000
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

# Train on pod
pod_trainer.train_on_pod(
    model=model,
    train_files='gs://my-bucket/train/*.tfrecord',
    val_files='gs://my-bucket/val/*.tfrecord',
    epochs=10,
    steps_per_epoch=1000,
    batch_size=2048
)
```

## 5. TPU Profiling and Debugging

### 5.1 Performance Profiling

```python
import tensorflow as tf
from tensorflow.python.profiler import profiler_v2 as profiler

class TPUProfiler:
    """Profiler for TPU performance analysis."""
    
    def __init__(self, logdir: str):
        """
        Initialize TPU Profiler.
        
        Args:
            logdir: Directory for profiler logs
        """
        self.logdir = logdir
    
    def profile_training_step(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        steps: int = 10
    ):
        """
        Profile training steps.
        
        Args:
            model: Keras model
            dataset: Training dataset
            steps: Number of steps to profile
        """
        # Start profiler
        tf.profiler.experimental.start(self.logdir)
        
        # Run training steps
        for step, (images, labels) in enumerate(dataset.take(steps)):
            with tf.profiler.experimental.Trace('train', step_num=step):
                loss = model.train_on_batch(images, labels)
            
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss}")
        
        # Stop profiler
        tf.profiler.experimental.stop()
        
        print(f"Profile saved to {self.logdir}")
        print("View with: tensorboard --logdir={self.logdir}")
    
    def analyze_tpu_utilization(self):
        """
        Analyze TPU utilization from profile.
        """
        # Load profile data
        print(f"Analyzing profile from {self.logdir}")
        print("Key metrics to check:")
        print("1. TPU Step Time - should be consistent")
        print("2. TPU Idle Time - should be minimal")
        print("3. Input Pipeline - should not be bottleneck")
        print("4. TPU Matrix Unit Utilization - should be high")
        print("5. Memory Bandwidth Utilization - check for saturation")


# Example usage
profiler = TPUProfiler(logdir='gs://my-bucket/profiler-logs')

# Profile training
profiler.profile_training_step(
    model=model,
    dataset=train_dataset,
    steps=10
)

# Analyze utilization
profiler.analyze_tpu_utilization()
```

## 6. Quick Reference Checklist

### TPU Setup
- [ ] Choose appropriate TPU version (v2/v3/v4)
- [ ] Select TPU topology (single-host or pod)
- [ ] Configure TPU runtime version
- [ ] Set up VPC networking
- [ ] Enable TPU API
- [ ] Create TPU service account with permissions

### Training Optimization
- [ ] Use bfloat16 mixed precision
- [ ] Optimize batch size (multiple of 8)
- [ ] Ensure batch size is divisible by number of cores
- [ ] Use drop_remainder=True in batching
- [ ] Enable XLA compilation
- [ ] Use tf.function for performance

### Data Pipeline
- [ ] Store data in GCS (preferably same region as TPU)
- [ ] Use TFRecord format
- [ ] Implement efficient data preprocessing
- [ ] Use dataset.cache() after expensive operations
- [ ] Enable dataset.prefetch(AUTOTUNE)
- [ ] Shard data across replicas
- [ ] Avoid Python operations in data pipeline

### Model Design
- [ ] Avoid dynamic shapes
- [ ] Use standard layer sizes (powers of 2)
- [ ] Minimize control flow operations
- [ ] Use batch normalization instead of layer normalization
- [ ] Avoid sparse operations
- [ ] Use supported operations (check TPU compatibility)

### Distributed Training
- [ ] Use TPUStrategy for distribution
- [ ] Scale learning rate with batch size
- [ ] Implement gradient accumulation for large models
- [ ] Use appropriate synchronization strategy
- [ ] Monitor training across replicas
- [ ] Save checkpoints to GCS

### Performance Monitoring
- [ ] Enable TensorBoard profiling
- [ ] Monitor TPU utilization
- [ ] Check for input pipeline bottlenecks
- [ ] Analyze step time distribution
- [ ] Monitor memory usage
- [ ] Track MXU (Matrix Unit) utilization

### Cost Optimization
- [ ] Use preemptible TPUs for fault-tolerant workloads
- [ ] Right-size TPU configuration
- [ ] Clean up idle TPU resources
- [ ] Use committed use discounts
- [ ] Monitor TPU usage with Cloud Monitoring
- [ ] Consider on-demand vs preemptible tradeoffs

### Debugging
- [ ] Enable TPU profiler
- [ ] Check for OOM errors
- [ ] Verify input shapes
- [ ] Test on smaller TPU configuration first
- [ ] Use tf.debugging assertions
- [ ] Monitor training metrics closely
- [ ] Check for NaN/Inf values

### Best Practices
- [ ] Start with v3-8 for development
- [ ] Scale to larger pods for production
- [ ] Use global batch size >= 1024
- [ ] Keep model and data in same region
- [ ] Implement checkpointing every N steps
- [ ] Use Cloud Storage FUSE for large datasets
- [ ] Test code on CPU/GPU before TPU deployment
