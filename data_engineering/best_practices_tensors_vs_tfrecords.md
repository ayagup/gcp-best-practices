# Best Practices for Tensors vs TFRecords Comparison on Google Cloud

## Overview

This guide compares Tensors and TFRecords, two fundamental data formats used in TensorFlow and machine learning workflows. Understanding when to use each format is crucial for optimizing data pipelines, training performance, and storage efficiency on Google Cloud.

## 1. Format Overview

### 1.1 Tensors vs TFRecords Comparison

```python
from typing import Dict, Any, List
import tensorflow as tf
import numpy as np
from dataclasses import dataclass

@dataclass
class FormatCharacteristic:
    """Data format characteristic details."""
    name: str
    tensors: str
    tfrecords: str
    notes: str

class TensorTFRecordComparator:
    """Comparator for Tensors and TFRecords formats."""
    
    def __init__(self):
        """Initialize Tensor TFRecord Comparator."""
        self.formats = self._initialize_formats()
        self.characteristics = self._initialize_characteristics()
    
    def _initialize_formats(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize format information.
        
        Returns:
            Dictionary with format details
        """
        return {
            'tensors': {
                'name': 'Tensors',
                'type': 'In-memory multi-dimensional arrays',
                'description': 'NumPy-like arrays optimized for GPU/TPU computation',
                'primary_use': 'Runtime computation and model operations',
                'key_features': [
                    'Direct computation support',
                    'GPU/TPU optimization',
                    'Dynamic shapes',
                    'Eager execution',
                    'Rich operations API',
                    'Automatic differentiation'
                ],
                'best_for': [
                    'Model training loops',
                    'Inference operations',
                    'Real-time processing',
                    'Small to medium datasets (fits in memory)',
                    'Research and prototyping',
                    'Interactive development'
                ],
                'data_types': [
                    'tf.float32', 'tf.int64', 'tf.string',
                    'tf.bool', 'tf.complex64', 'Custom dtypes'
                ],
                'storage': 'In-memory (RAM/GPU memory)',
                'serialization': 'Not directly serializable (requires conversion)',
                'performance': {
                    'read_speed': 'Instant (already in memory)',
                    'write_speed': 'N/A (not persisted)',
                    'memory_efficiency': 'Medium (raw arrays)',
                    'io_overhead': 'None (in-memory)'
                },
                'limitations': [
                    'Limited to available memory',
                    'Not persistent by default',
                    'Requires format conversion for storage',
                    'Not optimized for large datasets on disk'
                ]
            },
            'tfrecords': {
                'name': 'TFRecords',
                'type': 'Binary file format for serialized data',
                'description': 'Protocol buffer-based format for efficient data storage and streaming',
                'primary_use': 'Data storage, transfer, and training data pipelines',
                'key_features': [
                    'Efficient binary serialization',
                    'Sequential and parallel reads',
                    'Compression support (GZIP, ZLIB)',
                    'Large dataset support',
                    'tf.data integration',
                    'Sharding support'
                ],
                'best_for': [
                    'Large datasets (TB+ scale)',
                    'Training data pipelines',
                    'Data stored on GCS',
                    'Distributed training',
                    'Production ML workflows',
                    'Data preprocessing and caching'
                ],
                'data_types': [
                    'tf.train.Feature (bytes, int64, float)',
                    'Nested structures via Example/SequenceExample',
                    'Images, text, numerical data'
                ],
                'storage': 'Disk/Cloud Storage (persistent)',
                'serialization': 'Protocol Buffers (efficient binary)',
                'performance': {
                    'read_speed': 'High (optimized for sequential reads)',
                    'write_speed': 'High (batch writes)',
                    'memory_efficiency': 'High (streaming, no full load)',
                    'io_overhead': 'Low (optimized binary format)'
                },
                'limitations': [
                    'Requires serialization/deserialization',
                    'More complex to create',
                    'Less flexible than raw tensors',
                    'Requires parsing overhead'
                ]
            }
        }
    
    def _initialize_characteristics(self) -> List[FormatCharacteristic]:
        """
        Initialize characteristic comparison.
        
        Returns:
            List of format characteristics
        """
        return [
            FormatCharacteristic(
                'Primary Purpose',
                tensors='Runtime computation',
                tfrecords='Data storage and streaming',
                notes='Tensors for computation, TFRecords for I/O'
            ),
            FormatCharacteristic(
                'Memory Footprint',
                tensors='Full dataset in memory',
                tfrecords='Streaming (minimal memory)',
                notes='TFRecords better for large datasets'
            ),
            FormatCharacteristic(
                'Persistence',
                tensors='Volatile (in-memory)',
                tfrecords='Persistent (on disk)',
                notes='TFRecords survive process restarts'
            ),
            FormatCharacteristic(
                'Read Performance',
                tensors='Instant (already loaded)',
                tfrecords='Fast sequential reads',
                notes='Tensors faster if already in memory'
            ),
            FormatCharacteristic(
                'Write Performance',
                tensors='N/A (no direct storage)',
                tfrecords='Efficient batch writes',
                notes='TFRecords optimized for bulk writes'
            ),
            FormatCharacteristic(
                'Compression',
                tensors='No native compression',
                tfrecords='GZIP, ZLIB support',
                notes='TFRecords save storage space'
            ),
            FormatCharacteristic(
                'Parallelization',
                tensors='Via tf.data batching',
                tfrecords='Native sharding support',
                notes='TFRecords better for distributed training'
            ),
            FormatCharacteristic(
                'Data Types',
                tensors='All TF dtypes',
                tfrecords='bytes, int64, float only',
                notes='Tensors more flexible types'
            ),
            FormatCharacteristic(
                'GPU/TPU Support',
                tensors='Native optimization',
                tfrecords='Requires conversion to tensors',
                notes='Tensors directly used on accelerators'
            ),
            FormatCharacteristic(
                'Preprocessing',
                tensors='Real-time transformation',
                tfrecords='Pre-computed or streaming',
                notes='TFRecords can cache preprocessing'
            ),
            FormatCharacteristic(
                'Ease of Use',
                tensors='Simple, intuitive',
                tfrecords='More complex setup',
                notes='Tensors easier for beginners'
            ),
            FormatCharacteristic(
                'Debugging',
                tensors='Easy to inspect',
                tfrecords='Requires parsing to inspect',
                notes='Tensors better for debugging'
            ),
            FormatCharacteristic(
                'Cloud Storage',
                tensors='Requires conversion',
                tfrecords='Direct GCS integration',
                notes='TFRecords native cloud support'
            ),
            FormatCharacteristic(
                'Sharding',
                tensors='Manual implementation',
                tfrecords='Built-in sharding',
                notes='TFRecords better for distributed data'
            ),
            FormatCharacteristic(
                'Scalability',
                tensors='Limited by memory',
                tfrecords='TB+ scale support',
                notes='TFRecords for production scale'
            )
        ]
    
    def get_format_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Get format comparison."""
        return self.formats
    
    def get_characteristic_comparison(self) -> List[Dict[str, str]]:
        """Get characteristic comparison matrix."""
        return [
            {
                'characteristic': c.name,
                'tensors': c.tensors,
                'tfrecords': c.tfrecords,
                'notes': c.notes
            }
            for c in self.characteristics
        ]
    
    def recommend_format(
        self,
        dataset_size: str,
        use_case: str,
        storage_location: str,
        team_expertise: str
    ) -> Dict[str, Any]:
        """
        Recommend data format.
        
        Args:
            dataset_size: 'small', 'medium', 'large', 'very_large'
            use_case: 'prototyping', 'training', 'inference', 'production'
            storage_location: 'local', 'gcs', 'bigquery'
            team_expertise: 'beginner', 'intermediate', 'advanced'
            
        Returns:
            Recommendation dictionary
        """
        # Small datasets or prototyping
        if dataset_size == 'small' or use_case == 'prototyping':
            return {
                'recommendation': 'Tensors (with optional TFRecords caching)',
                'reason': 'Small data fits in memory, tensors are simpler',
                'workflow': 'Load data directly into tensors, convert to tf.data.Dataset',
                'alternatives': ['TFRecords if preparing for production scale']
            }
        
        # Large datasets
        if dataset_size in ['large', 'very_large']:
            return {
                'recommendation': 'TFRecords',
                'reason': 'Efficient streaming for datasets larger than memory',
                'workflow': 'Convert raw data to TFRecords, use tf.data pipeline',
                'alternatives': ['Tensors for small batches during debugging']
            }
        
        # Cloud storage
        if storage_location == 'gcs':
            return {
                'recommendation': 'TFRecords on GCS',
                'reason': 'Optimized for cloud storage and distributed training',
                'workflow': 'Store TFRecords on GCS, stream during training',
                'alternatives': ['BigQuery ML for structured data']
            }
        
        # Production training
        if use_case == 'production':
            return {
                'recommendation': 'TFRecords',
                'reason': 'Industry standard for production ML pipelines',
                'workflow': 'ETL to TFRecords, versioned storage, tf.data pipeline',
                'alternatives': ['Tensors for real-time inference only']
            }
        
        # Real-time inference
        if use_case == 'inference':
            return {
                'recommendation': 'Tensors',
                'reason': 'Direct computation without deserialization overhead',
                'workflow': 'Convert input to tensors, run model.predict()',
                'alternatives': ['TFRecords for batch inference']
            }
        
        # Beginners
        if team_expertise == 'beginner':
            return {
                'recommendation': 'Tensors',
                'reason': 'Easier to understand and debug',
                'workflow': 'NumPy → Tensors → Model, learn TFRecords later',
                'alternatives': ['TFRecords when scaling up']
            }
        
        # Default recommendation
        return {
            'recommendation': 'TFRecords for storage, Tensors for computation',
            'reason': 'Use both: TFRecords for data pipeline, Tensors for model',
            'workflow': 'TFRecords → tf.data → Tensors → Model',
            'alternatives': ['Pure tensors for small projects']
        }


# Example usage
comparator = TensorTFRecordComparator()

# Format comparison
formats = comparator.get_format_comparison()
print("Format Comparison:\n")
for key, info in formats.items():
    print(f"{info['name']}:")
    print(f"  Type: {info['type']}")
    print(f"  Primary use: {info['primary_use']}")
    print(f"  Storage: {info['storage']}")
    print(f"  Best for: {', '.join(info['best_for'][:3])}\n")

# Characteristic comparison
characteristics = comparator.get_characteristic_comparison()
print("Detailed Comparison:\n")
print(f"{'Characteristic':<20} {'Tensors':<30} {'TFRecords':<30}")
print("-" * 82)
for char in characteristics[:8]:
    print(f"{char['characteristic']:<20} {char['tensors']:<30} {char['tfrecords']:<30}")

# Recommendations
rec1 = comparator.recommend_format(
    dataset_size='small',
    use_case='prototyping',
    storage_location='local',
    team_expertise='beginner'
)

rec2 = comparator.recommend_format(
    dataset_size='large',
    use_case='production',
    storage_location='gcs',
    team_expertise='advanced'
)

print(f"\n\nRecommendation 1 (Small dataset, prototyping):")
print(f"  Format: {rec1['recommendation']}")
print(f"  Reason: {rec1['reason']}")
print(f"  Workflow: {rec1['workflow']}")

print(f"\nRecommendation 2 (Large dataset, production):")
print(f"  Format: {rec2['recommendation']}")
print(f"  Reason: {rec2['reason']}")
print(f"  Workflow: {rec2['workflow']}")
```

## 2. Practical Examples

### 2.1 Working with Tensors

```python
import tensorflow as tf
import numpy as np
from typing import Tuple

class TensorOperations:
    """Examples of working with Tensors."""
    
    def __init__(self):
        """Initialize Tensor Operations."""
        pass
    
    def create_tensors(self) -> Dict[str, tf.Tensor]:
        """
        Create tensors from different sources.
        
        Returns:
            Dictionary with tensor examples
        """
        # From NumPy array
        numpy_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        tensor_from_numpy = tf.constant(numpy_array)
        
        # From Python list
        tensor_from_list = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        
        # Random tensor
        random_tensor = tf.random.normal(shape=(3, 4), mean=0.0, stddev=1.0)
        
        # Zeros and ones
        zeros_tensor = tf.zeros((2, 3))
        ones_tensor = tf.ones((3, 2))
        
        # Range tensor
        range_tensor = tf.range(0, 10, 2)
        
        return {
            'from_numpy': tensor_from_numpy,
            'from_list': tensor_from_list,
            'random': random_tensor,
            'zeros': zeros_tensor,
            'ones': ones_tensor,
            'range': range_tensor
        }
    
    def tensor_operations(self, tensor1: tf.Tensor, tensor2: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Perform common tensor operations.
        
        Args:
            tensor1: First tensor
            tensor2: Second tensor
            
        Returns:
            Dictionary with operation results
        """
        return {
            'addition': tf.add(tensor1, tensor2),
            'multiplication': tf.multiply(tensor1, tensor2),
            'matrix_multiply': tf.matmul(tensor1, tensor2),
            'element_wise_square': tf.square(tensor1),
            'reduce_sum': tf.reduce_sum(tensor1),
            'reduce_mean': tf.reduce_mean(tensor1),
            'reshape': tf.reshape(tensor1, (-1,)),
            'transpose': tf.transpose(tensor1),
            'concat': tf.concat([tensor1, tensor2], axis=0)
        }
    
    def create_dataset_from_tensors(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32
    ) -> tf.data.Dataset:
        """
        Create tf.data.Dataset from tensors.
        
        Args:
            features: Feature array
            labels: Label array
            batch_size: Batch size
            
        Returns:
            TensorFlow Dataset
        """
        # Convert to tensors
        features_tensor = tf.constant(features)
        labels_tensor = tf.constant(labels)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((features_tensor, labels_tensor))
        
        # Apply transformations
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def tensor_to_numpy(self, tensor: tf.Tensor) -> np.ndarray:
        """
        Convert tensor to NumPy array.
        
        Args:
            tensor: TensorFlow tensor
            
        Returns:
            NumPy array
        """
        return tensor.numpy()


# Example usage
tensor_ops = TensorOperations()

# Create tensors
tensors = tensor_ops.create_tensors()
print("Created Tensors:")
print(f"  From NumPy shape: {tensors['from_numpy'].shape}")
print(f"  Random tensor:\n{tensors['random'].numpy()[:2]}")

# Perform operations
tensor_a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
tensor_b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
operations = tensor_ops.tensor_operations(tensor_a, tensor_b)

print(f"\nTensor Operations:")
print(f"  Addition:\n{operations['addition'].numpy()}")
print(f"  Matrix Multiply:\n{operations['matrix_multiply'].numpy()}")
print(f"  Reduce Mean: {operations['reduce_mean'].numpy():.2f}")

# Create dataset from tensors
features = np.random.randn(1000, 10).astype(np.float32)
labels = np.random.randint(0, 2, size=(1000,))
dataset = tensor_ops.create_dataset_from_tensors(features, labels, batch_size=32)

print(f"\nDataset from Tensors:")
for batch_features, batch_labels in dataset.take(1):
    print(f"  Batch features shape: {batch_features.shape}")
    print(f"  Batch labels shape: {batch_labels.shape}")
```

### 2.2 Working with TFRecords

```python
import tensorflow as tf
from typing import Dict, Any, List

class TFRecordOperations:
    """Examples of working with TFRecords."""
    
    def __init__(self):
        """Initialize TFRecord Operations."""
        pass
    
    def create_feature(self, value: Any) -> tf.train.Feature:
        """
        Create TFRecord feature from value.
        
        Args:
            value: Value to convert (bytes, int, float, or list)
            
        Returns:
            TFRecord Feature
        """
        if isinstance(value, bytes):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        elif isinstance(value, int):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        elif isinstance(value, float):
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        elif isinstance(value, list):
            if isinstance(value[0], int):
                return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
            elif isinstance(value[0], float):
                return tf.train.Feature(float_list=tf.train.FloatList(value=value))
        
        raise ValueError(f"Unsupported type: {type(value)}")
    
    def create_example(self, features_dict: Dict[str, Any]) -> tf.train.Example:
        """
        Create TFRecord Example from dictionary.
        
        Args:
            features_dict: Dictionary of features
            
        Returns:
            TFRecord Example
        """
        feature = {}
        
        for key, value in features_dict.items():
            feature[key] = self.create_feature(value)
        
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example
    
    def write_tfrecords(
        self,
        data: List[Dict[str, Any]],
        output_path: str,
        compression: str = None
    ) -> None:
        """
        Write data to TFRecord file.
        
        Args:
            data: List of feature dictionaries
            output_path: Output file path
            compression: Compression type ('GZIP', 'ZLIB', or None)
        """
        options = tf.io.TFRecordOptions(compression_type=compression)
        
        with tf.io.TFRecordWriter(output_path, options=options) as writer:
            for item in data:
                example = self.create_example(item)
                writer.write(example.SerializeToString())
        
        print(f"Wrote {len(data)} examples to {output_path}")
    
    def parse_example(self, example_proto: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Parse TFRecord example.
        
        Args:
            example_proto: Serialized example
            
        Returns:
            Dictionary of parsed features
        """
        # Define feature specification
        feature_spec = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'features': tf.io.FixedLenFeature([128], tf.float32)
        }
        
        # Parse example
        parsed_features = tf.io.parse_single_example(example_proto, feature_spec)
        
        # Decode image if needed
        if 'image' in parsed_features:
            image = tf.io.decode_jpeg(parsed_features['image'], channels=3)
            parsed_features['image'] = image
        
        return parsed_features
    
    def read_tfrecords(
        self,
        file_pattern: str,
        batch_size: int = 32,
        shuffle: bool = True,
        num_parallel_calls: int = tf.data.AUTOTUNE
    ) -> tf.data.Dataset:
        """
        Read TFRecord dataset.
        
        Args:
            file_pattern: TFRecord file pattern (supports wildcards)
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_parallel_calls: Parallel parsing threads
            
        Returns:
            TensorFlow Dataset
        """
        # List TFRecord files
        files = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
        
        # Read and parse
        dataset = files.interleave(
            lambda x: tf.data.TFRecordDataset(x),
            cycle_length=num_parallel_calls,
            num_parallel_calls=num_parallel_calls
        )
        
        # Parse examples
        dataset = dataset.map(self.parse_example, num_parallel_calls=num_parallel_calls)
        
        # Shuffle and batch
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def convert_dataset_to_tfrecords(
        self,
        dataset: tf.data.Dataset,
        output_path: str,
        num_shards: int = 10
    ) -> None:
        """
        Convert tf.data.Dataset to TFRecords with sharding.
        
        Args:
            dataset: Input dataset
            output_path: Output path pattern
            num_shards: Number of shards
        """
        # Calculate examples per shard
        dataset_size = tf.data.experimental.cardinality(dataset).numpy()
        examples_per_shard = dataset_size // num_shards
        
        for shard_id in range(num_shards):
            shard_path = f"{output_path}-{shard_id:05d}-of-{num_shards:05d}.tfrecord"
            
            # Create shard dataset
            shard_dataset = dataset.skip(shard_id * examples_per_shard).take(examples_per_shard)
            
            # Write shard
            with tf.io.TFRecordWriter(shard_path) as writer:
                for features, labels in shard_dataset:
                    # Convert to example
                    example = self.create_example({
                        'features': features.numpy().tolist(),
                        'label': int(labels.numpy())
                    })
                    writer.write(example.SerializeToString())
            
            print(f"Wrote shard {shard_id + 1}/{num_shards}: {shard_path}")


# Example usage
tfrecord_ops = TFRecordOperations()

# Create sample data
sample_data = [
    {
        'image': b'fake_image_bytes',
        'label': 0,
        'height': 224,
        'width': 224,
        'features': [0.1] * 128
    },
    {
        'image': b'fake_image_bytes_2',
        'label': 1,
        'height': 224,
        'width': 224,
        'features': [0.2] * 128
    }
]

# Write TFRecords
output_file = 'sample_data.tfrecord'
tfrecord_ops.write_tfrecords(sample_data, output_file, compression='GZIP')

# Read TFRecords
dataset = tfrecord_ops.read_tfrecords(output_file, batch_size=1, shuffle=False)

print(f"\nReading TFRecords:")
for batch in dataset.take(1):
    print(f"  Label: {batch['label'].numpy()}")
    print(f"  Height: {batch['height'].numpy()}")
    print(f"  Features shape: {batch['features'].shape}")
```

## 3. Conversion Between Formats

### 3.1 Tensor ↔ TFRecord Conversion

```python
import tensorflow as tf
import numpy as np
from typing import Tuple

class FormatConverter:
    """Convert between Tensors and TFRecords."""
    
    def __init__(self):
        """Initialize Format Converter."""
        pass
    
    def tensors_to_tfrecords(
        self,
        features: tf.Tensor,
        labels: tf.Tensor,
        output_path: str
    ) -> None:
        """
        Convert tensors to TFRecord file.
        
        Args:
            features: Feature tensor
            labels: Label tensor
            output_path: Output TFRecord path
        """
        with tf.io.TFRecordWriter(output_path) as writer:
            for i in range(features.shape[0]):
                feature_dict = {
                    'features': tf.train.Feature(
                        float_list=tf.train.FloatList(value=features[i].numpy())
                    ),
                    'label': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[int(labels[i].numpy())])
                    )
                }
                
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature_dict)
                )
                writer.write(example.SerializeToString())
        
        print(f"Converted {features.shape[0]} tensor examples to {output_path}")
    
    def tfrecords_to_tensors(
        self,
        tfrecord_path: str,
        feature_shape: Tuple[int, ...]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Load TFRecords into tensors (memory).
        
        Args:
            tfrecord_path: TFRecord file path
            feature_shape: Shape of feature tensor
            
        Returns:
            Tuple of (features_tensor, labels_tensor)
        """
        # Parse function
        def parse_fn(example_proto):
            feature_spec = {
                'features': tf.io.FixedLenFeature(feature_shape, tf.float32),
                'label': tf.io.FixedLenFeature([], tf.int64)
            }
            return tf.io.parse_single_example(example_proto, feature_spec)
        
        # Read dataset
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(parse_fn)
        
        # Load all into memory
        features_list = []
        labels_list = []
        
        for item in dataset:
            features_list.append(item['features'].numpy())
            labels_list.append(item['label'].numpy())
        
        # Convert to tensors
        features_tensor = tf.constant(np.array(features_list))
        labels_tensor = tf.constant(np.array(labels_list))
        
        print(f"Loaded {features_tensor.shape[0]} examples into tensors")
        return features_tensor, labels_tensor
    
    def numpy_to_tfrecords(
        self,
        numpy_array: np.ndarray,
        labels: np.ndarray,
        output_path: str
    ) -> None:
        """
        Convert NumPy arrays to TFRecords.
        
        Args:
            numpy_array: Feature array
            labels: Label array
            output_path: Output path
        """
        # Convert to tensors first
        features_tensor = tf.constant(numpy_array)
        labels_tensor = tf.constant(labels)
        
        # Write to TFRecords
        self.tensors_to_tfrecords(features_tensor, labels_tensor, output_path)


# Example usage
converter = FormatConverter()

# Create sample tensors
features_tensor = tf.random.normal((100, 10))
labels_tensor = tf.random.uniform((100,), minval=0, maxval=2, dtype=tf.int32)

# Convert tensors to TFRecords
output_path = 'converted_data.tfrecord'
converter.tensors_to_tfrecords(features_tensor, labels_tensor, output_path)

# Convert TFRecords back to tensors
loaded_features, loaded_labels = converter.tfrecords_to_tensors(
    output_path,
    feature_shape=(10,)
)

print(f"\nVerification:")
print(f"  Original features shape: {features_tensor.shape}")
print(f"  Loaded features shape: {loaded_features.shape}")
print(f"  Shapes match: {features_tensor.shape == loaded_features.shape}")
```

## 4. Performance Optimization

### 4.1 Best Practices

```python
import tensorflow as tf
import time
from typing import Dict, Any

class PerformanceOptimizer:
    """Optimize performance for both formats."""
    
    def __init__(self):
        """Initialize Performance Optimizer."""
        pass
    
    def benchmark_tensor_loading(
        self,
        data_size: int = 10000,
        feature_dim: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark tensor data loading.
        
        Args:
            data_size: Number of examples
            feature_dim: Feature dimension
            
        Returns:
            Timing results
        """
        # Create data
        features = tf.random.normal((data_size, feature_dim))
        labels = tf.random.uniform((data_size,), minval=0, maxval=10, dtype=tf.int32)
        
        # Benchmark: Direct tensor access
        start = time.time()
        for _ in range(10):
            _ = features[0:32]  # Simulate batch access
        tensor_time = time.time() - start
        
        # Benchmark: tf.data.Dataset from tensors
        start = time.time()
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        for _ in dataset.take(10):
            pass
        dataset_time = time.time() - start
        
        return {
            'direct_tensor_access_ms': tensor_time * 100,
            'tf_data_from_tensors_ms': dataset_time * 100,
            'data_size': data_size,
            'memory_mb': (features.numpy().nbytes + labels.numpy().nbytes) / (1024 * 1024)
        }
    
    def benchmark_tfrecord_loading(
        self,
        tfrecord_path: str,
        num_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark TFRecord loading.
        
        Args:
            tfrecord_path: Path to TFRecord file
            num_iterations: Number of iterations
            
        Returns:
            Timing results
        """
        # Parse function
        def parse_fn(example_proto):
            feature_spec = {
                'features': tf.io.FixedLenFeature([10], tf.float32),
                'label': tf.io.FixedLenFeature([], tf.int64)
            }
            return tf.io.parse_single_example(example_proto, feature_spec)
        
        # Benchmark without optimization
        start = time.time()
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(parse_fn).batch(32)
        for _ in dataset.take(num_iterations):
            pass
        unoptimized_time = time.time() - start
        
        # Benchmark with optimization
        start = time.time()
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(32)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        for _ in dataset.take(num_iterations):
            pass
        optimized_time = time.time() - start
        
        return {
            'unoptimized_ms': unoptimized_time * 1000,
            'optimized_ms': optimized_time * 1000,
            'speedup': unoptimized_time / optimized_time if optimized_time > 0 else 0
        }
    
    def get_optimization_tips(self) -> Dict[str, List[str]]:
        """
        Get optimization tips for both formats.
        
        Returns:
            Dictionary with tips
        """
        return {
            'tensors': [
                'Use tf.function for graph optimization',
                'Batch operations for GPU efficiency',
                'Use appropriate dtypes (float32 vs float64)',
                'Leverage XLA compilation',
                'Use tf.data for efficient pipelines',
                'Prefetch batches during training',
                'Use mixed precision training',
                'Pin memory for GPU transfer'
            ],
            'tfrecords': [
                'Use compression (GZIP) for storage',
                'Shard large datasets (100-1000 MB per shard)',
                'Enable parallel reading (cycle_length)',
                'Use parallel parsing (num_parallel_calls=AUTOTUNE)',
                'Prefetch batches (prefetch=AUTOTUNE)',
                'Cache preprocessed data when possible',
                'Use interleave for multi-file reading',
                'Store preprocessed features (not raw data)',
                'Use appropriate batch sizes',
                'Enable deterministic ordering when needed'
            ],
            'hybrid_approach': [
                'Use TFRecords for data storage/pipeline',
                'Convert to tensors for model computation',
                'Cache parsed TFRecords in memory',
                'Use tf.data for bridging both formats',
                'Preprocess once, train multiple times',
                'Use TFRecords for training, tensors for inference'
            ]
        }


# Example usage
optimizer = PerformanceOptimizer()

# Benchmark tensors
print("Tensor Loading Benchmark:")
tensor_results = optimizer.benchmark_tensor_loading(data_size=1000)
for key, value in tensor_results.items():
    print(f"  {key}: {value:.2f}")

# Get optimization tips
tips = optimizer.get_optimization_tips()
print("\nOptimization Tips for Tensors:")
for tip in tips['tensors'][:5]:
    print(f"  • {tip}")

print("\nOptimization Tips for TFRecords:")
for tip in tips['tfrecords'][:5]:
    print(f"  • {tip}")

print("\nHybrid Approach Tips:")
for tip in tips['hybrid_approach']:
    print(f"  • {tip}")
```

## 5. Quick Reference Checklist

### When to Use Tensors
- [ ] Dataset fits in memory
- [ ] Real-time inference
- [ ] Interactive development/debugging
- [ ] Prototyping and experimentation
- [ ] Direct GPU/TPU computation
- [ ] Small to medium datasets (<10 GB)
- [ ] Immediate data access needed
- [ ] Eager execution workflows

### When to Use TFRecords
- [ ] Large datasets (>10 GB)
- [ ] Production training pipelines
- [ ] Data stored on GCS
- [ ] Distributed training
- [ ] Need compression
- [ ] Streaming data processing
- [ ] Multiple epoch training
- [ ] Shareable data format

### Conversion Guidelines
- [ ] Preprocess once, save as TFRecords
- [ ] Load TFRecords → tensors for training
- [ ] Use tf.data.Dataset as bridge
- [ ] Shard large TFRecord files
- [ ] Version TFRecord datasets
- [ ] Document feature specifications

### Performance Optimization
- [ ] Use AUTOTUNE for parallelism
- [ ] Enable prefetching
- [ ] Compress TFRecords (GZIP)
- [ ] Batch efficiently
- [ ] Cache when possible
- [ ] Profile data pipelines
- [ ] Monitor I/O bottlenecks
- [ ] Use appropriate data types

### Best Practices
- [ ] Tensors for computation, TFRecords for I/O
- [ ] Start simple (tensors), scale up (TFRecords)
- [ ] Use tf.data for both formats
- [ ] Test with small subsets first
- [ ] Document data schemas
- [ ] Version data and code together
- [ ] Monitor memory usage
- [ ] Optimize based on bottlenecks
