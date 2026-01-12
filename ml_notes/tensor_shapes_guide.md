# Comprehensive Guide to Tensor Shapes and Indexing

## Overview

Understanding tensor shapes and indexing is fundamental to deep learning and machine learning engineering. This guide provides an exhaustive reference for working with tensor shapes, dimensions, and both positive and negative indexing across TensorFlow, PyTorch, and NumPy.

## Table of Contents

1. [Tensor Fundamentals](#1-tensor-fundamentals)
2. [Shape Basics](#2-shape-basics)
3. [Positive Shape Indices](#3-positive-shape-indices)
4. [Negative Shape Indices](#4-negative-shape-indices)
5. [Common Tensor Shapes](#5-common-tensor-shapes)
6. [Shape Manipulation](#6-shape-manipulation)
7. [Broadcasting Rules](#7-broadcasting-rules)
8. [Best Practices](#8-best-practices)

---

## 1. Tensor Fundamentals

### 1.1 What is a Tensor?

```python
import numpy as np
import tensorflow as tf
import torch

class TensorFundamentals:
    """Understanding tensor basics across frameworks."""
    
    def __init__(self):
        """Initialize Tensor Fundamentals."""
        pass
    
    def demonstrate_tensor_ranks(self):
        """
        Demonstrate tensors of different ranks (dimensions).
        
        Rank/Dimension Terminology:
        - Rank 0: Scalar (single number)
        - Rank 1: Vector (1D array)
        - Rank 2: Matrix (2D array)
        - Rank 3: 3D tensor
        - Rank 4+: Higher-dimensional tensors
        """
        examples = {}
        
        # Rank 0: Scalar
        scalar_np = np.array(42)
        scalar_tf = tf.constant(42)
        scalar_torch = torch.tensor(42)
        
        examples['rank_0_scalar'] = {
            'numpy': scalar_np,
            'shape': scalar_np.shape,  # ()
            'ndim': scalar_np.ndim,     # 0
            'description': 'Single number, no dimensions'
        }
        
        # Rank 1: Vector
        vector_np = np.array([1, 2, 3, 4, 5])
        vector_tf = tf.constant([1, 2, 3, 4, 5])
        vector_torch = torch.tensor([1, 2, 3, 4, 5])
        
        examples['rank_1_vector'] = {
            'numpy': vector_np,
            'shape': vector_np.shape,  # (5,)
            'ndim': vector_np.ndim,     # 1
            'description': '1D array, single axis'
        }
        
        # Rank 2: Matrix
        matrix_np = np.array([[1, 2, 3],
                              [4, 5, 6]])
        matrix_tf = tf.constant([[1, 2, 3],
                                 [4, 5, 6]])
        matrix_torch = torch.tensor([[1, 2, 3],
                                     [4, 5, 6]])
        
        examples['rank_2_matrix'] = {
            'numpy': matrix_np,
            'shape': matrix_np.shape,  # (2, 3)
            'ndim': matrix_np.ndim,     # 2
            'description': '2D array, rows and columns'
        }
        
        # Rank 3: 3D Tensor
        tensor_3d_np = np.array([[[1, 2], [3, 4]],
                                 [[5, 6], [7, 8]]])
        tensor_3d_tf = tf.constant([[[1, 2], [3, 4]],
                                    [[5, 6], [7, 8]]])
        tensor_3d_torch = torch.tensor([[[1, 2], [3, 4]],
                                        [[5, 6], [7, 8]]])
        
        examples['rank_3_tensor'] = {
            'numpy': tensor_3d_np,
            'shape': tensor_3d_np.shape,  # (2, 2, 2)
            'ndim': tensor_3d_np.ndim,     # 3
            'description': '3D array, depth/height/width'
        }
        
        # Rank 4: 4D Tensor (common in deep learning)
        tensor_4d_np = np.random.randn(32, 224, 224, 3)  # Batch of images
        
        examples['rank_4_tensor'] = {
            'numpy': tensor_4d_np,
            'shape': tensor_4d_np.shape,  # (32, 224, 224, 3)
            'ndim': tensor_4d_np.ndim,     # 4
            'description': 'Batch of images: (batch, height, width, channels)'
        }
        
        return examples


# Example usage
fundamentals = TensorFundamentals()
examples = fundamentals.demonstrate_tensor_ranks()

print("Tensor Ranks and Shapes:")
for name, info in examples.items():
    print(f"\n{name}:")
    print(f"  Shape: {info['shape']}")
    print(f"  Dimensions: {info['ndim']}")
    print(f"  Description: {info['description']}")
```

### 1.2 Shape vs Size vs Dimension

```python
class ShapeTerminology:
    """Understanding shape terminology."""
    
    def explain_terminology(self):
        """
        Explain different shape-related terms.
        
        Terms:
        - Shape: Tuple of dimension sizes (e.g., (3, 4, 5))
        - Rank/ndim: Number of dimensions (e.g., 3)
        - Size: Total number of elements (e.g., 60)
        - Axis: Individual dimension (e.g., axis 0, axis 1)
        """
        tensor = np.random.randn(3, 4, 5)
        
        terminology = {
            'shape': tensor.shape,           # (3, 4, 5)
            'rank_ndim': tensor.ndim,        # 3
            'size': tensor.size,             # 60 (3 × 4 × 5)
            'dtype': tensor.dtype,           # float64
            'axis_0_size': tensor.shape[0],  # 3
            'axis_1_size': tensor.shape[1],  # 4
            'axis_2_size': tensor.shape[2],  # 5
        }
        
        return terminology


# Example
terminology = ShapeTerminology()
info = terminology.explain_terminology()

print("\nShape Terminology:")
print(f"  Shape: {info['shape']}")
print(f"  Rank (ndim): {info['rank_ndim']}")
print(f"  Total size: {info['size']}")
print(f"  Axis 0 size: {info['axis_0_size']}")
print(f"  Axis 1 size: {info['axis_1_size']}")
print(f"  Axis 2 size: {info['axis_2_size']}")
```

---

## 2. Shape Basics

### 2.1 Reading Tensor Shapes

```python
class ShapeReader:
    """Reading and interpreting tensor shapes."""
    
    def read_shape(self, tensor_shape: tuple) -> dict:
        """
        Read and interpret tensor shape.
        
        Args:
            tensor_shape: Tuple representing shape
            
        Returns:
            Dictionary with shape interpretation
        """
        interpretation = {
            'shape': tensor_shape,
            'rank': len(tensor_shape),
            'total_elements': np.prod(tensor_shape) if tensor_shape else 1
        }
        
        # Common interpretations based on rank
        rank = len(tensor_shape)
        
        if rank == 0:
            interpretation['type'] = 'Scalar'
            interpretation['meaning'] = 'Single value'
            
        elif rank == 1:
            interpretation['type'] = 'Vector'
            interpretation['meaning'] = f'{tensor_shape[0]} elements'
            interpretation['axis_names'] = ['features']
            
        elif rank == 2:
            interpretation['type'] = 'Matrix'
            interpretation['meaning'] = f'{tensor_shape[0]} rows × {tensor_shape[1]} columns'
            interpretation['axis_names'] = ['rows (samples)', 'columns (features)']
            
        elif rank == 3:
            interpretation['type'] = '3D Tensor'
            interpretation['axis_names'] = ['depth/batch', 'height/time', 'width/features']
            interpretation['meaning'] = f'{tensor_shape[0]} × {tensor_shape[1]} × {tensor_shape[2]}'
            
        elif rank == 4:
            interpretation['type'] = '4D Tensor'
            interpretation['axis_names'] = ['batch', 'height', 'width', 'channels']
            interpretation['meaning'] = f'Batch: {tensor_shape[0]}, H: {tensor_shape[1]}, W: {tensor_shape[2]}, C: {tensor_shape[3]}'
            
        elif rank == 5:
            interpretation['type'] = '5D Tensor'
            interpretation['axis_names'] = ['batch', 'depth/time', 'height', 'width', 'channels']
            
        return interpretation


# Example usage
reader = ShapeReader()

shapes = [
    (),
    (100,),
    (32, 10),
    (32, 28, 28),
    (64, 224, 224, 3),
    (16, 10, 64, 64, 3)
]

print("\nShape Interpretations:")
for shape in shapes:
    info = reader.read_shape(shape)
    print(f"\nShape {shape}:")
    print(f"  Type: {info['type']}")
    print(f"  Rank: {info['rank']}")
    print(f"  Total elements: {info['total_elements']}")
    if 'meaning' in info:
        print(f"  Meaning: {info['meaning']}")
    if 'axis_names' in info:
        print(f"  Axis names: {info['axis_names']}")
```

---

## 3. Positive Shape Indices

### 3.1 Understanding Positive Indices

```python
class PositiveIndices:
    """Understanding positive shape indices."""
    
    def demonstrate_positive_indexing(self):
        """
        Demonstrate positive indexing for tensor shapes.
        
        Positive indices start from 0 (leftmost dimension).
        
        Example: Shape (2, 3, 4, 5)
        - Index 0 → 2 (first dimension)
        - Index 1 → 3 (second dimension)
        - Index 2 → 4 (third dimension)
        - Index 3 → 5 (fourth dimension)
        """
        tensor = np.random.randn(2, 3, 4, 5)
        shape = tensor.shape
        
        positive_indices = {}
        
        for i in range(len(shape)):
            positive_indices[f'axis_{i}'] = {
                'index': i,
                'size': shape[i],
                'description': self._get_axis_description(i, len(shape))
            }
        
        return positive_indices
    
    def _get_axis_description(self, axis: int, total_dims: int) -> str:
        """Get description for specific axis."""
        if total_dims == 4:  # Common for images
            descriptions = {
                0: 'Batch size',
                1: 'Height',
                2: 'Width',
                3: 'Channels'
            }
            return descriptions.get(axis, f'Dimension {axis}')
        elif total_dims == 3:  # Common for sequences
            descriptions = {
                0: 'Batch size',
                1: 'Sequence length',
                2: 'Features'
            }
            return descriptions.get(axis, f'Dimension {axis}')
        elif total_dims == 2:  # Common for tabular
            descriptions = {
                0: 'Samples/Rows',
                1: 'Features/Columns'
            }
            return descriptions.get(axis, f'Dimension {axis}')
        else:
            return f'Dimension {axis}'
    
    def access_shape_dimensions(self):
        """Demonstrate accessing shape dimensions."""
        tensor = np.random.randn(32, 64, 64, 3)  # Batch of images
        
        access_examples = {
            'full_shape': tensor.shape,           # (32, 64, 64, 3)
            'batch_size': tensor.shape[0],        # 32
            'height': tensor.shape[1],            # 64
            'width': tensor.shape[2],             # 64
            'channels': tensor.shape[3],          # 3
            'first_dim': tensor.shape[0],         # 32
            'second_dim': tensor.shape[1],        # 64
        }
        
        return access_examples


# Example usage
positive = PositiveIndices()
indices = positive.demonstrate_positive_indexing()

print("\nPositive Shape Indices:")
print("Tensor shape: (2, 3, 4, 5)")
for axis_name, info in indices.items():
    print(f"  {axis_name}: index={info['index']}, size={info['size']}, desc={info['description']}")

access = positive.access_shape_dimensions()
print("\nAccessing Shape Dimensions:")
for name, value in access.items():
    print(f"  {name}: {value}")
```

### 3.2 Positive Index Use Cases

```python
class PositiveIndexUseCases:
    """Common use cases for positive indices."""
    
    def common_operations(self):
        """Demonstrate common operations using positive indices."""
        tensor = np.random.randn(32, 100, 512)  # (batch, sequence, features)
        
        operations = {}
        
        # 1. Get batch size
        batch_size = tensor.shape[0]
        operations['get_batch_size'] = {
            'code': 'tensor.shape[0]',
            'result': batch_size,
            'description': 'Get first dimension (batch size)'
        }
        
        # 2. Get sequence length
        seq_length = tensor.shape[1]
        operations['get_sequence_length'] = {
            'code': 'tensor.shape[1]',
            'result': seq_length,
            'description': 'Get second dimension (sequence length)'
        }
        
        # 3. Get feature dimension
        features = tensor.shape[2]
        operations['get_features'] = {
            'code': 'tensor.shape[2]',
            'result': features,
            'description': 'Get third dimension (features)'
        }
        
        # 4. Reshape based on first dimension
        reshaped = tensor.reshape(tensor.shape[0], -1)
        operations['flatten_except_batch'] = {
            'code': 'tensor.reshape(tensor.shape[0], -1)',
            'original_shape': tensor.shape,
            'new_shape': reshaped.shape,
            'description': 'Flatten all except batch dimension'
        }
        
        # 5. Sum over specific axis
        summed = np.sum(tensor, axis=1)  # Sum over sequence dimension
        operations['sum_over_axis_1'] = {
            'code': 'np.sum(tensor, axis=1)',
            'original_shape': tensor.shape,
            'new_shape': summed.shape,
            'description': 'Sum over second dimension (sequence)'
        }
        
        return operations


# Example usage
use_cases = PositiveIndexUseCases()
ops = use_cases.common_operations()

print("\nPositive Index Use Cases:")
for op_name, details in ops.items():
    print(f"\n{op_name}:")
    for key, value in details.items():
        print(f"  {key}: {value}")
```

---

## 4. Negative Shape Indices

### 4.1 Understanding Negative Indices

```python
class NegativeIndices:
    """Understanding negative shape indices."""
    
    def demonstrate_negative_indexing(self):
        """
        Demonstrate negative indexing for tensor shapes.
        
        Negative indices count from the right (last dimension).
        
        Example: Shape (2, 3, 4, 5)
        - Index -1 → 5 (last dimension)
        - Index -2 → 4 (second-to-last dimension)
        - Index -3 → 3 (third-to-last dimension)
        - Index -4 → 2 (fourth-to-last dimension)
        
        Relationship: negative_index = positive_index - len(shape)
        """
        tensor = np.random.randn(2, 3, 4, 5)
        shape = tensor.shape
        
        negative_indices = {}
        
        for i in range(len(shape)):
            negative_idx = i - len(shape)
            negative_indices[f'dimension_{i}'] = {
                'positive_index': i,
                'negative_index': negative_idx,
                'size': shape[i],
                'access_positive': f'shape[{i}]',
                'access_negative': f'shape[{negative_idx}]',
                'same_value': shape[i] == shape[negative_idx]
            }
        
        return negative_indices
    
    def conversion_table(self):
        """Create conversion table between positive and negative indices."""
        shape = (2, 3, 4, 5, 6)
        conversions = []
        
        for pos_idx in range(len(shape)):
            neg_idx = pos_idx - len(shape)
            conversions.append({
                'positive': pos_idx,
                'negative': neg_idx,
                'dimension_size': shape[pos_idx],
                'interpretation_positive': self._interpret_position(pos_idx, len(shape), from_start=True),
                'interpretation_negative': self._interpret_position(neg_idx, len(shape), from_start=False)
            })
        
        return conversions
    
    def _interpret_position(self, index: int, total: int, from_start: bool) -> str:
        """Interpret index position."""
        if from_start:
            if index == 0:
                return "First dimension"
            elif index == total - 1:
                return "Last dimension"
            else:
                return f"{index + 1}th dimension from start"
        else:
            if index == -1:
                return "Last dimension"
            elif index == -total:
                return "First dimension"
            else:
                return f"{abs(index)}th dimension from end"


# Example usage
negative = NegativeIndices()
indices = negative.demonstrate_negative_indexing()

print("\nNegative Shape Indices:")
print("Tensor shape: (2, 3, 4, 5)")
for dim_name, info in indices.items():
    print(f"\n{dim_name}:")
    print(f"  Positive index: {info['positive_index']} → shape[{info['positive_index']}] = {info['size']}")
    print(f"  Negative index: {info['negative_index']} → shape[{info['negative_index']}] = {info['size']}")
    print(f"  Values match: {info['same_value']}")

print("\n\nConversion Table:")
conversions = negative.conversion_table()
print("Shape: (2, 3, 4, 5, 6)")
print(f"{'Positive':<10} {'Negative':<10} {'Size':<6} {'From Start':<25} {'From End':<25}")
print("-" * 80)
for conv in conversions:
    print(f"{conv['positive']:<10} {conv['negative']:<10} {conv['dimension_size']:<6} "
          f"{conv['interpretation_positive']:<25} {conv['interpretation_negative']:<25}")
```

### 4.2 Why Use Negative Indices?

```python
class NegativeIndexAdvantages:
    """Advantages of using negative indices."""
    
    def demonstrate_advantages(self):
        """
        Demonstrate why negative indices are useful.
        
        Advantages:
        1. Dimension-agnostic code
        2. Focus on last dimensions (common in ML)
        3. More readable for channel/feature dimensions
        4. Works across different batch sizes
        """
        advantages = {}
        
        # Advantage 1: Dimension-agnostic code
        # Works for both 3D and 4D tensors
        tensor_3d = np.random.randn(32, 64, 3)      # (batch, features, channels)
        tensor_4d = np.random.randn(32, 64, 64, 3)  # (batch, height, width, channels)
        
        # Using positive index (brittle):
        # channels_3d = tensor_3d.shape[2]  # OK
        # channels_4d = tensor_4d.shape[3]  # Different index!
        
        # Using negative index (robust):
        channels_3d = tensor_3d.shape[-1]  # Always works
        channels_4d = tensor_4d.shape[-1]  # Always works
        
        advantages['dimension_agnostic'] = {
            'tensor_3d_shape': tensor_3d.shape,
            'tensor_4d_shape': tensor_4d.shape,
            'channels_3d': channels_3d,
            'channels_4d': channels_4d,
            'benefit': 'Same code works for different ranks'
        }
        
        # Advantage 2: Last dimension focus (channels, features)
        image_tensor = np.random.randn(16, 224, 224, 3)
        
        advantages['last_dimension_focus'] = {
            'shape': image_tensor.shape,
            'channels_positive': image_tensor.shape[3],
            'channels_negative': image_tensor.shape[-1],
            'benefit': 'Negative index clearly indicates "last dimension" intent'
        }
        
        # Advantage 3: Feature dimension in sequences
        sequence_tensor = np.random.randn(8, 100, 512)  # (batch, seq_len, features)
        
        advantages['feature_dimension'] = {
            'shape': sequence_tensor.shape,
            'features_positive': sequence_tensor.shape[2],
            'features_negative': sequence_tensor.shape[-1],
            'benefit': 'Feature dimension naturally at the end'
        }
        
        # Advantage 4: Reshape operations
        batch_images = np.random.randn(32, 28, 28, 1)
        
        # Flatten while preserving batch dimension
        flattened_positive = batch_images.reshape(batch_images.shape[0], -1)
        flattened_negative = batch_images.reshape(batch_images.shape[0], -1)
        
        advantages['reshape_operations'] = {
            'original_shape': batch_images.shape,
            'flattened_shape': flattened_positive.shape,
            'benefit': 'Flexible reshaping based on first/last dimensions'
        }
        
        return advantages


# Example usage
neg_advantages = NegativeIndexAdvantages()
advantages = neg_advantages.demonstrate_advantages()

print("\nAdvantages of Negative Indices:")
for adv_name, details in advantages.items():
    print(f"\n{adv_name}:")
    for key, value in details.items():
        print(f"  {key}: {value}")
```

### 4.3 Common Negative Index Patterns

```python
class NegativeIndexPatterns:
    """Common patterns using negative indices."""
    
    def common_patterns(self):
        """Demonstrate common negative index patterns."""
        patterns = {}
        
        # Pattern 1: Get last dimension (channels/features)
        tensor = np.random.randn(32, 64, 64, 3)
        num_channels = tensor.shape[-1]
        
        patterns['get_last_dimension'] = {
            'code': 'tensor.shape[-1]',
            'tensor_shape': tensor.shape,
            'result': num_channels,
            'use_case': 'Get number of channels/features'
        }
        
        # Pattern 2: Get second-to-last dimension
        num_width = tensor.shape[-2]
        
        patterns['get_second_to_last'] = {
            'code': 'tensor.shape[-2]',
            'tensor_shape': tensor.shape,
            'result': num_width,
            'use_case': 'Get width or penultimate dimension'
        }
        
        # Pattern 3: Flatten all except last dimension
        reshaped = tensor.reshape(-1, tensor.shape[-1])
        
        patterns['flatten_except_last'] = {
            'code': 'tensor.reshape(-1, tensor.shape[-1])',
            'original_shape': tensor.shape,
            'new_shape': reshaped.shape,
            'use_case': 'Flatten spatial dimensions, keep channels'
        }
        
        # Pattern 4: Sum over second-to-last dimension
        summed = np.sum(tensor, axis=-2)
        
        patterns['sum_second_to_last'] = {
            'code': 'np.sum(tensor, axis=-2)',
            'original_shape': tensor.shape,
            'new_shape': summed.shape,
            'use_case': 'Pool over height dimension'
        }
        
        # Pattern 5: Transpose last two dimensions
        transposed = np.transpose(tensor, (0, 1, 3, 2))
        # Or using negative indices in swapaxes
        swapped = np.swapaxes(tensor, -2, -1)
        
        patterns['transpose_last_two'] = {
            'code': 'np.swapaxes(tensor, -2, -1)',
            'original_shape': tensor.shape,
            'new_shape': swapped.shape,
            'use_case': 'Swap width and channels'
        }
        
        # Pattern 6: Concatenate along last dimension
        tensor1 = np.random.randn(32, 64, 64, 3)
        tensor2 = np.random.randn(32, 64, 64, 5)
        concatenated = np.concatenate([tensor1, tensor2], axis=-1)
        
        patterns['concatenate_last_dim'] = {
            'code': 'np.concatenate([tensor1, tensor2], axis=-1)',
            'tensor1_shape': tensor1.shape,
            'tensor2_shape': tensor2.shape,
            'result_shape': concatenated.shape,
            'use_case': 'Combine features/channels'
        }
        
        return patterns


# Example usage
patterns_demo = NegativeIndexPatterns()
patterns = patterns_demo.common_patterns()

print("\nCommon Negative Index Patterns:")
for pattern_name, details in patterns.items():
    print(f"\n{pattern_name}:")
    for key, value in details.items():
        print(f"  {key}: {value}")
```

---

## 5. Common Tensor Shapes

### 5.1 ML/DL Common Shapes

```python
class CommonTensorShapes:
    """Common tensor shapes in machine learning."""
    
    def get_common_shapes(self):
        """
        Document common tensor shapes and their meanings.
        """
        common_shapes = {}
        
        # 1D: Features/Embeddings
        common_shapes['1d_features'] = {
            'shape': '(features,)',
            'example': (768,),
            'description': 'Single sample feature vector or embedding',
            'use_cases': ['Word embeddings', 'Feature vectors', 'Model outputs'],
            'accessing': {
                'features': 'shape[0] or shape[-1]'
            }
        }
        
        # 2D: Batch of samples
        common_shapes['2d_batch'] = {
            'shape': '(batch_size, features)',
            'example': (32, 768),
            'description': 'Batch of feature vectors (tabular data)',
            'use_cases': ['Dense layer input', 'Tabular data', 'Batch of embeddings'],
            'accessing': {
                'batch_size': 'shape[0]',
                'features': 'shape[1] or shape[-1]'
            }
        }
        
        # 3D: Sequences
        common_shapes['3d_sequences'] = {
            'shape': '(batch_size, sequence_length, features)',
            'example': (32, 100, 512),
            'description': 'Batch of sequences (NLP, time series)',
            'use_cases': ['LSTM input', 'Transformer input', 'Time series'],
            'accessing': {
                'batch_size': 'shape[0]',
                'sequence_length': 'shape[1] or shape[-2]',
                'features': 'shape[2] or shape[-1]'
            }
        }
        
        # 3D: Single image
        common_shapes['3d_image'] = {
            'shape': '(height, width, channels)',
            'example': (224, 224, 3),
            'description': 'Single image (TensorFlow format)',
            'use_cases': ['Image input', 'CNN input (single)'],
            'accessing': {
                'height': 'shape[0] or shape[-3]',
                'width': 'shape[1] or shape[-2]',
                'channels': 'shape[2] or shape[-1]'
            }
        }
        
        # 4D: Batch of images (TensorFlow/Keras format)
        common_shapes['4d_images_tf'] = {
            'shape': '(batch_size, height, width, channels)',
            'example': (32, 224, 224, 3),
            'description': 'Batch of images (TensorFlow/Keras format)',
            'use_cases': ['CNN input', 'Image classification', 'Object detection'],
            'format': 'NHWC (channels last)',
            'accessing': {
                'batch_size': 'shape[0]',
                'height': 'shape[1] or shape[-3]',
                'width': 'shape[2] or shape[-2]',
                'channels': 'shape[3] or shape[-1]'
            }
        }
        
        # 4D: Batch of images (PyTorch format)
        common_shapes['4d_images_torch'] = {
            'shape': '(batch_size, channels, height, width)',
            'example': (32, 3, 224, 224),
            'description': 'Batch of images (PyTorch format)',
            'use_cases': ['CNN input (PyTorch)', 'Image models'],
            'format': 'NCHW (channels first)',
            'accessing': {
                'batch_size': 'shape[0]',
                'channels': 'shape[1]',
                'height': 'shape[2] or shape[-2]',
                'width': 'shape[3] or shape[-1]'
            }
        }
        
        # 5D: Video/3D Images
        common_shapes['5d_video'] = {
            'shape': '(batch_size, time/depth, height, width, channels)',
            'example': (8, 16, 112, 112, 3),
            'description': 'Batch of videos or 3D images',
            'use_cases': ['Video classification', '3D CNN', 'Medical imaging'],
            'accessing': {
                'batch_size': 'shape[0]',
                'time_depth': 'shape[1]',
                'height': 'shape[2] or shape[-3]',
                'width': 'shape[3] or shape[-2]',
                'channels': 'shape[4] or shape[-1]'
            }
        }
        
        # Attention: (batch, heads, seq_len, head_dim)
        common_shapes['attention'] = {
            'shape': '(batch_size, num_heads, sequence_length, head_dimension)',
            'example': (32, 8, 100, 64),
            'description': 'Multi-head attention tensor',
            'use_cases': ['Transformer attention', 'Self-attention'],
            'accessing': {
                'batch_size': 'shape[0]',
                'num_heads': 'shape[1]',
                'sequence_length': 'shape[2] or shape[-2]',
                'head_dimension': 'shape[3] or shape[-1]'
            }
        }
        
        return common_shapes


# Example usage
common = CommonTensorShapes()
shapes = common.get_common_shapes()

print("\nCommon Tensor Shapes in ML/DL:")
for shape_name, info in shapes.items():
    print(f"\n{shape_name}:")
    print(f"  Shape: {info['shape']}")
    print(f"  Example: {info['example']}")
    print(f"  Description: {info['description']}")
    print(f"  Use cases: {', '.join(info['use_cases'])}")
    print(f"  Accessing dimensions:")
    for dim_name, access_code in info['accessing'].items():
        print(f"    {dim_name}: {access_code}")
```

---

## 6. Shape Manipulation

### 6.1 Reshaping Operations

```python
class ShapeManipulation:
    """Common shape manipulation operations."""
    
    def reshape_operations(self):
        """Demonstrate common reshape operations."""
        operations = {}
        
        # 1. Flatten (keep batch dimension)
        tensor = np.random.randn(32, 28, 28, 1)
        flattened = tensor.reshape(tensor.shape[0], -1)
        
        operations['flatten_keep_batch'] = {
            'original': tensor.shape,
            'result': flattened.shape,
            'code': 'tensor.reshape(tensor.shape[0], -1)',
            'explanation': 'Flatten all except batch: (32, 28, 28, 1) → (32, 784)'
        }
        
        # 2. Add dimension (unsqueeze)
        vector = np.random.randn(100)
        expanded = np.expand_dims(vector, axis=0)  # Add batch dimension
        
        operations['add_batch_dimension'] = {
            'original': vector.shape,
            'result': expanded.shape,
            'code': 'np.expand_dims(vector, axis=0)',
            'explanation': 'Add batch dimension: (100,) → (1, 100)'
        }
        
        # 3. Remove dimension (squeeze)
        tensor_with_extra = np.random.randn(32, 28, 28, 1)
        squeezed = np.squeeze(tensor_with_extra, axis=-1)
        
        operations['remove_dimension'] = {
            'original': tensor_with_extra.shape,
            'result': squeezed.shape,
            'code': 'np.squeeze(tensor, axis=-1)',
            'explanation': 'Remove last dimension: (32, 28, 28, 1) → (32, 28, 28)'
        }
        
        # 4. Transpose (swap dimensions)
        tensor_nhwc = np.random.randn(32, 224, 224, 3)  # TensorFlow format
        tensor_nchw = np.transpose(tensor_nhwc, (0, 3, 1, 2))  # PyTorch format
        
        operations['nhwc_to_nchw'] = {
            'original': tensor_nhwc.shape,
            'result': tensor_nchw.shape,
            'code': 'np.transpose(tensor, (0, 3, 1, 2))',
            'explanation': 'Convert TF to PyTorch format: (32, 224, 224, 3) → (32, 3, 224, 224)'
        }
        
        # 5. Flatten completely
        tensor = np.random.randn(32, 28, 28)
        flat = tensor.reshape(-1)
        
        operations['flatten_completely'] = {
            'original': tensor.shape,
            'result': flat.shape,
            'code': 'tensor.reshape(-1)',
            'explanation': 'Flatten to 1D: (32, 28, 28) → (25088,)'
        }
        
        # 6. Reshape to grid
        flat_tensor = np.random.randn(784)
        grid = flat_tensor.reshape(28, 28)
        
        operations['to_grid'] = {
            'original': flat_tensor.shape,
            'result': grid.shape,
            'code': 'tensor.reshape(28, 28)',
            'explanation': 'Reshape to grid: (784,) → (28, 28)'
        }
        
        # 7. Split and merge dimensions
        tensor = np.random.randn(32, 16, 64)  # (batch, seq, features)
        # Split features into heads
        batch, seq, features = tensor.shape
        num_heads = 8
        head_dim = features // num_heads
        reshaped = tensor.reshape(batch, seq, num_heads, head_dim)
        
        operations['split_dimension'] = {
            'original': tensor.shape,
            'result': reshaped.shape,
            'code': 'tensor.reshape(batch, seq, num_heads, head_dim)',
            'explanation': 'Split features: (32, 16, 64) → (32, 16, 8, 8)'
        }
        
        return operations


# Example usage
manipulator = ShapeManipulation()
ops = manipulator.reshape_operations()

print("\nShape Manipulation Operations:")
for op_name, details in ops.items():
    print(f"\n{op_name}:")
    print(f"  Original shape: {details['original']}")
    print(f"  Result shape: {details['result']}")
    print(f"  Code: {details['code']}")
    print(f"  Explanation: {details['explanation']}")
```

### 6.2 Using Negative Indices in Reshaping

```python
class NegativeIndexReshaping:
    """Using negative indices for flexible reshaping."""
    
    def flexible_reshaping(self):
        """Demonstrate flexible reshaping with negative indices."""
        examples = {}
        
        # Example 1: Flatten all except last dimension (works for any rank)
        for rank in [3, 4, 5]:
            if rank == 3:
                tensor = np.random.randn(32, 100, 512)
            elif rank == 4:
                tensor = np.random.randn(32, 64, 64, 3)
            else:
                tensor = np.random.randn(8, 16, 64, 64, 3)
            
            # This code works for all ranks!
            reshaped = tensor.reshape(-1, tensor.shape[-1])
            
            examples[f'flatten_except_last_rank_{rank}'] = {
                'original_shape': tensor.shape,
                'reshaped_shape': reshaped.shape,
                'code': 'tensor.reshape(-1, tensor.shape[-1])',
                'benefit': 'Same code works regardless of rank'
            }
        
        # Example 2: Add channel dimension if not present
        def ensure_channel_dimension(tensor):
            """Ensure tensor has channel dimension at the end."""
            if len(tensor.shape) == 2:
                # (H, W) → (H, W, 1)
                return np.expand_dims(tensor, axis=-1)
            else:
                # Already has channels
                return tensor
        
        tensor_2d = np.random.randn(28, 28)
        tensor_3d = np.random.randn(28, 28, 3)
        
        result_2d = ensure_channel_dimension(tensor_2d)
        result_3d = ensure_channel_dimension(tensor_3d)
        
        examples['ensure_channel_dim'] = {
            'input_2d_shape': tensor_2d.shape,
            'output_2d_shape': result_2d.shape,
            'input_3d_shape': tensor_3d.shape,
            'output_3d_shape': result_3d.shape,
            'code': 'np.expand_dims(tensor, axis=-1)',
            'benefit': 'Add dimension at the end'
        }
        
        # Example 3: Transpose last two dimensions (works for any rank)
        for rank in [3, 4]:
            if rank == 3:
                tensor = np.random.randn(32, 10, 20)
            else:
                tensor = np.random.randn(32, 64, 10, 20)
            
            transposed = np.swapaxes(tensor, -2, -1)
            
            examples[f'swap_last_two_rank_{rank}'] = {
                'original_shape': tensor.shape,
                'transposed_shape': transposed.shape,
                'code': 'np.swapaxes(tensor, -2, -1)',
                'benefit': 'Swap last two dimensions regardless of rank'
            }
        
        return examples


# Example usage
neg_reshape = NegativeIndexReshaping()
examples = neg_reshape.flexible_reshaping()

print("\nFlexible Reshaping with Negative Indices:")
for example_name, details in examples.items():
    print(f"\n{example_name}:")
    for key, value in details.items():
        print(f"  {key}: {value}")
```

---

## 7. Broadcasting Rules

### 7.1 Understanding Broadcasting

```python
class BroadcastingRules:
    """Understanding broadcasting with shapes."""
    
    def demonstrate_broadcasting(self):
        """
        Demonstrate broadcasting rules.
        
        Broadcasting Rules:
        1. If arrays have different ranks, prepend 1s to smaller rank
        2. Arrays are compatible if dimensions are equal or one is 1
        3. Dimensions of size 1 are stretched to match
        """
        examples = {}
        
        # Example 1: Scalar broadcasting
        tensor = np.random.randn(3, 4, 5)
        scalar = 2.0
        result = tensor * scalar
        
        examples['scalar_broadcast'] = {
            'tensor_shape': tensor.shape,
            'scalar_shape': '()',
            'result_shape': result.shape,
            'explanation': 'Scalar broadcasts to all elements'
        }
        
        # Example 2: Vector broadcasting
        tensor = np.random.randn(32, 100, 512)
        bias = np.random.randn(512)
        result = tensor + bias
        
        examples['vector_broadcast'] = {
            'tensor_shape': tensor.shape,     # (32, 100, 512)
            'bias_shape': bias.shape,         # (512,)
            'result_shape': result.shape,     # (32, 100, 512)
            'explanation': 'Bias broadcasts across first two dimensions'
        }
        
        # Example 3: Matrix broadcasting
        tensor = np.random.randn(32, 64, 64, 3)
        # Add per-channel scaling
        scale = np.random.randn(1, 1, 1, 3)
        result = tensor * scale
        
        examples['per_channel_broadcast'] = {
            'tensor_shape': tensor.shape,     # (32, 64, 64, 3)
            'scale_shape': scale.shape,       # (1, 1, 1, 3)
            'result_shape': result.shape,     # (32, 64, 64, 3)
            'explanation': 'Scale broadcasts across batch and spatial dimensions'
        }
        
        # Example 4: Using negative indices for broadcasting
        tensor = np.random.randn(32, 100, 512)
        # Add per-feature bias (works for any middle dimension size)
        bias = np.random.randn(tensor.shape[-1])
        result = tensor + bias
        
        examples['negative_index_broadcast'] = {
            'tensor_shape': tensor.shape,
            'bias_shape': bias.shape,
            'result_shape': result.shape,
            'explanation': 'Bias created using shape[-1] for flexibility'
        }
        
        return examples
    
    def check_broadcast_compatibility(self, shape1: tuple, shape2: tuple) -> dict:
        """
        Check if two shapes are broadcast-compatible.
        
        Args:
            shape1: First shape
            shape2: Second shape
            
        Returns:
            Dictionary with compatibility info
        """
        # Pad shapes to same length
        max_len = max(len(shape1), len(shape2))
        padded1 = (1,) * (max_len - len(shape1)) + shape1
        padded2 = (1,) * (max_len - len(shape2)) + shape2
        
        compatible = True
        result_shape = []
        
        for dim1, dim2 in zip(padded1, padded2):
            if dim1 == dim2:
                result_shape.append(dim1)
            elif dim1 == 1:
                result_shape.append(dim2)
            elif dim2 == 1:
                result_shape.append(dim1)
            else:
                compatible = False
                result_shape.append(None)
        
        return {
            'shape1': shape1,
            'shape2': shape2,
            'padded_shape1': padded1,
            'padded_shape2': padded2,
            'compatible': compatible,
            'result_shape': tuple(result_shape) if compatible else None
        }


# Example usage
broadcasting = BroadcastingRules()
examples = broadcasting.demonstrate_broadcasting()

print("\nBroadcasting Examples:")
for example_name, details in examples.items():
    print(f"\n{example_name}:")
    for key, value in details.items():
        print(f"  {key}: {value}")

# Check compatibility
print("\n\nBroadcast Compatibility Checks:")
test_cases = [
    ((32, 100, 512), (512,)),
    ((32, 64, 64, 3), (1, 1, 1, 3)),
    ((32, 100, 512), (32, 1, 512)),
    ((32, 100, 512), (16, 100, 512)),  # Incompatible
]

for shape1, shape2 in test_cases:
    result = broadcasting.check_broadcast_compatibility(shape1, shape2)
    print(f"\n{shape1} + {shape2}:")
    print(f"  Compatible: {result['compatible']}")
    if result['compatible']:
        print(f"  Result shape: {result['result_shape']}")
```

---

## 8. Best Practices

### 8.1 When to Use Positive vs Negative Indices

```python
class IndexingBestPractices:
    """Best practices for using positive vs negative indices."""
    
    def get_recommendations(self):
        """Get recommendations for index usage."""
        recommendations = {
            'use_positive_indices': {
                'scenarios': [
                    'Batch dimension (always first)',
                    'Sequence length in NLP',
                    'Height dimension in images',
                    'When dimension position is fixed by convention',
                    'When reading code should emphasize "from start"'
                ],
                'examples': [
                    'batch_size = tensor.shape[0]',
                    'seq_len = tensor.shape[1]',
                    'height = tensor.shape[1]  # for images'
                ]
            },
            
            'use_negative_indices': {
                'scenarios': [
                    'Feature/channel dimension (usually last)',
                    'When code should work across different ranks',
                    'Last few dimensions that are logically "at the end"',
                    'When emphasizing "from end" is clearer',
                    'Hidden dimensions in transformer models',
                    'Embedding dimensions'
                ],
                'examples': [
                    'num_channels = tensor.shape[-1]',
                    'features = tensor.shape[-1]',
                    'hidden_dim = tensor.shape[-1]',
                    'flatten = tensor.reshape(-1, tensor.shape[-1])'
                ]
            },
            
            'prefer_negative_for_flexibility': {
                'reason': 'Code that works across different tensor ranks',
                'example_scenario': 'Function that processes both 3D and 4D tensors',
                'code_example': '''
def get_num_channels(tensor):
    """Works for both (H, W, C) and (B, H, W, C)."""
    return tensor.shape[-1]  # Always gets channels
                '''
            },
            
            'prefer_positive_for_clarity': {
                'reason': 'When dimension meaning is tied to position',
                'example_scenario': 'Image processing with fixed format',
                'code_example': '''
def get_image_dimensions(image_batch):
    """Assuming format: (batch, height, width, channels)."""
    batch_size = image_batch.shape[0]
    height = image_batch.shape[1]
    width = image_batch.shape[2]
    channels = image_batch.shape[3]
    return batch_size, height, width, channels
                '''
            }
        }
        
        return recommendations


# Example usage
best_practices = IndexingBestPractices()
recommendations = best_practices.get_recommendations()

print("\nBest Practices for Index Usage:")
for category, info in recommendations.items():
    print(f"\n{category.upper().replace('_', ' ')}:")
    if 'scenarios' in info:
        print("  Scenarios:")
        for scenario in info['scenarios']:
            print(f"    - {scenario}")
        print("  Examples:")
        for example in info['examples']:
            print(f"    - {example}")
    else:
        for key, value in info.items():
            print(f"  {key}: {value}")
```

### 8.2 Common Pitfalls

```python
class CommonPitfalls:
    """Common pitfalls when working with tensor shapes."""
    
    def get_pitfalls(self):
        """Document common pitfalls and solutions."""
        pitfalls = {}
        
        # Pitfall 1: Forgetting about singleton dimensions
        pitfalls['singleton_dimensions'] = {
            'problem': 'Shape (32, 28, 28, 1) vs (32, 28, 28)',
            'issue': 'Operations may fail or produce unexpected results',
            'solution': 'Use np.squeeze() to remove or np.expand_dims() to add',
            'example': '''
# Remove singleton dimension
tensor_with_1 = np.random.randn(32, 28, 28, 1)
tensor_without_1 = np.squeeze(tensor_with_1, axis=-1)
# Or add singleton dimension
tensor = np.random.randn(32, 28, 28)
tensor_with_1 = np.expand_dims(tensor, axis=-1)
            '''
        }
        
        # Pitfall 2: Mixing TensorFlow and PyTorch formats
        pitfalls['format_mismatch'] = {
            'problem': 'NHWC (TF) vs NCHW (PyTorch)',
            'issue': 'Models expect specific channel position',
            'solution': 'Explicitly transpose or document expected format',
            'example': '''
# Convert TensorFlow to PyTorch format
tf_tensor = np.random.randn(32, 224, 224, 3)  # NHWC
torch_tensor = np.transpose(tf_tensor, (0, 3, 1, 2))  # NCHW

# Convert PyTorch to TensorFlow format
torch_tensor = np.random.randn(32, 3, 224, 224)  # NCHW
tf_tensor = np.transpose(torch_tensor, (0, 2, 3, 1))  # NHWC
            '''
        }
        
        # Pitfall 3: Hardcoding dimension indices
        pitfalls['hardcoded_indices'] = {
            'problem': 'Using fixed indices when shapes can vary',
            'issue': 'Code breaks when tensor rank changes',
            'solution': 'Use negative indices for last dimensions',
            'example': '''
# Bad: Hardcoded
num_features = tensor.shape[2]  # Breaks if rank changes

# Good: Flexible
num_features = tensor.shape[-1]  # Always gets last dimension
            '''
        }
        
        # Pitfall 4: Incorrect reshape
        pitfalls['incorrect_reshape'] = {
            'problem': 'Reshaping with incompatible sizes',
            'issue': 'Total elements must match',
            'solution': 'Use -1 for automatic dimension calculation',
            'example': '''
tensor = np.random.randn(32, 28, 28)  # 25088 elements

# Bad: Wrong size
# reshaped = tensor.reshape(32, 784)  # Error if dimensions don't match

# Good: Use -1
reshaped = tensor.reshape(32, -1)  # Automatically calculates: (32, 784)
reshaped = tensor.reshape(-1, 784)  # Automatically calculates: (32, 784)
            '''
        }
        
        # Pitfall 5: Broadcasting surprises
        pitfalls['broadcasting_surprises'] = {
            'problem': 'Unexpected broadcasting behavior',
            'issue': 'Silent errors or wrong results',
            'solution': 'Explicitly check shapes before operations',
            'example': '''
# Unexpected broadcasting
tensor1 = np.random.randn(32, 100, 512)
tensor2 = np.random.randn(100, 512)  # Missing batch dimension
result = tensor1 + tensor2  # Works but may not be intended

# Better: Explicit shapes
assert tensor1.shape[1:] == tensor2.shape, "Shape mismatch!"
result = tensor1 + tensor2[np.newaxis, :, :]  # Explicit broadcast
            '''
        }
        
        return pitfalls


# Example usage
pitfalls_demo = CommonPitfalls()
pitfalls = pitfalls_demo.get_pitfalls()

print("\nCommon Pitfalls:")
for pitfall_name, details in pitfalls.items():
    print(f"\n{pitfall_name.upper().replace('_', ' ')}:")
    print(f"  Problem: {details['problem']}")
    print(f"  Issue: {details['issue']}")
    print(f"  Solution: {details['solution']}")
    print(f"  Example:{details['example']}")
```

---

## Quick Reference Guide

### Shape Indexing Cheat Sheet

```
Tensor Shape: (2, 3, 4, 5, 6)

Positive Indices (from start):
  Index 0: 2  (first dimension)
  Index 1: 3  (second dimension)
  Index 2: 4  (third dimension)
  Index 3: 5  (fourth dimension)
  Index 4: 6  (fifth dimension)

Negative Indices (from end):
  Index -5: 2  (first dimension = 5th from end)
  Index -4: 3  (second dimension = 4th from end)
  Index -3: 4  (third dimension = 3rd from end)
  Index -2: 5  (fourth dimension = 2nd from end)
  Index -1: 6  (fifth dimension = last)

Conversion Formula:
  negative_index = positive_index - len(shape)
  positive_index = negative_index + len(shape)
```

### Common Operations

```python
# Get specific dimensions
batch_size = tensor.shape[0]      # Always batch (positive)
num_features = tensor.shape[-1]   # Always features (negative)

# Reshape operations
flattened = tensor.reshape(tensor.shape[0], -1)           # Flatten except batch
expanded = np.expand_dims(tensor, axis=-1)                # Add last dimension
squeezed = np.squeeze(tensor, axis=-1)                    # Remove last dimension
transposed = np.swapaxes(tensor, -2, -1)                  # Swap last two

# Broadcasting
bias = np.random.randn(tensor.shape[-1])                  # Create matching bias
result = tensor + bias                                     # Broadcasts correctly
```

### When to Use Which

**Use Positive Indices [0, 1, 2, ...] When:**
- Accessing batch dimension (always position 0)
- Dimension meaning is tied to absolute position
- Working with fixed-format data (e.g., specific image format)
- Code clarity requires "counting from start"

**Use Negative Indices [-1, -2, -3, ...] When:**
- Accessing feature/channel dimension (usually last)
- Code should work across different tensor ranks
- Last few dimensions are logically "at the end"
- Want flexibility and future-proof code

---

## GCP-Specific Considerations

### Vertex AI and TensorFlow

```python
# Vertex AI expects specific input shapes
# Images: (batch_size, height, width, channels) - NHWC format
input_tensor = np.random.randn(1, 224, 224, 3)

# For prediction
predictions = model.predict(input_tensor)

# Accessing output shape
output_features = predictions.shape[-1]  # Number of classes
```

### BigQuery ML with TensorFlow

```sql
-- BigQuery ML expects flattened features for tabular data
-- Shape handling happens automatically, but understanding helps:
CREATE OR REPLACE MODEL `project.dataset.model`
OPTIONS(
  model_type='DNN_CLASSIFIER',
  hidden_units=[128, 64, 32]  -- These define internal shape transformations
) AS
SELECT
  features,  -- Will be reshaped internally
  label
FROM `project.dataset.training_data`;
```

---

## Additional Resources

- **NumPy Shape Documentation**: https://numpy.org/doc/stable/reference/arrays.ndarray.html
- **TensorFlow Tensor Shapes**: https://www.tensorflow.org/guide/tensor
- **PyTorch Tensor Tutorial**: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
- **Vertex AI Training**: https://cloud.google.com/vertex-ai/docs/training/overview

---

This comprehensive guide covers tensor shapes, positive and negative indexing, and best practices for ML engineering on Google Cloud Platform!