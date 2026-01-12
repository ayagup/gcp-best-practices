# TPU: bfloat16 vs float32

## Overview
**bfloat16** (Brain Floating Point 16-bit) and **float32** (32-bit floating point) are numerical formats used in deep learning. TPUs are optimized for bfloat16, offering significant performance and memory advantages over float32.

## Data Type Comparison

| Feature | bfloat16 | float32 |
|---------|----------|---------|
| **Bit Width** | 16 bits | 32 bits |
| **Sign Bit** | 1 bit | 1 bit |
| **Exponent Bits** | 8 bits | 8 bits |
| **Mantissa Bits** | 7 bits | 23 bits |
| **Range** | Same as float32 (~10⁻³⁸ to 10³⁸) | ~10⁻³⁸ to 10³⁸ |
| **Precision** | ~3 decimal digits | ~7 decimal digits |
| **Memory Usage** | 50% of float32 | Baseline |
| **TPU Performance** | 2x faster | Baseline |

## Visual Representation

```
float32:  [Sign: 1 bit][Exponent: 8 bits][Mantissa: 23 bits]
          ├─┤├────────┤├──────────────────────────┤

bfloat16: [Sign: 1 bit][Exponent: 8 bits][Mantissa: 7 bits]
          ├─┤├────────┤├──────┤

float16:  [Sign: 1 bit][Exponent: 5 bits][Mantissa: 10 bits]
(IEEE)    ├─┤├────┤├──────────┤
```

## Why bfloat16 for TPUs?

### **1. Hardware Optimization**
TPUs have dedicated bfloat16 matrix multiply units (MXUs):
```python
# TPU architecture
# - bfloat16 MXUs: 128x128 systolic arrays
# - Optimized for matrix operations in bfloat16
# - Automatic mixed precision support
```

### **2. Performance Benefits**

```python
import tensorflow as tf

# Enable bfloat16 on TPU
policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
tf.keras.mixed_precision.set_global_policy(policy)

# Performance comparison
# bfloat16: ~2x faster training
# float32:  baseline speed
```

**Throughput Comparison:**
```
TPU v4 Pod (bfloat16): ~275 PFLOPS (int8 sparse)
TPU v4 Pod (bfloat16): ~137 PFLOPS (bfloat16 dense)
TPU v4 Pod (float32):  ~69 PFLOPS (float32)

Speed improvement: 2x with bfloat16 vs float32
```

### **3. Memory Efficiency**

```python
import numpy as np

# Memory comparison
model_size_float32 = 1_000_000  # 1M parameters
memory_float32 = model_size_float32 * 4  # 4 bytes per float32
memory_bfloat16 = model_size_float32 * 2  # 2 bytes per bfloat16

print(f"float32 memory: {memory_float32 / 1e6:.1f} MB")  # 4.0 MB
print(f"bfloat16 memory: {memory_bfloat16 / 1e6:.1f} MB")  # 2.0 MB
print(f"Memory savings: {(1 - memory_bfloat16/memory_float32) * 100:.0f}%")  # 50%
```

**Benefits:**
- ✅ 50% less memory usage
- ✅ Larger batch sizes
- ✅ Bigger models fit in memory
- ✅ Faster data transfers

## bfloat16 Advantages Over float16

### **Numerical Stability**

```python
# Range comparison
import numpy as np

# bfloat16: Same exponent range as float32
bfloat16_max = 3.38953e+38  # ~float32 max
bfloat16_min = 1.17549e-38  # ~float32 min

# float16: Limited range (commonly overflows)
float16_max = 65504
float16_min = 6.10352e-05

# Example: Gradient overflow with float16
large_gradient = 100000.0
# float16: OVERFLOW! ❌
# bfloat16: 100000.0 ✅ (within range)
```

### **No Loss Scaling Required**

```python
# float16 (IEEE) - requires loss scaling
from tensorflow.keras import mixed_precision

# With float16, need loss scaling to prevent underflow
policy = mixed_precision.Policy('mixed_float16')
# Gradients are scaled by 128 or more to prevent underflow

# bfloat16 - no loss scaling needed!
policy = mixed_precision.Policy('mixed_bfloat16')
# Gradients work naturally ✅
```

## Implementation on TPU

### **Method 1: Automatic Mixed Precision (Recommended)**

```python
import tensorflow as tf

# Connect to TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

# Create TPU strategy
strategy = tf.distribute.TPUStrategy(resolver)

# Enable bfloat16 mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
tf.keras.mixed_precision.set_global_policy(policy)

# Build model (automatically uses bfloat16)
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu'),  # bfloat16 compute
        tf.keras.layers.Dense(256, activation='relu'),  # bfloat16 compute
        tf.keras.layers.Dense(10)  # Output in float32
    ])
    
    # Note: Logits are automatically cast to float32 for loss calculation
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

# Train (automatically uses bfloat16)
model.fit(train_dataset, epochs=10)
```

### **Method 2: Explicit Casting**

```python
import tensorflow as tf

# Manual bfloat16 casting
@tf.function
def train_step(inputs, labels):
    # Cast inputs to bfloat16
    inputs = tf.cast(inputs, tf.bfloat16)
    
    with tf.GradientTape() as tape:
        # Forward pass in bfloat16
        predictions = model(inputs, training=True)
        
        # Cast to float32 for loss (more accurate)
        predictions = tf.cast(predictions, tf.float32)
        loss = loss_fn(labels, predictions)
    
    # Gradients computed in float32
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Update in bfloat16
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# Execute on TPU
with strategy.scope():
    for epoch in range(num_epochs):
        for batch_inputs, batch_labels in train_dataset:
            loss = train_step(batch_inputs, batch_labels)
```

### **Method 3: PyTorch on TPU**

```python
import torch
import torch_xla.core.xla_model as xm

# Enable bfloat16 on TPU
device = xm.xla_device()

# Create model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 10)
).to(device)

# Training loop with automatic mixed precision
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Automatically uses bfloat16 on TPU
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = model(data)
            loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        xm.optimizer_step(optimizer)  # TPU-optimized step
```

### **Method 4: JAX on TPU**

```python
import jax
import jax.numpy as jnp
from jax import random

# JAX automatically uses bfloat16 on TPU with policy
from jax.experimental import enable_x64
# Keep default behavior (bfloat16 on TPU)

# Define model
def model(params, x):
    # Operations automatically use bfloat16 on TPU
    x = jnp.dot(x, params['w1']) + params['b1']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['w2']) + params['b2']
    return x

# Loss function
@jax.jit
def loss_fn(params, x, y):
    preds = model(params, x)
    # Loss computed in float32 for stability
    return jnp.mean((preds - y) ** 2)

# Gradient computation (automatically bfloat16)
grad_fn = jax.grad(loss_fn)

# Training step
@jax.jit
def train_step(params, x, y, learning_rate):
    grads = grad_fn(params, x, y)
    # Update in bfloat16
    params = jax.tree_map(
        lambda p, g: p - learning_rate * g,
        params, grads
    )
    return params
```

## Mixed Precision Strategy

### **What Runs in bfloat16 vs float32**

```python
# Typical mixed precision setup
"""
bfloat16 Operations:
├── Matrix multiplications (Conv