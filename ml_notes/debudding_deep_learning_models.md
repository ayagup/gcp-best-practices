# Debugging Deep Learning Models: Common & Best Practices

## Overview

Model debugging is critical for identifying why a deep learning model isn't performing as expected. This guide covers systematic approaches to diagnose and fix common model issues.

## Common Model Problems

### 1. **Training Issues**
- Model not learning (loss not decreasing)
- Loss exploding (NaN/Inf values)
- Training too slow
- Overfitting
- Underfitting
- Unstable training

### 2. **Performance Issues**
- Low accuracy on training data
- Low accuracy on validation data
- Large train-val gap
- Poor generalization
- Inconsistent predictions

### 3. **Convergence Issues**
- Loss plateaus early
- Oscillating loss
- Slow convergence
- Divergence

## Systematic Debugging Approach

### Step 1: Verify the Setup

#### 1.1 Check Model Can Overfit a Single Batch
```python
import tensorflow as tf
import numpy as np

def test_overfit_single_batch(model, dataset, epochs=100):
    """Test if model can overfit a single batch (sanity check)"""
    
    # Get single batch
    for batch_x, batch_y in dataset.take(1):
        single_batch_x = batch_x
        single_batch_y = batch_y
    
    print(f"Batch shape: {single_batch_x.shape}")
    print(f"Labels shape: {single_batch_y.shape}")
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train on single batch
    print("\nTraining on single batch...")
    history = []
    
    for epoch in range(epochs):
        loss, acc = model.train_on_batch(single_batch_x, single_batch_y)
        history.append({'loss': loss, 'accuracy': acc})
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}")
    
    final_loss = history[-1]['loss']
    final_acc = history[-1]['accuracy']
    
    print(f"\nFinal: Loss={final_loss:.4f}, Acc={final_acc:.4f}")
    
    # Check if model learned
    if final_loss < 0.1 and final_acc > 0.95:
        print("✓ Model can learn - setup is correct")
        return True
    else:
        print("⚠️ Model failed to overfit single batch!")
        print("   Possible issues:")
        print("   - Learning rate too low")
        print("   - Model architecture too simple")
        print("   - Bug in loss function")
        print("   - Bug in model forward pass")
        return False

# Usage
test_overfit_single_batch(model, train_dataset)
```

#### 1.2 Verify Loss Function
```python
def verify_loss_function(model, dataset):
    """Verify loss function is correctly implemented"""
    
    for batch_x, batch_y in dataset.take(1):
        # Random predictions
        random_preds = tf.random.uniform(
            shape=(batch_x.shape[0], model.output_shape[-1]),
            minval=0,
            maxval=1
        )
        random_preds = random_preds / tf.reduce_sum(random_preds, axis=1, keepdims=True)
        
        # Perfect predictions
        perfect_preds = tf.one_hot(batch_y, depth=model.output_shape[-1])
        
        # Compute losses
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        random_loss = loss_fn(batch_y, random_preds).numpy()
        perfect_loss = loss_fn(batch_y, perfect_preds).numpy()
        
        # Get model loss
        model_preds = model(batch_x, training=False)
        model_loss = loss_fn(batch_y, model_preds).numpy()
        
        print("Loss Verification:")
        print(f"Random predictions loss: {random_loss:.4f}")
        print(f"Perfect predictions loss: {perfect_loss:.4f}")
        print(f"Model predictions loss: {model_loss:.4f}")
        
        # Expected loss for random guessing
        num_classes = model.output_shape[-1]
        expected_random_loss = -np.log(1.0 / num_classes)
        print(f"Expected random loss: {expected_random_loss:.4f}")
        
        # Sanity checks
        if perfect_loss > 0.1:
            print("⚠️ WARNING: Perfect predictions have high loss!")
            return False
        
        if abs(random_loss - expected_random_loss) > 0.5:
            print("⚠️ WARNING: Random loss differs from expected!")
            return False
        
        if model_loss < random_loss:
            print("✓ Model is better than random")
        else:
            print("⚠️ Model is no better than random guessing")
        
        return True

# Usage
verify_loss_function(model, train_dataset)
```

#### 1.3 Check Model Architecture
```python
def debug_model_architecture(model):
    """Debug model architecture issues"""
    
    print("=" * 60)
    print("MODEL ARCHITECTURE DEBUG")
    print("=" * 60)
    
    # Print summary
    model.summary()
    
    # Check for common issues
    print("\nArchitecture Checks:")
    
    # 1. Check output layer
    output_shape = model.output_shape
    print(f"1. Output shape: {output_shape}")
    
    if isinstance(output_shape[-1], int):
        print(f"   Number of output units: {output_shape[-1]}")
    
    # 2. Check activation functions
    print("\n2. Layer activations:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'activation'):
            print(f"   Layer {i} ({layer.name}): {layer.activation.__name__}")
    
    # 3. Check for BatchNorm before activation
    print("\n3. BatchNorm placement:")
    for i in range(len(model.layers) - 1):
        current_layer = model.layers[i]
        next_layer = model.layers[i + 1]
        
        if 'batch_norm' in current_layer.name.lower():
            if hasattr(next_layer, 'activation'):
                print(f"   ✓ BatchNorm at {i} before activation at {i+1}")
            else:
                print(f"   ⚠️ BatchNorm at {i} not before activation")
    
    # 4. Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"\n4. Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Non-trainable: {non_trainable_params:,}")
    
    # 5. Check for bottlenecks
    print("\n5. Potential bottlenecks:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'units'):
            if layer.units < 10:
                print(f"   ⚠️ Layer {i} ({layer.name}): Only {layer.units} units")
    
    print("=" * 60)

# Usage
debug_model_architecture(model)
```

### Step 2: Monitor Training Process

#### 2.1 Custom Training Loop with Detailed Logging
```python
def train_with_detailed_logging(model, train_dataset, val_dataset, epochs=10):
    """Training loop with detailed logging for debugging"""
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    train_loss_metric = tf.keras.metrics.Mean()
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'gradient_norms': [],
        'weight_norms': []
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training
        batch_losses = []
        gradient_norms = []
        
        for batch_idx, (x, y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                predictions = model(x, training=True)
                loss = loss_fn(y, predictions)
            
            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_weights)
            
            # Check for NaN/Inf in gradients
            for i, grad in enumerate(gradients):
                if grad is not None:
                    if tf.reduce_any(tf.math.is_nan(grad)):
                        print(f"⚠️ NaN gradient in layer {i} at batch {batch_idx}")
                    if tf.reduce_any(tf.math.is_inf(grad)):
                        print(f"⚠️ Inf gradient in layer {i} at batch {batch_idx}")
            
            # Compute gradient norm
            grad_norm = tf.sqrt(sum([tf.reduce_sum(tf.square(g)) for g in gradients if g is not None]))
            gradient_norms.append(grad_norm.numpy())
            
            # Apply gradients
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            
            # Update metrics
            train_loss_metric.update_state(loss)
            train_acc_metric.update_state(y, predictions)
            batch_losses.append(loss.numpy())
            
            # Log every 10 batches
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: "
                      f"Loss={loss.numpy():.4f}, "
                      f"GradNorm={grad_norm.numpy():.4f}")
        
        # Epoch metrics
        train_loss = train_loss_metric.result().numpy()
        train_acc = train_acc_metric.result().numpy()
        
        # Validation
        val_loss_metric = tf.keras.metrics.Mean()
        val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        
        for x, y in val_dataset:
            predictions = model(x, training=False)
            loss = loss_fn(y, predictions)
            val_loss_metric.update_state(loss)
            val_acc_metric.update_state(y, predictions)
        
        val_loss = val_loss_metric.result().numpy()
        val_acc = val_acc_metric.result().numpy()
        
        # Compute weight norms
        weight_norm = tf.sqrt(sum([tf.reduce_sum(tf.square(w)) for w in model.trainable_weights]))
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['gradient_norms'].append(np.mean(gradient_norms))
        history['weight_norms'].append(weight_norm.numpy())
        
        # Print epoch summary
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Avg Gradient Norm: {np.mean(gradient_norms):.4f}")
        print(f"  Weight Norm: {weight_norm.numpy():.4f}")
        
        # Check for issues
        if np.isnan(train_loss) or np.isinf(train_loss):
            print("⚠️ TRAINING STOPPED: NaN/Inf loss detected!")
            break
        
        if len(history['train_loss']) > 1:
            loss_change = history['train_loss'][-1] - history['train_loss'][-2]
            if abs(loss_change) < 1e-6:
                print("⚠️ WARNING: Loss not changing")
        
        # Reset metrics
        train_loss_metric.reset_states()
        train_acc_metric.reset_states()
    
    return history

# Usage
history = train_with_detailed_logging(model, train_dataset, val_dataset, epochs=10)
```

#### 2.2 Visualize Training Metrics
```python
import matplotlib.pyplot as plt

def visualize_training_debug(history):
    """Visualize training metrics for debugging"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Train-Val gap
    gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    axes[0, 2].plot(epochs, gap, 'g-')
    axes[0, 2].set_title('Train-Val Accuracy Gap')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Gap')
    axes[0, 2].axhline(y=0, color='r', linestyle='--')
    axes[0, 2].grid(True)
    
    # Gradient norms
    axes[1, 0].plot(epochs, history['gradient_norms'], 'purple')
    axes[1, 0].set_title('Gradient Norms')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    
    # Weight norms
    axes[1, 1].plot(epochs, history['weight_norms'], 'orange')
    axes[1, 1].set_title('Weight Norms')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Weight Norm')
    axes[1, 1].grid(True)
    
    # Loss change rate
    if len(history['train_loss']) > 1:
        loss_changes = np.diff(history['train_loss'])
        axes[1, 2].plot(range(2, len(epochs) + 1), loss_changes, 'brown')
        axes[1, 2].set_title('Loss Change Rate')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Δ Loss')
        axes[1, 2].axhline(y=0, color='r', linestyle='--')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_debug_plots.png', dpi=150)
    plt.show()
    
    # Diagnose issues
    print("\n" + "=" * 60)
    print("TRAINING DIAGNOSIS")
    print("=" * 60)
    
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    
    # Check for overfitting
    if final_train_acc > final_val_acc + 0.1:
        print("⚠️ OVERFITTING DETECTED")
        print(f"   Train accuracy ({final_train_acc:.3f}) >> Val accuracy ({final_val_acc:.3f})")
        print("   Recommendations:")
        print("   - Add dropout/regularization")
        print("   - Increase training data")
        print("   - Reduce model complexity")
        print("   - Add data augmentation")
    
    # Check for underfitting
    elif final_train_acc < 0.7:
        print("⚠️ UNDERFITTING DETECTED")
        print(f"   Train accuracy ({final_train_acc:.3f}) is low")
        print("   Recommendations:")
        print("   - Increase model complexity")
        print("   - Train for more epochs")
        print("   - Increase learning rate")
        print("   - Remove regularization")
    
    # Check if not learning
    elif abs(history['train_loss'][0] - history['train_loss'][-1]) < 0.01:
        print("⚠️ MODEL NOT LEARNING")
        print("   Loss is not decreasing")
        print("   Recommendations:")
        print("   - Increase learning rate")
        print("   - Check data preprocessing")
        print("   - Verify loss function")
        print("   - Check for vanishing gradients")
    
    # Check gradient issues
    avg_grad_norm = np.mean(history['gradient_norms'])
    if avg_grad_norm < 1e-7:
        print("⚠️ VANISHING GRADIENTS")
        print(f"   Average gradient norm: {avg_grad_norm:.2e}")
        print("   Recommendations:")
        print("   - Use batch normalization")
        print("   - Try ReLU activation")
        print("   - Use residual connections")
        print("   - Reduce network depth")
    
    elif avg_grad_norm > 100:
        print("⚠️ EXPLODING GRADIENTS")
        print(f"   Average gradient norm: {avg_grad_norm:.2e}")
        print("   Recommendations:")
        print("   - Use gradient clipping")
        print("   - Reduce learning rate")
        print("   - Use batch normalization")
        print("   - Check weight initialization")
    
    else:
        print("✓ Training appears healthy")
        print(f"   Final train acc: {final_train_acc:.3f}")
        print(f"   Final val acc: {final_val_acc:.3f}")
        print(f"   Average gradient norm: {avg_grad_norm:.2e}")
    
    print("=" * 60)

# Usage
visualize_training_debug(history)
```

### Step 3: Debug Specific Issues

#### 3.1 Vanishing/Exploding Gradients
```python
def check_gradient_flow(model, dataset):
    """Check for vanishing/exploding gradients"""
    
    # Get a batch
    for x, y in dataset.take(1):
        batch_x, batch_y = x, y
    
    # Compute gradients
    with tf.GradientTape() as tape:
        predictions = model(batch_x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(batch_y, predictions)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, model.trainable_weights)
    
    # Analyze gradient norms by layer
    print("=" * 60)
    print("GRADIENT FLOW ANALYSIS")
    print("=" * 60)
    
    gradient_info = []
    
    for i, (weight, grad) in enumerate(zip(model.trainable_weights, gradients)):
        if grad is not None:
            grad_norm = tf.norm(grad).numpy()
            grad_mean = tf.reduce_mean(tf.abs(grad)).numpy()
            grad_max = tf.reduce_max(tf.abs(grad)).numpy()
            
            weight_name = weight.name
            
            gradient_info.append({
                'layer': i,
                'name': weight_name,
                'norm': grad_norm,
                'mean': grad_mean,
                'max': grad_max
            })
            
            # Flag issues
            issue = ""
            if grad_norm < 1e-7:
                issue = "⚠️ VANISHING"
            elif grad_norm > 100:
                issue = "⚠️ EXPLODING"
            else:
                issue = "✓"
            
            print(f"{issue} Layer {i:2d} ({weight_name:30s}): "
                  f"norm={grad_norm:.2e}, mean={grad_mean:.2e}, max={grad_max:.2e}")
    
    # Visualize gradient flow
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    layers = [info['layer'] for info in gradient_info]
    norms = [info['norm'] for info in gradient_info]
    means = [info['mean'] for info in gradient_info]
    maxs = [info['max'] for info in gradient_info]
    
    axes[0].plot(layers, norms, 'b-o')
    axes[0].set_title('Gradient Norms by Layer')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Gradient Norm (log scale)')
    axes[0].set_yscale('log')
    axes[0].grid(True)
    
    axes[1].plot(layers, means, 'g-o')
    axes[1].set_title('Mean Absolute Gradient by Layer')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Mean |Gradient| (log scale)')
    axes[1].set_yscale('log')
    axes[1].grid(True)
    
    axes[2].plot(layers, maxs, 'r-o')
    axes[2].set_title('Max Absolute Gradient by Layer')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Max |Gradient| (log scale)')
    axes[2].set_yscale('log')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('gradient_flow.png', dpi=150)
    plt.show()
    
    print("=" * 60)
    
    return gradient_info

# Usage
gradient_info = check_gradient_flow(model, train_dataset)
```

#### 3.2 Debug Learning Rate
```python
def find_optimal_learning_rate(model, dataset, start_lr=1e-7, end_lr=10, num_steps=100):
    """Learning rate range test"""
    
    # Clone model
    test_model = tf.keras.models.clone_model(model)
    test_model.set_weights(model.get_weights())
    
    # Exponential learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        start_lr,
        decay_steps=1,
        decay_rate=(end_lr / start_lr) ** (1 / num_steps)
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    lrs = []
    losses = []
    
    print("Running learning rate range test...")
    
    step = 0
    for x, y in dataset.take(num_steps):
        with tf.GradientTape() as tape:
            predictions = test_model(x, training=True)
            loss = loss_fn(y, predictions)
        
        gradients = tape.gradient(loss, test_model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, test_model.trainable_weights))
        
        current_lr = optimizer.learning_rate(step).numpy()
        lrs.append(current_lr)
        losses.append(loss.numpy())
        
        if step % 10 == 0:
            print(f"Step {step}/{num_steps}: LR={current_lr:.2e}, Loss={loss.numpy():.4f}")
        
        step += 1
        
        # Stop if loss explodes
        if np.isnan(loss.numpy()) or loss.numpy() > losses[0] * 4:
            print("Stopping early - loss exploded")
            break
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate vs Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Smooth losses for easier reading
    smoothed_losses = np.convolve(losses, np.ones(5)/5, mode='valid')
    smoothed_lrs = lrs[2:-2]
    
    plt.plot(smoothed_lrs, smoothed_losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Smoothed Loss')
    plt.title('Learning Rate vs Smoothed Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('lr_range_test.png', dpi=150)
    plt.show()
    
    # Find optimal learning rate (steepest descent)
    gradients_lr = np.gradient(np.array(losses))
    min_gradient_idx = np.argmin(gradients_lr)
    optimal_lr = lrs[min_gradient_idx]
    
    print(f"\n💡 Suggested learning rate: {optimal_lr:.2e}")
    print(f"   (Use 1/10 of this for safe starting point: {optimal_lr/10:.2e})")
    
    return lrs, losses, optimal_lr

# Usage
lrs, losses, optimal_lr = find_optimal_learning_rate(model, train_dataset)
```

#### 3.3 Debug Overfitting
```python
def diagnose_overfitting(model, train_dataset, val_dataset, epochs=20):
    """Diagnose and analyze overfitting"""
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        verbose=1
    )
    
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Calculate overfitting metrics
    final_gap = train_acc[-1] - val_acc[-1]
    max_gap = max([t - v for t, v in zip(train_acc, val_acc)])
    
    # When did overfitting start?
    overfitting_epoch = None
    for i in range(len(val_loss) - 1):
        if val_loss[i+1] > val_loss[i] and train_loss[i+1] < train_loss[i]:
            overfitting_epoch = i
            break
    
    print("=" * 60)
    print("OVERFITTING ANALYSIS")
    print("=" * 60)
    print(f"Final train accuracy: {train_acc[-1]:.4f}")
    print(f"Final val accuracy: {val_acc[-1]:.4f}")
    print(f"Final accuracy gap: {final_gap:.4f}")
    print(f"Maximum accuracy gap: {max_gap:.4f}")
    
    if overfitting_epoch:
        print(f"Overfitting started at epoch: {overfitting_epoch}")
    
    # Diagnosis
    if final_gap > 0.15:
        print("\n⚠️ SEVERE OVERFITTING DETECTED")
        print("\nRecommendations (in order of priority):")
        print("1. Add dropout layers (start with 0.3-0.5)")
        print("2. Add L2 regularization to Dense layers")
        print("3. Use data augmentation")
        print("4. Collect more training data")
        print("5. Reduce model complexity")
        print("6. Add early stopping")
        
        # Suggest specific fixes
        print("\nSpecific code suggestions:")
        print("```python")
        print("# Add dropout")
        print("model.add(Dropout(0.5))")
        print("\n# Add L2 regularization")
        print("Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))")
        print("\n# Data augmentation")
        print("data_augmentation = tf.keras.Sequential([")
        print("    tf.keras.layers.RandomFlip('horizontal'),")
        print("    tf.keras.layers.RandomRotation(0.1),")
        print("    tf.keras.layers.RandomZoom(0.1),")
        print("])")
        print("```")
        
    elif final_gap > 0.05:
        print("\n⚠️ MODERATE OVERFITTING")
        print("\nRecommendations:")
        print("1. Add light dropout (0.2-0.3)")
        print("2. Use early stopping")
        print("3. Consider slight regularization")
        
    else:
        print("\n✓ Overfitting is under control")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_range = range(1, epochs + 1)
    
    axes[0].plot(epochs_range, train_acc, 'b-', label='Train')
    axes[0].plot(epochs_range, val_acc, 'r-', label='Validation')
    if overfitting_epoch:
        axes[0].axvline(x=overfitting_epoch, color='g', linestyle='--', 
                       label=f'Overfitting starts (epoch {overfitting_epoch})')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(epochs_range, train_loss, 'b-', label='Train')
    axes[1].plot(epochs_range, val_loss, 'r-', label='Validation')
    if overfitting_epoch:
        axes[1].axvline(x=overfitting_epoch, color='g', linestyle='--',
                       label=f'Overfitting starts (epoch {overfitting_epoch})')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('overfitting_analysis.png', dpi=150)
    plt.show()
    
    print("=" * 60)
    
    return history

# Usage
history = diagnose_overfitting(model, train_dataset, val_dataset)
```

#### 3.4 Analyze Model Predictions
```python
def analyze_predictions(model, dataset, class_names, num_samples=25):
    """Analyze model predictions to identify error patterns"""
    
    all_predictions = []
    all_labels = []
    all_images = []
    
    # Collect predictions
    for x, y in dataset.take(num_samples):
        predictions = model.predict(x, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        all_images.extend(x.numpy())
        all_predictions.extend(predicted_classes)
        all_labels.extend(y.numpy())
    
    all_images = np.array(all_images[:num_samples])
    all_predictions = np.array(all_predictions[:num_samples])
    all_labels = np.array(all_labels[:num_samples])
    
    # Find correct and incorrect predictions
    correct_mask = all_predictions == all_labels
    incorrect_mask = ~correct_mask
    
    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(incorrect_mask)[0]
    
    print("=" * 60)
    print("PREDICTION ANALYSIS")
    print("=" * 60)
    print(f"Total samples analyzed: {len(all_labels)}")
    print(f"Correct predictions: {np.sum(correct_mask)} ({np.mean(correct_mask)*100:.1f}%)")
    print(f"Incorrect predictions: {np.sum(incorrect_mask)} ({np.mean(incorrect_mask)*100:.1f}%)")
    
    # Confusion patterns
    if len(incorrect_indices) > 0:
        print("\nMost common misclassifications:")
        from collections import Counter
        misclass_pairs = [(all_labels[i], all_predictions[i]) for i in incorrect_indices]
        most_common = Counter(misclass_pairs).most_common(5)
        
        for (true_label, pred_label), count in most_common:
            true_name = class_names[true_label] if class_names else true_label
            pred_name = class_names[pred_label] if class_names else pred_label
            print(f"  {true_name} → {pred_name}: {count} times")
    
    # Visualize incorrect predictions
    if len(incorrect_indices) > 0:
        num_viz = min(9, len(incorrect_indices))
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(num_viz):
            idx = incorrect_indices[i]
            ax = axes[i]
            
            img = all_images[idx]
            if img.min() < 0:
                img = (img + 1) / 2
            
            if img.shape[-1] == 1:
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                ax.imshow(img)
            
            true_label = class_names[all_labels[idx]] if class_names else all_labels[idx]
            pred_label = class_names[all_predictions[idx]] if class_names else all_predictions[idx]
            
            ax.set_title(f'True: {true_label}\nPred: {pred_label}', color='red')
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_viz, 9):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('incorrect_predictions.png', dpi=150)
        plt.show()
    
    print("=" * 60)
    
    return all_predictions, all_labels

# Usage
predictions, labels = analyze_predictions(
    model, 
    val_dataset, 
    class_names=['cat', 'dog', 'bird']
)
```

### Step 4: Advanced Debugging

#### 4.1 Layer-wise Activation Analysis
```python
def analyze_layer_activations(model, dataset):
    """Analyze activations at each layer"""
    
    # Get a batch
    for x, y in dataset.take(1):
        sample_batch = x[:1]  # Use single sample
    
    # Create models for each layer
    layer_outputs = [layer.output for layer in model.layers if len(layer.output_shape) > 1]
    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # Get activations
    activations = activation_model.predict(sample_batch, verbose=0)
    
    print("=" * 60)
    print("LAYER ACTIVATION ANALYSIS")
    print("=" * 60)
    
    for i, (layer, activation) in enumerate(zip(model.layers, activations)):
        if len(activation.shape) <= 1:
            continue
        
        # Statistics
        mean_activation = np.mean(activation)
        std_activation = np.std(activation)
        max_activation = np.max(activation)
        min_activation = np.min(activation)
        
        # Check for dead neurons
        dead_percentage = np.mean(activation == 0) * 100
        
        print(f"\nLayer {i}: {layer.name}")
        print(f"  Shape: {activation.shape}")
        print(f"  Mean: {mean_activation:.4f}")
        print(f"  Std: {std_activation:.4f}")
        print(f"  Range: [{min_activation:.4f}, {max_activation:.4f}]")
        print(f"  Dead neurons: {dead_percentage:.1f}%")
        
        # Flag issues
        if dead_percentage > 50:
            print(f"  ⚠️ WARNING: >50% dead neurons!")
        if std_activation < 0.01:
            print(f"  ⚠️ WARNING: Very low activation variance")
        if max_activation > 1000:
            print(f"  ⚠️ WARNING: Very high activation values")
    
    # Visualize activations for convolutional layers
    conv_layers = [i for i, layer in enumerate(model.layers) 
                   if 'conv' in layer.name.lower()]
    
    if conv_layers:
        fig, axes = plt.subplots(len(conv_layers), 8, figsize=(16, 2*len(conv_layers)))
        if len(conv_layers) == 1:
            axes = axes.reshape(1, -1)
        
        for row, layer_idx in enumerate(conv_layers):
            activation = activations[layer_idx][0]  # First sample
            
            for col in range(min(8, activation.shape[-1])):
                ax = axes[row, col]
                ax.imshow(activation[:, :, col], cmap='viridis')
                ax.axis('off')
                if col == 0:
                    ax.set_title(f'Layer {layer_idx}', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('layer_activations.png', dpi=150)
        plt.show()
    
    print("=" * 60)

# Usage
analyze_layer_activations(model, train_dataset)
```

#### 4.2 Check Weight Initialization
```python
def analyze_weight_initialization(model):
    """Analyze if weights are properly initialized"""
    
    print("=" * 60)
    print("WEIGHT INITIALIZATION ANALYSIS")
    print("=" * 60)
    
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        
        if len(weights) == 0:
            continue
        
        print(f"\nLayer {i}: {layer.name}")
        
        # Analyze kernel weights
        if len(weights) > 0:
            kernel = weights[0]
            print(f"  Kernel shape: {kernel.shape}")
            print(f"  Kernel mean: {np.mean(kernel):.6f}")
            print(f"  Kernel std: {np.std(kernel):.6f}")
            print(f"  Kernel range: [{np.min(kernel):.6f}, {np.max(kernel):.6f}]")
            
            # Check for issues
            if np.std(kernel) < 0.001:
                print("  ⚠️ WARNING: Weights have very low variance!")
            if np.mean(kernel) > 0.1 or np.mean(kernel) < -0.1:
                print("  ⚠️ WARNING: Weights have high mean (not centered at 0)!")
            if np.max(np.abs(kernel)) > 10:
                print("  ⚠️ WARNING: Some weights are very large!")
        
        # Analyze bias
        if len(weights) > 1:
            bias = weights[1]
            print(f"  Bias shape: {bias.shape}")
            print(f"  Bias mean: {np.mean(bias):.6f}")
            print(f"  Bias std: {np.std(bias):.6f}")
            
            if np.max(np.abs(bias)) > 10:
                print("  ⚠️ WARNING: Some biases are very large!")
    
    print("=" * 60)

# Usage
analyze_weight_initialization(model)
```

#### 4.3 Integrated Gradients (Feature Importance)
```python
def compute_integrated_gradients(model, baseline, image, target_class, m_steps=50):
    """Compute integrated gradients for interpretability"""
    
    # Generate interpolated inputs
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
    interpolated_images = []
    
    for alpha in alphas:
        interpolated = baseline + alpha * (image - baseline)
        interpolated_images.append(interpolated)
    
    interpolated_images = tf.convert_to_tensor(interpolated_images, dtype=tf.float32)
    
    # Compute gradients
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        predictions = model(interpolated_images)
        target_predictions = predictions[:, target_class]
    
    gradients = tape.gradient(target_predictions, interpolated_images)
    
    # Approximate integral using trapezoidal rule
    grads = (gradients[:-1] + gradients[1:]) / 2.0
    integrated_grads = tf.reduce_mean(grads, axis=0)
    
    # Multiply by input difference
    integrated_grads = (image - baseline) * integrated_grads
    
    return integrated_grads

def visualize_attributions(model, image, true_label, class_names):
    """Visualize what the model is looking at"""
    
    # Create baseline (black image)
    baseline = tf.zeros_like(image)
    
    # Get prediction
    prediction = model.predict(tf.expand_dims(image, 0), verbose=0)[0]
    predicted_class = np.argmax(prediction)
    
    # Compute integrated gradients
    attributions = compute_integrated_gradients(
        model, baseline, image, predicted_class
    )
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    display_img = image.numpy()
    if display_img.min() < 0:
        display_img = (display_img + 1) / 2
    
    axes[0].imshow(display_img)
    axes[0].set_title(f'Original\nTrue: {class_names[true_label]}\n'
                     f'Pred: {class_names[predicted_class]}')
    axes[0].axis('off')
    
    # Attribution map
    attribution_map = tf.reduce_sum(tf.abs(attributions), axis=-1)
    axes[1].imshow(attribution_map, cmap='hot')
    axes[1].set_title('Attribution Map\n(What model looks at)')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(display_img)
    axes[2].imshow(attribution_map, cmap='hot', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('feature_attribution.png', dpi=150)
    plt.show()

# Usage
for images, labels in val_dataset.take(1):
    visualize_attributions(model, images[0], labels[0].numpy(), class_names)
```

## Common Solutions Reference

### Problem: Model not learning (loss not decreasing)

**Possible causes and solutions:**

```python
# 1. Learning rate too low
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Try 0.001 or 0.01

# 2. Wrong loss function
# For multi-class classification with integer labels:
model.compile(
    loss='sparse_categorical_crossentropy',  # Not 'categorical_crossentropy'
    optimizer='adam',
    metrics=['accuracy']
)

# 3. Data not preprocessed
# Normalize images to [0, 1] or [-1, 1]
images = images / 255.0

# 4. Vanishing gradients
# Add batch normalization
model.add(BatchNormalization())

# 5. Wrong activation in output layer
# For classification, use softmax:
model.add(Dense(num_classes, activation='softmax'))
```

### Problem: Loss becomes NaN

**Solutions:**

```python
# 1. Reduce learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# 2. Use gradient clipping
optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)

# 3. Check for log(0) or division by zero
# Use epsilon in custom losses:
loss = tf.reduce_mean(-tf.math.log(predictions + 1e-7))

# 4. Stabilize computations
# Use from_logits=True
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 5. Check data for NaN/Inf
assert not tf.reduce_any(tf.math.is_nan(x))
assert not tf.reduce_any(tf.math.is_inf(x))
```

### Problem: Overfitting

**Solutions:**

```python
# 1. Add dropout
model.add(Dropout(0.5))

# 2. Add L2 regularization
model.add(Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001)))

# 3. Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# 4. Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 5. Reduce model complexity
# Use fewer layers or fewer units per layer
```

### Problem: Underfitting

**Solutions:**

```python
# 1. Increase model capacity
model.add(Dense(256, activation='relu'))  # More units
model.add(Dense(256, activation='relu'))  # More layers

# 2. Train longer
model.fit(train_dataset, epochs=100)  # More epochs

# 3. Reduce regularization
model.add(Dropout(0.2))  # Lower dropout rate

# 4. Increase learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# 5. Better feature engineering
# Add more informative features
```

## GCP-Specific Debugging Tools

### 1. TensorBoard Integration
```python
# Enable TensorBoard logging
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='gs://my-bucket/logs',
    histogram_freq=1,
    profile_batch='500,520'
)

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[tensorboard_callback]
)

# View in Vertex AI
# The logs are automatically available in Vertex AI Training console
```

### 2. Vertex AI Model Evaluation
```python
from google.cloud import aiplatform

# Get model evaluation metrics
model = aiplatform.Model('projects/.../models/...')
evaluations = model.list_model_evaluations()

for evaluation in evaluations:
    print(f"Metrics: {evaluation.metrics}")
    print(f"Slice metrics: {evaluation.slice_dimensions}")
```

### 3. What-If Tool Integration
```python
# Load What-If Tool in Vertex AI Workbench
from witwidget.notebook.visualization import WitConfigBuilder
from witwidget.notebook.visualization import WitWidget

# Configure
config_builder = WitConfigBuilder(
    examples=test_examples,
    model_name='my_model'
)

# Display
WitWidget(config_builder)
```

## Debugging Checklist

- [ ] **Setup Verification**
  - [ ] Model can overfit single batch
  - [ ] Loss function is correct
  - [ ] Output layer activation is appropriate
  - [ ] Data preprocessing is correct
  
- [ ] **Training Monitoring**
  - [ ] Loss is decreasing
  - [ ] No NaN/Inf in loss or gradients
  - [ ] Learning rate is appropriate
  - [ ] Validation metrics tracked
  
- [ ] **Gradient Health**
  - [ ] Gradients are not vanishing (> 1e-7)
  - [ ] Gradients are not exploding (< 100)
  - [ ] Gradient flow is healthy across layers
  
- [ ] **Model Behavior**
  - [ ] Better than random guessing
  - [ ] Predictions make sense
  - [ ] No severe overfitting/underfitting
  - [ ] Activations are healthy
  
- [ ] **Performance**
  - [ ] Training speed is acceptable
  - [ ] Memory usage is reasonable
  - [ ] GPU utilization is high (if using GPU)

## Quick Debug Commands

```python
# Check if model learns at all
model.fit(train_dataset.take(1), epochs=100)

# Print model architecture
model.summary()

# Check trainable parameters
print(f"Trainable params: {model.count_params()}")

# Visualize model
tf.keras.utils.plot_model(model, show_shapes=True)

# Check batch shape and type
for x, y in dataset.take(1):
    print(f"X: {x.shape}, {x.dtype}")
    print(f"Y: {y.shape}, {y.dtype}")

# Verify output probabilities sum to 1
predictions = model.predict(x)
print(f"Prob sum: {predictions.sum(axis=1)}")

# Check for NaN in model weights
for w in model.weights:
    if tf.reduce_any(tf.math.is_nan(w)):
        print(f"NaN in {w.name}")
```

## Key Takeaways for GCP Data Engineer Exam

1. **Always start with sanity checks** - Can model overfit single batch?
2. **Monitor gradients** - Vanishing/exploding is common issue
3. **Use learning rate finder** - Don't guess learning rates
4. **Visualize training curves** - Loss, accuracy, gradients
5. **Check data pipeline first** - Most issues are data-related
6. **Use TensorBoard** - Built into Vertex AI, essential for debugging
7. **Test incrementally** - Start simple, add complexity
8. **Compare to baseline** - Always have a simple baseline model
9. **Use callbacks** - Early stopping, learning rate scheduling
10. **Profile performance** - Identify bottlenecks early

## Tools Summary

| Tool | Use Case | When to Use |
|------|----------|-------------|
| **model.fit() verbose** | Basic monitoring | Always |
| **TensorBoard** | Detailed metrics, profiling | Production training |
| **Custom training loop** | Gradient debugging | When training fails |
| **Learning rate finder** | Hyperparameter tuning | Before long training runs |
| **What-If Tool** | Model interpretability | Post-training analysis |
| **Vertex AI Profiler** | Performance bottlenecks | Slow training |
| **Model.summary()** | Architecture verification | Before training |