# Developing Deep Learning Models: Common & Best Practices

## Overview

This guide covers industry best practices for developing deep learning models from scratch to production, with focus on systematic approaches, common pitfalls, and GCP-specific implementations.

## Development Workflow

### Phase 1: Problem Definition & Setup

#### 1.1 Define the Problem Clearly
```python
"""
✓ Good Problem Definition:
- Binary classification: Detect fraudulent transactions
- Input: Transaction features (amount, time, location, etc.)
- Output: Probability of fraud (0-1)
- Success metric: F1-score > 0.85
- Latency requirement: < 100ms
- Dataset: 1M labeled transactions

✗ Poor Problem Definition:
- "Build an AI model"
- No clear metrics
- No constraints specified
"""

# Document your problem
problem_definition = {
    'task_type': 'binary_classification',  # classification, regression, etc.
    'input_shape': (None, 30),  # Feature dimensions
    'output_shape': (None, 2),  # Classes
    'success_metric': 'f1_score',
    'target_value': 0.85,
    'latency_requirement_ms': 100,
    'dataset_size': 1_000_000,
    'class_balance': {'fraud': 0.01, 'normal': 0.99}
}
```

#### 1.2 Establish Baselines First
```python
import tensorflow as tf
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def establish_baselines(X_train, y_train, X_val, y_val):
    """Always start with simple baselines"""
    
    results = {}
    
    # 1. Random baseline
    dummy_random = DummyClassifier(strategy='uniform')
    dummy_random.fit(X_train, y_train)
    random_acc = dummy_random.score(X_val, y_val)
    results['random'] = random_acc
    
    # 2. Most frequent class baseline
    dummy_frequent = DummyClassifier(strategy='most_frequent')
    dummy_frequent.fit(X_train, y_train)
    frequent_acc = dummy_frequent.score(X_val, y_val)
    results['most_frequent'] = frequent_acc
    
    # 3. Simple logistic regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_acc = lr.score(X_val, y_val)
    results['logistic_regression'] = lr_acc
    
    # 4. Simple shallow neural network
    simple_nn = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    simple_nn.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    simple_nn.fit(X_train, y_train, epochs=10, verbose=0, validation_split=0.2)
    _, nn_acc = simple_nn.evaluate(X_val, y_val, verbose=0)
    results['simple_nn'] = nn_acc
    
    print("=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    for name, score in results.items():
        print(f"{name:20s}: {score:.4f}")
    print("=" * 60)
    print(f"💡 Your model should beat: {max(results.values()):.4f}")
    print("=" * 60)
    
    return results

# Usage
baseline_results = establish_baselines(X_train, y_train, X_val, y_val)
```

### Phase 2: Data Preparation

#### 2.1 Data Splitting Strategy
```python
import numpy as np
from sklearn.model_selection import train_test_split

def split_data_properly(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Best practice: 70% train, 15% validation, 15% test
    """
    
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, 
        test_size=val_size_adjusted, 
        random_state=random_state,
        stratify=y_trainval
    )
    
    print("Data Split Summary:")
    print(f"Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Verify no data leakage
    train_ids = set(range(len(X_train)))
    val_ids = set(range(len(X_train), len(X_train) + len(X_val)))
    test_ids = set(range(len(X_train) + len(X_val), len(X)))
    
    assert len(train_ids & val_ids) == 0, "Train-Val overlap!"
    assert len(train_ids & test_ids) == 0, "Train-Test overlap!"
    assert len(val_ids & test_ids) == 0, "Val-Test overlap!"
    
    print("✓ No data leakage between splits")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Usage
X_train, X_val, X_test, y_train, y_val, y_test = split_data_properly(X, y)
```

#### 2.2 Feature Preprocessing Pipeline
```python
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf

class FeaturePreprocessor:
    """Reusable preprocessing pipeline"""
    
    def __init__(self, numerical_features, categorical_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.scaler = StandardScaler()
        self.categorical_encoders = {}
        
    def fit(self, X):
        """Fit preprocessors on training data only"""
        
        # Fit numerical scaler
        if self.numerical_features:
            self.scaler.fit(X[self.numerical_features])
        
        # Fit categorical encoders
        for col in self.categorical_features:
            unique_values = X[col].unique()
            self.categorical_encoders[col] = {
                val: idx for idx, val in enumerate(unique_values)
            }
        
        return self
    
    def transform(self, X):
        """Transform data using fitted preprocessors"""
        
        X_transformed = X.copy()
        
        # Scale numerical features
        if self.numerical_features:
            X_transformed[self.numerical_features] = self.scaler.transform(
                X[self.numerical_features]
            )
        
        # Encode categorical features
        for col in self.categorical_features:
            X_transformed[col] = X[col].map(
                self.categorical_encoders[col]
            ).fillna(0)  # Unknown categories -> 0
        
        return X_transformed
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)

# Usage
preprocessor = FeaturePreprocessor(
    numerical_features=['amount', 'time'],
    categorical_features=['merchant_category', 'card_type']
)

# Fit on training data ONLY
X_train_processed = preprocessor.fit_transform(X_train)

# Transform validation and test using fitted preprocessor
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)
```

#### 2.3 Create Efficient Data Pipeline
```python
def create_tf_dataset(X, y, batch_size=32, shuffle=True, augment=False):
    """Create optimized TensorFlow dataset"""
    
    # Convert to tensors
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Shuffle (before batching)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    
    # Batch
    dataset = dataset.batch(batch_size)
    
    # Augmentation (if applicable, e.g., for images)
    if augment:
        def augment_fn(images, labels):
            images = tf.image.random_flip_left_right(images)
            images = tf.image.random_brightness(images, 0.2)
            return images, labels
        
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Performance optimizations
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Create datasets
train_dataset = create_tf_dataset(
    X_train_processed, y_train, 
    batch_size=32, 
    shuffle=True,
    augment=True
)

val_dataset = create_tf_dataset(
    X_val_processed, y_val,
    batch_size=32,
    shuffle=False,
    augment=False
)

test_dataset = create_tf_dataset(
    X_test_processed, y_test,
    batch_size=32,
    shuffle=False,
    augment=False
)
```

### Phase 3: Model Architecture Design

#### 3.1 Start Simple, Then Increase Complexity
```python
def build_model_v1_simple(input_shape, num_classes):
    """Version 1: Simplest possible model"""
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ], name='simple_model_v1')
    
    return model

def build_model_v2_deeper(input_shape, num_classes):
    """Version 2: Add depth"""
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ], name='deeper_model_v2')
    
    return model

def build_model_v3_with_regularization(input_shape, num_classes):
    """Version 3: Add regularization to prevent overfitting"""
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ], name='regularized_model_v3')
    
    return model

def build_model_v4_optimized(input_shape, num_classes):
    """Version 4: Production-ready with all best practices"""
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # First block
    x = tf.keras.layers.Dense(
        128, 
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Second block with skip connection
    skip = x
    x = tf.keras.layers.Dense(
        128,
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, skip])  # Residual connection
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Third block
    x = tf.keras.layers.Dense(
        64,
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer='glorot_uniform'
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='optimized_model_v4')
    
    return model

# Progressive development
print("Testing Model V1 (Simple)...")
model_v1 = build_model_v1_simple(input_shape=(30,), num_classes=2)
model_v1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_v1 = model_v1.fit(train_dataset, validation_data=val_dataset, epochs=10, verbose=0)

print("Testing Model V2 (Deeper)...")
model_v2 = build_model_v2_deeper(input_shape=(30,), num_classes=2)
model_v2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_v2 = model_v2.fit(train_dataset, validation_data=val_dataset, epochs=10, verbose=0)

print("Testing Model V3 (With Regularization)...")
model_v3 = build_model_v3_with_regularization(input_shape=(30,), num_classes=2)
model_v3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_v3 = model_v3.fit(train_dataset, validation_data=val_dataset, epochs=10, verbose=0)

# Compare results
print("\nModel Comparison:")
print(f"V1 Val Accuracy: {max(history_v1.history['val_accuracy']):.4f}")
print(f"V2 Val Accuracy: {max(history_v2.history['val_accuracy']):.4f}")
print(f"V3 Val Accuracy: {max(history_v3.history['val_accuracy']):.4f}")
```

#### 3.2 Architecture Design Principles
```python
"""
BEST PRACTICES FOR ARCHITECTURE DESIGN

1. INPUT LAYER
   ✓ Use Input() layer explicitly
   ✓ Document expected input shape
   ✓ Handle variable batch size with None

2. HIDDEN LAYERS
   ✓ Start with 1-2 layers
   ✓ Use ReLU activation for hidden layers
   ✓ Layer sizes: decreasing pattern (128 → 64 → 32)
   ✓ Add BatchNormalization after Dense layers
   ✓ Add Dropout for regularization (0.2-0.5)

3. OUTPUT LAYER
   ✓ Binary classification: 1 unit + sigmoid OR 2 units + softmax
   ✓ Multi-class classification: num_classes units + softmax
   ✓ Regression: 1 unit + linear (no activation)

4. WEIGHT INITIALIZATION
   ✓ ReLU: 'he_normal' or 'he_uniform'
   ✓ Tanh/Sigmoid: 'glorot_normal' or 'glorot_uniform'

5. REGULARIZATION
   ✓ L2 regularization: 0.001 - 0.01
   ✓ Dropout: 0.2 - 0.5
   ✓ Early stopping: patience=5-10

6. BATCH NORMALIZATION
   ✓ Place AFTER activation (recommended)
   ✓ Or BEFORE activation (alternative)
   ✓ Don't use with Dropout in same layer

7. SKIP CONNECTIONS
   ✓ Use for very deep networks (>10 layers)
   ✓ Helps with gradient flow
   ✓ Improves training stability
"""

def get_recommended_architecture(task_type, input_shape, output_shape):
    """Get recommended architecture for common tasks"""
    
    architectures = {
        'simple_tabular': {
            'layers': [128, 64],
            'dropout': 0.3,
            'batch_norm': True,
            'l2_reg': 0.001
        },
        'complex_tabular': {
            'layers': [256, 128, 64, 32],
            'dropout': 0.4,
            'batch_norm': True,
            'l2_reg': 0.001
        },
        'image_classification': {
            'base': 'MobileNetV2 or EfficientNet',
            'top_layers': [256, 128],
            'dropout': 0.5,
            'batch_norm': True,
            'data_augmentation': True
        },
        'text_classification': {
            'embedding_dim': 128,
            'lstm_units': 128,
            'dense_layers': [64],
            'dropout': 0.4,
            'batch_norm': True
        }
    }
    
    return architectures.get(task_type, architectures['simple_tabular'])
```

### Phase 4: Training Configuration

#### 4.1 Optimizer Selection
```python
def configure_optimizer(learning_rate=0.001, optimizer_type='adam'):
    """Configure optimizer with best practices"""
    
    optimizers = {
        'adam': tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0  # Gradient clipping
        ),
        'adamw': tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,
            clipnorm=1.0
        ),
        'sgd': tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True,
            clipnorm=1.0
        ),
        'rmsprop': tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            momentum=0.9,
            clipnorm=1.0
        )
    }
    
    return optimizers.get(optimizer_type, optimizers['adam'])

"""
OPTIMIZER SELECTION GUIDE:

Adam (Default Choice):
  ✓ Works well out of the box
  ✓ Adaptive learning rate
  ✓ Good for most problems
  Use: General purpose, starting point

AdamW (Better Generalization):
  ✓ Adam with decoupled weight decay
  ✓ Better generalization than Adam
  ✓ Recommended for production
  Use: When overfitting is a concern

SGD with Momentum:
  ✓ Better final performance (if tuned well)
  ✓ Requires learning rate scheduling
  ✓ More stable than Adam for very deep networks
  Use: When you have time to tune LR

RMSprop:
  ✓ Good for RNNs
  ✓ Handles non-stationary objectives
  Use: Recurrent neural networks
"""
```

#### 4.2 Learning Rate Scheduling
```python
def create_learning_rate_schedule(initial_lr=0.001, schedule_type='reduce_on_plateau'):
    """Create learning rate schedule"""
    
    schedules = {
        'reduce_on_plateau': tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,          # Reduce by half
            patience=5,          # After 5 epochs without improvement
            min_lr=1e-7,         # Don't go below this
            verbose=1
        ),
        
        'exponential_decay': tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        ),
        
        'cosine_decay': tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=10000,
            alpha=0.1
        ),
        
        'one_cycle': tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: initial_lr * (1 + np.cos(epoch * np.pi / 100)) / 2
        )
    }
    
    return schedules.get(schedule_type)

# Usage
lr_schedule = create_learning_rate_schedule(initial_lr=0.001, schedule_type='reduce_on_plateau')
```

#### 4.3 Callbacks Configuration
```python
def create_callbacks(model_name, patience=10):
    """Create comprehensive callback list"""
    
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'models/{model_name}_best.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=f'logs/{model_name}',
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch='500,520'  # Profile specific batches
        ),
        
        # CSV logger
        tf.keras.callbacks.CSVLogger(
            filename=f'logs/{model_name}_training.csv',
            separator=',',
            append=False
        ),
        
        # Learning rate scheduler
        create_learning_rate_schedule(schedule_type='reduce_on_plateau'),
        
        # Custom callback for monitoring
        CustomMonitoringCallback()
    ]
    
    return callbacks

class CustomMonitoringCallback(tf.keras.callbacks.Callback):
    """Custom callback to monitor training"""
    
    def on_epoch_end(self, epoch, logs=None):
        """Monitor training at end of each epoch"""
        
        if logs is None:
            logs = {}
        
        # Check for overfitting
        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        
        if val_loss > train_loss * 1.5:
            print(f"\n⚠️ WARNING: Possible overfitting at epoch {epoch+1}")
            print(f"   Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        
        # Check for underfitting
        train_acc = logs.get('accuracy', 0)
        if train_acc < 0.7 and epoch > 10:
            print(f"\n⚠️ WARNING: Low training accuracy at epoch {epoch+1}: {train_acc:.4f}")
        
        # Check for unstable training
        if np.isnan(train_loss) or np.isinf(train_loss):
            print(f"\n❌ ERROR: NaN/Inf loss detected at epoch {epoch+1}")
            self.model.stop_training = True

# Usage
callbacks = create_callbacks(model_name='fraud_detector_v1', patience=10)
```

### Phase 5: Training Process

#### 5.1 Complete Training Pipeline
```python
def train_model_complete(
    model,
    train_dataset,
    val_dataset,
    model_name,
    epochs=100,
    initial_lr=0.001,
    class_weights=None
):
    """Complete training pipeline with best practices"""
    
    print("=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)
    
    # 1. Compile model
    model.compile(
        optimizer=configure_optimizer(learning_rate=initial_lr, optimizer_type='adam'),
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    # 2. Print model summary
    model.summary()
    
    # 3. Create callbacks
    callbacks = create_callbacks(model_name=model_name, patience=10)
    
    # 4. Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,  # Handle imbalanced data
        verbose=1
    )
    
    # 5. Evaluate on validation set
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    val_results = model.evaluate(val_dataset, verbose=0)
    for metric_name, value in zip(model.metrics_names, val_results):
        print(f"{metric_name:15s}: {value:.4f}")
    
    # 6. Plot training history
    plot_training_history(history, model_name)
    
    # 7. Save final model
    model.save(f'models/{model_name}_final.h5')
    print(f"\n✓ Model saved: models/{model_name}_final.h5")
    
    return history, model

def plot_training_history(history, model_name):
    """Plot training history"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['loss', 'accuracy', 'precision', 'recall']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        if metric in history.history:
            ax.plot(history.history[metric], label=f'Train {metric}')
            if f'val_{metric}' in history.history:
                ax.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
            
            ax.set_title(f'{metric.capitalize()} over Epochs')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_training_history.png', dpi=150)
    plt.show()

# Usage
history, trained_model = train_model_complete(
    model=model_v4,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    model_name='fraud_detector_v4',
    epochs=100,
    initial_lr=0.001,
    class_weights={0: 1.0, 1: 99.0}  # Handle class imbalance
)
```

#### 5.2 Hyperparameter Tuning
```python
import keras_tuner as kt

def build_tunable_model(hp):
    """Build model with tunable hyperparameters"""
    
    model = tf.keras.Sequential()
    
    # Tunable input layer
    model.add(tf.keras.layers.Input(shape=(30,)))
    
    # Tunable number of layers
    for i in range(hp.Int('num_layers', min_value=1, max_value=4)):
        # Tunable units per layer
        model.add(tf.keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(
                hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='log')
            )
        ))
        
        # Tunable dropout
        model.add(tf.keras.layers.Dropout(
            rate=hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)
        ))
    
    # Output layer
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    
    # Tunable learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create tuner
tuner = kt.Hyperband(
    build_tunable_model,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory='tuning',
    project_name='fraud_detection'
)

# Search for best hyperparameters
tuner.search(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
)

# Get best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\nBest Hyperparameters:")
for key, value in best_hps.values.items():
    print(f"{key}: {value}")

# Build best model
best_model = tuner.hypermodel.build(best_hps)
```

### Phase 6: Model Evaluation

#### 6.1 Comprehensive Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def evaluate_model_comprehensive(model, test_dataset, class_names):
    """Comprehensive model evaluation"""
    
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # 1. Collect predictions
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    for x_batch, y_batch in test_dataset:
        predictions = model.predict(x_batch, verbose=0)
        y_pred_proba.extend(predictions)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(y_batch.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # 2. Basic metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    print("\n1. Overall Metrics:")
    print(f"   Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"   Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"   Recall:    {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"   F1-Score:  {f1_score(y_true, y_pred, average='weighted'):.4f}")
    
    # 3. Classification report
    print("\n2. Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 4. Confusion matrix
    print("\n3. Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=150)
    plt.show()
    
    # 5. ROC curve and AUC
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=(10, 8))
    
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/roc_curves.png', dpi=150)
    plt.show()
    
    # 6. Precision-Recall curve
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    plt.figure(figsize=(10, 8))
    
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(
            (y_true == i).astype(int), 
            y_pred_proba[:, i]
        )
        ap = average_precision_score((y_true == i).astype(int), y_pred_proba[:, i])
        
        plt.plot(recall, precision, label=f'{class_names[i]} (AP = {ap:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/pr_curves.png', dpi=150)
    plt.show()
    
    # 7. Error analysis
    print("\n4. Error Analysis:")
    errors = y_true != y_pred
    error_rate = np.mean(errors)
    print(f"   Error rate: {error_rate:.4f} ({np.sum(errors)} errors out of {len(y_true)})")
    
    # Most confident wrong predictions
    wrong_confidence = []
    for i in range(len(y_true)):
        if errors[i]:
            confidence = y_pred_proba[i, y_pred[i]]
            wrong_confidence.append((i, confidence, y_true[i], y_pred[i]))
    
    wrong_confidence.sort(key=lambda x: x[1], reverse=True)
    
    print("\n   Most confident wrong predictions:")
    for idx, conf, true_label, pred_label in wrong_confidence[:5]:
        print(f"   Sample {idx}: Predicted {class_names[pred_label]} "
              f"(confidence: {conf:.3f}), True: {class_names[true_label]}")
    
    print("=" * 60)
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm
    }

# Usage
evaluation_results = evaluate_model_comprehensive(
    model=trained_model,
    test_dataset=test_dataset,
    class_names=['Normal', 'Fraud']
)
```

### Phase 7: Model Optimization

#### 7.1 Model Compression
```python
def compress_model(model, compression_type='pruning'):
    """Apply model compression techniques"""
    
    if compression_type == 'pruning':
        import tensorflow_model_optimization as tfmot
        
        # Define pruning schedule
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=0.5,
                begin_step=0,
                end_step=1000
            )
        }
        
        # Apply pruning
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
            model, **pruning_params
        )
        
        return model_for_pruning
    
    elif compression_type == 'quantization':
        # Post-training quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_model = converter.convert()
        
        # Save quantized model
        with open('models/quantized_model.tflite', 'wb') as f:
            f.write(quantized_model)
        
        print("✓ Quantized model saved")
        return quantized_model
    
    elif compression_type == 'distillation':
        # Knowledge distillation (create smaller student model)
        student_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        # Train student to mimic teacher
        # (Implementation depends on specific use case)
        return student_model

# Usage
compressed_model = compress_model(trained_model, compression_type='pruning')
```

#### 7.2 Model Export and Serving
```python
def prepare_model_for_serving(model, model_name, export_path='serving_models'):
    """Prepare model for production serving"""
    
    import os
    import time
    
    # 1. Save in TensorFlow SavedModel format
    version = str(int(time.time()))
    model_path = os.path.join(export_path, model_name, version)
    
    tf.saved_model.save(model, model_path)
    print(f"✓ SavedModel exported to: {model_path}")
    
    # 2. Convert to TensorFlow Lite (for mobile/edge)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    tflite_path = f'{export_path}/{model_name}.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✓ TFLite model exported to: {tflite_path}")
    
    # 3. Create model signature for Vertex AI
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 30], dtype=tf.float32)])
    def serve_fn(input_data):
        return model(input_data)
    
    # 4. Generate model card
    model_card = f"""
    # {model_name} Model Card
    
    ## Model Details
    - Version: {version}
    - Task: Binary Classification
    - Framework: TensorFlow {tf.__version__}
    - Input Shape: (None, 30)
    - Output Shape: (None, 2)
    
    ## Performance Metrics
    - Accuracy: {evaluation_results.get('accuracy', 'N/A')}
    - Precision: {evaluation_results.get('precision', 'N/A')}
    - Recall: {evaluation_results.get('recall', 'N/A')}
    
    ## Usage
    ```python
    import tensorflow as tf
    model = tf.saved_model.load('{model_path}')
    predictions = model(input_data)
    ```
    
    ## Deployment
    - Vertex AI Endpoint: [To be configured]
    - Latency Target: < 100ms
    - Throughput: 1000 QPS
    """
    
    with open(f'{model_path}/model_card.md', 'w') as f:
        f.write(model_card)
    
    print("✓ Model card generated")
    
    return model_path

# Usage
serving_path = prepare_model_for_serving(
    model=trained_model,
    model_name='fraud_detector'
)
```

## GCP-Specific Best Practices

### 1. Vertex AI Training
```python
from google.cloud import aiplatform

def train_on_vertex_ai(
    project_id,
    location,
    display_name,
    script_path,
    container_uri,
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1
):
    """Train model on Vertex AI"""
    
    aiplatform.init(project=project_id, location=location)
    
    # Create custom training job
    job = aiplatform.CustomTrainingJob(
        display_name=display_name,
        script_path=script_path,
        container_uri=container_uri,
        requirements=['tensorflow==2.13.0', 'pandas', 'numpy'],
        model_serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-13:latest'
    )
    
    # Run training
    model = job.run(
        replica_count=1,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        args=['--epochs', '100', '--batch-size', '32'],
        environment_variables={'TF_CPP_MIN_LOG_LEVEL': '2'}
    )
    
    print(f"✓ Model trained and uploaded: {model.resource_name}")
    
    return model

# Usage
vertex_model = train_on_vertex_ai(
    project_id='my-project',
    location='us-central1',
    display_name='fraud-detector-training',
    script_path='train.py',
    container_uri='gcr.io/my-project/training:latest'
)
```

### 2. Model Deployment
```python
def deploy_to_vertex_ai(model, endpoint_display_name):
    """Deploy model to Vertex AI endpoint"""
    
    # Create endpoint
    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_display_name
    )
    
    # Deploy model
    endpoint.deploy(
        model=model,
        deployed_model_display_name=endpoint_display_name,
        machine_type='n1-standard-4',
        min_replica_count=1,
        max_replica_count=10,
        traffic_percentage=100,
        accelerator_type='NVIDIA_TESLA_T4',
        accelerator_count=1
    )
    
    print(f"✓ Model deployed to endpoint: {endpoint.resource_name}")
    
    return endpoint

# Usage
endpoint = deploy_to_vertex_ai(
    model=vertex_model,
    endpoint_display_name='fraud-detector-endpoint'
)
```

### 3. Model Monitoring
```python
def setup_model_monitoring(endpoint, alert_email):
    """Set up model monitoring in Vertex AI"""
    
    from google.cloud.aiplatform import model_monitoring
    
    # Configure monitoring
    monitoring_config = model_monitoring.ModelMonitoringConfig(
        alert_config=model_monitoring.AlertConfig(
            email_alert_config=model_monitoring.EmailAlertConfig(
                user_emails=[alert_email]
            )
        ),
        objective_configs=[
            model_monitoring.ObjectiveConfig(
                training_dataset=model_monitoring.TrainingDataset(
                    target_field='label'
                ),
                training_prediction_skew_detection_config=model_monitoring.SkewDetectionConfig(
                    skew_thresholds={
                        'feature1': 0.1,
                        'feature2': 0.1
                    }
                ),
                prediction_drift_detection_config=model_monitoring.DriftDetectionConfig(
                    drift_thresholds={
                        'feature1': 0.1,
                        'feature2': 0.1
                    }
                )
            )
        ]
    )
    
    endpoint.update(monitoring_config=monitoring_config)
    
    print("✓ Model monitoring configured")

# Usage
setup_model_monitoring(
    endpoint=endpoint,
    alert_email='ml-team@company.com'
)
```

## Development Checklist

### Pre-Training
- [ ] Problem clearly defined with success metrics
- [ ] Baseline models established
- [ ] Data split properly (train/val/test)
- [ ] Data preprocessing pipeline created
- [ ] Feature engineering completed
- [ ] Data leakage checked
- [ ] Class imbalance handled

### Model Architecture
- [ ] Started with simple model
- [ ] Architecture appropriate for task
- [ ] Input/output layers correct
- [ ] Activation functions appropriate
- [ ] Weight initialization correct
- [ ] Model summary reviewed

### Training Configuration
- [ ] Appropriate optimizer selected
- [ ] Learning rate tuned
- [ ] Loss function correct
- [ ] Metrics defined
- [ ] Callbacks configured (early stopping, checkpointing)
- [ ] Regularization added if needed

### Training Process
- [ ] Training progress monitored
- [ ] Overfitting/underfitting checked
- [ ] Gradient flow healthy
- [ ] Training time acceptable
- [ ] Best model saved

### Evaluation
- [ ] Model evaluated on test set
- [ ] Confusion matrix analyzed
- [ ] Error patterns identified
- [ ] Performance meets requirements
- [ ] Model interpretability checked

### Production Preparation
- [ ] Model compressed if needed
- [ ] Model exported in serving format
- [ ] Model card created
- [ ] API endpoint tested
- [ ] Monitoring configured
- [ ] Documentation complete

## Common Pitfalls to Avoid

```python
"""
❌ COMMON MISTAKES:

1. DATA LEAKAGE
   - Fitting preprocessors on entire dataset
   - Using test data for any decisions
   - Temporal leakage in time series

2. PREPROCESSING ERRORS
   - Not normalizing inputs
   - Inconsistent preprocessing train/test
   - Forgetting to save preprocessing state

3. ARCHITECTURE MISTAKES
   - Wrong output activation (e.g., ReLU for classification)
   - Too complex model from start
   - No regularization leading to overfitting

4. TRAINING ISSUES
   - Learning rate too high/low
   - Not using validation set
   - Training too long without early stopping
   - Ignoring class imbalance

5. EVALUATION ERRORS
   - Only looking at accuracy (ignoring precision/recall)
   - Not checking confusion matrix
   - Evaluating on training data
   - Not analyzing errors

6. PRODUCTION MISTAKES
   - Not version controlling models
   - No monitoring in production
   - Not handling edge cases
   - Missing documentation

✓ BEST PRACTICES:

1. INCREMENTAL DEVELOPMENT
   - Start simple, add complexity gradually
   - Compare each version to previous

2. EXPERIMENT TRACKING
   - Log all experiments with metrics
   - Version control code and data
   - Document decisions and results

3. REPRODUCIBILITY
   - Set random seeds
   - Save preprocessing steps
   - Document environment

4. TESTING
   - Unit test preprocessing
   - Integration test pipeline
   - Load test serving endpoint

5. DOCUMENTATION
   - Model card with metrics
   - API documentation
   - Deployment guide
   - Monitoring dashboard
"""
```

## Key Takeaways for GCP Data Engineer Exam

1. **Start with baselines** - Simple models first, then complexity
2. **Proper data splitting** - 70/15/15, stratified, no leakage
3. **Preprocessing pipeline** - Fit on train only, reusable
4. **Progressive development** - V1 simple → V2 deeper → V3 regularized
5. **Comprehensive callbacks** - Early stopping, checkpointing, TensorBoard
6. **Multiple metrics** - Not just accuracy (precision, recall, F1, AUC)
7. **Error analysis** - Understand failure modes
8. **Model compression** - Pruning, quantization for efficiency
9. **Vertex AI integration** - Training, deployment, monitoring
10. **Production readiness** - SavedModel format, monitoring, documentation

## Tools & Frameworks Summary

| Tool | Purpose | When to Use |
|------|---------|-------------|
| **TensorFlow/Keras** | Model development | All projects |
| **Keras Tuner** | Hyperparameter tuning | Optimization phase |
| **TensorBoard** | Monitoring & visualization | During training |
| **TF Model Optimization** | Compression | Before deployment |
| **Vertex AI Training** | Managed training | Production workloads |
| **Vertex AI Endpoints** | Model serving | Production deployment |
| **Model Monitoring** | Performance tracking | Production models |
| **MLflow** | Experiment tracking | Research phase |
| **DVC** | Data version control | Large datasets |