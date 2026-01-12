# Regularization Algorithms in Machine Learning

## Overview
**Regularization** techniques prevent overfitting by adding constraints or penalties to the model, improving generalization to unseen data. These methods reduce model complexity and variance.

## 1. L1 Regularization (Lasso)

### **Concept**
Adds absolute value of weights to the loss function, encouraging sparsity.

### **Mathematical Formula**
```
Loss = Original_Loss + λ × Σ|w_i|

Where:
- λ (lambda): Regularization strength
- w_i: Model weights
```

### **Implementation**

```python
# TensorFlow/Keras
from tensorflow import keras
from tensorflow.keras import regularizers

model = keras.Sequential([
    keras.layers.Dense(
        128, 
        activation='relu',
        kernel_regularizer=regularizers.L1(l1=0.01)  # λ = 0.01
    ),
    keras.layers.Dense(64, activation='relu',
        kernel_regularizer=regularizers.L1(0.01)
    ),
    keras.layers.Dense(10, activation='softmax')
])

# PyTorch
import torch.nn as nn

class L1RegularizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# Add L1 penalty in training loop
def train_step(model, data, target, optimizer, lambda_l1=0.01):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    
    # Add L1 regularization
    l1_penalty = sum(p.abs().sum() for p in model.parameters())
    loss = loss + lambda_l1 * l1_penalty
    
    loss.backward()
    optimizer.step()
    return loss.item()

# Scikit-learn (Lasso Regression)
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)  # alpha is λ
lasso.fit(X_train, y_train)
```

### **Characteristics**
- ✅ **Feature selection**: Drives some weights to exactly zero
- ✅ **Sparse models**: Reduces model size
- ✅ **Interpretability**: Identifies important features
- ⚠️ **Non-differentiable at zero**: Can cause optimization issues
- ⚠️ **Computationally expensive**: For high-dimensional data

### **Use Cases**
```python
# Use L1 when:
use_cases = {
    'Feature Selection': 'Identify important features in high-dimensional data',
    'Genomics': 'Select relevant genes from thousands',
    'Text Classification': 'Identify key words/features',
    'Compressed Models': 'Deploy on edge devices with limited memory'
}

# Example: Feature selection
from sklearn.linear_model import LassoCV

# Cross-validated Lasso
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_train, y_train)

# Get selected features (non-zero coefficients)
selected_features = np.where(lasso_cv.coef_ != 0)[0]
print(f"Selected {len(selected_features)} out of {X_train.shape[1]} features")
```

## 2. L2 Regularization (Ridge)

### **Concept**
Adds squared magnitude of weights to the loss function, penalizing large weights.

### **Mathematical Formula**
```
Loss = Original_Loss + λ × Σ(w_i²)

Where:
- λ (lambda): Regularization strength
- w_i: Model weights
```

### **Implementation**

```python
# TensorFlow/Keras
model = keras.Sequential([
    keras.layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.L2(l2=0.01)  # λ = 0.01
    ),
    keras.layers.Dense(64, activation='relu',
        kernel_regularizer=regularizers.L2(0.01)
    ),
    keras.layers.Dense(10, activation='softmax')
])

# PyTorch (using weight_decay in optimizer)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # L2 regularization (λ = 0.01)
)

# Scikit-learn (Ridge Regression)
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)  # alpha is λ
ridge.fit(X_train, y_train)

# Manual implementation in PyTorch
def train_step_l2(model, data, target, optimizer, lambda_l2=0.01):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    
    # Add L2 regularization
    l2_penalty = sum((p**2).sum() for p in model.parameters())
    loss = loss + lambda_l2 * l2_penalty
    
    loss.backward()
    optimizer.step()
    return loss.item()
```

### **Characteristics**
- ✅ **Smooth penalties**: Differentiable everywhere
- ✅ **Reduces overfitting**: Shrinks all weights proportionally
- ✅ **Computationally efficient**: Easy to optimize
- ✅ **Stable training**: No discontinuities
- ⚠️ **No feature selection**: Keeps all features (non-zero weights)
- ⚠️ **Less interpretable**: Doesn't eliminate features

### **Use Cases**
```python
# Use L2 when:
use_cases = {
    'Deep Learning': 'Default choice for neural networks',
    'Multicollinearity': 'Handle correlated features',
    'Ridge Regression': 'Stabilize linear regression',
    'General Purpose': 'When feature selection is not needed'
}

# Example: Ridge with cross-validation
from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5)
ridge_cv.fit(X_train, y_train)
print(f"Best alpha: {ridge_cv.alpha_}")
```

## 3. Elastic Net (L1 + L2)

### **Concept**
Combines L1 and L2 penalties, balancing feature selection and weight shrinkage.

### **Mathematical Formula**
```
Loss = Original_Loss + λ₁ × Σ|w_i| + λ₂ × Σ(w_i²)

Or with mixing parameter:
Loss = Original_Loss + λ × [α × Σ|w_i| + (1-α) × Σ(w_i²)]

Where:
- α (alpha): Mixing ratio (0 = L2 only, 1 = L1 only)
- λ (lambda): Overall regularization strength
```

### **Implementation**

```python
# TensorFlow/Keras
model = keras.Sequential([
    keras.layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.L1L2(l1=0.01, l2=0.01)
    ),
    keras.layers.Dense(10, activation='softmax')
])

# Scikit-learn
from sklearn.linear_model import ElasticNet, ElasticNetCV

# Basic Elastic Net
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)  # l1_ratio is α
elastic.fit(X_train, y_train)

# Cross-validated Elastic Net
elastic_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
    alphas=[0.001, 0.01, 0.1, 1.0],
    cv=5
)
elastic_cv.fit(X_train, y_train)
print(f"Best l1_ratio: {elastic_cv.l1_ratio_}")
print(f"Best alpha: {elastic_cv.alpha_}")

# PyTorch
class ElasticNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

def elastic_net_loss(model, output, target, lambda_l1=0.01, lambda_l2=0.01):
    mse_loss = nn.MSELoss()(output, target)
    
    # L1 penalty
    l1_penalty = sum(p.abs().sum() for p in model.parameters())
    
    # L2 penalty
    l2_penalty = sum((p**2).sum() for p in model.parameters())
    
    total_loss = mse_loss + lambda_l1 * l1_penalty + lambda_l2 * l2_penalty
    return total_loss
```

### **Characteristics**
- ✅ **Best of both worlds**: Feature selection + weight shrinkage
- ✅ **Handles multicollinearity**: Better than Lasso alone
- ✅ **Grouped selection**: Selects correlated features together
- ✅ **Flexible**: Tune L1/L2 ratio for specific needs
- ⚠️ **More hyperparameters**: Need to tune both λ₁ and λ₂ (or α and λ)

### **Use Cases**
```python
# Use Elastic Net when:
use_cases = {
    'Genomics': 'Many correlated features (genes)',
    'Finance': 'Correlated predictors in time series',
    'High-dimensional sparse data': 'More features than samples',
    'Uncertain about L1 vs L2': 'Let model decide the balance'
}
```

## 4. Dropout

### **Concept**
Randomly drops neurons during training, preventing co-adaptation and improving generalization.

### **Mathematical Formula**
```
During Training:
y = f(x) × mask, where mask ~ Bernoulli(keep_prob)

During Inference:
y = f(x) × keep_prob  (or no scaling if using inverted dropout)
```

### **Implementation**

```python
# TensorFlow/Keras
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),  # Drop 50% of neurons
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),  # Drop 30% of neurons
    keras.layers.Dense(10, activation='softmax')
])

# PyTorch
class DropoutModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # Only active during training
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Training mode
model.train()  # Enables dropout
output = model(input_data)

# Inference mode
model.eval()  # Disables dropout
with torch.no_grad():
    predictions = model(test_data)

# Custom dropout implementation
class CustomDropout:
    def __init__(self, rate=0.5):
        self.rate = rate
    
    def __call__(self, x, training=True):
        if not training:
            return x
        
        # Inverted dropout (recommended)
        keep_prob = 1 - self.rate
        mask = (np.random.random(x.shape) < keep_prob) / keep_prob
        return x * mask
```

### **Variants**

#### **Spatial Dropout (for CNNs)**
```python
# Drops entire feature maps instead of individual pixels
from tensorflow.keras.layers import SpatialDropout2D

model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu'),
    SpatialDropout2D(0.2),  # Drop 20% of feature maps
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

# PyTorch
import torch.nn as nn

spatial_dropout = nn.Dropout2d(0.2)  # For 2D feature maps
```

#### **DropConnect**
```python
# Drops connections (weights) instead of neurons
class DropConnect(nn.Module):
    def __init__(self, in_features, out_features, drop_prob=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.ones_like(self.weight) * (1 - self.drop_prob))
            masked_weight = self.weight * mask
        else:
            masked_weight = self.weight * (1 - self.drop_prob)
        
        return torch.matmul(x, masked_weight.t())
```

#### **Alpha Dropout (for SELU)**
```python
# Maintains mean and variance for SELU activation
model = keras.Sequential([
    keras.layers.Dense(512, activation='selu'),
    keras.layers.AlphaDropout(0.1),  # For SELU networks
    keras.layers.Dense(256, activation='selu'),
    keras.layers.AlphaDropout(0.1),
    keras.layers.Dense(10, activation='softmax')
])
```

### **Characteristics**
- ✅ **Simple and effective**: Easy to implement
- ✅ **Model ensemble**: Approximates training multiple models
- ✅ **Reduces co-adaptation**: Forces robust feature learning
- ✅ **No overhead at inference**: (with inverted dropout)
- ⚠️ **Slower training**: Due to stochastic nature
- ⚠️ **Hyperparameter tuning**: Need to find optimal drop rate

### **Best Practices**
```python
# Recommended dropout rates by layer type
dropout_rates = {
    'Input layer': 0.1-0.2,      # Low dropout
    'Hidden layers': 0.3-0.5,    # Medium dropout
    'Large networks': 0.5-0.8,   # Higher dropout
    'Small networks': 0.2-0.3,   # Lower dropout
    'Convolutional layers': 0.1-0.25,  # Lower than dense layers
    'Recurrent layers': 0.2-0.3  # Careful with RNNs
}

# Rule of thumb: Start with 0.5, tune based on validation performance
```

## 5. Batch Normalization

### **Concept**
Normalizes activations within mini-batches, reducing internal covariate shift and acting as regularization.

### **Mathematical Formula**
```
For batch B:
μ_B = (1/|B|) × Σ x_i          (batch mean)
σ²_B = (1/|B|) × Σ(x_i - μ_B)²  (batch variance)

Normalized:
x̂_i = (x_i - μ_B) / √(σ²_B + ε)

Scaled and shifted:
y_i = γ × x̂_i + β

Where γ, β are learnable parameters
```

### **Implementation**

```python
# TensorFlow/Keras
model = keras.Sequential([
    keras.layers.Dense(512),
    keras.layers.BatchNormalization(),  # Before or after activation
    keras.layers.Activation('relu'),
    keras.layers.Dense(256),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(10, activation='softmax')
])

# PyTorch
class BatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = torch.relu(x)
        x = self.bn2(self.fc2(x))
        x = torch.relu(x)
        x = self.fc3(x)
        return x

# For CNNs
class ConvBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)  # 2D batch norm for images
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        return x
```

### **Characteristics**
- ✅ **Faster training**: Allows higher learning rates
- ✅ **Reduces internal covariate shift**: Stabilizes distribution
- ✅ **Regularization effect**: Acts like dropout (batch-level noise)
- ✅ **Reduces need for dropout**: Can replace dropout in some cases
- ⚠️ **Batch size dependency**: Poor performance with small batches
- ⚠️ **Different train/test behavior**: Uses running statistics at inference
- ⚠️ **Memory overhead**: Stores running mean/variance

### **Variants**

#### **Layer Normalization**
```python
# For RNNs and Transformers (batch-independent)
layer_norm = keras.layers.LayerNormalization()

# PyTorch
layer_norm = nn.LayerNorm(normalized_shape=512)

# Normalizes across features, not batch
# Better for variable batch sizes and RNNs
```

#### **Group Normalization**
```python
# For small batch sizes (e.g., object detection)
# PyTorch only
group_norm = nn.GroupNorm(num_groups=32, num_channels=256)

# Divides channels into groups and normalizes within groups
```

#### **Instance Normalization**
```python
# For style transfer (normalizes per instance)
instance_norm = nn.InstanceNorm2d(64)

# Normalizes each sample independently
```

### **Best Practices**
```python
# Where to place batch normalization
placement_strategies = {
    'After linear/conv': 'BN before activation (modern practice)',
    'After activation': 'BN after activation (original paper)',
    'Skip connections': 'BN before addition in ResNets'
}

# Example: ResNet-style block
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Add residual after BN
        x = torch.relu(x)
        return x
```

## 6. Early Stopping

### **Concept**
Stops training when validation performance stops improving, preventing overfitting.

### **Implementation**

```python
# TensorFlow/Keras
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',      # Metric to monitor
    patience=10,             # Epochs to wait before stopping
    restore_best_weights=True,  # Restore best model
    min_delta=0.001,         # Minimum improvement threshold
    mode='min'               # 'min' for loss, 'max' for accuracy
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[early_stopping]
)

# PyTorch - Manual implementation
class EarlyStoppingPyTorch:
    def __init__(self, patience=10, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
    
    def save_checkpoint(self, model):
        self.best_model = model.state_dict().copy()

# Usage
early_stopping = EarlyStoppingPyTorch(patience=10)

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    
    early_stopping(val_loss, model)
    
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        model.load_state_dict(early_stopping.best_model)
        break
```

### **Advanced Early Stopping**

```python
# Multiple metrics monitoring
class AdvancedEarlyStopping:
    def __init__(self, patience=10, metrics=['val_loss', 'val_accuracy']):
        self.patience = patience
        self.metrics = metrics
        self.counters = {m: 0 for m in metrics}
        self.best_scores = {m: None for m in metrics}
        self.early_stop = False
    
    def __call__(self, current_metrics, model):
        should_stop = True
        
        for metric, value in current_metrics.items():
            if metric not in self.metrics:
                continue
            
            if self.best_scores[metric] is None:
                self.best_scores[metric] = value
                self.counters[metric] = 0
                should_stop = False
            elif self._is_improvement(metric, value):
                self.best_scores[metric] = value
                self.counters[metric] = 0
                should_stop = False
            else:
                self.counters[metric] += 1
        
        # Stop if ALL metrics haven't improved
        if all(c >= self.patience for c in self.counters.values()):
            self.early_stop = True
    
    def _is_improvement(self, metric, value):
        if 'loss' in metric:
            return value < self.best_scores[metric]
        else:  # accuracy, auc, etc.
            return value > self.best_scores[metric]
```

### **Characteristics**
- ✅ **Simple and effective**: Easy to implement
- ✅ **Prevents overfitting**: Stops before performance degrades
- ✅ **Saves computation**: Avoids unnecessary training
- ✅ **Automatic**: No manual monitoring required
- ⚠️ **Validation set required**: Needs held-out data
- ⚠️ **Can stop too early**: May miss delayed improvements
- ⚠️ **Noisy metrics**: Validation loss can fluctuate

### **Best Practices**
```python
# Recommended configurations
early_stopping_configs = {
    'Small datasets': {
        'patience': 20,      # More patience for noisy signals
        'min_delta': 0.001
    },
    'Large datasets': {
        'patience': 5,       # Less patience, clearer signals
        'min_delta': 0.0001
    },
    'Deep networks': {
        'patience': 15,      # Allow time for convergence
        'restore_best_weights': True
    }
}
```

## 7. Data Augmentation

### **Concept**
Artificially expands training data by applying transformations, reducing overfitting.

### **Image Augmentation**

```python
# TensorFlow/Keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define augmentation pipeline
datagen = ImageDataGenerator(
    rotation_range=20,           # Random rotation ±20 degrees
    width_shift_range=0.2,       # Horizontal shift
    height_shift_range=0.2,      # Vertical shift
    horizontal_flip=True,        # Random horizontal flip
    zoom_range=0.2,              # Random zoom
    shear_range=0.2,             # Shear transformation
    fill_mode='nearest',         # Fill mode for new pixels
    brightness_range=[0.8, 1.2]  # Brightness adjustment
)

# Apply to training data
train_generator = datagen.flow(
    X_train, y_train,
    batch_size=32
)

model.fit(train_generator, epochs=50)

# Using tf.keras.layers (preferred in TF 2.x)
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2),
    keras.layers.RandomTranslation(0.1, 0.1),
    keras.layers.RandomContrast(0.2),
])

# Add to model
model = keras.Sequential([
    data_augmentation,  # Apply during training only
    keras.layers.Conv2D(32, 3, activation='relu'),
    # ... rest of model
])

# PyTorch (torchvision)
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(
    'path/to/train',
    transform=train_transforms
)

# Advanced: Albumentations (more options)
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.Transpose(),
    A.GaussNoise(p=0.2),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.1),
        A.PiecewiseAffine(p=0.3),
    ], p=0.2),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

### **Text Augmentation**

```python
# Synonym replacement
import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_src='wordnet')
text = "The quick brown fox jumps over the lazy dog"
augmented_text = aug.augment(text)
# Output: "The fast brown fox jumps over the lazy dog"

# Back translation
aug = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de',
    to_model_name='facebook/wmt19-de-en'
)
augmented_text = aug.augment(text)

# Random insertion/deletion
import nlpaug.augmenter.word as naw

aug_insert = naw.RandomWordAug(action='insert')
aug_delete = naw.RandomWordAug(action='delete')
aug_swap = naw.RandomWordAug(action='swap')

# Contextual word embeddings
aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased',
    action='substitute'
)
augmented_text = aug.augment(text)
```

### **Tabular Data Augmentation**

```python
# SMOTE (Synthetic Minority Over-sampling Technique)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Gaussian noise injection
def add_noise(X, noise_level=0.05):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

X_augmented = add_noise(X_train)

# Mixup (interpolation between samples)
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Use in training
x_mixed, y_a, y_b, lam = mixup_data(x_batch, y_batch)
loss = lam * criterion(model(x_mixed), y_a) + (1 - lam) * criterion(model(x_mixed), y_b)
```

### **Advanced Techniques**

#### **CutMix**
```python
# CutMix for images
def cutmix(images, labels, alpha=1.0):
    indices = torch.randperm(images.size(0))
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]
    
    lam = np.random.beta(alpha, alpha)
    
    # Generate random box
    W, H = images.size(2), images.size(3)
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    
    images[:, :, x1:x2, y1:y2] = shuffled_images[:, :, x1:x2, y1:y2]
    
    # Adjust lambda based on actual cut area
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    
    return images, labels, shuffled_labels, lam
```

#### **AutoAugment**
```python
# Automatically learned augmentation policies
from torchvision.transforms import AutoAugmentPolicy, AutoAugment

# ImageNet policy
autoaugment = AutoAugment(policy=AutoAugmentPolicy.IMAGENET)

# Apply in transform pipeline
transform = transforms.Compose([
    autoaugment,
    transforms.ToTensor(),
])
```

### **Characteristics**
- ✅ **Increases effective dataset size**: More training examples
- ✅ **Improves generalization**: Exposes model to variations
- ✅ **Domain-specific**: Tailored to data type
- ✅ **No inference cost**: Only applied during training
- ⚠️ **Can introduce unrealistic data**: Careful with transformations
- ⚠️ **Slower training**: More computation per epoch
- ⚠️ **Hyperparameter tuning**: Need to tune augmentation strength

## 8. Weight Constraints

### **Concept**
Constrains weight values during training, preventing them from growing too large.

### **Implementation**

```python
# TensorFlow/Keras
from tensorflow.keras import constraints

model = keras.Sequential([
    keras.layers.Dense(
        128,
        activation='relu',
        kernel_constraint=constraints.MaxNorm(max_value=3.0)  # Constrain ||w|| ≤ 3
    ),
    keras.layers.Dense(
        64,
        kernel_constraint=constraints.UnitNorm()  # Constrain ||w|| = 1
    ),
    keras.layers.Dense(
        32,
        kernel_constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0)
    ),
    keras.layers.Dense(10, activation='softmax')
])

# PyTorch - Manual implementation
class MaxNormConstraint:
    def __init__(self, max_norm=3.0):
        self.max_norm = max_norm
    
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            norm = w.norm(2, dim=1, keepdim=True)
            desired = torch.clamp(norm, 0, self.max_norm)
            module.weight.data = w * (desired / (norm + 1e-8))

# Apply constraint after optimizer step
max_norm_constraint = MaxNormConstraint(max_norm=3.0)

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch), labels)
        loss.backward()
        optimizer.step()
        
        # Apply constraint
        model.apply(max_norm_constraint)
```

### **Types of Constraints**

```python
# Max-Norm Constraint
# ||w|| ≤ c for each weight vector
max_norm = constraints.MaxNorm(max_value=2.0, axis=0)

# Unit-Norm Constraint
# ||w|| = 1 (normalize weights)
unit_norm = constraints.UnitNorm(axis=0)

# Non-Negativity Constraint
# w ≥ 0 (all weights positive)
non_neg = constraints.NonNeg()

# Min-Max Constraint
# min ≤ w ≤ max
min_max = constraints.MinMaxNorm(min_value=-1.0, max_value=1.0)
```

### **Characteristics**
- ✅ **Prevents weight explosion**: Keeps weights bounded
- ✅ **Stabilizes training**: Especially for deep networks
- ✅ **Complements other regularization**: Works with L1/L2
- ⚠️ **Less common**: Not as widely used as other methods
- ⚠️ **Hard constraints**: Can conflict with optimization

## 9. MinDiff (Fairness Regularization)

### **Concept**
MinDiff is a fairness-aware regularization technique that minimizes the difference in model predictions between different demographic groups or slices. It helps reduce bias and improve fairness by adding a penalty term when the model behaves differently for protected groups.

### **Mathematical Formula**
```
Loss = Original_Loss + λ × MinDiff_Loss

MinDiff_Loss = E[|f(x_A) - f(x_B)|²]

Where:
- f(x_A): Model predictions on group A (e.g., sensitive group)
- f(x_B): Model predictions on group B (e.g., reference group)
- λ (lambda): MinDiff weight (importance of fairness constraint)
- E[·]: Expected value over paired examples
```

### **Implementation**

#### **TensorFlow with TensorFlow Model Remediation**

```python
# Install: pip install tensorflow-model-remediation
import tensorflow as tf
from tensorflow_model_remediation import min_diff
from tensorflow_model_remediation.min_diff import losses

# Step 1: Prepare MinDiff data
# Create dataset with pairs from different groups
def create_mindiff_dataset(sensitive_group_data, reference_group_data):
    """
    sensitive_group_data: Data from underrepresented/protected group
    reference_group_data: Data from majority/reference group
    """
    mindiff_data = min_diff.keras.utils.pack_min_diff_data(
        original_dataset=train_dataset,
        sensitive_group_dataset=sensitive_group_data,
        nonsensitive_group_dataset=reference_group_data
    )
    return mindiff_data

# Step 2: Build model with MinDiff
original_model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Wrap model with MinDiff
mindiff_model = min_diff.keras.MinDiffModel(
    original_model=original_model,
    loss=losses.MMDLoss(),  # Maximum Mean Discrepancy loss
    loss_weight=1.0  # λ parameter
)

# Step 3: Compile
mindiff_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 4: Train
history = mindiff_model.fit(
    mindiff_data,
    epochs=10,
    validation_data=val_mindiff_data
)

# Example with custom loss
mindiff_model = min_diff.keras.MinDiffModel(
    original_model=original_model,
    loss=losses.AbsoluteDifferenceLoss(),  # L1 difference
    # Or: losses.PredictionRateLoss()  # Equalizes positive prediction rates
    loss_weight=0.5
)
```

#### **Manual Implementation (PyTorch)**

```python
import torch
import torch.nn as nn

class MinDiffLoss(nn.Module):
    def __init__(self, loss_type='mmd', lambda_weight=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.lambda_weight = lambda_weight
    
    def forward(self, pred_sensitive, pred_reference):
        """
        pred_sensitive: Predictions on sensitive group
        pred_reference: Predictions on reference group
        """
        if self.loss_type == 'mmd':
            # Maximum Mean Discrepancy
            loss = self._mmd_loss(pred_sensitive, pred_reference)
        elif self.loss_type == 'abs_diff':
            # Absolute difference
            loss = torch.mean(torch.abs(pred_sensitive - pred_reference))
        elif self.loss_type == 'squared_diff':
            # Squared difference
            loss = torch.mean((pred_sensitive - pred_reference) ** 2)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return self.lambda_weight * loss
    
    def _mmd_loss(self, x, y):
        """Maximum Mean Discrepancy with RBF kernel"""
        xx = torch.mean(self._rbf_kernel(x, x))
        yy = torch.mean(self._rbf_kernel(y, y))
        xy = torch.mean(self._rbf_kernel(x, y))
        return xx + yy - 2 * xy
    
    def _rbf_kernel(self, x, y, gamma=1.0):
        """Radial Basis Function kernel"""
        dist = torch.cdist(x.unsqueeze(1), y.unsqueeze(1), p=2)
        return torch.exp(-gamma * dist ** 2)

# Training loop with MinDiff
class FairModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# Initialize
model = FairModel(input_size=20, hidden_size=128, output_size=1)
criterion = nn.BCELoss()
mindiff_loss = MinDiffLoss(loss_type='mmd', lambda_weight=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training with MinDiff
def train_with_mindiff(model, train_loader, sensitive_loader, reference_loader):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        # Iterate through batches
        for (batch_data, batch_labels), sensitive_batch, reference_batch in \
            zip(train_loader, sensitive_loader, reference_loader):
            
            optimizer.zero_grad()
            
            # Original task loss
            predictions = model(batch_data)
            task_loss = criterion(predictions, batch_labels)
            
            # MinDiff loss (fairness penalty)
            pred_sensitive = model(sensitive_batch[0])
            pred_reference = model(reference_batch[0])
            fairness_loss = mindiff_loss(pred_sensitive, pred_reference)
            
            # Combined loss
            total_loss_batch = task_loss + fairness_loss
            
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

# Usage
train_with_mindiff(model, train_loader, sensitive_loader, reference_loader)
```

#### **Scikit-learn Compatible Wrapper**

```python
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class MinDiffClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, lambda_weight=1.0, max_iter=100):
        self.base_estimator = base_estimator
        self.lambda_weight = lambda_weight
        self.max_iter = max_iter
    
    def fit(self, X, y, sensitive_groups=None):
        """
        X: Feature matrix
        y: Labels
        sensitive_groups: Binary array indicating sensitive group membership
        """
        if sensitive_groups is None:
            # Fallback to regular training
            self.base_estimator.fit(X, y)
            return self
        
        # Split into sensitive and reference groups
        X_sensitive = X[sensitive_groups == 1]
        X_reference = X[sensitive_groups == 0]
        y_sensitive = y[sensitive_groups == 1]
        y_reference = y[sensitive_groups == 0]
        
        # Train with fairness constraint (simplified)
        # In practice, this would use gradient-based optimization
        for iteration in range(self.max_iter):
            # Fit on all data
            self.base_estimator.fit(X, y)
            
            # Check predictions on both groups
            pred_sensitive = self.base_estimator.predict_proba(X_sensitive)[:, 1]
            pred_reference = self.base_estimator.predict_proba(X_reference)[:, 1]
            
            # Calculate disparity
            disparity = np.abs(np.mean(pred_sensitive) - np.mean(pred_reference))
            
            if disparity < 0.01:  # Convergence threshold
                break
            
            # Adjust sample weights to reduce disparity
            # (This is a simplified heuristic approach)
            sample_weights = np.ones(len(X))
            if np.mean(pred_sensitive) > np.mean(pred_reference):
                sample_weights[sensitive_groups == 1] *= (1 - self.lambda_weight * 0.1)
            else:
                sample_weights[sensitive_groups == 0] *= (1 - self.lambda_weight * 0.1)
            
            # Refit with adjusted weights
            if hasattr(self.base_estimator, 'fit') and 'sample_weight' in \
               self.base_estimator.fit.__code__.co_varnames:
                self.base_estimator.fit(X, y, sample_weight=sample_weights)
        
        return self
    
    def predict(self, X):
        return self.base_estimator.predict(X)
    
    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

# Usage
from sklearn.ensemble import RandomForestClassifier

base_model = RandomForestClassifier(n_estimators=100, random_state=42)
fair_model = MinDiffClassifier(base_estimator=base_model, lambda_weight=1.0)

# Train with sensitive group indicators
sensitive_groups = (df['gender'] == 'female').values  # Example
fair_model.fit(X_train, y_train, sensitive_groups=sensitive_groups)
```

### **MinDiff Loss Functions**

#### **1. Maximum Mean Discrepancy (MMD)**
```python
# Measures distribution distance using kernel methods
# Best for: Overall distribution matching

loss = losses.MMDLoss(
    kernel='gaussian',  # or 'laplacian'
    predictions_transform='sigmoid'  # Transform logits if needed
)
```

#### **2. Absolute Difference Loss**
```python
# Penalizes absolute difference in predictions
# Best for: Simple fairness constraints

loss = losses.AbsoluteDifferenceLoss()
```

#### **3. Prediction Rate Loss**
```python
# Equalizes positive prediction rates across groups
# Best for: Demographic parity

loss = losses.PredictionRateLoss()
```

### **Complete Example: Fair Lending Model**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow_model_remediation import min_diff

# Step 1: Load and prepare data
# Assume we have a lending dataset with protected attribute 'race'
train_df = pd.read_csv('lending_data.csv')

# Identify groups
sensitive_group = train_df[train_df['race'] == 'minority']
reference_group = train_df[train_df['race'] == 'majority']

# Create datasets
def df_to_dataset(dataframe, labels, batch_size=32):
    features = dataframe.drop(['race', 'approved'], axis=1)
    ds = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

# Original training data
train_dataset = df_to_dataset(
    train_df,
    train_df['approved'].values,
    batch_size=32
)

# MinDiff datasets (without labels for fairness constraint)
sensitive_dataset = tf.data.Dataset.from_tensor_slices(
    dict(sensitive_group.drop(['race', 'approved'], axis=1))
).batch(32)

reference_dataset = tf.data.Dataset.from_tensor_slices(
    dict(reference_group.drop(['race', 'approved'], axis=1))
).batch(32)

# Pack MinDiff data
mindiff_dataset = min_diff.keras.utils.pack_min_diff_data(
    original_dataset=train_dataset,
    sensitive_group_dataset=sensitive_dataset,
    nonsensitive_group_dataset=reference_dataset
)

# Step 2: Build model
def create_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

base_model = create_model(input_shape=train_df.shape[1] - 2)

# Step 3: Apply MinDiff
mindiff_model = min_diff.keras.MinDiffModel(
    original_model=base_model,
    loss=min_diff.losses.MMDLoss(),
    loss_weight=1.5  # Tune based on fairness-accuracy tradeoff
)

mindiff_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

# Step 4: Train
history = mindiff_model.fit(
    mindiff_dataset,
    epochs=20,
    validation_data=val_mindiff_dataset,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)

# Step 5: Evaluate fairness
def evaluate_fairness(model, data_df):
    """Evaluate model fairness across groups"""
    predictions = model.predict(data_df.drop(['race', 'approved'], axis=1))
    
    for group in data_df['race'].unique():
        group_data = data_df[data_df['race'] == group]
        group_preds = predictions[data_df['race'] == group]
        
        approval_rate = np.mean(group_preds > 0.5)
        avg_score = np.mean(group_preds)
        
        print(f"{group}:")
        print(f"  Approval Rate: {approval_rate:.2%}")
        print(f"  Average Score: {avg_score:.4f}")
        print()
    
    # Calculate demographic parity difference
    minority_rate = np.mean(predictions[data_df['race'] == 'minority'] > 0.5)
    majority_rate = np.mean(predictions[data_df['race'] == 'majority'] > 0.5)
    dp_diff = abs(minority_rate - majority_rate)
    
    print(f"Demographic Parity Difference: {dp_diff:.4f}")
    print(f"Fair model: {dp_diff < 0.1}")  # Common threshold

# Evaluate
print("Without MinDiff:")
evaluate_fairness(base_model, test_df)

print("\nWith MinDiff:")
evaluate_fairness(mindiff_model, test_df)
```

### **Hyperparameter Tuning**

```python
# Tune lambda_weight for fairness-accuracy tradeoff
def tune_mindiff_weight(weights=[0.1, 0.5, 1.0, 2.0, 5.0]):
    results = []
    
    for weight in weights:
        print(f"\nTraining with lambda={weight}")
        
        # Create model
        base_model = create_model(input_shape=X_train.shape[1])
        mindiff_model = min_diff.keras.MinDiffModel(
            original_model=base_model,
            loss=min_diff.losses.MMDLoss(),
            loss_weight=weight
        )
        
        mindiff_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        mindiff_model.fit(
            mindiff_dataset,
            epochs=10,
            verbose=0
        )
        
        # Evaluate
        accuracy = mindiff_model.evaluate(test_dataset, verbose=0)[1]
        
        # Calculate fairness metric
        predictions = mindiff_model.predict(test_data)
        fairness_metric = calculate_demographic_parity(predictions, test_labels, groups)
        
        results.append({
            'lambda': weight,
            'accuracy': accuracy,
            'fairness_violation': fairness_metric
        })
        
        print(f"Accuracy: {accuracy:.4f}, Fairness Violation: {fairness_metric:.4f}")
    
    return pd.DataFrame(results)

# Run tuning
results_df = tune_mindiff_weight()

# Plot tradeoff curve
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(results_df['fairness_violation'], results_df['accuracy'], 'o-')
plt.xlabel('Fairness Violation (lower is better)')
plt.ylabel('Accuracy')
plt.title('Fairness-Accuracy Tradeoff')
plt.grid(True)
for i, row in results_df.iterrows():
    plt.annotate(f"λ={row['lambda']}", (row['fairness_violation'], row['accuracy']))
plt.show()
```

### **Characteristics**
- ✅ **Reduces bias**: Minimizes prediction disparities across groups
- ✅ **Flexible**: Multiple loss functions for different fairness notions
- ✅ **Integration-friendly**: Works with existing models
- ✅ **Measurable**: Quantifiable fairness improvements
- ⚠️ **Accuracy tradeoff**: May reduce overall accuracy slightly
- ⚠️ **Requires group labels**: Need sensitive attribute information
- ⚠️ **Hyperparameter tuning**: Need to balance fairness vs accuracy

### **Use Cases**

```python
# MinDiff is ideal for:
use_cases = {
    'Lending/Credit': 'Fair approval rates across demographics',
    'Hiring': 'Reduce bias in candidate screening',
    'Healthcare': 'Equal treatment recommendations across groups',
    'Criminal Justice': 'Fair risk assessment across ethnicities',
    'Advertising': 'Equal opportunity for ad targeting',
    'Education': 'Fair admissions decisions'
}

# Example: Healthcare model
# Goal: Ensure treatment recommendations are fair across age groups

# Define groups
young_patients = patients_df[patients_df['age'] < 40]
older_patients = patients_df[patients_df['age'] >= 40]

# Create MinDiff datasets
young_dataset = create_dataset(young_patients)
older_dataset = create_dataset(older_patients)

# Apply MinDiff
mindiff_model = min_diff.keras.MinDiffModel(
    original_model=treatment_model,
    loss=min_diff.losses.MMDLoss(),
    loss_weight=1.0
)
```

### **Fairness Metrics to Monitor**

```python
def calculate_fairness_metrics(predictions, labels, sensitive_attr):
    """Calculate common fairness metrics"""
    from sklearn.metrics import confusion_matrix
    
    metrics = {}
    
    for group in np.unique(sensitive_attr):
        group_mask = (sensitive_attr == group)
        group_preds = predictions[group_mask]
        group_labels = labels[group_mask]
        
        # Positive prediction rate (Demographic Parity)
        ppr = np.mean(group_preds > 0.5)
        
        # True Positive Rate (Equal Opportunity)
        tn, fp, fn, tp = confusion_matrix(
            group_labels,
            group_preds > 0.5
        ).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # False Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics[group] = {
            'positive_rate': ppr,
            'true_positive_rate': tpr,
            'false_positive_rate': fpr
        }
    
    # Calculate disparities
    groups = list(metrics.keys())
    demographic_parity = abs(
        metrics[groups[0]]['positive_rate'] - 
        metrics[groups[1]]['positive_rate']
    )
    
    equal_opportunity = abs(
        metrics[groups[0]]['true_positive_rate'] - 
        metrics[groups[1]]['true_positive_rate']
    )
    
    print("Fairness Metrics:")
    print(f"  Demographic Parity Difference: {demographic_parity:.4f}")
    print(f"  Equal Opportunity Difference: {equal_opportunity:.4f}")
    print(f"  Fair (< 0.1): {demographic_parity < 0.1 and equal_opportunity < 0.1}")
    
    return metrics

# Usage after training
predictions = model.predict(X_test)
fairness_metrics = calculate_fairness_metrics(
    predictions,
    y_test,
    sensitive_attribute_test
)
```

### **Best Practices**

```python
# 1. Start with baseline (no MinDiff)
baseline_model.fit(train_data)
baseline_fairness = evaluate_fairness(baseline_model)

# 2. Add MinDiff with low weight
mindiff_model = MinDiffModel(baseline_model, loss_weight=0.1)
mindiff_model.fit(mindiff_data)

# 3. Gradually increase weight until fairness threshold met
for weight in [0.5, 1.0, 2.0, 5.0]:
    test_model = MinDiffModel(baseline_model, loss_weight=weight)
    test_model.fit(mindiff_data)
    fairness = evaluate_fairness(test_model)
    if fairness < threshold:
        break

# 4. Monitor both accuracy and fairness
callbacks = [
    FairnessMetricCallback(sensitive_data),
    EarlyStopping(monitor='val_loss', patience=5)
]

# 5. Use appropriate loss function
# - MMDLoss: General distribution matching
# - AbsoluteDifferenceLoss: Simple mean difference
# - PredictionRateLoss: Demographic parity
```

### **Integration with Other Regularization**

```python
# Combine MinDiff with other regularization techniques
model = keras.Sequential([
    keras.layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=keras.regularizers.L2(0.01)  # L2 regularization
    ),
    keras.layers.BatchNormalization(),  # Batch normalization
    keras.layers.Dropout(0.3),  # Dropout
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

# Wrap with MinDiff
mindiff_model = min_diff.keras.MinDiffModel(
    original_model=model,
    loss=min_diff.losses.MMDLoss(),
    loss_weight=1.0  # Fairness regularization
)

# Train with all regularization techniques
history = mindiff_model.fit(
    mindiff_dataset,
    epochs=50,
    callbacks=[
        EarlyStopping(patience=10, restore_best_weights=True)  # Early stopping
    ]
)
```

### **Limitations and Considerations**

```python
# When MinDiff may not be suitable:
limitations = {
    'Small group sizes': 'Unstable if sensitive group has < 100 examples',
    'Multiple protected attributes': 'Complex to balance across many groups',
    'Accuracy-critical applications': 'May sacrifice too much accuracy',
    'Unlabeled groups': 'Requires known group membership',
    'Intersectionality': 'Difficult to handle intersecting identities'
}

# Alternative approaches:
alternatives = {
    'Reweighting': 'Adjust sample weights instead of loss',
    'Adversarial Debiasing': 'Use adversarial networks for fairness',
    'Post-processing': 'Adjust predictions after training',
    'Fair Representation': 'Learn fair feature representations'
}
```

## 10. Noise Injection

### **Concept**
Adds noise to inputs, weights, or gradients during training to improve robustness.

### **Implementation**

```python
# Input noise (Gaussian)
class GaussianNoise(keras.layers.Layer):
    def __init__(self, stddev=0.1, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev
    
    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(
                shape=tf.shape(inputs),
                mean=0.0,
                stddev=self.stddev
            )
            return inputs + noise
        return inputs

model = keras.Sequential([
    GaussianNoise(stddev=0.1),  # Add noise to inputs
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# PyTorch
class GaussianNoisePyTorch(nn.Module):
    def __init__(self, stddev=0.1):
        super().__init__()
        self.stddev = stddev
    
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x

# Weight noise
def add_weight_noise(model, noise_std=0.01):
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                noise = torch.randn_like(param) * noise_std
                param.add_(noise)

# Gradient noise (for optimization)
class NoisySGD(torch.optim.SGD):
    def __init__(self, params, lr, noise_std=0.01, **kwargs):
        super().__init__(params, lr, **kwargs)
        self.noise_std = noise_std
    
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    noise = torch.randn_like(p.grad) * self.noise_std
                    p.grad.add_(noise)
        
        super().step(closure)
```

### **Characteristics**
- ✅ **Improves robustness**: Model less sensitive to perturbations
- ✅ **Acts as regularization**: Similar effect to dropout
- ✅ **Helps escape local minima**: Adds stochasticity to optimization
- ⚠️ **Can slow convergence**: Too much noise hurts training
- ⚠️ **Hyperparameter sensitive**: Need to tune noise level

## Regularization Comparison Table

| Technique | Prevents Overfitting | Feature Selection | Computational Cost | Best For |
|-----------|---------------------|-------------------|-------------------|----------|
| **L1 (Lasso)** | ✅ Moderate | ✅ Yes | Low | High-dimensional sparse data |
| **L2 (Ridge)** | ✅ Strong | ❌ No | Low | General purpose, deep learning |
| **Elastic Net** | ✅ Strong | ✅ Yes | Low | Correlated features |
| **Dropout** | ✅ Strong | ❌ No | Medium | Deep neural networks |
| **Batch Norm** | ✅ Moderate | ❌ No | Medium | CNNs, deep networks |
| **Early Stopping** | ✅ Strong | ❌ No | None | All models (universal) |
| **Data Augmentation** | ✅ Very Strong | ❌ No | High | Limited data scenarios |
| **Weight Constraints** | ✅ Moderate | ❌ No | Low | Deep networks, RNNs |
| **MinDiff** | ✅ Moderate | ❌ No | Medium | Fairness-critical applications |
| **Noise Injection** | ✅ Moderate | ❌ No | Medium | Adversarial robustness |

## Combining Regularization Techniques

```python
# Best practice: Combine multiple techniques
model = keras.Sequential([
    # Input augmentation
    keras.layers.GaussianNoise(0.1),
    
    # Dense layer with L2 and dropout
    keras.layers.Dense(
        512,
        activation='relu',
        kernel_regularizer=regularizers.L2(0.01),  # L2
        kernel_constraint=constraints.MaxNorm(3.0)  # Weight constraint
    ),
    keras.layers.BatchNormalization(),  # Batch norm
    keras.layers.Dropout(0.5),  # Dropout
    
    # Another dense layer
    keras.layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=regularizers.L2(0.01)
    ),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    # Output
    keras.layers.Dense(10, activation='softmax')
])

# Compile with optimizer
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with early stopping and data augmentation
history = model.fit(
    train_dataset,  # With data augmentation
    validation_data=val_dataset,
    epochs=100,
    callbacks=[
        EarlyStopping(patience=10, restore_best_weights=True)
    ]
)
```

## Choosing the Right Regularization

```python
# Decision tree for regularization selection
def choose_regularization(problem_type, data_size, features):
    """
    problem_type: 'classification', 'regression', 'generation'
    data_size: 'small' (<10k), 'medium' (10k-1M), 'large' (>1M)
    features: 'low' (<100), 'medium' (100-10k), 'high' (>10k)
    """
    recommendations = []
    
    # Always use early stopping
    recommendations.append('EarlyStopping')
    
    # L2 regularization for most cases
    if problem_type in ['classification', 'regression']:
        recommendations.append('L2 (weight_decay)')
    
    # Feature selection for high-dimensional data
    if features == 'high':
        recommendations.append('L1 or Elastic Net')
    
    # Data augmentation for small datasets
    if data_size == 'small':
        recommendations.append('Data Augmentation (strong)')
    
    # Dropout for deep networks
    if problem_type in ['classification', 'generation']:
        recommendations.append('Dropout (0.3-0.5)')
    
    # Batch normalization for CNNs and deep networks
    if problem_type in ['classification', 'generation']:
        recommendations.append('Batch Normalization')
    
    return recommendations

# Example usage
print(choose_regularization('classification', 'small', 'high'))
# Output: ['EarlyStopping', 'L2 (weight_decay)', 'L1 or Elastic Net', 
#          'Data Augmentation (strong)', 'Dropout (0.3-0.5)', 'Batch Normalization']
```

## Summary

### **Quick Reference**

```python
# Standard recipe for deep learning
standard_regularization = {
    'L2': 0.01,                    # Weight decay in optimizer
    'Dropout': 0.3-0.5,           # After dense layers
    'Batch Normalization': True,   # Before or after activations
    'Early Stopping': {
        'patience': 10,
        'restore_best_weights': True
    },
    'Data Augmentation': 'task-specific'
}

# For interpretability (feature selection)
interpretable_regularization = {
    'L1 or Elastic Net': 'primary',
    'Early Stopping': 'always'
}

# For small datasets
small_data_regularization = {
    'Data Augmentation': 'aggressive',
    'Dropout': 0.5-0.7,  # Higher dropout
    'L2': 0.01-0.1,      # Stronger L2
    'Early Stopping': 'conservative (patience=20)'
}
```

---

**Pro Tip**: Start with L2 regularization + Early Stopping as a baseline. Add Dropout and Batch Normalization for deep networks. Use Data Augmentation when you have limited training data. Combine multiple techniques for best results!

Similar code found with 2 license types