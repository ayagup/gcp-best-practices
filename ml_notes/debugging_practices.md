# Debugging Deep Learning Datasets: Common & Best Practices

## Overview

Dataset issues are the #1 cause of poor model performance in deep learning. Proper dataset debugging can save weeks of wasted training time and compute resources.

## Common Dataset Problems

### 1. **Data Quality Issues**
- Missing values
- Corrupted files
- Incorrect labels
- Duplicate records
- Inconsistent formats
- Invalid/out-of-range values

### 2. **Data Distribution Issues**
- Class imbalance
- Train/validation/test leakage
- Distribution shift between splits
- Outliers and anomalies
- Biased sampling

### 3. **Pipeline Issues**
- Incorrect preprocessing
- Wrong data types
- Memory bottlenecks
- Slow data loading
- Incorrect batching

## Best Practices for Dataset Debugging

### 1. **Start with Data Inspection**

#### 1.1 Basic Statistics
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and inspect
df = pd.read_csv('data.csv')

# Basic info
print(df.info())
print(df.describe())
print(df.head())

# Check for missing values
print(df.isnull().sum())
print(f"Total missing: {df.isnull().sum().sum()}")

# Check duplicates
print(f"Duplicates: {df.duplicated().sum()}")

# Check data types
print(df.dtypes)

# Value counts for categorical
for col in df.select_dtypes(include=['object']).columns:
    print(f"\n{col}:")
    print(df[col].value_counts())
```

#### 1.2 Visualize Distributions
```python
# Numerical features
df.hist(bins=50, figsize=(20, 15))
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.show()

# Box plots for outliers
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for idx, col in enumerate(df.select_dtypes(include=[np.number]).columns[:6]):
    ax = axes[idx // 3, idx % 3]
    df.boxplot(column=col, ax=ax)
plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()
```

#### 1.3 Class Distribution (Classification)
```python
import matplotlib.pyplot as plt

# Class distribution
class_counts = df['label'].value_counts()
print("Class distribution:")
print(class_counts)
print(f"\nImbalance ratio: {class_counts.max() / class_counts.min():.2f}")

# Visualize
plt.figure(figsize=(10, 6))
class_counts.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Check if classes are properly represented in splits
train_dist = train_df['label'].value_counts(normalize=True)
val_dist = val_df['label'].value_counts(normalize=True)
test_dist = test_df['label'].value_counts(normalize=True)

comparison = pd.DataFrame({
    'Train': train_dist,
    'Validation': val_dist,
    'Test': test_dist
})
print("\nClass distribution across splits:")
print(comparison)
```

### 2. **Validate Data Pipeline**

#### 2.1 Check Data Loading
```python
import tensorflow as tf

# Create simple dataset
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.batch(32)

# Inspect first batch
for batch_x, batch_y in dataset.take(1):
    print(f"Batch X shape: {batch_x.shape}")
    print(f"Batch Y shape: {batch_y.shape}")
    print(f"X dtype: {batch_x.dtype}")
    print(f"Y dtype: {batch_y.dtype}")
    print(f"X range: [{batch_x.numpy().min()}, {batch_x.numpy().max()}]")
    print(f"Y unique values: {np.unique(batch_y.numpy())}")
    
# Check for NaN/Inf
for batch_x, batch_y in dataset.take(10):
    if tf.reduce_any(tf.math.is_nan(batch_x)):
        print("WARNING: NaN values found in X!")
    if tf.reduce_any(tf.math.is_inf(batch_x)):
        print("WARNING: Inf values found in X!")
```

#### 2.2 Visualize Random Samples
```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_samples(dataset, num_samples=9, class_names=None):
    """Visualize random samples from dataset"""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    
    for i, (images, labels) in enumerate(dataset.take(1)):
        for j in range(min(num_samples, len(images))):
            ax = axes[j]
            
            # Handle different image formats
            img = images[j].numpy()
            if img.shape[-1] == 1:  # Grayscale
                ax.imshow(img.squeeze(), cmap='gray')
            else:  # RGB
                # Denormalize if needed
                if img.min() < 0:
                    img = (img + 1) / 2
                ax.imshow(img)
            
            label = labels[j].numpy()
            title = f"Label: {class_names[label] if class_names else label}"
            ax.set_title(title)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150)
    plt.show()

# Usage
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.batch(32).shuffle(1000)

class_names = ['cat', 'dog', 'bird']
visualize_samples(train_dataset, class_names=class_names)
```

#### 2.3 Check Preprocessing Pipeline
```python
def debug_preprocessing(image_path, preprocess_fn):
    """Debug image preprocessing step by step"""
    import cv2
    
    # Load original
    original = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Apply preprocessing
    processed = preprocess_fn(original_rgb)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_rgb)
    axes[0].set_title(f'Original\nShape: {original_rgb.shape}\nRange: [{original_rgb.min()}, {original_rgb.max()}]')
    axes[0].axis('off')
    
    # Handle normalized images
    if processed.min() < 0:
        display_img = (processed + 1) / 2
    else:
        display_img = processed
    
    axes[1].imshow(display_img)
    axes[1].set_title(f'Processed\nShape: {processed.shape}\nRange: [{processed.min():.2f}, {processed.max():.2f}]')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return original_rgb, processed

# Usage
def my_preprocess(img):
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
    return img

debug_preprocessing('sample_image.jpg', my_preprocess)
```

### 3. **Detect Data Leakage**

#### 3.1 Check for Identical Samples
```python
def check_data_leakage(train_df, val_df, test_df, id_column=None):
    """Check for data leakage between splits"""
    
    if id_column:
        # Check by ID
        train_ids = set(train_df[id_column])
        val_ids = set(val_df[id_column])
        test_ids = set(test_df[id_column])
        
        train_val_overlap = train_ids & val_ids
        train_test_overlap = train_ids & test_ids
        val_test_overlap = val_ids & test_ids
        
        print(f"Train-Val overlap: {len(train_val_overlap)} samples")
        print(f"Train-Test overlap: {len(train_test_overlap)} samples")
        print(f"Val-Test overlap: {len(val_test_overlap)} samples")
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print("⚠️ WARNING: Data leakage detected!")
            return False
    
    # Check by entire row (slower)
    train_hashes = set(pd.util.hash_pandas_object(train_df).values)
    val_hashes = set(pd.util.hash_pandas_object(val_df).values)
    test_hashes = set(pd.util.hash_pandas_object(test_df).values)
    
    train_val_overlap = len(train_hashes & val_hashes)
    train_test_overlap = len(train_hashes & test_hashes)
    val_test_overlap = len(val_hashes & test_hashes)
    
    print(f"\nRow-based check:")
    print(f"Train-Val duplicate rows: {train_val_overlap}")
    print(f"Train-Test duplicate rows: {train_test_overlap}")
    print(f"Val-Test duplicate rows: {val_test_overlap}")
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("⚠️ WARNING: Duplicate rows found across splits!")
        return False
    
    print("✓ No data leakage detected")
    return True

# Usage
check_data_leakage(train_df, val_df, test_df, id_column='customer_id')
```

#### 3.2 Check Feature Distribution Similarity
```python
from scipy.stats import ks_2samp

def check_distribution_shift(train_df, test_df, numerical_cols):
    """Check if train and test come from same distribution"""
    
    results = []
    for col in numerical_cols:
        statistic, pvalue = ks_2samp(train_df[col].dropna(), 
                                      test_df[col].dropna())
        results.append({
            'feature': col,
            'ks_statistic': statistic,
            'p_value': pvalue,
            'different_dist': pvalue < 0.05
        })
    
    results_df = pd.DataFrame(results)
    print(results_df)
    
    suspicious = results_df[results_df['different_dist']]
    if len(suspicious) > 0:
        print(f"\n⚠️ WARNING: {len(suspicious)} features show distribution shift:")
        print(suspicious['feature'].tolist())
    else:
        print("\n✓ No significant distribution shift detected")
    
    return results_df

# Usage
numerical_cols = train_df.select_dtypes(include=[np.number]).columns
check_distribution_shift(train_df, test_df, numerical_cols)
```

### 4. **Validate Labels**

#### 4.1 Check Label Consistency
```python
def validate_labels(df, label_column, expected_labels=None):
    """Validate label quality"""
    
    print("Label Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Unique labels: {df[label_column].nunique()}")
    print(f"Missing labels: {df[label_column].isnull().sum()}")
    
    # Check label values
    actual_labels = set(df[label_column].dropna().unique())
    print(f"\nActual labels: {sorted(actual_labels)}")
    
    if expected_labels:
        expected_set = set(expected_labels)
        unexpected = actual_labels - expected_set
        missing = expected_set - actual_labels
        
        if unexpected:
            print(f"⚠️ Unexpected labels found: {unexpected}")
        if missing:
            print(f"⚠️ Expected labels missing: {missing}")
        if not unexpected and not missing:
            print("✓ All labels are valid")
    
    # Check for label encoding issues
    if df[label_column].dtype in ['int64', 'int32']:
        min_label = df[label_column].min()
        max_label = df[label_column].max()
        print(f"\nLabel range: [{min_label}, {max_label}]")
        
        # Should start from 0 for most frameworks
        if min_label != 0:
            print(f"⚠️ WARNING: Labels don't start from 0")
        
        # Check for gaps
        expected_range = set(range(min_label, max_label + 1))
        actual_range = set(df[label_column].dropna().unique())
        gaps = expected_range - actual_range
        if gaps:
            print(f"⚠️ WARNING: Missing label values: {sorted(gaps)}")

# Usage
validate_labels(df, 'label', expected_labels=[0, 1, 2, 3, 4])
```

#### 4.2 Visual Label Inspection
```python
def inspect_labels_visually(dataset, class_names, num_samples=25):
    """Manually inspect if labels match images"""
    
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    axes = axes.ravel()
    
    for batch_x, batch_y in dataset.take(1):
        for i in range(min(num_samples, len(batch_x))):
            ax = axes[i]
            
            img = batch_x[i].numpy()
            if img.min() < 0:
                img = (img + 1) / 2
            
            if img.shape[-1] == 1:
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                ax.imshow(img)
            
            label_idx = batch_y[i].numpy()
            label_name = class_names[label_idx] if class_names else label_idx
            ax.set_title(f'{label_name}', fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('label_verification.png', dpi=150)
    plt.show()
    
    print("Review the images above and verify labels are correct")

# Usage
inspect_labels_visually(train_dataset, class_names=['cat', 'dog', 'bird'])
```

### 5. **Check Data Augmentation**

#### 5.1 Visualize Augmentation Effects
```python
def visualize_augmentation(image, augmentation_fn, num_augmentations=9):
    """Visualize effect of data augmentation"""
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(num_augmentations):
        augmented = augmentation_fn(image)
        
        if isinstance(augmented, tf.Tensor):
            augmented = augmented.numpy()
        
        # Denormalize if needed
        if augmented.min() < 0:
            augmented = (augmented + 1) / 2
        
        axes[i].imshow(augmented)
        axes[i].set_title(f'Augmentation {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_samples.png', dpi=150)
    plt.show()

# Example augmentation function
def augment_fn(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    return image

# Usage
sample_image = train_images[0]
visualize_augmentation(sample_image, augment_fn)
```

#### 5.2 Check Augmentation Sanity
```python
def check_augmentation_sanity(dataset, num_batches=10):
    """Ensure augmentation doesn't break data"""
    
    issues = {
        'nan_values': 0,
        'inf_values': 0,
        'out_of_range': 0,
        'wrong_shape': 0
    }
    
    expected_shape = None
    
    for i, (images, labels) in enumerate(dataset.take(num_batches)):
        # Check first batch for expected shape
        if expected_shape is None:
            expected_shape = images.shape[1:]
        
        # Check for NaN
        if tf.reduce_any(tf.math.is_nan(images)):
            issues['nan_values'] += 1
        
        # Check for Inf
        if tf.reduce_any(tf.math.is_inf(images)):
            issues['inf_values'] += 1
        
        # Check range (assuming normalized to [0, 1] or [-1, 1])
        img_min = tf.reduce_min(images)
        img_max = tf.reduce_max(images)
        
        if img_min < -1.1 or img_max > 1.1:
            issues['out_of_range'] += 1
            print(f"Batch {i}: Range [{img_min:.2f}, {img_max:.2f}]")
        
        # Check shape consistency
        if images.shape[1:] != expected_shape:
            issues['wrong_shape'] += 1
    
    print("Augmentation Sanity Check:")
    print(f"Batches checked: {num_batches}")
    for issue, count in issues.items():
        status = "✓" if count == 0 else "⚠️"
        print(f"{status} {issue}: {count}")
    
    if sum(issues.values()) == 0:
        print("\n✓ All augmentation checks passed!")
    else:
        print("\n⚠️ Issues detected in augmentation pipeline!")
    
    return issues

# Usage
augmented_dataset = train_dataset.map(
    lambda x, y: (augment_fn(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)
check_augmentation_sanity(augmented_dataset)
```

### 6. **Monitor Data Loading Performance**

#### 6.1 Profile Data Pipeline
```python
import time

def profile_data_pipeline(dataset, num_batches=100):
    """Profile data loading speed"""
    
    print("Profiling data pipeline...")
    
    # Warmup
    for _ in dataset.take(5):
        pass
    
    # Time loading
    start_time = time.time()
    batch_times = []
    
    for i, batch in enumerate(dataset.take(num_batches)):
        batch_start = time.time()
        _ = batch  # Force execution
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        if i % 10 == 0:
            print(f"Batch {i}: {batch_time*1000:.2f}ms")
    
    total_time = time.time() - start_time
    
    print(f"\nPipeline Performance:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average batch time: {np.mean(batch_times)*1000:.2f}ms")
    print(f"Min batch time: {np.min(batch_times)*1000:.2f}ms")
    print(f"Max batch time: {np.max(batch_times)*1000:.2f}ms")
    print(f"Throughput: {num_batches / total_time:.2f} batches/sec")
    
    # Check for bottlenecks
    if np.max(batch_times) > 3 * np.median(batch_times):
        print("⚠️ WARNING: Inconsistent batch loading times detected!")
        print("   Consider using .cache() or .prefetch()")

# Usage
profile_data_pipeline(train_dataset)
```

#### 6.2 Optimize Pipeline
```python
def create_optimized_pipeline(file_paths, labels, batch_size=32):
    """Create optimized tf.data pipeline"""
    
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    # Shuffle
    dataset = dataset.shuffle(buffer_size=len(file_paths))
    
    # Load and preprocess with parallelism
    def load_and_preprocess(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    dataset = dataset.map(
        load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Cache after expensive operations
    dataset = dataset.cache()
    
    # Batch
    dataset = dataset.batch(batch_size)
    
    # Augmentation (after batching for efficiency)
    def augment_batch(images, labels):
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_brightness(images, 0.2)
        return images, labels
    
    dataset = dataset.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Prefetch
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Compare performance
print("Without optimization:")
profile_data_pipeline(unoptimized_dataset, num_batches=50)

print("\nWith optimization:")
optimized_dataset = create_optimized_pipeline(file_paths, labels)
profile_data_pipeline(optimized_dataset, num_batches=50)
```

### 7. **Validate Train/Val/Test Splits**

```python
def validate_splits(train_df, val_df, test_df, label_column):
    """Comprehensive split validation"""
    
    print("=" * 60)
    print("SPLIT VALIDATION REPORT")
    print("=" * 60)
    
    # 1. Size check
    total = len(train_df) + len(val_df) + len(test_df)
    train_pct = len(train_df) / total * 100
    val_pct = len(val_df) / total * 100
    test_pct = len(test_df) / total * 100
    
    print(f"\n1. Split Sizes:")
    print(f"   Train: {len(train_df)} ({train_pct:.1f}%)")
    print(f"   Val:   {len(val_df)} ({val_pct:.1f}%)")
    print(f"   Test:  {len(test_df)} ({test_pct:.1f}%)")
    
    # 2. Class distribution
    print(f"\n2. Class Distribution:")
    train_dist = train_df[label_column].value_counts(normalize=True).sort_index()
    val_dist = val_df[label_column].value_counts(normalize=True).sort_index()
    test_dist = test_df[label_column].value_counts(normalize=True).sort_index()
    
    dist_df = pd.DataFrame({
        'Train': train_dist,
        'Val': val_dist,
        'Test': test_dist
    })
    print(dist_df)
    
    # Check if distributions are similar
    max_diff = (dist_df.max(axis=1) - dist_df.min(axis=1)).max()
    if max_diff > 0.1:
        print(f"   ⚠️ WARNING: Class distribution varies by >{max_diff*100:.1f}% across splits")
    else:
        print("   ✓ Class distributions are consistent")
    
    # 3. Check for leakage
    print(f"\n3. Data Leakage Check:")
    check_data_leakage(train_df, val_df, test_df)
    
    # 4. Feature statistics comparison
    print(f"\n4. Feature Statistics:")
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != label_column]
    
    for col in numerical_cols[:5]:  # Check first 5 features
        train_mean = train_df[col].mean()
        val_mean = val_df[col].mean()
        test_mean = test_df[col].mean()
        
        max_mean_diff = max(abs(train_mean - val_mean), abs(train_mean - test_mean))
        relative_diff = max_mean_diff / train_mean if train_mean != 0 else 0
        
        status = "✓" if relative_diff < 0.2 else "⚠️"
        print(f"   {status} {col}: Train={train_mean:.2f}, Val={val_mean:.2f}, Test={test_mean:.2f}")
    
    print("\n" + "=" * 60)

# Usage
validate_splits(train_df, val_df, test_df, label_column='label')
```

### 8. **Debug Common Issues**

#### 8.1 Check for File Corruption (Images)
```python
import cv2
from PIL import Image

def check_image_files(file_paths, num_samples=100):
    """Check for corrupted image files"""
    
    corrupted = []
    invalid_format = []
    wrong_channels = []
    
    sample_paths = np.random.choice(file_paths, min(num_samples, len(file_paths)), replace=False)
    
    for path in sample_paths:
        try:
            # Try with PIL
            img = Image.open(path)
            img.verify()
            
            # Try with OpenCV
            img = cv2.imread(path)
            if img is None:
                corrupted.append(path)
                continue
            
            # Check channels
            if len(img.shape) != 3 or img.shape[2] not in [1, 3, 4]:
                wrong_channels.append(path)
                
        except Exception as e:
            invalid_format.append((path, str(e)))
    
    print(f"Checked {len(sample_paths)} images:")
    print(f"✓ Valid: {len(sample_paths) - len(corrupted) - len(invalid_format)}")
    print(f"⚠️ Corrupted: {len(corrupted)}")
    print(f"⚠️ Invalid format: {len(invalid_format)}")
    print(f"⚠️ Wrong channels: {len(wrong_channels)}")
    
    if corrupted:
        print(f"\nCorrupted files (first 5):")
        for path in corrupted[:5]:
            print(f"  - {path}")
    
    if invalid_format:
        print(f"\nInvalid format (first 5):")
        for path, error in invalid_format[:5]:
            print(f"  - {path}: {error}")
    
    return corrupted, invalid_format, wrong_channels

# Usage
all_image_paths = train_df['image_path'].tolist()
check_image_files(all_image_paths, num_samples=1000)
```

#### 8.2 Memory Usage Monitoring
```python
import psutil
import os

def monitor_memory_usage(dataset, num_batches=10):
    """Monitor memory usage during data loading"""
    
    process = psutil.Process(os.getpid())
    
    print("Monitoring memory usage...")
    memory_samples = []
    
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory: {initial_memory:.2f} MB")
    
    for i, batch in enumerate(dataset.take(num_batches)):
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_samples.append(current_memory)
        
        if i % 2 == 0:
            print(f"Batch {i}: {current_memory:.2f} MB")
    
    final_memory = memory_samples[-1]
    memory_increase = final_memory - initial_memory
    
    print(f"\nMemory Summary:")
    print(f"Initial: {initial_memory:.2f} MB")
    print(f"Final: {final_memory:.2f} MB")
    print(f"Increase: {memory_increase:.2f} MB")
    print(f"Average per batch: {memory_increase / num_batches:.2f} MB")
    
    if memory_increase > 1000:  # More than 1GB increase
        print("⚠️ WARNING: High memory usage detected!")
        print("   Consider using .cache() with a file path instead of memory")

# Usage
monitor_memory_usage(train_dataset)
```

## GCP-Specific Debugging Tools

### 1. **Vertex AI Data Labeling**
```python
from google.cloud import aiplatform

def check_labeling_quality(dataset_name):
    """Check data labeling quality in Vertex AI"""
    
    aiplatform.init(project='your-project', location='us-central1')
    
    dataset = aiplatform.ImageDataset(dataset_name)
    
    # Get annotation statistics
    annotations = dataset.list_annotations()
    
    print(f"Total annotations: {len(annotations)}")
    # Add quality checks...
```

### 2. **TensorFlow Data Validation (TFDV)**
```python
import tensorflow_data_validation as tfdv

# Generate statistics
train_stats = tfdv.generate_statistics_from_dataframe(train_df)
val_stats = tfdv.generate_statistics_from_dataframe(val_df)

# Visualize
tfdv.visualize_statistics(train_stats)

# Infer schema
schema = tfdv.infer_schema(train_stats)

# Detect anomalies
anomalies = tfdv.validate_statistics(val_stats, schema)
tfdv.display_anomalies(anomalies)

# Detect drift
drift_anomalies = tfdv.validate_statistics(
    statistics=val_stats,
    schema=schema,
    previous_statistics=train_stats
)
tfdv.display_anomalies(drift_anomalies)
```

### 3. **BigQuery Data Profiling**
```python
from google.cloud import bigquery

def profile_bigquery_data(project_id, dataset_id, table_id):
    """Profile data in BigQuery"""
    
    client = bigquery.Client(project=project_id)
    
    # Basic statistics
    query = f"""
    SELECT
        COUNT(*) as total_rows,
        COUNT(DISTINCT label) as unique_labels,
        AVG(feature1) as avg_feature1,
        STDDEV(feature1) as std_feature1,
        MIN(feature1) as min_feature1,
        MAX(feature1) as max_feature1
    FROM `{project_id}.{dataset_id}.{table_id}`
    """
    
    results = client.query(query).to_dataframe()
    print(results)
    
    # Check for nulls
    null_query = f"""
    SELECT
        COUNTIF(label IS NULL) as null_labels,
        COUNTIF(feature1 IS NULL) as null_feature1
    FROM `{project_id}.{dataset_id}.{table_id}`
    """
    
    null_results = client.query(null_query).to_dataframe()
    print(null_results)

# Usage
profile_bigquery_data('my-project', 'my_dataset', 'training_data')
```

## Debugging Checklist

- [ ] **Data Quality**
  - [ ] No missing values (or handled appropriately)
  - [ ] No duplicate records
  - [ ] No corrupted files
  - [ ] Data types are correct
  - [ ] Value ranges are valid

- [ ] **Labels**
  - [ ] Labels are correct and consistent
  - [ ] Label encoding starts from 0
  - [ ] No missing labels
  - [ ] Class distribution is acceptable

- [ ] **Splits**
  - [ ] No data leakage between splits
  - [ ] Class distribution is consistent across splits
  - [ ] Feature distributions are similar
  - [ ] Split ratios are appropriate (e.g., 70/15/15)

- [ ] **Pipeline**
  - [ ] Data loads correctly
  - [ ] Preprocessing is correct
  - [ ] Augmentation works as expected
  - [ ] No NaN/Inf values after preprocessing
  - [ ] Batch shapes are correct
  - [ ] Data types match model expectations

- [ ] **Performance**
  - [ ] Data loading is not a bottleneck
  - [ ] Memory usage is acceptable
  - [ ] Pipeline uses prefetching
  - [ ] Caching is used where appropriate

- [ ] **Distribution**
  - [ ] No class imbalance (or handled)
  - [ ] No outliers (or handled)
  - [ ] Features are properly scaled
  - [ ] No distribution shift between train/test

## Common Debugging Commands

```python
# Quick dataset inspection
for batch in dataset.take(1):
    print(f"Batch shapes: {[x.shape for x in batch]}")
    print(f"Dtypes: {[x.dtype for x in batch]}")
    print(f"Value ranges: {[(x.numpy().min(), x.numpy().max()) for x in batch]}")

# Check for NaN/Inf
tf.debugging.check_numerics(tensor, "checking for NaN/Inf")

# Enable eager execution for debugging
tf.config.run_functions_eagerly(True)

# Set logging level
tf.get_logger().setLevel('INFO')

# Profile with TensorBoard
tf.profiler.experimental.start('logdir')
# ... run training ...
tf.profiler.experimental.stop()
```

## Key Takeaways for GCP Data Engineer Exam

1. **Always inspect data first** - Don't start training blindly
2. **Use TFDV** for automated data validation and drift detection
3. **Check for data leakage** - Critical for model reliability
4. **Visualize samples** - Catch labeling errors early
5. **Profile data pipeline** - Prevent training bottlenecks
6. **Use tf.data best practices** - cache(), prefetch(), parallel loading
7. **Monitor class distribution** - Handle imbalance appropriately
8. **Validate preprocessing** - Ensure normalization/augmentation is correct
9. **Test on small subset first** - Quick iteration during debugging
10. **Use BigQuery/TFDV** for large-scale data profiling on GCP

## Tools Summary

| Tool | Use Case | Scale |
|------|----------|-------|
| **Pandas profiling** | Quick EDA | Small-Medium |
| **TFDV** | Schema validation, drift detection | Large |
| **TensorBoard** | Pipeline profiling | Any |
| **BigQuery** | SQL-based data profiling | Very Large |
| **Vertex AI Data Labeling** | Label quality checks | Any |
| **tf.data.experimental.stats** | Pipeline statistics | Any |