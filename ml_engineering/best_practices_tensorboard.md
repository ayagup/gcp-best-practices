# TensorBoard: Features and Best Practices

## Overview

TensorBoard is TensorFlow's visualization toolkit that provides the visualization and tooling needed for machine learning experimentation. It enables tracking and visualizing metrics such as loss and accuracy, visualizing the model graph, viewing histograms of weights, biases, or other tensors as they change over time, displaying images, text, and audio data, and profiling TensorFlow programs.

## Key Features

### 1. Scalars Dashboard
- **Purpose**: Tracks scalar metrics over time (loss, accuracy, learning rate, etc.)
- **Capabilities**:
  - Line charts showing metric evolution
  - Smoothing controls for noisy data
  - Comparison across multiple runs
  - Custom scalar summaries
  - Regular expression filtering

### 2. Graphs Dashboard
- **Purpose**: Visualizes computational graph structure
- **Capabilities**:
  - Model architecture visualization
  - Op-level graph exploration
  - Conceptual graph view
  - Node statistics and metadata
  - Device placement information

### 3. Distributions and Histograms
- **Purpose**: Shows how tensor distributions change over time
- **Capabilities**:
  - Weight and bias distributions
  - Activation distributions
  - Gradient flow analysis
  - Histogram mode and distribution overlay
  - Time-series view of distributions

### 4. Images Dashboard
- **Purpose**: Displays image data throughout training
- **Capabilities**:
  - Input image visualization
  - Feature map visualization
  - Generated image tracking (GANs, autoencoders)
  - Image augmentation validation
  - Multiple images per step

### 5. Audio Dashboard
- **Purpose**: Embeds playable audio widgets
- **Capabilities**:
  - Audio sample playback
  - Waveform visualization
  - Multiple audio formats support
  - Training progress audio tracking

### 6. Text Dashboard
- **Purpose**: Displays text data and sequences
- **Capabilities**:
  - Generated text samples
  - Text classification results
  - Markdown rendering
  - Sequence-to-sequence outputs

### 7. PR Curves Dashboard
- **Purpose**: Precision-Recall curve visualization
- **Capabilities**:
  - Binary and multi-class classification metrics
  - Threshold optimization
  - F1 score visualization
  - ROC curve comparison

### 8. Profiler
- **Purpose**: Performance analysis and optimization
- **Capabilities**:
  - Overview page with performance summary
  - Trace viewer for detailed timeline
  - GPU kernel statistics
  - Memory profile analysis
  - TensorFlow op profile
  - Input pipeline analysis
  - Recommendation engine for optimization

### 9. HParams Dashboard
- **Purpose**: Hyperparameter tuning visualization
- **Capabilities**:
  - Parallel coordinates view
  - Scatter plot matrix
  - Table view of all experiments
  - Filtering and sorting
  - Metric comparison across hyperparameters

### 10. Embeddings Projector
- **Purpose**: Interactive visualization of high-dimensional data
- **Capabilities**:
  - PCA (Principal Component Analysis)
  - t-SNE visualization
  - UMAP support
  - Custom projection
  - Nearest neighbor search
  - Metadata filtering

### 11. What-If Tool
- **Purpose**: Model understanding and fairness analysis
- **Capabilities**:
  - Interactive model probing
  - Feature attribution
  - Fairness metrics
  - Counterfactual analysis
  - Partial dependence plots

### 12. Mesh Dashboard
- **Purpose**: 3D mesh and point cloud visualization
- **Capabilities**:
  - 3D object visualization
  - Point cloud rendering
  - Animation over time
  - Multiple mesh formats

## Google Cloud Integration

### Vertex AI TensorBoard

**Managed TensorBoard Service**:
- Fully managed and scalable
- Enterprise-grade security
- Persistent storage in Cloud Storage
- Shared experiments across teams
- Integration with Vertex AI Training
- Integration with Vertex AI Pipelines

**Key Features**:
- One-click TensorBoard deployment
- Multi-user collaboration
- Automatic experiment tracking
- Region-specific instances
- IAM-based access control
- VPC Service Controls support

**Creating Vertex AI TensorBoard**:
```python
from google.cloud import aiplatform

aiplatform.init(project='your-project-id', location='us-central1')

tensorboard = aiplatform.Tensorboard.create(
    display_name='my-tensorboard',
    description='Team experiment tracking'
)
```

## Best Practices

### 1. Logging Strategy

**Use Hierarchical Naming**:
```python
# Good practice
tf.summary.scalar('train/loss', loss, step=epoch)
tf.summary.scalar('train/accuracy', accuracy, step=epoch)
tf.summary.scalar('validation/loss', val_loss, step=epoch)
tf.summary.scalar('validation/accuracy', val_accuracy, step=epoch)

# Avoid flat structure
tf.summary.scalar('loss', loss, step=epoch)
```

**Log at Appropriate Frequencies**:
```python
# Log training metrics every N steps
if global_step % log_frequency == 0:
    tf.summary.scalar('train/loss', loss, step=global_step)

# Log validation metrics every epoch
if step % steps_per_epoch == 0:
    tf.summary.scalar('validation/accuracy', val_acc, step=epoch)

# Log expensive visualizations less frequently
if global_step % (log_frequency * 10) == 0:
    tf.summary.image('predictions', images, step=global_step)
```

### 2. Experiment Organization

**Use Clear Naming Conventions**:
```python
import datetime

# Include key hyperparameters in run name
run_name = f"lr{learning_rate}_bs{batch_size}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
log_dir = f"logs/{run_name}"
```

**Organize by Experiment Type**:
```
logs/
├── baseline/
│   ├── run_001/
│   └── run_002/
├── architecture_search/
│   ├── resnet50/
│   └── efficientnet/
└── hyperparameter_tuning/
    ├── lr_0.001/
    └── lr_0.0001/
```

### 3. Performance Optimization

**Minimize Logging Overhead**:
```python
# Use conditional logging
if step % 100 == 0:
    with tf.summary.record_if(True):
        tf.summary.scalar('metrics/loss', loss, step=step)
else:
    with tf.summary.record_if(False):
        pass  # Skip expensive summary operations
```

**Batch Summary Operations**:
```python
# Log multiple metrics together
with tf.summary.record_if(should_log):
    tf.summary.scalar('train/loss', loss, step=step)
    tf.summary.scalar('train/accuracy', accuracy, step=step)
    tf.summary.scalar('train/learning_rate', lr, step=step)
```

**Use Profiler Strategically**:
```python
# Profile only a few steps
tf.profiler.experimental.start('logs/profiler')
for step in range(start_step, start_step + profile_steps):
    train_step()
tf.profiler.experimental.stop()
```

### 4. Visualization Best Practices

**Image Visualization**:
```python
# Limit number of images to avoid memory issues
max_images = 4
tf.summary.image(
    'predictions',
    images[:max_images],
    step=step,
    max_outputs=max_images
)

# Normalize images for better visualization
normalized_images = (images - images.min()) / (images.max() - images.min())
tf.summary.image('normalized_inputs', normalized_images, step=step)
```

**Histogram Logging**:
```python
# Log weight distributions
for layer in model.layers:
    if hasattr(layer, 'kernel'):
        tf.summary.histogram(
            f'weights/{layer.name}/kernel',
            layer.kernel,
            step=step
        )
    if hasattr(layer, 'bias') and layer.bias is not None:
        tf.summary.histogram(
            f'weights/{layer.name}/bias',
            layer.bias,
            step=step
        )
```

**Gradient Monitoring**:
```python
with tf.GradientTape() as tape:
    predictions = model(inputs, training=True)
    loss = loss_fn(labels, predictions)

gradients = tape.gradient(loss, model.trainable_variables)

# Log gradient norms
for grad, var in zip(gradients, model.trainable_variables):
    if grad is not None:
        tf.summary.histogram(
            f'gradients/{var.name}',
            grad,
            step=step
        )
        tf.summary.scalar(
            f'gradient_norm/{var.name}',
            tf.norm(grad),
            step=step
        )
```

### 5. Hyperparameter Tracking

**Using HParams Plugin**:
```python
from tensorboard.plugins.hparams import api as hp

# Define hyperparameters
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.0001, 0.01))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32, 64, 128]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))

# Define metrics
METRIC_ACCURACY = hp.Metric('accuracy', display_name='Accuracy')
METRIC_LOSS = hp.Metric('loss', display_name='Loss')

# Log experiment configuration
with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_LEARNING_RATE, HP_BATCH_SIZE, HP_OPTIMIZER],
        metrics=[METRIC_ACCURACY, METRIC_LOSS]
    )

# Log each trial
def run_trial(hparams, trial_dir):
    with tf.summary.create_file_writer(trial_dir).as_default():
        hp.hparams(hparams)
        accuracy = train_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
```

### 6. Model Comparison

**Compare Multiple Runs**:
```python
# Create separate log directories for each configuration
configs = [
    {'lr': 0.001, 'optimizer': 'adam'},
    {'lr': 0.01, 'optimizer': 'adam'},
    {'lr': 0.001, 'optimizer': 'sgd'},
]

for i, config in enumerate(configs):
    log_dir = f"logs/comparison/run_{i}_lr{config['lr']}_opt{config['optimizer']}"
    # Train with this configuration and log to log_dir
```

**Use Tags for Filtering**:
```python
# Add tags to differentiate experiment types
tf.summary.text(
    'experiment_info',
    f"Type: baseline\nDate: {datetime.now()}\nDescription: Initial model",
    step=0
)
```

### 7. Debugging Techniques

**Track Model Layers**:
```python
# Monitor layer activations
class DebugCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir)
    
    def on_epoch_end(self, epoch, logs=None):
        with self.file_writer.as_default():
            for layer in self.model.layers:
                if hasattr(layer, 'kernel'):
                    weights = layer.kernel
                    tf.summary.histogram(
                        f'layer_weights/{layer.name}',
                        weights,
                        step=epoch
                    )
```

**Detect Training Issues**:
```python
# Monitor for NaN and Inf values
def check_numerics(tensors, step):
    for name, tensor in tensors.items():
        has_nan = tf.reduce_any(tf.math.is_nan(tensor))
        has_inf = tf.reduce_any(tf.math.is_inf(tensor))
        
        tf.summary.scalar(f'debug/{name}/has_nan', 
                         tf.cast(has_nan, tf.float32), step=step)
        tf.summary.scalar(f'debug/{name}/has_inf',
                         tf.cast(has_inf, tf.float32), step=step)
```

### 8. Production Deployment

**Conditional Logging for Production**:
```python
import os

ENABLE_TENSORBOARD = os.getenv('ENABLE_TENSORBOARD', 'false').lower() == 'true'

if ENABLE_TENSORBOARD:
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        update_freq='epoch'
    )
    callbacks.append(tensorboard_callback)
```

**Secure Access**:
```python
# For Vertex AI TensorBoard
from google.cloud import aiplatform

# Use IAM for access control
tensorboard = aiplatform.Tensorboard(
    tensorboard_name='projects/PROJECT_ID/locations/LOCATION/tensorboards/TENSORBOARD_ID'
)

# Grant access to team members via IAM
# roles/aiplatform.tensorboardUser - View access
# roles/aiplatform.tensorboardEditor - Edit access
```

### 9. Custom Visualizations

**Custom Scalars Layout**:
```python
from tensorboard.plugins.custom_scalar import layout_pb2

# Create custom dashboard layout
layout_summary = layout_pb2.Layout(
    category=[
        layout_pb2.Category(
            title='Training Metrics',
            chart=[
                layout_pb2.Chart(
                    title='Loss Comparison',
                    multiline=layout_pb2.MultilineChartContent(
                        tag=[r'train/loss', r'validation/loss']
                    )
                ),
                layout_pb2.Chart(
                    title='Accuracy Comparison',
                    multiline=layout_pb2.MultilineChartContent(
                        tag=[r'train/accuracy', r'validation/accuracy']
                    )
                )
            ]
        )
    ]
)
```

**Custom Plugin Development**:
```python
# Create custom TensorBoard plugin for specific visualizations
# Useful for domain-specific metrics or visualizations
```

### 10. Integration with Training Frameworks

**Keras Integration**:
```python
# Built-in TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,  # Log histograms every epoch
    write_graph=True,
    write_images=True,
    update_freq='epoch',  # or 'batch' or integer
    profile_batch='10,20',  # Profile batches 10-20
    embeddings_freq=1
)

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    callbacks=[tensorboard_callback]
)
```

**PyTorch Integration**:
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=log_dir)

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Training step
        output = model(data)
        loss = criterion(output, target)
        
        # Log metrics
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('train/loss', loss.item(), global_step)
        
    # Log learning rate
    writer.add_scalar('train/learning_rate', 
                     optimizer.param_groups[0]['lr'], epoch)
    
    # Log model graph (once)
    if epoch == 0:
        writer.add_graph(model, data)

writer.close()
```

**Vertex AI Training Integration**:
```python
from google.cloud import aiplatform

# Create Vertex AI TensorBoard instance
tensorboard = aiplatform.Tensorboard.create(
    display_name='vertex-training-tensorboard'
)

# Create training job with TensorBoard
job = aiplatform.CustomTrainingJob(
    display_name='training-with-tensorboard',
    script_path='trainer/task.py',
    container_uri='gcr.io/cloud-aiplatform/training/tf-cpu.2-8:latest',
    requirements=['tensorflow==2.8.0'],
    model_serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-8:latest'
)

model = job.run(
    replica_count=1,
    machine_type='n1-standard-4',
    tensorboard=tensorboard.resource_name,
    service_account='your-service-account@project.iam.gserviceaccount.com'
)
```

## Common Patterns and Use Cases

### 1. Computer Vision

```python
# Log sample predictions
def log_predictions(model, val_data, step, num_samples=4):
    images, labels = next(iter(val_data))
    predictions = model.predict(images[:num_samples])
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples*3))
    for i in range(num_samples):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f'True: {labels[i]}')
        axes[i, 1].imshow(images[i])
        axes[i, 1].set_title(f'Pred: {predictions[i].argmax()}')
    
    # Log to TensorBoard
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    
    tf.summary.image('predictions/comparison', image, step=step)
```

### 2. Natural Language Processing

```python
# Log attention weights
def log_attention(attention_weights, tokens, step):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attention_weights, cmap='viridis')
    
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    
    plt.colorbar(im, ax=ax)
    
    # Convert to image and log
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    
    tf.summary.image('attention/weights', image, step=step)
```

### 3. Reinforcement Learning

```python
# Log episode metrics
def log_episode(episode_reward, episode_length, epsilon, step):
    tf.summary.scalar('rl/episode_reward', episode_reward, step=step)
    tf.summary.scalar('rl/episode_length', episode_length, step=step)
    tf.summary.scalar('rl/epsilon', epsilon, step=step)
    
    # Log Q-value distributions
    if step % 100 == 0:
        tf.summary.histogram('rl/q_values', q_values, step=step)
```

### 4. Generative Models

```python
# Log generated samples
def log_generated_images(generator, epoch, noise_dim=100, num_examples=16):
    noise = tf.random.normal([num_examples, noise_dim])
    generated_images = generator(noise, training=False)
    
    # Arrange in grid
    grid_size = int(np.sqrt(num_examples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i, :, :, 0], cmap='gray')
        ax.axis('off')
    
    # Log to TensorBoard
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    
    tf.summary.image('generated/samples', image, step=epoch)
```

## Troubleshooting

### Common Issues

**1. TensorBoard Not Showing Data**:
```python
# Ensure writer is properly flushed
writer.flush()

# Or use context manager
with tf.summary.create_file_writer(log_dir).as_default():
    tf.summary.scalar('loss', loss, step=step)
```

**2. High Memory Usage**:
```python
# Reduce histogram logging frequency
# Log fewer images per summary
# Use lower image resolution
# Clear Keras backend periodically
import gc
if step % 1000 == 0:
    gc.collect()
    tf.keras.backend.clear_session()
```

**3. Slow Training with TensorBoard**:
```python
# Reduce update frequency
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    update_freq=1000  # Update every 1000 batches instead of every batch
)

# Disable profiling after initial run
# Disable histogram logging for large models
```

**4. Port Already in Use**:
```bash
# Use different port
tensorboard --logdir=logs --port=6007

# Kill existing TensorBoard process (Windows PowerShell)
Get-Process -Name tensorboard | Stop-Process -Force
```

**5. Graph Not Showing**:
```python
# Ensure graph is logged
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    return loss

# Log the graph
tf.summary.trace_on(graph=True, profiler=False)
train_step(sample_x, sample_y)
with writer.as_default():
    tf.summary.trace_export(name="model_trace", step=0)
```

## Performance Benchmarking

### Profiling Best Practices

```python
# Profile a specific training range
from tensorflow.python.profiler import profiler_v2 as profiler

logdir = 'logs/profiler'
tf.profiler.experimental.start(logdir)

# Profile for a few steps
for step in range(start_step, start_step + 100):
    train_step()
    
    # Collect trace after warmup
    if step == start_step + 10:
        tf.profiler.experimental.start(logdir)
    elif step == start_step + 20:
        tf.profiler.experimental.stop()

tf.profiler.experimental.stop()
```

### Analyzing Profiler Output

1. **Overview Page**: Check GPU utilization and identify bottlenecks
2. **Trace Viewer**: Analyze timeline of operations
3. **Op Profile**: Identify expensive operations
4. **Input Pipeline**: Optimize data loading
5. **Memory Profile**: Track memory allocation and usage

## Security and Compliance

### Access Control

```python
# For Vertex AI TensorBoard
# Use IAM roles:
# - roles/aiplatform.tensorboardViewer: Read-only access
# - roles/aiplatform.tensorboardUser: View and experiment access
# - roles/aiplatform.tensorboardEditor: Full access

# Set up VPC Service Controls for data isolation
# Enable audit logging for compliance
```

### Data Privacy

```python
# Avoid logging sensitive data
# Use differential privacy techniques if needed
# Sanitize text and image data before logging

def sanitize_text(text):
    # Remove PII before logging
    sanitized = remove_emails(text)
    sanitized = remove_phone_numbers(sanitized)
    return sanitized

tf.summary.text('examples', sanitize_text(sample_text), step=step)
```

## Advanced Features

### 1. What-If Tool Integration

```python
# Analyze model fairness and performance
from witwidget.notebook.visualization import WitWidget, WitConfigBuilder

config_builder = WitConfigBuilder(
    test_examples,
    feature_names
).set_model_name('my_model').set_target_feature('label')

WitWidget(config_builder, height=800)
```

### 2. Embeddings Visualization

```python
# Log embeddings for visualization
from tensorboard.plugins import projector

# Save embeddings
embedding_var = tf.Variable(embeddings, name='embeddings')
checkpoint = tf.train.Checkpoint(embedding=embedding_var)
checkpoint.save(os.path.join(log_dir, 'embedding.ckpt'))

# Configure projector
config = projector.ProjectorConfig()
embedding_config = config.embeddings.add()
embedding_config.tensor_name = 'embedding/.ATTRIBUTES/VARIABLE_VALUE'
embedding_config.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)
```

### 3. Multi-Worker Training

```python
# Log from chief worker only
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    # Create model
    model = create_model()
    
# Log only from chief worker
if strategy.cluster_resolver.task_type == 'chief':
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
else:
    tensorboard_callback = None

callbacks = [c for c in [tensorboard_callback] if c is not None]
model.fit(dataset, callbacks=callbacks)
```

## Cost Optimization

### Vertex AI TensorBoard

**Best Practices for Cost**:
- Delete unused TensorBoard instances
- Use appropriate regions to minimize data transfer costs
- Archive old experiments to Cloud Storage
- Use lifecycle policies for automatic cleanup
- Monitor usage with Cloud Monitoring

```python
# Delete old TensorBoard instance
tensorboard.delete()

# Set up lifecycle policy for old logs
from google.cloud import storage

bucket = storage.Client().bucket('tensorboard-logs')
rule = storage.lifecycle.LifecycleRuleDelete(age=90)  # Delete after 90 days
bucket.add_lifecycle_delete_rule(age=90)
bucket.patch()
```

## Conclusion

TensorBoard is an essential tool for ML experimentation and model development. By following these best practices:

1. **Organize experiments systematically** with clear naming and hierarchical structure
2. **Log strategically** to balance detail with performance
3. **Use appropriate visualizations** for different data types
4. **Leverage hyperparameter tracking** for systematic tuning
5. **Profile performance** to identify bottlenecks
6. **Integrate with Google Cloud** for team collaboration and scalability
7. **Implement security measures** for production deployments
8. **Optimize costs** through efficient resource management

TensorBoard, especially when integrated with Vertex AI, provides a powerful platform for ML experimentation, model understanding, and team collaboration in Google Cloud environments.

## Additional Resources

- [TensorBoard Official Documentation](https://www.tensorflow.org/tensorboard)
- [Vertex AI TensorBoard Documentation](https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-overview)
- [TensorBoard GitHub Repository](https://github.com/tensorflow/tensorboard)
- [TensorBoard Profiler Guide](https://www.tensorflow.org/guide/profiler)
- [Vertex AI Training Integration](https://cloud.google.com/vertex-ai/docs/training/overview)
